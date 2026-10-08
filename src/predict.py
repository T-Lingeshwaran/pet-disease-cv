"""
PET HEALTH AI
Prediction + Grad-CAM

Pipeline:

                INPUT IMAGE
                     |
                     v
             MODALITY CLASSIFIER
                /          \
               /            \
            EYE              SKIN
             |                |
             v                v
          CSF-Net       EfficientNetV2-S
             |                |
             v                v
       Eye condition    Skin condition
             \                /
              \              /
               v            v
                  Grad-CAM

Usage:
    python src/predict.py "path/to/image.jpg"
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.models import mobilenet_v3_small


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODALITY_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "modality"
    / "best_model.pth"
)

SKIN_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "skin_efficientnet_v2s"
    / "best_model.pth"
)

EYE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "eye_csfnet"
    / "best_csfnet.pth"
)

GRADCAM_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam"
)

SKIN_GRADCAM_DIR = (
    GRADCAM_DIR
    / "skin"
)

EYE_GRADCAM_DIR = (
    GRADCAM_DIR
    / "eye"
)

SKIN_GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EYE_GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

MODALITY_IMAGE_SIZE = 224

SKIN_IMAGE_SIZE = 384

EYE_IMAGE_SIZE = 224


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).view(1, 3, 1, 1)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def load_image(image_path):
    """Load an image as RGB."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    try:
        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as exc:
        raise RuntimeError(
            f"Could not open image:\n"
            f"{image_path}\n\n"
            f"{exc}"
        )

    return image


def load_checkpoint(path):
    """Load a PyTorch checkpoint."""

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    return torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )


def clean_state_dict(state_dict):
    """
    Remove DataParallel 'module.' prefix
    if present.
    """

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    return cleaned


def get_state_dict(checkpoint):
    """
    Extract state_dict from common checkpoint formats.
    """

    if not isinstance(
        checkpoint,
        dict
    ):
        raise RuntimeError(
            "Invalid checkpoint format."
        )

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    # Check whether checkpoint itself is a state dict.
    if all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Could not find model_state_dict "
        "or state_dict in checkpoint."
    )


def get_classes(
    checkpoint,
    fallback_classes
):
    """
    Get class names saved inside checkpoint.
    """

    if isinstance(
        checkpoint,
        dict
    ):

        for key in (
            "class_names",
            "classes",
        ):

            if key in checkpoint:

                value = checkpoint[key]

                if isinstance(
                    value,
                    (list, tuple)
                ):
                    return list(value)

    return list(
        fallback_classes
    )


def resize_tensor(
    image,
    size
):
    """
    PIL RGB image -> [1,3,H,W] tensor.

    Values remain in [0,1].
    """

    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    return tensor


def normalize_imagenet(
    tensor
):
    """ImageNet normalization."""

    mean = IMAGENET_MEAN.to(
        tensor.device
    )

    std = IMAGENET_STD.to(
        tensor.device
    )

    return (
        tensor - mean
    ) / std


# ============================================================
# COLOR CONVERSION
# ============================================================

def rgb_to_hsv_tensor(
    rgb
):
    """
    RGB -> HSV.

    Input:
        [B,3,H,W], values [0,1]

    Output:
        [B,3,H,W]
    """

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    maxc = torch.max(
        rgb,
        dim=1
    ).values

    minc = torch.min(
        rgb,
        dim=1
    ).values

    delta = maxc - minc

    h = torch.zeros_like(
        maxc
    )

    mask = delta > 1e-6

    mask_r = (
        mask
        & (maxc == r)
    )

    h[mask_r] = (
        (
            g[mask_r] - b[mask_r]
        )
        / delta[mask_r]
    ) % 6.0

    mask_g = (
        mask
        & (maxc == g)
    )

    h[mask_g] = (
        (
            b[mask_g] - r[mask_g]
        )
        / delta[mask_g]
        + 2.0
    )

    mask_b = (
        mask
        & (maxc == b)
    )

    h[mask_b] = (
        (
            r[mask_b] - g[mask_b]
        )
        / delta[mask_b]
        + 4.0
    )

    h = h / 6.0

    s = torch.zeros_like(
        maxc
    )

    nonzero = maxc > 1e-6

    s[nonzero] = (
        delta[nonzero]
        / maxc[nonzero]
    )

    v = maxc

    return torch.stack(
        [h, s, v],
        dim=1
    )


def rgb_to_ycbcr_tensor(
    rgb
):
    """
    RGB -> YCbCr.

    Input:
        [B,3,H,W], values [0,1]

    Output:
        [B,3,H,W]
    """

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    y = (
        0.299 * r
        + 0.587 * g
        + 0.114 * b
    )

    cb = (
        -0.168736 * r
        -0.331264 * g
        + 0.500000 * b
        + 0.5
    )

    cr = (
        0.500000 * r
        -0.418688 * g
        -0.081312 * b
        + 0.5
    )

    return torch.stack(
        [y, cb, cr],
        dim=1
    ).clamp(
        0,
        1
    )


# ============================================================
# MODALITY MODEL
# ============================================================

def build_modality_model():
    """
    EfficientNet-B0 binary classifier.

    Expected classes:
        eye
        skin
    """

    model = models.efficientnet_b0(
        weights=None
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        2
    )

    return model


def predict_modality(
    image
):
    """
    Detect whether image is eye or skin.
    """

    print(
        "\nLoading modality classifier..."
    )

    checkpoint = load_checkpoint(
        MODALITY_CHECKPOINT
    )

    model = build_modality_model()

    state_dict = clean_state_dict(
        get_state_dict(checkpoint)
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    tensor = resize_tensor(
        image,
        MODALITY_IMAGE_SIZE
    )

    tensor = normalize_imagenet(
        tensor
    )

    tensor = tensor.to(
        DEVICE
    )

    # Modality does not need gradients.
    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    classes = get_classes(
        checkpoint,
        [
            "eye",
            "skin"
        ]
    )

    predicted_index = int(
        probabilities.argmax().item()
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )

    class_lower = (
        predicted_class
        .strip()
        .lower()
    )

    if "skin" in class_lower:

        modality = "skin"

    elif "eye" in class_lower:

        modality = "eye"

    else:

        modality = (
            "skin"
            if predicted_index == 1
            else "eye"
        )

    return (
        modality,
        confidence
    )


# ============================================================
# SKIN MODEL
# ============================================================

def build_skin_model(
    num_classes
):
    """
    EfficientNetV2-S skin classifier.
    """

    model = models.efficientnet_v2_s(
        weights=None
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(
        nn.Dropout(
            p=0.3
        ),
        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model


# ============================================================
# ROBUST GRAD-CAM
# ============================================================

class GradCAM:
    """
    Grad-CAM implementation.

    IMPORTANT:

    The GradCAM object MUST be created BEFORE
    the model forward pass.

    The forward hook captures activations during
    the forward pass.

    A tensor hook captures the gradient during
    backward().
    """

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = (
            target_layer
        )

        self.activations = None

        self.gradients = None

        # Hook is installed immediately.
        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

        # This is the important part.
        #
        # We attach the gradient hook to the
        # actual activation tensor.

        if (
            isinstance(
                output,
                torch.Tensor
            )
            and output.requires_grad
        ):

            output.register_hook(
                self._save_gradient
            )

    def _save_gradient(
        self,
        gradient
    ):

        self.gradients = gradient

    def remove_hooks(
        self
    ):

        if (
            self.forward_handle
            is not None
        ):

            self.forward_handle.remove()

            self.forward_handle = None

    def generate(
        self,
        output,
        class_index,
        output_size
    ):
        """
        Generate Grad-CAM.

        output:
            model logits [B,C]

        class_index:
            predicted class index

        output_size:
            (height,width)
        """

        # ----------------------------------------------------
        # The forward hook MUST have captured activation.
        # ----------------------------------------------------

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activation was not captured. "
                "The GradCAM hook must be created BEFORE "
                "the model forward pass."
            )

        # Clear old gradient.
        self.gradients = None

        # Clear model gradients.
        self.model.zero_grad(
            set_to_none=True
        )

        # Selected class score.
        score = output[
            :,
            class_index
        ].sum()

        # Backpropagation.
        score.backward()

        # ----------------------------------------------------
        # Gradient should now exist.
        # ----------------------------------------------------

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradient was not captured. "
                "The target activation did not receive "
                "a gradient."
            )

        activations = (
            self.activations
        )

        gradients = (
            self.gradients
        )

        # ----------------------------------------------------
        # Grad-CAM weights
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted feature maps
        # ----------------------------------------------------

        cam = (
            weights
            * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Positive influence.
        cam = F.relu(
            cam
        )

        # ----------------------------------------------------
        # Resize CAM
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        cam = cam[
            0,
            0
        ]

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Normalize 0..1
        # ----------------------------------------------------

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 1e-8:

            cam /= maximum

        return cam


# ============================================================
# SKIN PREDICTION
# ============================================================

def predict_skin(
    image,
    top_k=3
):
    """
    Predict skin condition.

    IMPORTANT:
    GradCAM is created BEFORE model(image).
    """

    print(
        "\nLoading EfficientNetV2-S skin model..."
    )

    checkpoint = load_checkpoint(
        SKIN_CHECKPOINT
    )

    fallback_classes = [
        "Bacterial_dermatosis",
        "demodicosis",
        "Dermatitis",
        "Flea Allergy",
        "Fungal_infections",
        "hotspot",
        "Hypersensitivity",
        "Hypersensitivity_allergic_dermatosis",
        "mange",
        "ringworm",
        "Scabies",
    ]

    classes = get_classes(
        checkpoint,
        fallback_classes
    )

    model = build_skin_model(
        len(classes)
    )

    state_dict = clean_state_dict(
        get_state_dict(checkpoint)
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    tensor = resize_tensor(
        image,
        SKIN_IMAGE_SIZE
    )

    tensor = normalize_imagenet(
        tensor
    )

    tensor = tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Install Grad-CAM BEFORE forward.
    # --------------------------------------------------------

    target_layer = (
        model.features[-1]
    )

    cam_generator = GradCAM(
        model,
        target_layer
    )

    try:

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        top_k = min(
            top_k,
            len(classes)
        )

        values, indices = torch.topk(
            probabilities,
            k=top_k
        )

        predictions = []

        for value, index in zip(
            values.detach().cpu(),
            indices.detach().cpu()
        ):

            index = int(
                index.item()
            )

            predictions.append({
                "index": index,
                "class": classes[index],
                "confidence": float(
                    value.item()
                )
            })

        target_class = int(
            indices[0].item()
        )

        # DO NOT remove the hook here.
        #
        # It is needed for Grad-CAM after this function.

        return (
            model,
            tensor,
            logits,
            predictions,
            target_class,
            cam_generator
        )

    except Exception:

        cam_generator.remove_hooks()

        raise


# ============================================================
# EYE MODEL
# ============================================================

class MobileNetFeatureExtractor(
    nn.Module
):
    """
    MobileNetV3-Small feature extractor.
    """

    def __init__(
        self
    ):

        super().__init__()

        backbone = mobilenet_v3_small(
            weights=None
        )

        self.features = (
            backbone.features
        )

        self.avgpool = (
            backbone.avgpool
        )

        self.projection = (
            backbone.classifier[0]
        )

        self.output_dim = 1024

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.avgpool(
            x
        )

        x = torch.flatten(
            x,
            1
        )

        x = self.projection(
            x
        )

        return x


class CSFNet(
    nn.Module
):
    """
    CSF-Net-inspired eye model.

    Three branches:

        RGB
        HSV
        YCbCr

    Feature fusion:

        MultiheadAttention
        LayerNorm
        Mean fusion
        MLP classifier
    """

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.rgb_branch = (
            MobileNetFeatureExtractor()
        )

        self.hsv_branch = (
            MobileNetFeatureExtractor()
        )

        self.ycbcr_branch = (
            MobileNetFeatureExtractor()
        )

        feature_dim = 1024

        self.attention = (
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=1,
                batch_first=True
            )
        )

        self.norm = nn.LayerNorm(
            feature_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                512
            ),

            nn.BatchNorm1d(
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.35
            ),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(
        self,
        rgb,
        hsv,
        ycbcr
    ):

        rgb_features = (
            self.rgb_branch(
                rgb
            )
        )

        hsv_features = (
            self.hsv_branch(
                hsv
            )
        )

        ycbcr_features = (
            self.ycbcr_branch(
                ycbcr
            )
        )

        tokens = torch.stack(
            [
                rgb_features,
                hsv_features,
                ycbcr_features
            ],
            dim=1
        )

        attended, _ = (
            self.attention(
                tokens,
                tokens,
                tokens
            )
        )

        attended = self.norm(
            attended + tokens
        )

        fused = attended.mean(
            dim=1
        )

        logits = self.classifier(
            fused
        )

        return logits


# ============================================================
# EYE INPUTS
# ============================================================

def prepare_eye_inputs(
    image
):
    """
    Prepare RGB, HSV and YCbCr tensors.
    """

    rgb = resize_tensor(
        image,
        EYE_IMAGE_SIZE
    )

    hsv = rgb_to_hsv_tensor(
        rgb
    )

    ycbcr = rgb_to_ycbcr_tensor(
        rgb
    )

    rgb = normalize_imagenet(
        rgb
    )

    rgb = rgb.to(
        DEVICE
    )

    hsv = hsv.to(
        DEVICE
    )

    ycbcr = ycbcr.to(
        DEVICE
    )

    return (
        rgb,
        hsv,
        ycbcr
    )


# ============================================================
# EYE GRAD-CAM
# ============================================================

class EyeGradCAM:
    """
    Grad-CAM for all three eye branches.

    The three GradCAM hooks are created BEFORE
    the eye model forward pass.
    """

    def __init__(
        self,
        model
    ):

        self.model = model

        self.rgb_cam = GradCAM(
            model,
            model.rgb_branch.features[-1]
        )

        self.hsv_cam = GradCAM(
            model,
            model.hsv_branch.features[-1]
        )

        self.ycbcr_cam = GradCAM(
            model,
            model.ycbcr_branch.features[-1]
        )

    def generate(
        self,
        logits,
        class_index,
        output_size
    ):
        """
        Backpropagate selected class and
        combine RGB/HSV/YCbCr CAMs.
        """

        # Check activations first.
        generators = [
            self.rgb_cam,
            self.hsv_cam,
            self.ycbcr_cam
        ]

        for generator in generators:

            if generator.activations is None:

                raise RuntimeError(
                    "Eye Grad-CAM activation was not "
                    "captured. Make sure EyeGradCAM was "
                    "created before the model forward pass."
                )

        self.model.zero_grad(
            set_to_none=True
        )

        score = logits[
            :,
            class_index
        ].sum()

        score.backward()

        cams = []

        for generator in generators:

            if generator.gradients is None:

                continue

            activations = (
                generator.activations
            )

            gradients = (
                generator.gradients
            )

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            cam = (
                weights
                * activations
            ).sum(
                dim=1,
                keepdim=True
            )

            cam = F.relu(
                cam
            )

            cam = F.interpolate(
                cam,
                size=output_size,
                mode="bilinear",
                align_corners=False
            )

            cam = cam[
                0,
                0
            ]

            cam = (
                cam
                .detach()
                .cpu()
                .numpy()
            )

            cam -= cam.min()

            maximum = cam.max()

            if maximum > 1e-8:

                cam /= maximum

            cams.append(
                cam
            )

        if len(cams) == 0:

            raise RuntimeError(
                "No eye Grad-CAM branch produced "
                "a gradient."
            )

        combined = np.mean(
            np.stack(cams),
            axis=0
        )

        combined -= combined.min()

        maximum = combined.max()

        if maximum > 1e-8:

            combined /= maximum

        return combined

    def remove_hooks(
        self
    ):

        self.rgb_cam.remove_hooks()

        self.hsv_cam.remove_hooks()

        self.ycbcr_cam.remove_hooks()


# ============================================================
# EYE PREDICTION
# ============================================================

def predict_eye(
    image,
    top_k=3
):
    """
    Predict eye condition.

    Grad-CAM hooks are created BEFORE
    the forward pass.
    """

    print(
        "\nLoading CSF-Net eye model..."
    )

    checkpoint = load_checkpoint(
        EYE_CHECKPOINT
    )

    fallback_classes = [
        "Blepharitis",
        "Cataract",
        "Cherry_Eye",
        "Conjunctivitis",
        "Corneal_Sequestrum",
        "Corneal_Ulcer",
        "Glaucoma",
        "Health",
        "Non_ulcerative",
    ]

    classes = get_classes(
        checkpoint,
        fallback_classes
    )

    model = CSFNet(
        len(classes)
    )

    state_dict = clean_state_dict(
        get_state_dict(checkpoint)
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    (
        rgb,
        hsv,
        ycbcr
    ) = prepare_eye_inputs(
        image
    )

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Create all three hooks BEFORE forward.
    # --------------------------------------------------------

    cam_generator = EyeGradCAM(
        model
    )

    try:

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            rgb,
            hsv,
            ycbcr
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        top_k = min(
            top_k,
            len(classes)
        )

        values, indices = torch.topk(
            probabilities,
            k=top_k
        )

        predictions = []

        for value, index in zip(
            values.detach().cpu(),
            indices.detach().cpu()
        ):

            index = int(
                index.item()
            )

            predictions.append({
                "index": index,
                "class": classes[index],
                "confidence": float(
                    value.item()
                )
            })

        target_class = int(
            indices[0].item()
        )

        return (
            model,
            (
                rgb,
                hsv,
                ycbcr
            ),
            logits,
            predictions,
            target_class,
            cam_generator
        )

    except Exception:

        cam_generator.remove_hooks()

        raise


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    cam,
    output_path,
    title
):
    """
    Save:
        ORIGINAL | GRAD-CAM

    side by side for easier interpretation.
    """

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original = np.asarray(
        image.convert("RGB")
    )

    original_height = original.shape[0]
    original_width = original.shape[1]

    # --------------------------------------------------------
    # Resize CAM to original image size
    # --------------------------------------------------------

    cam = cv2.resize(
        cam,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Convert CAM to heatmap
    # --------------------------------------------------------

    cam_uint8 = np.uint8(
        255
        * np.clip(
            cam,
            0,
            1
        )
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Create Grad-CAM overlay
    # --------------------------------------------------------

    original_float = original.astype(
        np.float32
    )

    heatmap_float = heatmap.astype(
        np.float32
    )

    overlay = (
        0.55 * original_float
        + 0.45 * heatmap_float
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Convert both images to BGR for OpenCV
    # --------------------------------------------------------

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    overlay_bgr = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Add labels
    # --------------------------------------------------------

    label_height = 60

    original_panel = cv2.copyMakeBorder(
        original_bgr,
        label_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    overlay_panel = cv2.copyMakeBorder(
        overlay_bgr,
        label_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # Original label
    cv2.putText(
        original_panel,
        "ORIGINAL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Grad-CAM label
    cv2.putText(
        overlay_panel,
        f"GRAD-CAM - {title}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Put images side by side
    # --------------------------------------------------------

    combined = cv2.hconcat(
        [
            original_panel,
            overlay_panel
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        str(output_path),
        combined
    )


# ============================================================
# SKIN GRAD-CAM
# ============================================================

def generate_skin_gradcam(
    model,
    cam_generator,
    logits,
    target_class,
    image,
    predicted_class
):
    """
    Generate skin Grad-CAM.
    """

    print(
        "\nGenerating skin Grad-CAM..."
    )

    try:

        cam = cam_generator.generate(
            logits,
            target_class,
            (
                SKIN_IMAGE_SIZE,
                SKIN_IMAGE_SIZE
            )
        )

    finally:

        cam_generator.remove_hooks()

    output_path = (
        SKIN_GRADCAM_DIR
        / "skin_gradcam.png"
    )

    create_gradcam_overlay(
        image,
        cam,
        output_path,
        f"SKIN - {predicted_class}"
    )

    return output_path


# ============================================================
# EYE GRAD-CAM
# ============================================================

def generate_eye_gradcam(
    model,
    cam_generator,
    logits,
    target_class,
    image,
    predicted_class
):
    """
    Generate combined eye Grad-CAM.
    """

    print(
        "\nGenerating eye Grad-CAM..."
    )

    try:

        cam = cam_generator.generate(
            logits,
            target_class,
            (
                EYE_IMAGE_SIZE,
                EYE_IMAGE_SIZE
            )
        )

    finally:

        cam_generator.remove_hooks()

    output_path = (
        EYE_GRADCAM_DIR
        / "eye_gradcam.png"
    )

    create_gradcam_overlay(
        image,
        cam,
        output_path,
        f"EYE - {predicted_class}"
    )

    return output_path


# ============================================================
# PRINT PREDICTIONS
# ============================================================

def print_predictions(
    predictions,
    title
):

    print_header(
        title
    )

    for i, prediction in enumerate(
        predictions,
        start=1
    ):

        print(
            f"{i}. "
            f"{prediction['class']:<40}"
            f"{prediction['confidence'] * 100:>7.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Command-line argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python src/predict.py "path/to/image.jpg"'
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "GPU: Not available - using CPU"
        )

    print()

    print("=" * 70)

    print(
        "                 PET HEALTH AI"
    )

    print(
        "          Prediction + Grad-CAM"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    print(
        f"\nImage: {image_path}"
    )

    image = load_image(
        image_path
    )

    print(
        f"Size : "
        f"{image.width} x {image.height}"
    )

    # ========================================================
    # STEP 1
    # MODALITY
    # ========================================================

    (
        modality,
        router_confidence
    ) = predict_modality(
        image
    )

    print_header(
        "IMAGE ROUTING"
    )

    print(
        f"Detected modality : "
        f"{modality.upper()}"
    )

    print(
        f"Router confidence : "
        f"{router_confidence * 100:.2f}%"
    )

    # ========================================================
    # SKIN
    # ========================================================

    if modality == "skin":

        (
            skin_model,
            skin_tensor,
            skin_logits,
            predictions,
            target_class,
            skin_cam_generator
        ) = predict_skin(
            image
        )

        print_predictions(
            predictions,
            "SKIN CONDITION"
        )

        predicted_class = (
            predictions[0]["class"]
        )

        output_path = (
            generate_skin_gradcam(
                skin_model,
                skin_cam_generator,
                skin_logits,
                target_class,
                image,
                predicted_class
            )
        )

        print(
            "\nGrad-CAM saved:"
        )

        print(
            output_path
        )

        del skin_model
        del skin_tensor
        del skin_logits

    # ========================================================
    # EYE
    # ========================================================

    elif modality == "eye":

        (
            eye_model,
            eye_inputs,
            eye_logits,
            predictions,
            target_class,
            eye_cam_generator
        ) = predict_eye(
            image
        )

        print_predictions(
            predictions,
            "EYE CONDITION"
        )

        predicted_class = (
            predictions[0]["class"]
        )

        output_path = (
            generate_eye_gradcam(
                eye_model,
                eye_cam_generator,
                eye_logits,
                target_class,
                image,
                predicted_class
            )
        )

        print(
            "\nGrad-CAM saved:"
        )

        print(
            output_path
        )

        del eye_model
        del eye_inputs
        del eye_logits

    else:

        raise RuntimeError(
            f"Unknown modality: {modality}"
        )

    # --------------------------------------------------------
    # GPU cleanup
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    print()

    print("=" * 70)

    print(
        "Prediction completed successfully."
    )

    print("=" * 70)

    print()


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()