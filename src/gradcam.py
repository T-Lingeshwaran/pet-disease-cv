from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# BASIC GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        output,
        class_index
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        score = output[:, class_index].sum()

        score.backward(
            retain_graph=True
        )

        activations = self.activations
        gradients = self.gradients

        if activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Global-average-pool gradients
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted feature maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = F.relu(
            cam
        )

        # ----------------------------------------------------
        # Resize to input size
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 1e-8:

            cam /= max_value

        return cam


# ============================================================
# IMAGE OVERLAY
# ============================================================

def overlay_cam(
    original_image,
    cam,
    alpha=0.45
):

    original = np.asarray(
        original_image
    ).astype(
        np.uint8
    )

    height, width = (
        original.shape[:2]
    )

    cam = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        original.astype(np.float32)
        * (1 - alpha)
        +
        heatmap.astype(np.float32)
        * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    return overlay


# ============================================================
# SAVE CAM
# ============================================================

def save_cam(
    original_image,
    cam,
    output_path,
    alpha=0.45
):

    overlay = overlay_cam(
        original_image,
        cam,
        alpha
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        Image
    )

    return output_path