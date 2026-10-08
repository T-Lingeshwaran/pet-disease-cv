# ============================================================
# PET HEALTH VLM
# Qwen2.5-VL-3B-Instruct
#
# Usage:
#
#   python src/vlm_explain.py "path/to/image.jpg"
#
# With an existing CV prediction:
#
#   python src/vlm_explain.py "image.jpg" ^
#       --prediction "Ringworm" ^
#       --confidence 95.2
#
# ============================================================

import argparse
import gc
import os

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Keep visual token count relatively small for RTX 3050 6 GB.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

MAX_NEW_TOKENS = 256


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

    print("=" * 70)
    print("GPU detected")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(0))

else:

    DEVICE = "cpu"

    print("WARNING: CUDA not available.")
    print("VLM will run on CPU and will be much slower.")


# ============================================================
# LOAD MODEL
# ============================================================

def load_vlm():

    print()
    print("=" * 70)
    print("Loading Qwen2.5-VL-3B-Instruct")
    print("=" * 70)

    print("Model:", MODEL_NAME)
    print("Quantization: 4-bit")

    # --------------------------------------------------------
    # 4-bit configuration
    # --------------------------------------------------------

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if DEVICE == "cuda":

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

    else:

        # CPU fallback.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    model.eval()

    print("VLM loaded successfully.")

    return model, processor


# ============================================================
# PROMPT
# ============================================================

def build_prompt(prediction=None, confidence=None):

    prediction_context = ""

    if prediction:

        if confidence is not None:

            prediction_context = f"""
Our computer-vision classifier predicted:

Condition: {prediction}
Confidence: {confidence:.2f}%

Treat this prediction as an input from another model.
Do NOT blindly agree with it.
Instead, assess whether the visible image provides
reasonable visual evidence for that prediction.
"""

        else:

            prediction_context = f"""
Our computer-vision classifier predicted:

Condition: {prediction}

Treat this prediction as an input from another model.
Do NOT blindly agree with it.
"""

    prompt = f"""
You are a cautious veterinary image-analysis assistant.

Analyze the provided pet image.

{prediction_context}

Your task is NOT to make a definitive veterinary diagnosis.

Describe only visible image evidence.

Please provide the response in this format:

OBSERVATION:
Briefly describe what is visibly present.

ANATOMICAL REGION:
Identify the relevant body region.

VISUAL FEATURES:
List the most important visible features such as:
- redness
- swelling
- cloudiness
- discharge
- hair loss
- scaling
- crusting
- circular lesions
- pigmentation changes
- opacity
- irritation
- other clearly visible abnormalities

MODEL CONSISTENCY:
If a computer-vision prediction was provided, explain whether
the visible image appears broadly consistent with it,
partially consistent, or not clearly consistent.

IMPORTANT:
Do not claim that the image proves a disease.
Do not invent symptoms that cannot be seen.
Do not provide medication or treatment instructions.
Mention when image quality or viewpoint limits interpretation.

Keep the answer concise and medically cautious.
"""

    return prompt


# ============================================================
# RUN VLM
# ============================================================

def analyze_image(
    model,
    processor,
    image_path,
    prediction=None,
    confidence=None,
):

    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Verify image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    print()
    print("Image:", image_path)
    print("Size :", image.size)

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        prediction=prediction,
        confidence=confidence,
    )

    # --------------------------------------------------------
    # Qwen conversation
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # --------------------------------------------------------
    # Prepare vision inputs
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move tensors to model device.
    if DEVICE == "cuda":

        inputs = {
            key: value.to(model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print()
    print("Generating VLM explanation...")

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # --------------------------------------------------------
    # Remove input tokens from output
    # --------------------------------------------------------

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del inputs
    del generated_ids
    del generated_ids_trimmed

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return output_text


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Pet Health Vision-Language Model"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to pet image",
    )

    parser.add_argument(
        "--prediction",
        type=str,
        default=None,
        help="Prediction from your CV model",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="CV model confidence percentage",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model, processor = load_vlm()

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    result = analyze_image(
        model=model,
        processor=processor,
        image_path=args.image,
        prediction=args.prediction,
        confidence=args.confidence,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VLM RESULT")
    print("=" * 70)
    print()

    print(result)

    print()
    print("=" * 70)
    print("END")
    print("=" * 70)


if __name__ == "__main__":
    main()