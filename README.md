🐾 Pet Health Computer Vision

An explainable computer vision system for AI-assisted veterinary screening of dogs and cats from images. The project combines deep-learning classification, modality routing, visual explainability, and vision-language models to analyze common eye and skin conditions.

The system is designed as a screening and decision-support prototype, not a replacement for veterinary diagnosis.

🔬 Project Overview

The pipeline automatically determines whether an input image belongs to the eye or skin/coat modality and routes it to a task-specific model:

                    Pet Image
                       │
                       ▼
               Modality Classifier
                 ┌─────┴─────┐
                 │           │
                Eye         Skin
                 │           │
                 ▼           ▼
             MCFA-Net     CSTF-Net
                 │           │
                 ▼           ▼
          Eye Condition  Skin Condition
                 │           │
                 └─────┬─────┘
                       ▼
              Grad-CAM / LayerCAM
                       │
                       ▼
              Visual Explanation
                       │
                       ▼
                Vision-Language
                  Explanation
👁️ Eye Analysis

The eye module evaluates multiple ocular conditions using a combined dog/cat dataset.

The project evaluates several architectures, including:

EfficientNet-B0
DenseNet-121
CSF-Net-inspired architecture
MCFA-Net — Multi-Color Feature Attention Network

MCFA-Net combines RGB, HSV, and YCbCr representations using branch attention and cross-color feature fusion.

🐕 Skin Analysis

The skin module performs multi-class skin-condition classification.

Models evaluated include:

EfficientNet-B0
EfficientNetV2-S
ConvNeXt-Tiny
CSTF-Net — Color- and Spatial/Texture Feature Network

CSTF-Net combines RGB appearance with complementary color and texture representations to improve class-balanced skin-condition recognition.

🔎 Explainable AI

Predictions are investigated using:

Grad-CAM
LayerCAM
Branch-level activation analysis
Baseline vs proposed-model comparisons

The goal is to determine whether model predictions are supported by visually relevant ocular or skin regions rather than treating classification accuracy alone as sufficient evidence.

🤖 Vision-Language Explanation

A vision-language model can provide a cautious textual interpretation of the image and model attribution.

The VLM is instructed to describe:

visible visual observations
relevant anatomical/skin regions
relationship to the model prediction
whether the highlighted evidence appears focused or diffuse
uncertainty and limitations

The system explicitly avoids presenting the output as a definitive veterinary diagnosis.

📊 Evaluation

The project evaluates models using:

Accuracy
Macro Precision
Macro Recall
Macro F1-score
Per-class precision/recall/F1
Confusion matrices
Training/validation curves
Grad-CAM and LayerCAM qualitative analysis

Particular emphasis is placed on macro-averaged metrics because the datasets contain substantial class imbalance.

⚠️ Disclaimer

This project is an AI-assisted veterinary screening/research prototype. Model predictions and visual explanations should not be interpreted as definitive medical or veterinary diagnoses. Clinical assessment by a qualified veterinarian is required for diagnosis and treatment decisions.