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
                    LayerCAM
                       │
                       ▼
              Visual Explanation
                       │
                       ▼
                Vision-Language
                  Explanation

The system performs:

Modality detection
Task-specific classification
LayerCAM visualization
Optional vision-language explanation
👁️ Eye Analysis

The eye module evaluates multiple ocular conditions using a combined dog/cat dataset.

The project evaluates several architectures, including:

EfficientNet-B0
DenseNet-121
CSF-Net-inspired architecture
MCFA-Net — Multi-Color Feature Attention Network
MCFA-Net

MCFA-Net combines RGB, HSV, and YCbCr representations using branch attention and cross-color feature fusion.

RGB ───────┐
HSV ───────┼──► Feature Projection
YCbCr ─────┘
                 │
                 ▼
          Branch Attention
                 │
                 ▼
       Cross-Color Transformer
                 │
                 ▼
           Feature Fusion
                 │
                 ▼
          Eye Classification
🐕 Skin Analysis

The skin module performs multi-class skin-condition classification.

Models evaluated include:

EfficientNet-B0
EfficientNetV2-S
ConvNeXt-Tiny
CSTF-Net — Color-Spatial-Texture Fusion Network
CSTF-Net

CSTF-Net combines complementary visual representations:

RGB semantic features
HSV color features
Texture features derived from image gradients

These representations are combined using learnable branch weighting and feature fusion before classification.

RGB ───────┐
HSV ───────┼──► Branch Feature Extraction
Texture ───┘
                 │
                 ▼
          Branch Weighting
                 │
                 ▼
       Cross-Feature Transformer
                 │
                 ▼
            Feature Fusion
                 │
                 ▼
         Skin Classification
🔎 Explainable AI

The current inference pipeline uses LayerCAM as its visual explainability method.

LayerCAM is used to investigate whether the regions influencing the classifier correspond to visually relevant areas of the eye or skin.

The system produces a single LayerCAM visualization for each prediction.

For CSTF-Net, the visualization combines attribution information from its RGB, color, and texture branches while being presented simply as:

CSTF-Net LayerCAM

For MCFA-Net:

MCFA-Net LayerCAM

The explainability analysis helps assess whether predictions are supported by relevant image regions rather than relying on classification accuracy alone.

🤖 Vision-Language Explanation

A vision-language model can provide a cautious textual interpretation of the image together with the LayerCAM visualization.

The VLM is instructed to describe:

visible visual observations
relevant anatomical or skin regions
visible features
relationship between the highlighted evidence and the model prediction
whether the highlighted evidence appears relevant
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
LayerCAM qualitative analysis

Particular emphasis is placed on macro-averaged metrics because the datasets contain substantial class imbalance.

🧪 Model Comparison

The project compares baseline, comparison, and proposed architectures rather than assuming that the proposed architecture is automatically superior.

Eye
Model	Role
EfficientNet-B0	Baseline
DenseNet-121	Comparison
CSF-Net-inspired	Existing architecture comparison
MCFA-Net	Proposed
Skin
Model	Role
EfficientNet-B0	Baseline
EfficientNetV2-S	Comparison
ConvNeXt-Tiny	Comparison
CSTF-Net	Proposed
🖼️ Explainability Outputs

Generated LayerCAM visualizations are stored under:

outputs/
└── layercam/
    ├── eye/
    │   └── <image_name>/
    │       └── mcfa_v2_layercam.png
    │
    └── skin/
        └── <image_name>/
            └── cstf_layercam.png

The system produces one primary LayerCAM visualization per prediction rather than separate RGB, color, texture, or Grad-CAM outputs.

⚠️ Disclaimer

This project is an AI-assisted veterinary screening/research prototype.

Model predictions and LayerCAM visualizations should not be interpreted as definitive medical or veterinary diagnoses. Visual attribution maps indicate regions that influenced a model prediction; they do not constitute ground-truth lesion annotations.

Clinical assessment by a qualified veterinarian is required for diagnosis and treatment decisions.