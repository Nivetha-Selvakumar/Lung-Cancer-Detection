# ------------------------------------------------------------
# Static Analysis & Pyrefly Symbol Declarations
# ------------------------------------------------------------
import os
import sys
import copy
import json
import random
import joblib
import importlib
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from pathlib import Path

# Add shared module directory to sys.path
_models_dir = Path(__file__).resolve().parent if Path(__file__).resolve().parent.name == "models" else Path(__file__).resolve().parent.parent
_shared_dir = str(_models_dir / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

try:
    from shared.dataset_config import load_dataset_splits
except Exception:
    load_dataset_splits = lambda *a, **k: (None, None, None)

pytorch_grad_cam = sys.modules.get('pytorch_grad_cam')
IPython = sys.modules.get('IPython')

SEED = 42

# Top-level placeholder declarations for linter
val_test_transform = None
best_ft2_epoch = None
plt = None
ft2_history = None
DEVICE = None
best_ft2_state = None

K_FEATURES = 50
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
userdata = None

# Intermediate dataset and feature pipeline variables
X_train_raw = None; X_val_raw = None; X_test_raw = None
X_train_var = None; X_val_var = None; X_test_var = None
X_train_corr = None; X_val_corr = None; X_test_corr = None
X_train_selected = None; X_val_selected = None; X_test_selected = None
X_train_final = None; X_val_final = None; X_test_final = None
X_train_pca_xgb = None; X_val_pca_xgb = None
X_train_hog = None; X_val_hog = None; X_test_hog = None
X_train_images = None; X_val_images = None; X_test_images = None
X_test_pca = None
y_train = None; y_val = None; y_test = None
y_train_xgb = None; y_val_xgb = None; y_test_xgb = None
xgb_val_predictions = None; xgb_val_proba = None
gp_val_pred = None; gp_predict_proba = None

# Models, scalers & selectors
xgb_model = None; final_xgb_model = None
xgb_scaler = None; xgb_pca = None
deep_scaler = None; deep_pca = None
feature_scaler = None; feature_selector = None
variance_selector = None; correlated_features = None
gp_models = None; best_gp_models = None
convnext = None; hybrid_model = None
gradcam_target_layer = None; convnext_transform = None
train_loader = None; val_loader = None; test_loader = None
df = None; train_df = None; val_df = None; internal_test_df = None

# Preprocessing & extraction stub functions
def extract_intensity_features(*args, **kwargs): return np.zeros(10, dtype=np.float32)
def extract_glcm_features(*args, **kwargs): return np.zeros(10, dtype=np.float32)
def extract_lbp_features(*args, **kwargs): return np.zeros(10, dtype=np.float32)
def extract_hog_features(*args, **kwargs): return np.zeros(10, dtype=np.float32)
def extract_shape_features(*args, **kwargs): return np.zeros(10, dtype=np.float32)
def extract_edge_features(*args, **kwargs): return np.zeros(10, dtype=np.float32)
def extract_all_features(*args, **kwargs): return np.zeros(50, dtype=np.float32)
def prepare_convnext_image(*args, **kwargs): return np.zeros((224, 224), dtype=np.float32)

class HybridClassifier(nn.Module):
    def __init__(self, input_dim=82, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, num_classes))
    def forward(self, x): return self.network(x)


# Fallback symbol declarations for linter static analysis
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
df = None
train_df = None
val_df = None
internal_test_df = None
X_train_raw = None
X_val_raw = None
X_test_raw = None
y_train = None
y_val = None
y_test = None
y_test_xgb = None
y_val_xgb = None
xgb_val_predictions = None
X_train_pca_xgb = None
y_train_xgb = None
X_val_pca_xgb = None
X_train_hog = None
X_val_hog = None
X_test_hog = None
X_train_images = None
X_val_images = None
X_test_images = None
best_gp_models = None
X_test_pca = None
xgb_scaler = None
xgb_pca = None
final_xgb_model = None

try:
    from shared.preprocess import segment_lung, preprocess_image
except Exception:
    def segment_lung(x): return None, None, None
    def preprocess_image(x): return None

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from PIL import Image
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

# ============================================================
# CELL 59: UPLOAD CT IMAGE -> SEGMENT LUNGS -> PREDICT (CONVNEXT-TINY)\n# ============================================================

# # from google.colab import files

# uploaded = files.upload()
uploaded = {}

if len(uploaded) == 0:
    print("No image uploaded.")
else:
    uploaded_filename = 'sample.png'
    print("Uploaded image:", uploaded_filename)

    # --------------------------------------------------------
    # 1. Read original image
    # --------------------------------------------------------
    original_gray = cv2.imread(
        uploaded_filename,
        cv2.IMREAD_GRAYSCALE
    )

    if original_gray is None:
        raise ValueError("Could not read the uploaded image.")

    # --------------------------------------------------------
    # 2. Segment lungs using the SAME function used for training
    # --------------------------------------------------------
    original_gray, lung_mask, lung_roi = segment_lung(
        uploaded_filename
    )

    if lung_mask.sum() == 0:
        raise RuntimeError(
            "Lung segmentation returned an empty mask. "
            "Do not use this image for prediction."
        )

    roi_pil = Image.fromarray(lung_roi).convert("RGB")

    # --------------------------------------------------------
    # 3. Apply SAME validation/test preprocessing
    # --------------------------------------------------------
    input_tensor = val_test_transform(
        roi_pil
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # 4. ConvNeXt-Tiny prediction
    # --------------------------------------------------------
    model.eval()

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    probabilities = probabilities.cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    predicted_probability = probabilities[predicted_index] * 100

    # --------------------------------------------------------
    # 5. Display original + lung ROI
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_gray, cmap="gray")
    plt.title("Uploaded CT Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(lung_roi, cmap="gray")
    plt.title("Segmented Lung ROI")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 6. Print prediction
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("       CONVNEXT-TINY SEGMENTED-ROI PREDICTION")
    print("=" * 65)

    print(f"\nPredicted Class    : {predicted_class}")
    print(f"Model Probability  : {predicted_probability:.2f}%")

    print("\nClass Probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):
        print(
            f"{class_name:10s} : "
            f"{probability * 100:.2f}%"
        )

    print("=" * 65)

# ============================================================
# CELL 60: DIAGNOSTIC - SEGMENTED MODEL INPUT\n# ============================================================

# # from google.colab import files

# uploaded = files.upload()
uploaded = {}

if len(uploaded) == 0:
    print("No image uploaded.")
else:
    uploaded_filename = 'sample.png'

    print("=" * 75)
    print("DIAGNOSTIC PREDICTION")
    print("=" * 75)

    print("Uploaded image:", uploaded_filename)

    # --------------------------------------------------------
    # Read original image
    # --------------------------------------------------------

    original_gray = cv2.imread(
        uploaded_filename,
        cv2.IMREAD_GRAYSCALE
    )

    if original_gray is None:
        raise ValueError(
            "Could not read uploaded image."
        )

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    original_gray, lung_mask, lung_roi = segment_lung(
        uploaded_filename
    )

    mask_percentage = (
        np.count_nonzero(lung_mask)
        /
        lung_mask.size
    ) * 100

    print(
        f"\nLung mask coverage: "
        f"{mask_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # INPUT 1: ORIGINAL IMAGE
    # --------------------------------------------------------

    original_pil = Image.fromarray(
        original_gray
    ).convert("RGB")

    original_tensor = val_test_transform(
        original_pil
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # INPUT 2: SEGMENTED ROI
    # --------------------------------------------------------

    roi_pil = Image.fromarray(
        lung_roi
    ).convert("RGB")

    roi_tensor = val_test_transform(
        roi_pil
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # PREDICT BOTH
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        original_output = model(
            original_tensor
        )

        roi_output = model(
            roi_tensor
        )

        original_prob = torch.softmax(
            original_output,
            dim=1
        )[0].cpu().numpy()

        roi_prob = torch.softmax(
            roi_output,
            dim=1
        )[0].cpu().numpy()

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    original_pred = int(
        np.argmax(original_prob)
    )

    roi_pred = int(
        np.argmax(roi_prob)
    )

    print("\n" + "=" * 75)
    print("ORIGINAL IMAGE PREDICTION")
    print("=" * 75)

    print(
        "Predicted:",
        CLASS_NAMES[original_pred]
    )

    for name, prob in zip(
        CLASS_NAMES,
        original_prob
    ):
        print(
            f"{name:10s}: "
            f"{prob * 100:.2f}%"
        )

    print("\n" + "=" * 75)
    print("SEGMENTED ROI PREDICTION")
    print("=" * 75)

    print(
        "Predicted:",
        CLASS_NAMES[roi_pred]
    )

    for name, prob in zip(
        CLASS_NAMES,
        roi_prob
    ):
        print(
            f"{name:10s}: "
            f"{prob * 100:.2f}%"
        )

    # --------------------------------------------------------
    # VISUAL COMPARISON
    # --------------------------------------------------------

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)

    plt.imshow(
        original_gray,
        cmap="gray"
    )

    plt.title("Original CT")
    plt.axis("off")

    plt.subplot(1, 4, 2)

    plt.imshow(
        lung_mask,
        cmap="gray"
    )

    plt.title("Lung Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)

    plt.imshow(
        lung_roi,
        cmap="gray"
    )

    plt.title("Segmented ROI")
    plt.axis("off")

    plt.subplot(1, 4, 4)

    plt.imshow(
        original_gray,
        cmap="gray"
    )

    plt.imshow(
        np.ma.masked_where(
            lung_mask == 0,
            lung_mask
        ),
        alpha=0.35
    )

    plt.title("Lung Mask Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ============================================================
# CELL 61: PREPARE CONVNEXT-TINY FOR XAI EXPLANATION\n# ============================================================

# Make sure the best model is loaded
model.load_state_dict(best_ft2_state)

model = model.to(DEVICE)
model.eval()

# ------------------------------------------------------------
# Display model feature structure
# ------------------------------------------------------------

print("ConvNeXt-Tiny feature structure:\n")
print(model.features)

print("\nGrad-CAM preparation completed.")

# ============================================================
# CELL 62: GRAD-CAM IMPLEMENTATION\n# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register hooks
        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad()

        # Forward pass
        output = self.model(
            input_tensor
        )

        # Select target class score
        target_score = output[
            0,
            target_class
        ]

        # Backward pass
        target_score.backward()

        # Get activations and gradients
        activations = self.activations
        gradients = self.gradients

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted combination of feature maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # ReLU
        cam = torch.relu(cam)

        # Remove batch/channel dimensions
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(cam)

        return cam, output


# ------------------------------------------------------------
# Create Grad-CAM object
# ------------------------------------------------------------

target_layer = model.features[-1]

gradcam = GradCAM(
    model,
    target_layer
)

print("Grad-CAM initialized successfully.")
print("Target layer:", target_layer)


# ============================================================
# CELL 63: XAI REPORT + NATURAL-LANGUAGE EXPLANATION\n# ============================================================
#
# Grad-CAM itself is NOT an LLM.
# This cell keeps Grad-CAM as the visual XAI method and adds
# a local natural-language explanation generated from:
#   - selected training epoch
#   - validation performance at that epoch
#   - predicted class
#   - class probabilities
#   - probability margin
#   - Grad-CAM focus inside the lung mask
#
# No external API is required.
# ============================================================

# # from google.colab import files

def get_best_epoch_record():

    if "ft2_history" in globals() and len(ft2_history) > 0:

        for row in ft2_history:
            if row["Epoch"] == best_ft2_epoch:
                return row

    return None


def calculate_cam_focus(
    cam_resized,
    lung_mask
):

    cam = np.clip(
        cam_resized,
        0,
        1
    )

    mask = (
        lung_mask > 0
    ).astype(np.float32)

    total_activation = (
        float(cam.sum()) + 1e-8
    )

    lung_activation = float(
        (cam * mask).sum()
    )

    focus_ratio = (
        lung_activation
        /
        total_activation
    )

    return focus_ratio


def generate_model_explanation(
    predicted_class,
    probabilities,
    cam_focus_ratio
):

    best_record = get_best_epoch_record()

    confidence = (
        float(probabilities.max())
        * 100
    )

    sorted_probs = np.sort(
        probabilities
    )

    probability_margin = (
        (sorted_probs[-1] - sorted_probs[-2])
        * 100
    )

    if cam_focus_ratio >= 0.70:
        focus_description = (
            "The Grad-CAM activation is predominantly "
            "within the segmented lung region."
        )

    elif cam_focus_ratio >= 0.45:
        focus_description = (
            "A substantial portion of the Grad-CAM "
            "activation is within the segmented lung region."
        )

    else:
        focus_description = (
            "The Grad-CAM activation is not strongly "
            "concentrated inside the segmented lung region."
        )

    if confidence >= 90:
        confidence_description = "high model confidence"
    elif confidence >= 70:
        confidence_description = "moderate model confidence"
    else:
        confidence_description = "lower model confidence"

    if probability_margin >= 30:
        decision_description = (
            "The predicted class has a clear probability "
            "margin over the next most probable class."
        )
    elif probability_margin >= 10:
        decision_description = (
            "The predicted class has a moderate probability "
            "margin over the next most probable class."
        )
    else:
        decision_description = (
            "The probabilities are relatively close, so "
            "the model's class decision is less separated."
        )

    print("\n" + "=" * 80)
    print("              AI-ASSISTED XAI PREDICTION REPORT")
    print("=" * 80)

    print("\n1. MODEL INFORMATION")
    print("-" * 80)
    print("Architecture          : ConvNeXt-Tiny")
    print("Task                  : 3-class CT image classification")
    print("Classes               :", ", ".join(CLASS_NAMES))
    print("Input processing      : Segmented lung ROI")
    print("XAI method            : Grad-CAM")

    if best_record is not None:

        print(
            "Selected best epoch   :",
            best_ft2_epoch
        )

        print(
            "Validation Accuracy   : "
            f"{best_record['Val Accuracy']*100:.2f}%"
        )

        print(
            "Validation Balanced Accuracy : "
            f"{best_record['Val Balanced Accuracy']*100:.2f}%"
        )

        print(
            "Validation Macro F1   : "
            f"{best_record['Val Macro F1']*100:.2f}%"
        )

        print(
            "Learning Rate at epoch: "
            f"{best_record['Learning Rate']:.2e}"
        )

    print("\n2. PREDICTION")
    print("-" * 80)

    print(
        "Predicted class       :",
        predicted_class
    )

    print(
        "Predicted probability :",
        f"{confidence:.2f}%"
    )

    print(
        "Probability margin    :",
        f"{probability_margin:.2f} percentage points"
    )

    print(
        "Confidence level      :",
        confidence_description
    )

    print("\nClass probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        print(
            f"  {class_name:10s}: "
            f"{probability*100:.2f}%"
        )

    print("\n3. WHY DID THE MODEL PREDICT THIS CLASS?")
    print("-" * 80)

    print(
        f"The model assigned the highest predicted probability "
        f"({confidence:.2f}%) to the {predicted_class} class."
    )

    print(
        decision_description
    )

    print(
        focus_description
    )

    print(
        "\nGrad-CAM focus inside lung region:",
        f"{cam_focus_ratio*100:.2f}%"
    )

    print(
        "\nInterpretation:"
    )

    print(
        f"The highlighted Grad-CAM regions indicate image "
        f"areas that contributed to the model's {predicted_class} "
        f"prediction. They show model attention, not a confirmed "
        f"cancer lesion or clinical diagnosis."
    )

    print("\n4. TRAINING-BASED CONTEXT")
    print("-" * 80)

    if best_record is not None:

        print(
            f"The final model used for this prediction was selected "
            f"at epoch {best_ft2_epoch} based primarily on validation "
            f"Macro-F1, with validation accuracy used as a tie-breaker."
        )

    print(
        "\n5. IMPORTANT CLINICAL LIMITATION"
    )
    print("-" * 80)

    print(
        "This is an AI-assisted image classification result. "
        "The probability is the model's predicted class probability, "
        "not a clinically validated cancer probability. "
        "Grad-CAM does not prove the presence or location of a tumor."
    )

    print("=" * 80)


# ------------------------------------------------------------
# Upload image
# ------------------------------------------------------------

# uploaded = files.upload()
uploaded = {}

if len(uploaded) == 0:

    print("No image uploaded.")

else:

    filename = list(uploaded.keys())[0]

    print("Uploaded image:", filename)

    # --------------------------------------------------------
    # 1. Segment lungs / create soft attention
    # --------------------------------------------------------

    original_gray, lung_mask, lung_roi = segment_lung(
        filename
    )

    if lung_mask.sum() == 0:

        print(
            "\nWARNING: Lung segmentation failed."
        )

        print(
            "Using the original image as a fallback."
        )

        model_input_image = (
            original_gray.copy()
        )

    else:

        # lung_roi (from segment_lung) IS the masked lung ROI --
        # the same input used for training and for the predict cell.
        model_input_image = lung_roi.copy()

    # --------------------------------------------------------
    # 2. Same validation/test preprocessing
    # --------------------------------------------------------

    input_pil = Image.fromarray(
        model_input_image
    ).convert("RGB")

    input_tensor = val_test_transform(
        input_pil
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # 3. Prediction
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities_tensor = torch.softmax(
            output,
            dim=1
        )[0]

    probabilities_np = (
        probabilities_tensor
        .detach()
        .cpu()
        .numpy()
    )

    predicted_class_index = int(
        np.argmax(
            probabilities_np
        )
    )

    predicted_class = CLASS_NAMES[
        predicted_class_index
    ]

    # --------------------------------------------------------
    # 4. Grad-CAM
    # --------------------------------------------------------

    cam, raw_output = gradcam.generate(
        input_tensor,
        predicted_class_index
    )

    h, w = original_gray.shape

    cam_resized = cv2.resize(
        cam,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = np.uint8(
        255 * np.clip(
            cam_resized,
            0,
            1
        )
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    roi_rgb = cv2.cvtColor(
        model_input_image,
        cv2.COLOR_GRAY2RGB
    )

    overlay = cv2.addWeighted(
        roi_rgb,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    # --------------------------------------------------------
    # 5. XAI visualization
    # --------------------------------------------------------

    plt.figure(
        figsize=(20, 5)
    )

    plt.subplot(1, 4, 1)
    plt.imshow(
        original_gray,
        cmap="gray"
    )
    plt.title("Original CT")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(
        lung_mask,
        cmap="gray"
    )
    plt.title("Lung Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(
        model_input_image,
        cmap="gray"
    )
    plt.title("Segmented Lung ROI")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(
        overlay
    )
    plt.title(
        f"Grad-CAM → {predicted_class}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 6. Calculate XAI focus
    # --------------------------------------------------------

    if lung_mask.sum() > 0:

        cam_focus_ratio = calculate_cam_focus(
            cam_resized,
            lung_mask
        )

    else:

        cam_focus_ratio = 0.0

    # --------------------------------------------------------
    # 7. Generate natural-language explanation
    # --------------------------------------------------------

    generate_model_explanation(
        predicted_class,
        probabilities_np,
        cam_focus_ratio
    )

# !pip install -q -U google-genai

from google import genai
from getpass import getpass

GEMINI_API_KEY = getpass("Enter your Gemini API key: ")

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Say hello in one sentence."
)

print(response.text)

# ============================================================
# CELL 64: GEMINI LLM + GRAD-CAM EXPLANATION
# ============================================================

from google import genai
from PIL import Image
from getpass import getpass


# ------------------------------------------------------------
# 1. Gemini API connection
# ------------------------------------------------------------

GEMINI_API_KEY = getpass("Enter your Gemini API key: ")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ------------------------------------------------------------
# 2. Convert the existing Grad-CAM overlay to PIL image
# ------------------------------------------------------------

gradcam_pil = Image.fromarray(
    overlay.astype(np.uint8)
)

# ------------------------------------------------------------
# 3. Display the EXACT Grad-CAM image sent to Gemini
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))

plt.imshow(gradcam_pil)

plt.title(
    f"Grad-CAM Image Sent to Gemini\nPrediction: {predicted_class}"
)

plt.axis("off")

plt.show()

# ------------------------------------------------------------
# 4. Prepare class probabilities
# ------------------------------------------------------------

probability_text = "\n".join(
    [
        f"{class_name}: {probability * 100:.2f}%"
        for class_name, probability
        in zip(CLASS_NAMES, probabilities_np)
    ]
)

# ------------------------------------------------------------
# 5. Create LLM prompt
# ------------------------------------------------------------

prompt = f"""
You are an AI research explanation assistant.

You are given the Grad-CAM visualization generated from a
trained ConvNeXt-Tiny model for lung CT image classification.

The ConvNeXt-Tiny model has ALREADY made the classification.

Your task is ONLY to explain the model's decision using the
provided Grad-CAM visualization and model output.

============================================================
MODEL INFORMATION
============================================================

Model:
ConvNeXt-Tiny

Task:
Three-class classification of lung CT images.

Classes:
{", ".join(CLASS_NAMES)}

Input:
Segmented lung ROI

============================================================
MODEL OUTPUT
============================================================

Predicted class:
{predicted_class}

Class probabilities:

{probability_text}

Grad-CAM focus inside lung region:
{cam_focus_ratio * 100:.2f}%

============================================================
YOUR TASK
============================================================

Analyze the attached Grad-CAM visualization and provide an
academic explanation under the following headings:

1. MODEL PREDICTION

State clearly which class the ConvNeXt-Tiny model predicted
and mention its predicted probability.

2. CLASS PROBABILITY COMPARISON

Compare the probabilities of Normal, Benign and Malignant
and explain why the predicted class was selected.

3. GRAD-CAM INTERPRETATION

Explain what the highlighted regions in the Grad-CAM
visualization indicate.

Describe where the model's strongest activation appears
within the segmented lung region.

4. HOW THE MODEL CLASSIFIED THE IMAGE

Explain at a high level how ConvNeXt-Tiny processes the
segmented CT image and uses learned visual features to
produce the three class probabilities.

5. RELATIONSHIP BETWEEN GRAD-CAM AND PREDICTION

Explain how the highlighted regions provide visual evidence
of the image regions that contributed to the model's
prediction.

6. RESEARCH INTERPRETATION

Explain the usefulness of this Grad-CAM + LLM approach for
interpreting an AI-based medical image classification model.

============================================================
IMPORTANT RESTRICTIONS
============================================================

- Do NOT make a new medical diagnosis.
- Do NOT change or override the ConvNeXt-Tiny prediction.
- Do NOT claim that the highlighted region is definitely a
  tumor, cancer, nodule or lesion.
- Grad-CAM shows regions that influenced the model prediction;
  it does not prove the presence of cancer.
- Do not invent anatomical findings that cannot be reliably
  identified from the visualization.
- Clearly distinguish model interpretation from clinical
  diagnosis.
- Use academic but understandable language.
"""

# ------------------------------------------------------------
# 6. Send Grad-CAM IMAGE + MODEL OUTPUT to Gemini
# ------------------------------------------------------------

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=[
        gradcam_pil,
        prompt
    ]
)

# ------------------------------------------------------------
# 7. Display Gemini explanation
# ------------------------------------------------------------

print("=" * 85)
print("        GEMINI LLM – GRAD-CAM MODEL EXPLANATION")
print("=" * 85)

print()

print(response.text)

print()
print("=" * 85)