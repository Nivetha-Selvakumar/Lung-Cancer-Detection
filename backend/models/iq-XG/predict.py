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
# CELL 41 - UPLOAD CT IMAGE AND PREDICT
# XGBoost + HOG + PCA (segmentation-aware, same as training)
# ============================================================

# # from google.colab import files
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog

# ------------------------------------------------------------
# 1. Upload image
# ------------------------------------------------------------

# uploaded = files.upload()
uploaded = {}

if len(uploaded) == 0:
    print("No image uploaded.")
else:
    uploaded_filename = 'sample.png'
    image_path = uploaded_filename

    print("Uploaded image:", uploaded_filename)

    # --------------------------------------------------------
    # 2. Read original image
    # --------------------------------------------------------

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        raise ValueError("Unable to read the uploaded image.")

    print("Original image size:",
          original_image.shape[1], "x", original_image.shape[0])

    # --------------------------------------------------------
    # 3. SAME segmentation-aware preprocessing used during
    #    training (segment_lung -> resize -> denoise -> CLAHE
    #    -> mask -> normalize). Using preprocess_image() here
    #    directly guarantees this matches training exactly.
    # --------------------------------------------------------

    _, lung_mask, _ = segment_lung(image_path)

    lung_mask_resized = cv2.resize(
        lung_mask,
        (224, 224),
        interpolation=cv2.INTER_NEAREST
    )

    normalized_image = preprocess_image(image_path)

    processed_image = (normalized_image * 255.0).astype(np.uint8)

    enhanced_image = processed_image  # used below for the
    # candidate-region visualization; already segmented + CLAHE'd

    lung_coverage = (np.sum(lung_mask_resized > 0) / lung_mask_resized.size) * 100

    print(f"Lung mask coverage: {lung_coverage:.2f}%")

    # --------------------------------------------------------
    # 4. HOG feature extraction
    # SAME parameters used during training
    # --------------------------------------------------------

    hog_features = hog(
        normalized_image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    # Convert to 2D
    hog_features = hog_features.reshape(1, -1)

    print("HOG feature shape:", hog_features.shape)

    # --------------------------------------------------------
    # 5. Apply trained StandardScaler
    # --------------------------------------------------------

    scaled_features = xgb_scaler.transform(hog_features)

    # --------------------------------------------------------
    # 6. Apply trained PCA
    # --------------------------------------------------------

    pca_features = xgb_pca.transform(scaled_features)

    print("PCA feature shape:", pca_features.shape)

    # --------------------------------------------------------
    # 7. XGBoost prediction
    # --------------------------------------------------------

    probabilities = final_xgb_model.predict_proba(pca_features)[0]

    predicted_class_index = np.argmax(probabilities)

    class_names = [
        "Normal",
        "Benign",
        "Malignant"
    ]

    predicted_class = class_names[predicted_class_index]

    predicted_probability = probabilities[predicted_class_index] * 100

    # --------------------------------------------------------
    # 8. Display prediction
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("              LUNG CT CLASSIFICATION RESULT")
    print("=" * 60)

    print(f"\nPredicted Class : {predicted_class}")
    print(f"Model Probability : {predicted_probability:.2f}%")

    print("\nClass Probabilities:")

    for class_name, probability in zip(
        class_names,
        probabilities
    ):
        print(
            f"{class_name:10s} : "
            f"{probability * 100:.2f}%"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # 9. Candidate abnormal-region visualization
    #
    # IMPORTANT:
    # This is an image-processing visualization.
    # It is NOT a cancer localization model.
    # --------------------------------------------------------

    # Smooth image
    blurred = cv2.GaussianBlur(
        enhanced_image,
        (0, 0),
        sigmaX=7
    )

    # Local contrast / intensity difference
    local_difference = cv2.absdiff(
        enhanced_image,
        blurred
    )

    # Normalize difference
    local_difference = cv2.normalize(
        local_difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Threshold
    _, candidate_mask = cv2.threshold(
        local_difference,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find candidate contours
    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # 10. Select meaningful candidate regions
    # --------------------------------------------------------

    image_area = 224 * 224

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore extremely small and extremely large regions
        if (
            area > image_area * 0.002
            and area < image_area * 0.20
        ):
            valid_contours.append(
                (area, contour)
            )

    # Sort by area
    valid_contours = sorted(
        valid_contours,
        key=lambda x: x[0],
        reverse=True
    )

    # Keep top few candidate regions
    selected_contours = valid_contours[:3]

    # --------------------------------------------------------
    # 11. Create red-highlighted image
    # --------------------------------------------------------

    display_image = cv2.cvtColor(
        processed_image,
        cv2.COLOR_GRAY2RGB
    )

    if len(selected_contours) > 0:

        for area, contour in selected_contours:

            x, y, w, h = cv2.boundingRect(contour)

            # Draw red contour
            cv2.drawContours(
                display_image,
                [contour],
                -1,
                (255, 0, 0),
                2
            )

            # Draw red ellipse around candidate region
            center = (
                x + w // 2,
                y + h // 2
            )

            axes = (
                max(w // 2, 8),
                max(h // 2, 8)
            )

            cv2.ellipse(
                display_image,
                center,
                axes,
                0,
                0,
                360,
                (255, 0, 0),
                3
            )

        region_status = (
            "Candidate abnormal region highlighted"
        )

    else:

        region_status = (
            "No clear candidate region detected"
        )

    # --------------------------------------------------------
    # 12. Display results
    # --------------------------------------------------------

    plt.figure(figsize=(16, 5))

    # Original
    plt.subplot(1, 3, 1)

    plt.imshow(
        original_image,
        cmap="gray"
    )

    plt.title("Uploaded CT Image")
    plt.axis("off")

    # Processed
    plt.subplot(1, 3, 2)

    plt.imshow(
        processed_image,
        cmap="gray"
    )

    plt.title("Preprocessed Image (224 × 224)")
    plt.axis("off")

    # Highlighted
    plt.subplot(1, 3, 3)

    plt.imshow(display_image)

    plt.title("Candidate Region Highlighted")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 13. Final interpretation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if predicted_class == "Normal":

        print(
            f"The XGBoost model classified this image as "
            f"NORMAL ({predicted_probability:.2f}%)."
        )

    elif predicted_class == "Benign":

        print(
            f"The XGBoost model classified this image as "
            f"BENIGN ({predicted_probability:.2f}%)."
        )

    else:

        print(
            f"The XGBoost model classified this image as "
            f"MALIGNANT ({predicted_probability:.2f}%)."
        )

    print(f"\n{region_status}")

    print(
        "\nIMPORTANT:"
        "\nThe red-highlighted region is only a "
        "candidate region identified using image-processing "
        "techniques."
        "\nIt is NOT a confirmed cancer lesion and is NOT "
        "the direct explanation of the XGBoost prediction."
        "\nThe classification probability is produced by "
        "the trained XGBoost model."
        "\nClinical interpretation must be performed by a "
        "qualified medical professional."
    )

    print("=" * 60)