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

# Exact shared preprocessing.

# ============================================================
# SOURCE COLAB CELL 7
# ============================================================

# ============================================================
# CELL 7 — IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {
            "valid": False,
            "height": 0,
            "width": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan
        }

    return {
        "valid": True,
        "height": img.shape[0],
        "width": img.shape[1],
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "min": int(np.min(img)),
        "max": int(np.max(img))
    }


# Check ALL 53 development images
# quality_records = []
# 
# if 'df' in globals():
#     for _, row in df.iterrows():
# 
#     result = check_image_quality(row["filepath"])
# 
#     quality_records.append({
#         "filepath": row["filepath"],
#         "class": row["class"],
#         **result
#     })
# 
# quality_df = pd.DataFrame(quality_records)
# 
# 
# ============================================================
# QUALITY SUMMARY
# ============================================================
# 
# print("==============================================")
# print("IMAGE QUALITY CHECK")
# print("==============================================")
# 
# print("Total images checked :", len(quality_df))
# print("Unreadable images    :", (~quality_df["valid"]).sum())
# 
# print("\nImage dimensions:")
# print(
#     "Height:",
#     quality_df["height"].min(),
#     "to",
#     quality_df["height"].max()
# )
# 
# print(
#     "Width :",
#     quality_df["width"].min(),
#     "to",
#     quality_df["width"].max()
# )
# 
# print("\nIntensity statistics:")
# print(
#     "Mean intensity:",
#     f"{quality_df['mean'].mean():.2f}"
# )
# 
# print(
#     "Average standard deviation:",
#     f"{quality_df['std'].mean():.2f}"
# )
# 
# ============================================================
# FLAG POTENTIALLY PROBLEMATIC IMAGES
# ============================================================
# 
# low_contrast = quality_df[
#     quality_df["std"] < 10
# ]
# 
# very_small = quality_df[
#     (quality_df["height"] < 100) |
#     (quality_df["width"] < 100)
# ]
# 
# print("\n==============================================")
# print("QUALITY FLAGS")
# print("==============================================")
# 
# print(
#     "Low-contrast images (std < 10):",
#     len(low_contrast)
# )
# 
# print(
#     "Very small images (<100 px):",
#     len(very_small)
# )
# 
# ============================================================
# DISPLAY FLAGS
# ============================================================
# 
# if len(low_contrast) > 0:
# 
#     print("\nLow-contrast images:")
#     print(
#         low_contrast[
#             ["filepath", "class", "height", "width", "mean", "std"]
#         ]
#     )
# 
# if len(very_small) > 0:
# 
#     print("\nVery small images:")
#     print(
#         very_small[
#             ["filepath", "class", "height", "width"]
#         ]
#     )
# 
# print("\n[OK] Quality checking completed.")
# 
# 
# ============================================================
# SOURCE COLAB CELL 9
# ============================================================
# 
# ============================================================
# CELL 8 — IMPROVED LUNG-FIELD SEGMENTATION
# ============================================================

import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def segment_lung(image_input):

    # --------------------------------------------------------
    # 1. Read image
    # --------------------------------------------------------
    if isinstance(image_input, (str, Path)):
        original = cv2.imread(
            str(image_input),
            cv2.IMREAD_GRAYSCALE
        )
    else:
        original = np.asarray(image_input).copy()

        if original.ndim == 3:
            original = cv2.cvtColor(
                original,
                cv2.COLOR_BGR2GRAY
            )

    if original is None:
        raise ValueError("Unable to read image.")

    h, w = original.shape

    # --------------------------------------------------------
    # 2. Gaussian smoothing
    # --------------------------------------------------------
    blurred = cv2.GaussianBlur(
        original,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # 3. Otsu threshold
    #    Dark regions become foreground
    # --------------------------------------------------------
    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 4. Restrict analysis to central thoracic region
    #
    # This prevents external dark regions and the lower
    # scanner/table area from becoming lung components.
    # --------------------------------------------------------
    thorax_region = np.zeros_like(threshold)

    x1 = int(0.08 * w)
    x2 = int(0.92 * w)

    y1 = int(0.08 * h)
    y2 = int(0.86 * h)

    thorax_region[y1:y2, x1:x2] = 255

    threshold = cv2.bitwise_and(
        threshold,
        thorax_region
    )

    # --------------------------------------------------------
    # 5. Morphological opening
    # --------------------------------------------------------
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1
    )

    # --------------------------------------------------------
    # 6. Small closing
    # --------------------------------------------------------
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=1
    )

    # --------------------------------------------------------
    # 7. Connected-component analysis
    # --------------------------------------------------------
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )
    )

    lung_mask = np.zeros_like(cleaned)

    # Minimum component size
    min_area = 0.0015 * h * w

    center_x = w / 2
    center_y = h / 2

    # --------------------------------------------------------
    # 8. Score candidate components
    #
    # Components are evaluated according to:
    # - size
    # - position
    # - whether they fall in a plausible lung region
    # --------------------------------------------------------
    candidates = []

    for label_id in range(1, num_labels):

        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        cx, cy = centroids[label_id]

        # Ignore components too close to the outer image
        if cx < 0.08*w or cx > 0.92*w:
            continue

        if cy < 0.08*h or cy > 0.84*h:
            continue

        # Distance from vertical center
        vertical_distance = abs(cy - center_y) / h

        # Prefer components on either side of the mediastinum
        horizontal_distance = abs(cx - center_x) / w

        # Lung candidates should generally be away from the
        # exact central mediastinal region.
        if horizontal_distance < 0.04:
            continue

        candidates.append({
            "label": label_id,
            "area": area,
            "cx": cx,
            "cy": cy,
            "vertical_distance": vertical_distance
        })

    # --------------------------------------------------------
    # 9. Process left and right lung regions separately
    # --------------------------------------------------------
    left_candidates = [
        c for c in candidates
        if c["cx"] < center_x
    ]

    right_candidates = [
        c for c in candidates
        if c["cx"] >= center_x
    ]

    # Sort by area
    left_candidates = sorted(
        left_candidates,
        key=lambda x: x["area"],
        reverse=True
    )

    right_candidates = sorted(
        right_candidates,
        key=lambda x: x["area"],
        reverse=True
    )

    # --------------------------------------------------------
    # 10. Keep multiple components per lung
    #
    # This is important for abnormal/malignant CTs where
    # the lung may be fragmented by pathology.
    # --------------------------------------------------------
    selected_left = left_candidates[:5]
    selected_right = right_candidates[:5]

    for candidate in selected_left + selected_right:
        lung_mask[
            labels == candidate["label"]
        ] = 255

    # --------------------------------------------------------
    # 11. Morphological refinement
    # --------------------------------------------------------
    lung_mask = cv2.morphologyEx(
        lung_mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2
    )

    # Fill internal holes
    binary = lung_mask > 0

    binary = ndi.binary_fill_holes(
        binary
    )

    lung_mask = (
        binary.astype(np.uint8) * 255
    )

    # --------------------------------------------------------
    # 12. Remove tiny remaining components
    # --------------------------------------------------------
    num_labels2, labels2, stats2, _ = (
        cv2.connectedComponentsWithStats(
            lung_mask,
            connectivity=8
        )
    )

    final_mask = np.zeros_like(lung_mask)

    for label_id in range(1, num_labels2):

        area = stats2[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area >= 0.002 * h * w:
            final_mask[
                labels2 == label_id
            ] = 255

    lung_mask = final_mask

    # --------------------------------------------------------
    # 13. Create segmented ROI
    # --------------------------------------------------------
    segmented_roi = original.copy()

    segmented_roi[
        lung_mask == 0
    ] = 0

    return (
        original,
        threshold,
        lung_mask,
        segmented_roi
    )


# ============================================================
# SOURCE COLAB CELL 10
# ============================================================

# ============================================================
# CELL 8B — VISUALIZE LUNG SEGMENTATION
# ============================================================

# Select one image from each class
# sample_paths = []
# 
# for class_name in CLASS_NAMES:
# 
#     sample = train_df[
#         train_df["class"] == class_name
#     ].iloc[0]["filepath"]
# 
#     sample_paths.append(
#         (class_name, sample)
#     )
# 
# 
# fig, axes = plt.subplots(
#     len(sample_paths),
#     4,
#     figsize=(16, 12)
# )
# 
# for row, (class_name, image_path) in enumerate(sample_paths):
# 
#     original, threshold, lung_mask, segmented_roi = (
#         segment_lung(image_path)
#     )
# 
    # Original
#     axes[row, 0].imshow(
#         original,
#         cmap="gray"
#     )
#     axes[row, 0].set_title(
#         f"{class_name} — Original"
#     )
# 
    # Otsu threshold
#     axes[row, 1].imshow(
#         threshold,
#         cmap="gray"
#     )
#     axes[row, 1].set_title(
#         "Otsu Threshold"
#     )
# 
    # Lung mask
#     axes[row, 2].imshow(
#         lung_mask,
#         cmap="gray"
#     )
#     axes[row, 2].set_title(
#         "Lung Mask"
#     )
# 
    # Segmented ROI
#     axes[row, 3].imshow(
#         segmented_roi,
#         cmap="gray"
#     )
#     axes[row, 3].set_title(
#         "Segmented Lung ROI"
#     )
# 
#     for col in range(4):
#         axes[row, col].axis("off")
# 
# plt.tight_layout()
# plt.show()
# 
# 
# ============================================================
# SOURCE COLAB CELL 12
# ============================================================
# 
# ============================================================
# CELL 9 — CT IMAGE PREPROCESSING
# ============================================================
# 
def preprocess_image(
    image_input,
    target_size=(224, 224)
):
    """
    Complete preprocessing pipeline.

    1. Lung-field segmentation
    2. Resize
    3. Median filtering
    4. CLAHE contrast enhancement
    5. Apply lung mask
    6. Normalize to [0, 1]
    """

    # --------------------------------------------------------
    # 1. Lung-field segmentation
    # --------------------------------------------------------
    original, threshold, lung_mask, segmented_roi = (
        segment_lung(image_input)
    )

    # --------------------------------------------------------
    # 2. Resize image and mask
    # --------------------------------------------------------
    resized_roi = cv2.resize(
        segmented_roi,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    resized_mask = cv2.resize(
        lung_mask,
        target_size,
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # 3. Median filtering
    # --------------------------------------------------------
    denoised = cv2.medianBlur(
        resized_roi,
        3
    )

    # --------------------------------------------------------
    # 4. CLAHE
    # --------------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    # --------------------------------------------------------
    # 5. Apply lung mask
    # --------------------------------------------------------
    enhanced[
        resized_mask == 0
    ] = 0

    # --------------------------------------------------------
    # 6. Normalize to [0, 1]
    # --------------------------------------------------------
    normalized = (
        enhanced.astype(np.float32) / 255.0
    )

    return {
        "original": original,
        "segmented_roi": segmented_roi,
        "resized": resized_roi,
        "denoised": denoised,
        "enhanced": enhanced,
        "mask": resized_mask,
        "normalized": normalized
    }


# ============================================================
# SOURCE COLAB CELL 13
# ============================================================

# ============================================================
# CELL 9B — VISUALIZE COMPLETE PREPROCESSING
# ============================================================

# One image from each class
preprocessing_samples = []

# for class_name in CLASS_NAMES:
# 
#     sample = train_df[
#         train_df["class"] == class_name
#     ].iloc[0]["filepath"]
# 
#     preprocessing_samples.append(
#         (class_name, sample)
#     )
# 
# 
# fig, axes = plt.subplots(
#     len(preprocessing_samples),
#     5,
#     figsize=(18, 11)
# )
# 
# 
# for row, (class_name, image_path) in enumerate(
#     preprocessing_samples
# ):
# 
#     result = preprocess_image(
#         image_path
#     )
# 
    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------
#     axes[row, 0].imshow(
#         result["original"],
#         cmap="gray"
#     )
# 
#     axes[row, 0].set_title(
#         f"{class_name}\nOriginal"
#     )
# 
    # --------------------------------------------------------
    # Segmented ROI
    # --------------------------------------------------------
#     axes[row, 1].imshow(
#         result["segmented_roi"],
#         cmap="gray"
#     )
# 
#     axes[row, 1].set_title(
#         "Segmented ROI"
#     )
# 
    # --------------------------------------------------------
    # Denoised
    # --------------------------------------------------------
#     axes[row, 2].imshow(
#         result["denoised"],
#         cmap="gray"
#     )
# 
#     axes[row, 2].set_title(
#         "Median Filter"
#     )
# 
    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------
#     axes[row, 3].imshow(
#         result["enhanced"],
#         cmap="gray"
#     )
# 
#     axes[row, 3].set_title(
#         "CLAHE"
#     )
# 
    # --------------------------------------------------------
    # Normalized
    # --------------------------------------------------------
#     axes[row, 4].imshow(
#         result["normalized"],
#         cmap="gray",
#         vmin=0,
#         vmax=1
#     )
# 
#     axes[row, 4].set_title(
#         "Normalized [0,1]"
#     )
# 
#     for col in range(5):
#         axes[row, col].axis("off")
# 
# 
# plt.tight_layout()
# plt.show()