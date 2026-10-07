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

# ============================================================
# CELL 2 - IMPORT LIBRARIES
# ============================================================


import os
import zipfile
import random
import time
import warnings

import numpy as np
import pandas as pd

import cv2
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# Suppress unnecessary warnings
# ------------------------------------------------------------

warnings.filterwarnings(
    "ignore"
)


print("[OK] NumPy imported")
print("[OK] Pandas imported")
print("[OK] OpenCV imported")
print("[OK] Scikit-learn imported")
print("[OK] XGBoost imported")
print("[OK] All imports completed successfully.")

# ============================================================
# CELL 12 - LUNG SEGMENTATION FUNCTION (CLASSICAL IMAGE PROCESSING)
# ============================================================
#
# This is the SAME lung-field segmentation used in the Deep
# Learning (ConvNeXt) notebook: deterministic classical image
# processing, not a trained network. Using the identical
# function in both notebooks means any accuracy difference you
# see between XGBoost and ConvNeXt is due to the CLASSIFIER,
# not to two different segmentation methods.
#
# Why segmentation belongs here too:
#   HOG features are computed from image gradients over the
#   WHOLE image. Without segmentation, the strongest gradients
#   in a CT slice usually come from the rib cage / chest wall
#   and the scanner table -- not from lung tissue. Those
#   irrelevant edges dominate the HOG histogram and get passed
#   into PCA and XGBoost as if they were diagnostic signal.
#   Masking out everything except the lung field first means
#   HOG only describes texture/structure inside the lungs.
#
# Algorithm:
#   1. Otsu-threshold the slice, inverted, so dark regions
#      (air: background + lungs) become foreground.
#   2. Remove any blob touching the image border (removes the
#      background outside the body, leaving only internal air
#      pockets -> the lungs).
#   3. Morphological opening to drop thin noise.
#   4. Keep the largest 1-2 components (left + right lung).
#   5. Fill holes so vessels/nodules (brighter than air) stay
#      INSIDE the lung ROI instead of being punched out.
#   6. Morphological closing + small dilation to smooth the
#      boundary without clipping peripheral nodules.

def segment_lung(image_path):
    """
    Deterministic lung-field segmentation for a 2-D grayscale
    CT slice.

    Returns
    -------
    original : np.ndarray (H, W), uint8
    mask     : np.ndarray (H, W), uint8   binary lung mask {0,1}
    roi      : np.ndarray (H, W), uint8   original image with the
                                           mask applied
    """

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w = original.shape

    blurred = cv2.GaussianBlur(original, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = (binary > 0).astype(np.uint8)

    labeled, _ = ndi.label(binary)

    border_labels = set(labeled[0, :].tolist())
    border_labels |= set(labeled[-1, :].tolist())
    border_labels |= set(labeled[:, 0].tolist())
    border_labels |= set(labeled[:, -1].tolist())
    border_labels.discard(0)

    if border_labels:
        border_cleared = np.where(
            np.isin(labeled, list(border_labels)), 0, binary
        ).astype(np.uint8)
    else:
        border_cleared = binary

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(border_cleared, cv2.MORPH_OPEN, kernel_open, iterations=1)

    labeled2, num2 = ndi.label(opened)

    if num2 == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return original, empty_mask, original.copy()

    sizes = ndi.sum(opened, labeled2, range(1, num2 + 1))
    min_area = 0.003 * h * w

    candidate_labels = [i + 1 for i, s in enumerate(sizes) if s > min_area]

    if len(candidate_labels) == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return original, empty_mask, original.copy()

    candidate_labels = sorted(
        candidate_labels, key=lambda lab: sizes[lab - 1], reverse=True
    )[:2]

    mask = np.isin(labeled2, candidate_labels).astype(np.uint8)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    roi = original.copy()
    roi[mask == 0] = 0

    return original, mask, roi


# ============================================================
# CELL 14 - IMAGE PREPROCESSING FUNCTION (WITH SEGMENTATION)
# ============================================================
#
# Preprocessing steps:
#
#     1. Segment the lung field  (segment_lung, Cell 12)
#     2. Resize the ORIGINAL and the MASK to 224 x 224
#     3. Denoising (on the full resized grayscale image, so
#        CLAHE below still sees a normal intensity range)
#     4. CLAHE contrast enhancement
#     5. Apply the lung mask (zero out everything outside the
#        lungs)
#     6. Normalize pixel values to [0, 1]
#
# NOTE:
# Segmentation is classical image processing (Otsu threshold +
# connected components), not a learned/trained operation, so no
# model weights are needed here.
# ============================================================


IMAGE_SIZE = (
    224,
    224
)


def preprocess_image(
    image_path
):

    """
    Load, segment and preprocess one CT image.

    Returns:
        Preprocessed lung-ROI image as float32 array.
    """

    # --------------------------------------------------------
    # SEGMENT LUNGS
    # --------------------------------------------------------

    original, mask, _ = segment_lung(image_path)

    # --------------------------------------------------------
    # RESIZE IMAGE AND MASK
    # --------------------------------------------------------

    image = cv2.resize(
        original,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    mask_resized = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # DENOISING
    # --------------------------------------------------------
    #
    # Median filtering reduces small noise while preserving
    # important boundaries reasonably well.
    # --------------------------------------------------------

    image = cv2.medianBlur(
        image,
        3
    )

    # --------------------------------------------------------
    # CONTRAST ENHANCEMENT
    # --------------------------------------------------------
    #
    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    #
    # Applied BEFORE masking, on the full image, so CLAHE still
    # sees normal chest-wall intensities as local reference and
    # does not degenerate on all-zero tiles outside the lungs.
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(
        image
    )

    # --------------------------------------------------------
    # APPLY LUNG MASK
    # --------------------------------------------------------
    #
    # Zero out everything outside the segmented lung field so
    # HOG gradients below are computed only from lung tissue.
    # If segmentation failed (empty mask), fall back to the
    # full image rather than returning an all-black image.
    # --------------------------------------------------------

    if mask_resized.sum() > 0:
        image[mask_resized == 0] = 0

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------
    #
    # Convert:
    #
    #     0 - 255
    #
    # into:
    #
    #     0 - 1
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0

    return image