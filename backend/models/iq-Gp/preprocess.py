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
# EXTRACTED FROM ORIGINAL COLAB CELL 11
# Required imports from original Cell 1 are included only so
# this file can run independently.
# ============================================================

import cv2
import numpy as np

# ============================================================
# GP PREPROCESSING PIPELINE
# ============================================================
#
# IQ-OTH/NCCD CT images
#
# Preprocessing:
#
#   Original CT image
#          ↓
#   Resize to 224 × 224
#          ↓
#   Denoising
#          ↓
#   Normalization
#          ↓
#   ROI extraction
#
# NOTE:
# The ROI step below is a computational image-processing
# approximation. It is NOT a clinically validated lung
# segmentation algorithm.
# ============================================================


TARGET_SIZE = (224, 224)


def preprocess_ct_image(
    image_path,
    return_all=False
):

    """
    Preprocess one IQ-OTH/NCCD CT image.

    Parameters
    ----------
    image_path : str
        Path to the CT image.

    return_all : bool
        If True, return all preprocessing stages.
        If False, return only the final ROI image.

    Returns
    -------
    numpy array
        Preprocessed ROI image.
    """


    # ========================================================
    # STEP 1 — LOAD IMAGE
    # ========================================================

    original = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )


    if original is None:

        raise ValueError(
            f"Unable to read image:\n{image_path}"
        )


    # ========================================================
    # STEP 2 — RESIZE
    # ========================================================

    resized = cv2.resize(

        original,

        TARGET_SIZE,

        interpolation=cv2.INTER_AREA

    )


    # ========================================================
    # STEP 3 — DENOISING
    # ========================================================
    #
    # Gaussian filtering is used to reduce small-scale noise.
    # ========================================================

    denoised = cv2.GaussianBlur(

        resized,

        (5, 5),

        sigmaX=0

    )


    # ========================================================
    # STEP 4 — NORMALIZATION
    # ========================================================
    #
    # Convert image intensity to the range [0, 255].
    # ========================================================

    normalized = cv2.normalize(

        denoised,

        None,

        alpha=0,

        beta=255,

        norm_type=cv2.NORM_MINMAX

    )


    # ========================================================
    # STEP 5 — ROI EXTRACTION
    # ========================================================
    #
    # Otsu thresholding is used to obtain a broad foreground
    # region. Morphological operations remove small regions
    # and close small gaps.
    #
    # This is an image-processing ROI approximation.
    # ========================================================

    _, roi_mask = cv2.threshold(

        normalized,

        0,

        255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU

    )


    # Morphological kernel

    kernel = np.ones(

        (5, 5),

        dtype=np.uint8

    )


    # Remove small isolated regions

    roi_mask = cv2.morphologyEx(

        roi_mask,

        cv2.MORPH_OPEN,

        kernel

    )


    # Close small gaps

    roi_mask = cv2.morphologyEx(

        roi_mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    # Apply mask

    roi = cv2.bitwise_and(

        normalized,

        normalized,

        mask=roi_mask

    )


    # ========================================================
    # RETURN
    # ========================================================

    if return_all:

        return {

            "original": original,

            "resized": resized,

            "denoised": denoised,

            "normalized": normalized,

            "roi_mask": roi_mask,

            "roi": roi

        }


    return roi