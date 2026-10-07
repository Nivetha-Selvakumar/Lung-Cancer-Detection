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
# EXTRACTED FROM ORIGINAL COLAB CELL 14
# Required imports from original Cells 1 and 11 are included
# only as imports/dependency context. The feature algorithm
# itself is unchanged.
# ============================================================

import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern

from preprocess import preprocess_ct_image

# ============================================================
# GENETIC PROGRAMMING FEATURE EXTRACTION
# ============================================================
#
# Features used:
#
# 1. HOG
#    Histogram of Oriented Gradients
#    Captures structural and edge information.
#
# 2. LBP
#    Local Binary Pattern
#    Captures local texture information.
#
# 3. INTENSITY FEATURES
#    Statistical information from the CT image.
#
# These complementary features provide GP with information
# about:
#
#     Structure + Texture + Intensity
#
# ============================================================




def extract_gp_features(
    image_path
):

    """
    Extract a compact feature vector from one
    preprocessed IQ-OTH/NCCD CT image.
    """


    # ========================================================
    # STEP 1 — PREPROCESS IMAGE
    # ========================================================

    roi = preprocess_ct_image(

        image_path

    )


    # Convert pixel values to [0,1]

    image_float = (

        roi.astype(
            np.float32
        )

        / 255.0

    )


    # ========================================================
    # STEP 2 — HOG FEATURES
    # ========================================================
    #
    # HOG describes local edge and shape information.
    #
    # orientations = 9
    # pixels_per_cell = 16 × 16
    # cells_per_block = 2 × 2
    # ========================================================


    hog_features = hog(

        image_float,

        orientations=9,

        pixels_per_cell=(16, 16),

        cells_per_block=(2, 2),

        block_norm="L2-Hys",

        feature_vector=True

    )


    # ========================================================
    # STEP 3 — LBP FEATURES
    # ========================================================
    #
    # LBP describes local texture patterns.
    # ========================================================


    radius = 2

    points = 8 * radius


    lbp = local_binary_pattern(

        roi,

        points,

        radius,

        method="uniform"

    )


    # Number of histogram bins

    n_bins = points + 2


    lbp_histogram, _ = np.histogram(

        lbp.ravel(),

        bins=np.arange(
            0,
            n_bins + 1
        ),

        range=(
            0,
            n_bins
        )

    )


    # Normalize LBP histogram

    lbp_histogram = (

        lbp_histogram.astype(
            np.float32
        )

        /

        (
            lbp_histogram.sum()
            + 1e-8
        )

    )


    # ========================================================
    # STEP 4 — INTENSITY FEATURES
    # ========================================================
    #
    # Statistical information from the CT image.
    # ========================================================


    intensity_features = np.array([

        np.mean(
            image_float
        ),

        np.std(
            image_float
        ),

        np.min(
            image_float
        ),

        np.max(
            image_float
        ),

        np.percentile(
            image_float,
            10
        ),

        np.percentile(
            image_float,
            25
        ),

        np.percentile(
            image_float,
            50
        ),

        np.percentile(
            image_float,
            75
        ),

        np.percentile(
            image_float,
            90
        )

    ], dtype=np.float32)


    # ========================================================
    # STEP 5 — COMBINE ALL FEATURES
    # ========================================================


    combined_features = np.concatenate([

        hog_features,

        lbp_histogram,

        intensity_features

    ])


    return combined_features.astype(
        np.float32
    )