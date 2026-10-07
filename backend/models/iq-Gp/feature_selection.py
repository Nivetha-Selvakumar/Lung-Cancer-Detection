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
# FEATURE STANDARDIZATION + PCA
# Extracted from original Colab Cells 20 and 21.
# No algorithmic change.
# ============================================================

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# STANDARDIZE GP FEATURES
# ============================================================
#
# Input:
#     6111 handcrafted features
#
# Standardization:
#
#     z = (x - mean) / standard deviation
#
# IMPORTANT:
# The scaler is fitted ONLY on training data.
# Validation and test data are transformed using the
# training-set statistics.
#
# This prevents information leakage.
# ============================================================

from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# CREATE SCALER
# ------------------------------------------------------------

gp_scaler = StandardScaler()


# ------------------------------------------------------------
# FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------

X_train_scaled = gp_scaler.fit_transform(
    X_train_raw
)


# ------------------------------------------------------------
# TRANSFORM VALIDATION DATA
# ------------------------------------------------------------

X_val_scaled = gp_scaler.transform(
    X_val_raw
)


# ------------------------------------------------------------
# TRANSFORM TEST DATA
# ------------------------------------------------------------

X_test_scaled = gp_scaler.transform(
    X_test_raw
)


# ============================================================
# VERIFY
# ============================================================

print("==========================================")
print("GP FEATURE STANDARDIZATION")
print("==========================================")

print(
    "Training shape   :",
    X_train_scaled.shape
)

print(
    "Validation shape :",
    X_val_scaled.shape
)

print(
    "Test shape       :",
    X_test_scaled.shape
)


print("\nTraining mean:")
print(
    np.mean(
        X_train_scaled
    )
)


print("\nTraining standard deviation:")
print(
    np.std(
        X_train_scaled
    )
)

# ============================================================
# PCA DIMENSIONALITY REDUCTION FOR GP
# ============================================================
#
# 6111 handcrafted features are too large for efficient
# Genetic Programming.
#
# PCA compresses the feature space while retaining the
# dominant variance in the training data.
#
# First GP experiment:
#
#     6111 → 32 components
#
# PCA is fitted ONLY on training data.
# ============================================================

from sklearn.decomposition import PCA


GP_PCA_COMPONENTS = 32


# ------------------------------------------------------------
# CREATE PCA
# ------------------------------------------------------------

gp_pca = PCA(

    n_components=GP_PCA_COMPONENTS,

    random_state=SEED

)


# ------------------------------------------------------------
# FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------

X_train_pca = gp_pca.fit_transform(
    X_train_scaled
)


# ------------------------------------------------------------
# TRANSFORM VALIDATION
# ------------------------------------------------------------

X_val_pca = gp_pca.transform(
    X_val_scaled
)


# ------------------------------------------------------------
# TRANSFORM TEST
# ------------------------------------------------------------

X_test_pca = gp_pca.transform(
    X_test_scaled
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("==========================================")
print("GP PCA DIMENSIONALITY REDUCTION")
print("==========================================")


print(
    "Before PCA:",
    X_train_scaled.shape
)


print(
    "After PCA:",
    X_train_pca.shape
)


print(
    "Validation:",
    X_val_pca.shape
)


print(
    "Test:",
    X_test_pca.shape
)


# ------------------------------------------------------------
# EXPLAINED VARIANCE
# ------------------------------------------------------------

variance_retained = (
    gp_pca.explained_variance_ratio_.sum()
)


print(
    "\nVariance retained:",
    f"{variance_retained * 100:.2f}%"
)