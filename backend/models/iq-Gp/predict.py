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
# EXTRACTED FROM ORIGINAL COLAB FINAL PREDICTION CELLS
# Cells: 52 and 56
# This is the exact test-set prediction workflow from the notebook.
# It is NOT an invented single-image upload predictor.
# ============================================================

import numpy as np



# ==================== ORIGINAL COLAB CELL 52 ====================

# ============================================================
# FINAL GP MODEL
# TEST SET PREDICTION
# ============================================================
#
# IMPORTANT:
#
# The test set is being used for FINAL evaluation only.
#
# We are NOT changing the model based on these results.
# ============================================================


print("============================================================")
print("FINAL GP MODEL - TEST PREDICTION")
print("============================================================")


# ------------------------------------------------------------
# TEST PROBABILITIES
# ------------------------------------------------------------

test_normal_probability = (

    best_gp_models["Normal"]
    .predict_proba(

        X_test_pca

    )[:, 1]

)


test_benign_probability = (

    best_gp_models["Benign"]
    .predict_proba(

        X_test_pca

    )[:, 1]

)


test_malignant_probability = (

    best_gp_models["Malignant"]
    .predict_proba(

        X_test_pca

    )[:, 1]

)


# ------------------------------------------------------------
# COMBINE THREE GP SCORES
# ------------------------------------------------------------

test_score_matrix = np.column_stack([

    test_normal_probability,

    test_benign_probability,

    test_malignant_probability

])


# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------

test_score_sum = (

    test_score_matrix.sum(

        axis=1,

        keepdims=True

    )

)


test_score_sum = np.maximum(

    test_score_sum,

    1e-8

)


gp_test_probabilities = (

    test_score_matrix
    /
    test_score_sum

)


# ------------------------------------------------------------
# FINAL TEST PREDICTIONS
# ------------------------------------------------------------

gp_test_predictions = np.argmax(

    gp_test_probabilities,

    axis=1

)


print(
    "Test samples:",
    len(gp_test_predictions)
)


print(
    "Probability matrix:",
    gp_test_probabilities.shape
)


print(
    "Prediction vector:",
    gp_test_predictions.shape
)


print("\n[OK] Final GP test prediction completed.")


# ==================== ORIGINAL COLAB CELL 56 ====================

# ============================================================
# FINAL GP TEST PREDICTION DISTRIBUTION
# ============================================================


print("============================================================")
print("FINAL GP TEST PREDICTION DISTRIBUTION")
print("============================================================")


class_names = [

    "Normal",
    "Benign",
    "Malignant"

]


for class_index, class_name in enumerate(

    class_names

):

    predicted_count = np.sum(

        gp_test_predictions
        ==
        class_index

    )


    actual_count = np.sum(

        y_test
        ==
        class_index

    )


    print(

        f"{class_name:<12} | "
        f"Actual: {actual_count:3d} | "
        f"Predicted: {predicted_count:3d}"

    )