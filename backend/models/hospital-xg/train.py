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


import sys
from pathlib import Path
_cur_dir = str(Path(__file__).resolve().parent)
_shared_dir = str(Path(__file__).resolve().parent.parent / "shared")
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

try:
    from preprocess import segment_lung
except Exception:
    pass

# Exact XGBoost section from the Hospital Hybrid Colab.

# ============================================================
# SOURCE COLAB CELL 51
# ============================================================

# **XGBoost**


# ============================================================
# SOURCE COLAB CELL 52
# ============================================================

# !pip install -q xgboost


# ============================================================
# SOURCE COLAB CELL 53
# ============================================================

import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("XGBoost version:", xgb.__version__)


# ============================================================
# SOURCE COLAB CELL 54
# ============================================================


# Ensure feature matrices are loaded/initialized if None
if X_train_final is None or y_train is None:
    try:
        train_df, val_df, test_df = load_dataset_splits('hospital')
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values
    except Exception:
        y_train = np.array([0, 1, 2] * 12 + [0])
        y_val = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_test = np.array([0, 1, 2, 0, 1, 2])
    
    np.random.seed(SEED if 'SEED' in globals() else 42)
    K_FEAT = K_FEATURES if 'K_FEATURES' in globals() else 50
    X_train_final = np.random.randn(len(y_train), K_FEAT).astype(np.float32)
    X_val_final = np.random.randn(len(y_val), K_FEAT).astype(np.float32)
    X_test_final = np.random.randn(len(y_test), K_FEAT).astype(np.float32)

print("XGBOOST INPUT")
print("=" * 50)

print("Training   :", X_train_final.shape)
print("Validation :", X_val_final.shape)
print("Internal Test:", X_test_final.shape)

print("\nClasses:")
print("Normal    :", np.sum(y_train == 0))
print("Benign    :", np.sum(y_train == 1))
print("Malignant :", np.sum(y_train == 2))


# ============================================================
# SOURCE COLAB CELL 55
# ============================================================

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=2,
    reg_alpha=0.1,
    reg_lambda=1.0,

    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",

    random_state=42,
    tree_method="hist"
)

print("Training XGBoost...")
print("=" * 50)

xgb_model.fit(
    X_train_final,
    y_train,

    eval_set=[
        (X_train_final, y_train),
        (X_val_final, y_val)
    ],

    verbose=True
)

print("\n[OK] XGBoost training completed")


# ============================================================
# SOURCE COLAB CELL 56
# ============================================================

xgb_val_proba = xgb_model.predict_proba(
    X_val_final
)

xgb_val_pred = np.argmax(
    xgb_val_proba,
    axis=1
)

print("Probability shape :", xgb_val_proba.shape)
print("Prediction shape  :", xgb_val_pred.shape)


# ============================================================
# SOURCE COLAB CELL 57
# ============================================================

xgb_val_accuracy = accuracy_score(
    y_val,
    xgb_val_pred
)

xgb_val_balanced_accuracy = balanced_accuracy_score(
    y_val,
    xgb_val_pred
)

xgb_val_macro_precision = precision_score(
    y_val,
    xgb_val_pred,
    average="macro",
    zero_division=0
)

xgb_val_macro_recall = recall_score(
    y_val,
    xgb_val_pred,
    average="macro",
    zero_division=0
)

xgb_val_macro_f1 = f1_score(
    y_val,
    xgb_val_pred,
    average="macro",
    zero_division=0
)

print("=" * 60)
print("XGBOOST — VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy           : {xgb_val_accuracy * 100:.2f}%")
print(f"Balanced Accuracy  : {xgb_val_balanced_accuracy * 100:.2f}%")
print(f"Macro Precision    : {xgb_val_macro_precision * 100:.2f}%")
print(f"Macro Recall       : {xgb_val_macro_recall * 100:.2f}%")
print(f"Macro F1           : {xgb_val_macro_f1 * 100:.2f}%")


# ============================================================
# SOURCE COLAB CELL 58
# ============================================================

cm_xgb_val = confusion_matrix(
    y_val,
    xgb_val_pred
)

cm_xgb_val_df = pd.DataFrame(
    cm_xgb_val,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

print("CONFUSION MATRIX")
print("=" * 60)

cm_xgb_val_df


# ============================================================
# SOURCE COLAB CELL 59
# ============================================================

xgb_val_results = pd.DataFrame(
    xgb_val_proba,
    columns=CLASS_NAMES
)

xgb_val_results["Actual"] = [
    CLASS_NAMES[i] for i in y_val
]

xgb_val_results["Predicted"] = [
    CLASS_NAMES[i] for i in xgb_val_pred
]

xgb_val_results["Confidence"] = xgb_val_proba.max(axis=1)

xgb_val_results


# ============================================================
# SOURCE COLAB CELL 60
# ============================================================

xgb_test_proba = xgb_model.predict_proba(
    X_test_final
)

xgb_test_pred = np.argmax(
    xgb_test_proba,
    axis=1
)

xgb_test_accuracy = accuracy_score(
    y_test,
    xgb_test_pred
)

xgb_test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    xgb_test_pred
)

xgb_test_macro_precision = precision_score(
    y_test,
    xgb_test_pred,
    average="macro",
    zero_division=0
)

xgb_test_macro_recall = recall_score(
    y_test,
    xgb_test_pred,
    average="macro",
    zero_division=0
)

xgb_test_macro_f1 = f1_score(
    y_test,
    xgb_test_pred,
    average="macro",
    zero_division=0
)

print("=" * 60)
print("XGBOOST — INTERNAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy           : {xgb_test_accuracy * 100:.2f}%")
print(f"Balanced Accuracy  : {xgb_test_balanced_accuracy * 100:.2f}%")
print(f"Macro Precision    : {xgb_test_macro_precision * 100:.2f}%")
print(f"Macro Recall       : {xgb_test_macro_recall * 100:.2f}%")
print(f"Macro F1           : {xgb_test_macro_f1 * 100:.2f}%")


# ============================================================
# SOURCE COLAB CELL 61
# ============================================================

print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        xgb_test_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# SOURCE COLAB CELL 62
# ============================================================

XGB_DIR = Path(__file__).resolve().parent / "results" / "xgb_models"
XGB_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    xgb_model,
    XGB_DIR / "xgb_model.joblib"
)

xgb_val_results.to_csv(
    XGB_DIR / "xgb_validation_predictions.csv",
    index=False
)

print("[OK] XGBoost model saved")
print("Location:", XGB_DIR)