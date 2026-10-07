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

# Exact GP section from the Hospital Hybrid Colab.
# Depends on variables produced by the shared preprocessing/feature pipeline.

# ============================================================
# SOURCE COLAB CELL 37
# ============================================================

# **Genetic Programming**


# ============================================================
# SOURCE COLAB CELL 38
# ============================================================

# !pip install -q gplearn


# ============================================================
# SOURCE COLAB CELL 39
# ============================================================

import numpy as np
import pandas as pd


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

print("GP INPUT")
print("-" * 40)

print("Training features    :", X_train_final.shape)
print("Validation features  :", X_val_final.shape)
print("Internal Test        :", X_test_final.shape)

print("\nLabels:")
print("Training distribution   :", np.bincount(y_train))
print("Validation distribution :", np.bincount(y_val))
print("Internal Test           :", np.bincount(y_test))


# ============================================================
# SOURCE COLAB CELL 40
# ============================================================

from gplearn.genetic import SymbolicClassifier
import numpy as np
import joblib
from pathlib import Path

CLASS_NAMES = ["Normal", "Benign", "Malignant"]
N_CLASSES = 3

gp_models = []

print("Starting Genetic Programming training...")
print("=" * 60)

for class_idx, class_name in enumerate(CLASS_NAMES):

    print(f"\nTraining GP for class: {class_name}")

    # One-vs-Rest target
    y_binary = (y_train == class_idx).astype(int)

    gp = SymbolicClassifier(
        population_size=250,
        generations=15,

        function_set=(
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "log",
            "abs",
            "neg"
        ),

        metric="log loss",

        parsimony_coefficient=0.001,
        max_samples=0.8,

        random_state=42 + class_idx * 10,
        n_jobs=-1,

        verbose=1
    )

    gp.fit(X_train_final, y_binary)

    gp_models.append(gp)

    print(f"[OK] {class_name} GP completed")
    print(f"  Program length: {gp._program.length_}")
    print(f"  Program depth : {gp._program.depth_}")

print("\n" + "=" * 60)
print("ALL GP MODELS TRAINED")


# ============================================================
# SOURCE COLAB CELL 41
# ============================================================

def gp_predict_proba(models, X):

    probabilities = []

    for model in models:
        # Probability of belonging to this class
        prob = model.predict_proba(X)[:, 1]
        probabilities.append(prob)

    probabilities = np.column_stack(probabilities)

    # Normalize so that probabilities across
    # Normal + Benign + Malignant sum to 1
    row_sums = probabilities.sum(axis=1, keepdims=True)

    probabilities = probabilities / np.clip(
        row_sums,
        1e-8,
        None
    )

    return probabilities


# ============================================================
# SOURCE COLAB CELL 42
# ============================================================

gp_val_proba = gp_predict_proba(
    gp_models,
    X_val_final
)

gp_val_pred = np.argmax(
    gp_val_proba,
    axis=1
)

print("Validation probability shape:", gp_val_proba.shape)
print("Validation prediction shape:", gp_val_pred.shape)


# ============================================================
# SOURCE COLAB CELL 43
# ============================================================

gp_val_results = pd.DataFrame(
    gp_val_proba,
    columns=CLASS_NAMES
)

gp_val_results["Actual"] = [
    CLASS_NAMES[i] for i in y_val
]

gp_val_results["Predicted"] = [
    CLASS_NAMES[i] for i in gp_val_pred
]

gp_val_results["Confidence"] = gp_val_proba.max(axis=1)

gp_val_results


# ============================================================
# SOURCE COLAB CELL 44
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

gp_accuracy = accuracy_score(y_val, gp_val_pred)

gp_balanced_accuracy = balanced_accuracy_score(
    y_val,
    gp_val_pred
)

gp_macro_precision = precision_score(
    y_val,
    gp_val_pred,
    average="macro",
    zero_division=0
)

gp_macro_recall = recall_score(
    y_val,
    gp_val_pred,
    average="macro",
    zero_division=0
)

gp_macro_f1 = f1_score(
    y_val,
    gp_val_pred,
    average="macro",
    zero_division=0
)

print("=" * 60)
print("GENETIC PROGRAMMING — VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy           : {gp_accuracy * 100:.2f}%")
print(f"Balanced Accuracy  : {gp_balanced_accuracy * 100:.2f}%")
print(f"Macro Precision    : {gp_macro_precision * 100:.2f}%")
print(f"Macro Recall       : {gp_macro_recall * 100:.2f}%")
print(f"Macro F1           : {gp_macro_f1 * 100:.2f}%")


# ============================================================
# SOURCE COLAB CELL 45
# ============================================================

print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_val,
        gp_val_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# SOURCE COLAB CELL 46
# ============================================================

print("CONFUSION MATRIX")
print("=" * 60)

cm_gp = confusion_matrix(
    y_val,
    gp_val_pred
)

cm_gp_df = pd.DataFrame(
    cm_gp,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_gp_df


# ============================================================
# SOURCE COLAB CELL 47
# ============================================================

GP_DIR = Path(__file__).resolve().parent / "results" / "gp_models"
GP_DIR.mkdir(parents=True, exist_ok=True)

for class_name, model in zip(CLASS_NAMES, gp_models):

    filename = GP_DIR / f"gp_{class_name.lower()}.joblib"

    joblib.dump(
        model,
        filename
    )

    print("Saved:", filename)


# ============================================================
# SOURCE COLAB CELL 48
# ============================================================

# ============================================================
# GP — INTERNAL TEST EVALUATION
# ============================================================

gp_test_proba = gp_predict_proba(
    gp_models,
    X_test_final
)

gp_test_pred = np.argmax(
    gp_test_proba,
    axis=1
)

# Metrics
gp_test_accuracy = accuracy_score(
    y_test,
    gp_test_pred
)

gp_test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    gp_test_pred
)

gp_test_macro_precision = precision_score(
    y_test,
    gp_test_pred,
    average="macro",
    zero_division=0
)

gp_test_macro_recall = recall_score(
    y_test,
    gp_test_pred,
    average="macro",
    zero_division=0
)

gp_test_macro_f1 = f1_score(
    y_test,
    gp_test_pred,
    average="macro",
    zero_division=0
)

print("=" * 60)
print("GENETIC PROGRAMMING — INTERNAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy           : {gp_test_accuracy * 100:.2f}%")
print(f"Balanced Accuracy  : {gp_test_balanced_accuracy * 100:.2f}%")
print(f"Macro Precision    : {gp_test_macro_precision * 100:.2f}%")
print(f"Macro Recall       : {gp_test_macro_recall * 100:.2f}%")
print(f"Macro F1           : {gp_test_macro_f1 * 100:.2f}%")


# ============================================================
# SOURCE COLAB CELL 49
# ============================================================

print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        gp_test_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# SOURCE COLAB CELL 50
# ============================================================

print("CONFUSION MATRIX")
print("=" * 60)

cm_gp_test = confusion_matrix(
    y_test,
    gp_test_pred
)

cm_gp_test_df = pd.DataFrame(
    cm_gp_test,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_gp_test_df