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
gp_predictions = None
best_gp_predictions = None
gp_test_predictions = None
best_gp_iteration = None

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
# EXTRACTED FROM ORIGINAL COLAB METRIC/EVALUATION CELLS
# Cells: 31,32,33,34,35,50,51,53,54,55,56
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix



# ==================== ORIGINAL COLAB CELL 31 ====================

# ============================================================
# GP ITERATION 1
# OVERALL VALIDATION ACCURACY
# ============================================================


from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# CORRECT
# ------------------------------------------------------------

correct_predictions = np.sum(

    gp_predictions == y_val

)


# ------------------------------------------------------------
# TOTAL
# ------------------------------------------------------------

total_predictions = len(
    y_val
)


# ------------------------------------------------------------
# INCORRECT
# ------------------------------------------------------------

incorrect_predictions = (

    total_predictions
    -
    correct_predictions

)


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

gp_accuracy = (

    correct_predictions
    /
    total_predictions

)


print("==========================================")
print("GP ITERATION 1")
print("OVERALL VALIDATION ACCURACY")
print("==========================================")


print(
    "Correct predictions   :",
    correct_predictions
)


print(
    "Incorrect predictions :",
    incorrect_predictions
)


print(
    "Total predictions     :",
    total_predictions
)


print(
    f"\nOverall Accuracy      : "
    f"{gp_accuracy * 100:.2f}%"
)


# ==================== ORIGINAL COLAB CELL 32 ====================

# ============================================================
# GP ITERATION 1
# BALANCED PERFORMANCE METRICS
# ============================================================


from sklearn.metrics import (

    balanced_accuracy_score,

    f1_score,

    precision_score,

    recall_score

)


# ------------------------------------------------------------
# BALANCED ACCURACY
# ------------------------------------------------------------

gp_balanced_accuracy = (

    balanced_accuracy_score(

        y_val,

        gp_predictions

    )

)


# ------------------------------------------------------------
# MACRO PRECISION
# ------------------------------------------------------------

gp_macro_precision = (

    precision_score(

        y_val,

        gp_predictions,

        average="macro",

        zero_division=0

    )

)


# ------------------------------------------------------------
# MACRO RECALL
# ------------------------------------------------------------

gp_macro_recall = (

    recall_score(

        y_val,

        gp_predictions,

        average="macro",

        zero_division=0

    )

)


# ------------------------------------------------------------
# MACRO F1
# ------------------------------------------------------------

gp_macro_f1 = (

    f1_score(

        y_val,

        gp_predictions,

        average="macro",

        zero_division=0

    )

)


# ------------------------------------------------------------
# WEIGHTED F1
# ------------------------------------------------------------

gp_weighted_f1 = (

    f1_score(

        y_val,

        gp_predictions,

        average="weighted",

        zero_division=0

    )

)


# ============================================================
# DISPLAY
# ============================================================

print("==========================================")
print("GP ITERATION 1")
print("VALIDATION METRICS")
print("==========================================")


print(
    f"Accuracy          : "
    f"{gp_accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy : "
    f"{gp_balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision   : "
    f"{gp_macro_precision:.4f}"
)


print(
    f"Macro Recall      : "
    f"{gp_macro_recall:.4f}"
)


print(
    f"Macro F1          : "
    f"{gp_macro_f1:.4f}"
)


print(
    f"Weighted F1       : "
    f"{gp_weighted_f1:.4f}"
)


# ==================== ORIGINAL COLAB CELL 33 ====================

# ============================================================
# GP ITERATION 1
# CLASSIFICATION REPORT
# ============================================================


from sklearn.metrics import classification_report


print("==========================================")
print("GP ITERATION 1")
print("CLASSIFICATION REPORT")
print("==========================================")


print(

    classification_report(

        y_val,

        gp_predictions,

        labels=[

            0,
            1,
            2

        ],

        target_names=[

            "Normal",
            "Benign",
            "Malignant"

        ],

        digits=4,

        zero_division=0

    )

)


# ==================== ORIGINAL COLAB CELL 34 ====================

# ============================================================
# GP ITERATION 1
# CONFUSION MATRIX
# ============================================================


from sklearn.metrics import confusion_matrix



cm_gp = confusion_matrix(

    y_val,

    gp_predictions,

    labels=[

        0,
        1,
        2

    ]

)


print("==========================================")
print("GP ITERATION 1")
print("CONFUSION MATRIX")
print("==========================================")


print(
    cm_gp
)


# ============================================================
# PLOT
# ============================================================


plt.figure(

    figsize=(7, 6)

)


sns.heatmap(

    cm_gp,

    annot=True,

    fmt="d",

    xticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ],

    yticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ]

)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.title(
    "GP Iteration 1 - Validation Confusion Matrix"
)


plt.tight_layout()


plt.show()


# ==================== ORIGINAL COLAB CELL 35 ====================

# ============================================================
# GP ITERATION 1
# CLASS-WISE PERFORMANCE
# ============================================================
#
# This is particularly important because the dataset is
# imbalanced.
# ============================================================


print("==========================================")
print("GP ITERATION 1")
print("CLASS-WISE PERFORMANCE")
print("==========================================")


for class_index, class_name in enumerate([

    "Normal",
    "Benign",
    "Malignant"

]):


    # Number of actual samples

    actual_count = np.sum(

        y_val == class_index

    )


    # Number correctly classified

    correct_count = np.sum(

        (

            y_val == class_index

        )

        &

        (

            gp_predictions == class_index

        )

    )


    # Calculate recall / class accuracy

    if actual_count > 0:

        class_recall = (

            correct_count
            /
            actual_count

        )

    else:

        class_recall = 0.0


    print(

        f"{class_name:<12}: "
        f"{correct_count}/"
        f"{actual_count} correct "
        f"("
        f"{class_recall * 100:.2f}%"
        f")"

    )


# ==================== ORIGINAL COLAB CELL 50 ====================

# ============================================================
# FINAL GP VALIDATION CLASSIFICATION REPORT
# ============================================================


print("============================================================")
print("FINAL GP MODEL - VALIDATION REPORT")
print("============================================================")


print(
    classification_report(

        y_val,

        best_gp_predictions,

        labels=[0, 1, 2],

        target_names=[

            "Normal",
            "Benign",
            "Malignant"

        ],

        digits=4,

        zero_division=0

    )
)


# ==================== ORIGINAL COLAB CELL 51 ====================

# ============================================================
# FINAL GP VALIDATION CONFUSION MATRIX
# ============================================================


final_gp_val_cm = confusion_matrix(

    y_val,

    best_gp_predictions,

    labels=[0, 1, 2]

)


print("============================================================")
print("FINAL GP VALIDATION CONFUSION MATRIX")
print("============================================================")


print(
    final_gp_val_cm
)


plt.figure(
    figsize=(7, 6)
)


sns.heatmap(

    final_gp_val_cm,

    annot=True,

    fmt="d",

    xticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ],

    yticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ]

)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.title(
    f"Final GP Validation Confusion Matrix "
    f"(Iteration {best_gp_iteration})"
)


plt.tight_layout()


plt.show()


# ==================== ORIGINAL COLAB CELL 53 ====================

# ============================================================
# FINAL GP TEST PERFORMANCE
# ============================================================


gp_test_accuracy = accuracy_score(

    y_test,

    gp_test_predictions

)


gp_test_balanced_accuracy = (

    balanced_accuracy_score(

        y_test,

        gp_test_predictions

    )

)


gp_test_macro_precision = (

    precision_score(

        y_test,

        gp_test_predictions,

        average="macro",

        zero_division=0

    )

)


gp_test_macro_recall = (

    recall_score(

        y_test,

        gp_test_predictions,

        average="macro",

        zero_division=0

    )

)


gp_test_macro_f1 = (

    f1_score(

        y_test,

        gp_test_predictions,

        average="macro",

        zero_division=0

    )

)


gp_test_weighted_f1 = (

    f1_score(

        y_test,

        gp_test_predictions,

        average="weighted",

        zero_division=0

    )


)


print("============================================================")
print("FINAL GP TEST PERFORMANCE")
print("============================================================")


print(
    f"Accuracy           : "
    f"{gp_test_accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy  : "
    f"{gp_test_balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision    : "
    f"{gp_test_macro_precision:.4f}"
)


print(
    f"Macro Recall       : "
    f"{gp_test_macro_recall:.4f}"
)


print(
    f"Macro F1           : "
    f"{gp_test_macro_f1:.4f}"
)


print(
    f"Weighted F1        : "
    f"{gp_test_weighted_f1:.4f}"
)


# ==================== ORIGINAL COLAB CELL 54 ====================

# ============================================================
# FINAL GP TEST CLASSIFICATION REPORT
# ============================================================


print("============================================================")
print("FINAL GP TEST CLASSIFICATION REPORT")
print("============================================================")


print(

    classification_report(

        y_test,

        gp_test_predictions,

        labels=[0, 1, 2],

        target_names=[

            "Normal",
            "Benign",
            "Malignant"

        ],

        digits=4,

        zero_division=0

    )

)


# ==================== ORIGINAL COLAB CELL 55 ====================

# ============================================================
# FINAL GP TEST CONFUSION MATRIX
# ============================================================


final_gp_test_cm = confusion_matrix(

    y_test,

    gp_test_predictions,

    labels=[0, 1, 2]

)


print("============================================================")
print("FINAL GP TEST CONFUSION MATRIX")
print("============================================================")


print(
    final_gp_test_cm
)


plt.figure(
    figsize=(7, 6)
)


sns.heatmap(

    final_gp_test_cm,

    annot=True,

    fmt="d",

    xticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ],

    yticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ]

)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.title(
    f"Final GP Test Confusion Matrix "
    f"(Iteration {best_gp_iteration})"
)


plt.tight_layout()


plt.show()


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