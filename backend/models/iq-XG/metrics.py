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
X_test_pca_xgb = None

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
# CELL 28 - XGBOOST BASELINE VALIDATION METRICS
# ============================================================


xgb_val_accuracy = accuracy_score(

    y_val_xgb,

    xgb_val_predictions

)


xgb_val_balanced_accuracy = (

    balanced_accuracy_score(

        y_val_xgb,

        xgb_val_predictions

    )

)


xgb_val_macro_precision = (

    precision_score(

        y_val_xgb,

        xgb_val_predictions,

        average="macro",

        zero_division=0

    )

)


xgb_val_macro_recall = (

    recall_score(

        y_val_xgb,

        xgb_val_predictions,

        average="macro",

        zero_division=0

    )

)


xgb_val_macro_f1 = (

    f1_score(

        y_val_xgb,

        xgb_val_predictions,

        average="macro",

        zero_division=0

    )

)


xgb_val_weighted_f1 = (

    f1_score(

        y_val_xgb,

        xgb_val_predictions,

        average="weighted",

        zero_division=0

    )


)


print("=" * 60)
print("XGBOOST BASELINE VALIDATION PERFORMANCE")
print("=" * 60)


print(
    f"Accuracy           : "
    f"{xgb_val_accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy  : "
    f"{xgb_val_balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision    : "
    f"{xgb_val_macro_precision:.4f}"
)


print(
    f"Macro Recall       : "
    f"{xgb_val_macro_recall:.4f}"
)


print(
    f"Macro F1          : "
    f"{xgb_val_macro_f1:.4f}"
)


print(
    f"Weighted F1       : "
    f"{xgb_val_weighted_f1:.4f}"
)

# ============================================================
# CELL 29 - XGBOOST BASELINE CLASSIFICATION REPORT
# ============================================================


print("=" * 60)
print("XGBOOST BASELINE CLASSIFICATION REPORT")
print("=" * 60)


print(

    classification_report(

        y_val_xgb,

        xgb_val_predictions,

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

# ============================================================
# CELL 30 - XGBOOST BASELINE CONFUSION MATRIX
# ============================================================


xgb_baseline_cm = confusion_matrix(

    y_val_xgb,

    xgb_val_predictions,

    labels=[0, 1, 2]

)


print("=" * 60)
print("XGBOOST BASELINE CONFUSION MATRIX")
print("=" * 60)


print(
    xgb_baseline_cm
)


plt.figure(

    figsize=(7, 6)

)


sns.heatmap(

    xgb_baseline_cm,

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
    "XGBoost Baseline Validation Confusion Matrix"
)


plt.tight_layout()


plt.show()

# ============================================================
# CELL 31 - XGBOOST AUTOMATIC EXPERIMENT LOOP
# ============================================================
#
# We will test several XGBoost configurations automatically.
#
# The TEST SET is NOT used for selecting the model.
#
# Model selection is based only on VALIDATION performance.
#
# Metrics recorded:
#
#     Accuracy
#     Balanced Accuracy
#     Macro Precision
#     Macro Recall
#     Macro F1
#     Weighted F1
#
# ============================================================


# ------------------------------------------------------------
# EXPERIMENT CONFIGURATIONS
# ------------------------------------------------------------

xgb_experiments = [

    {
        "name": "XGB-1",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "name": "XGB-2",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "name": "XGB-3",
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "name": "XGB-4",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "name": "XGB-5",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9
    }

]


# ============================================================
# STORAGE
# ============================================================


xgb_results = []

xgb_models = {}

xgb_predictions = {}

xgb_probabilities = {}


# ============================================================
# RUN EXPERIMENTS
# ============================================================


for config in xgb_experiments:

    experiment_name = config["name"]


    print("\n" + "=" * 65)

    print(
        f"TRAINING {experiment_name}"
    )

    print("=" * 65)


    print(
        "Trees          :",
        config["n_estimators"]
    )

    print(
        "Max depth      :",
        config["max_depth"]
    )

    print(
        "Learning rate  :",
        config["learning_rate"]
    )


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=config[
            "n_estimators"
        ],

        max_depth=config[
            "max_depth"
        ],

        learning_rate=config[
            "learning_rate"
        ],

        subsample=config[
            "subsample"
        ],

        colsample_bytree=config[
            "colsample_bytree"
        ],

        min_child_weight=2,

        reg_alpha=0.0,

        reg_lambda=1.0,

        objective="multi:softprob",

        num_class=3,

        eval_metric="mlogloss",

        random_state=SEED,

        n_jobs=-1,

        tree_method="hist"

    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    start_time = time.time()


    model.fit(

        X_train_pca_xgb,

        y_train_xgb,

        eval_set=[

            (
                X_val_pca_xgb,
                y_val_xgb
            )

        ],

        verbose=False

    )


    elapsed_time = (

        time.time()
        -
        start_time

    )


    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba(

        X_val_pca_xgb

    )


    predictions = np.argmax(

        probabilities,

        axis=1

    )


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_val_xgb,

        predictions

    )


    balanced_accuracy = (

        balanced_accuracy_score(

            y_val_xgb,

            predictions

        )

    )


    macro_precision = (

        precision_score(

            y_val_xgb,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    macro_recall = (

        recall_score(

            y_val_xgb,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    macro_f1 = (

        f1_score(

            y_val_xgb,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    weighted_f1 = (

        f1_score(

            y_val_xgb,

            predictions,

            average="weighted",

            zero_division=0

        )

    )


    # --------------------------------------------------------
    # STORE MODEL
    # --------------------------------------------------------

    xgb_models[
        experiment_name
    ] = model


    xgb_predictions[
        experiment_name
    ] = predictions


    xgb_probabilities[
        experiment_name
    ] = probabilities


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    xgb_results.append({

        "Experiment":
            experiment_name,

        "Trees":
            config["n_estimators"],

        "Max Depth":
            config["max_depth"],

        "Learning Rate":
            config["learning_rate"],

        "Accuracy":
            accuracy,

        "Balanced Accuracy":
            balanced_accuracy,

        "Macro Precision":
            macro_precision,

        "Macro Recall":
            macro_recall,

        "Macro F1":
            macro_f1,

        "Weighted F1":
            weighted_f1,

        "Training Time (s)":
            elapsed_time

    })


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print(
        f"\nAccuracy          : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Training Time     : "
        f"{elapsed_time:.2f} seconds"
    )


print("\n")
print("=" * 65)
print("ALL XGBOOST EXPERIMENTS COMPLETED")
print("=" * 65)

# ============================================================
# CELL 32 - XGBOOST EXPERIMENT COMPARISON
# ============================================================


xgb_results_df = pd.DataFrame(
    xgb_results
)


# ------------------------------------------------------------
# Sort by Macro F1
# ------------------------------------------------------------
#
# Macro F1 gives equal importance to all three classes.
# This is particularly useful because the dataset is imbalanced.
# ------------------------------------------------------------

xgb_results_df = (

    xgb_results_df
    .sort_values(
        by="Macro F1",
        ascending=False
    )
    .reset_index(
        drop=True
    )

)


print("=" * 65)
print("XGBOOST EXPERIMENT RESULTS")
print("=" * 65)


print(
    xgb_results_df
)

# ============================================================
# CELL 33 - SELECT FINAL XGBOOST MODEL
# ============================================================
#
# Model selection is performed ONLY using the validation set.
#
# Primary selection metric:
#
#     Macro F1
#
# Reason:
#     Macro F1 gives equal importance to:
#
#         Normal
#         Benign
#         Malignant
#
# This prevents the larger Malignant class from dominating
# model selection.
#
# The test set is still completely untouched.
# ============================================================


# ------------------------------------------------------------
# GET BEST EXPERIMENT
# ------------------------------------------------------------

best_xgb_experiment = (

    xgb_results_df
    .iloc[0]["Experiment"]

)


# ------------------------------------------------------------
# GET FINAL MODEL
# ------------------------------------------------------------

final_xgb_model = xgb_models[
    best_xgb_experiment
]


# ------------------------------------------------------------
# GET BEST VALIDATION PREDICTIONS
# ------------------------------------------------------------

final_xgb_val_predictions = (

    xgb_predictions[
        best_xgb_experiment
    ]

)


final_xgb_val_probabilities = (

    xgb_probabilities[
        best_xgb_experiment
    ]

)


# ============================================================
# DISPLAY FINAL MODEL
# ============================================================

print("=" * 65)
print("FINAL XGBOOST MODEL SELECTED")
print("=" * 65)


print(
    "Selected experiment:",
    best_xgb_experiment
)


best_row = xgb_results_df.iloc[0]


print(
    "Trees:",
    int(best_row["Trees"])
)


print(
    "Max Depth:",
    int(best_row["Max Depth"])
)


print(
    "Learning Rate:",
    best_row["Learning Rate"]
)


print(
    "\nValidation Macro F1:",
    f"{best_row['Macro F1']:.4f}"
)


print(
    "Validation Accuracy:",
    f"{best_row['Accuracy'] * 100:.2f}%"
)


print(
    "Validation Balanced Accuracy:",
    f"{best_row['Balanced Accuracy'] * 100:.2f}%"
)


print(
    "\n[OK] Final XGBoost model selected."
)

# ============================================================
# CELL 34 - FINAL XGBOOST VALIDATION REPORT
# ============================================================


print("=" * 65)
print("FINAL XGBOOST VALIDATION CLASSIFICATION REPORT")
print("=" * 65)


print(

    classification_report(

        y_val_xgb,

        final_xgb_val_predictions,

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

# ============================================================
# CELL 35 - FINAL XGBOOST VALIDATION CONFUSION MATRIX
# ============================================================


final_xgb_val_cm = confusion_matrix(

    y_val_xgb,

    final_xgb_val_predictions,

    labels=[0, 1, 2]

)


print("=" * 65)
print("FINAL XGBOOST VALIDATION CONFUSION MATRIX")
print("=" * 65)


print(
    final_xgb_val_cm
)


plt.figure(

    figsize=(7, 6)

)


sns.heatmap(

    final_xgb_val_cm,

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
    "Final XGBoost Validation Confusion Matrix"
)


plt.tight_layout()


plt.show()

# ============================================================
# CELL 36 - FINAL XGBOOST TEST PREDICTION
# ============================================================
#
# IMPORTANT:
#
# The test set was NOT used for:
#
#     - preprocessing fitting
#     - scaler fitting
#     - PCA fitting
#     - model selection
#     - hyperparameter selection
#
# It is used now only for final evaluation.
# ============================================================


print("=" * 65)
print("FINAL XGBOOST TEST PREDICTION")
print("=" * 65)


test_probabilities = (

    final_xgb_model.predict_proba(

        X_test_pca_xgb

    )

)


test_predictions = np.argmax(

    test_probabilities,

    axis=1

)


print(
    "Test samples:",
    len(test_predictions)
)


print(
    "Probability matrix:",
    test_probabilities.shape
)


print(
    "Prediction vector:",
    test_predictions.shape
)


print(
    "\n[OK] Final test prediction completed."
)

# ============================================================
# CELL 37 - FINAL XGBOOST TEST PERFORMANCE
# ============================================================


final_test_accuracy = accuracy_score(

    y_test_xgb,

    test_predictions

)


final_test_balanced_accuracy = (

    balanced_accuracy_score(

        y_test_xgb,

        test_predictions

    )

)


final_test_macro_precision = (

    precision_score(

        y_test_xgb,

        test_predictions,

        average="macro",

        zero_division=0

    )

)


final_test_macro_recall = (

    recall_score(

        y_test_xgb,

        test_predictions,

        average="macro",

        zero_division=0

    )

)


final_test_macro_f1 = (

    f1_score(

        y_test_xgb,

        test_predictions,

        average="macro",

        zero_division=0

    )

)


final_test_weighted_f1 = (

    f1_score(

        y_test_xgb,

        test_predictions,

        average="weighted",

        zero_division=0

    )

)


# ============================================================
# DISPLAY
# ============================================================


print("=" * 65)
print("FINAL XGBOOST TEST PERFORMANCE")
print("=" * 65)


print(
    f"Accuracy            : "
    f"{final_test_accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy   : "
    f"{final_test_balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision     : "
    f"{final_test_macro_precision:.4f}"
)


print(
    f"Macro Recall        : "
    f"{final_test_macro_recall:.4f}"
)


print(
    f"Macro F1            : "
    f"{final_test_macro_f1:.4f}"
)


print(
    f"Weighted F1         : "
    f"{final_test_weighted_f1:.4f}"
)

# ============================================================
# CELL 38 - FINAL XGBOOST TEST CLASSIFICATION REPORT
# ============================================================


print("=" * 65)
print("FINAL XGBOOST TEST CLASSIFICATION REPORT")
print("=" * 65)


print(

    classification_report(

        y_test_xgb,

        test_predictions,

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

# ============================================================
# CELL 39 - FINAL XGBOOST TEST CONFUSION MATRIX
# ============================================================


final_xgb_test_cm = confusion_matrix(

    y_test_xgb,

    test_predictions,

    labels=[0, 1, 2]

)


print("=" * 65)
print("FINAL XGBOOST TEST CONFUSION MATRIX")
print("=" * 65)


print(
    final_xgb_test_cm
)


plt.figure(

    figsize=(7, 6)

)


sns.heatmap(

    final_xgb_test_cm,

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
    "Final XGBoost Test Confusion Matrix"
)


plt.tight_layout()


plt.show()

# ============================================================
# CELL 40 - INDIVIDUAL TEST PREDICTIONS
# ============================================================


print("=" * 65)
print("FIRST 30 XGBOOST TEST PREDICTIONS")
print("=" * 65)


for i in range(

    min(
        30,
        len(test_predictions)
    )

):


    actual_class = CLASS_NAMES[
        y_test_xgb[i]
    ]


    predicted_class = CLASS_NAMES[
        test_predictions[i]
    ]


    status = (

        "[OK] CORRECT"

        if y_test_xgb[i]
        ==
        test_predictions[i]

        else

        "✗ INCORRECT"

    )


    confidence = (

        test_probabilities[i][
            test_predictions[i]
        ]

        * 100

    )


    print(

        f"{i + 1:02d}. "
        f"Actual: {actual_class:<10} | "
        f"Predicted: {predicted_class:<10} | "
        f"Confidence: {confidence:.2f}% | "
        f"{status}"

    )