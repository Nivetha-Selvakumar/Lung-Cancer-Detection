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
X_test_fused = None
FEEDBACK_DIR = None

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

# Exact original-vs-candidate evaluation implementation.

# ============================================================
# SOURCE COLAB CELL 130
# ============================================================

# ============================================================
# CHECK AVAILABLE INTERNAL TEST VARIABLES
# ============================================================

print("=" * 70)
print("SEARCHING FOR INTERNAL TEST DATA")
print("=" * 70)

possible_names = [
    "X_test_fused",
    "y_test_fused",
    "X_internal_test",
    "y_internal_test",
    "X_internal",
    "y_internal",
    "X_test",
    "y_test",
    "X_fused_test",
    "y_fused_test",
    "X_test_hybrid",
    "y_test_hybrid"
]

for name in possible_names:

    if name in globals():

        value = globals()[name]

        try:
            print(
                f"{name:<20} "
                f"shape={value.shape} "
                f"type={type(value)}"
            )

        except AttributeError:

            print(
                f"{name:<20} "
                f"type={type(value)}"
            )


print("\n" + "=" * 70)
print("OTHER POSSIBLE FEATURE MATRICES")
print("=" * 70)

for name, value in globals().items():

    if name.startswith("_"):
        continue

    try:

        if hasattr(value, "shape"):

            shape = value.shape

            # Look for matrices involving 82 features
            if (
                len(shape) == 2
                and 82 in shape
            ):

                print(
                    f"{name:<30} shape={shape}"
                )

    except Exception:
        pass


# ============================================================
# SOURCE COLAB CELL 131
# ============================================================

# ============================================================
# CELL 51 — ORIGINAL vs CANDIDATE MODEL EVALUATION
# ============================================================
#
# Same internal test set:
#       X_test_fused = (6, 82)
#       y_test       = (6,)
#
# Original model is NOT modified.
#
# Candidate is accepted only if it improves the selected
# evaluation criteria.
# ============================================================


import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from sklearn.metrics import (

    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. VERIFY TEST DATA
# ============================================================

print("=" * 70)
print("INTERNAL TEST DATA")
print("=" * 70)

print(
    "X_test_fused:",
    X_test_fused.shape
)

print(
    "y_test:",
    y_test.shape
)


if X_test_fused.shape != (6, 82):

    raise ValueError(
        "Expected X_test_fused shape "
        "(6, 82), but obtained "
        f"{X_test_fused.shape}"
    )


if y_test.shape != (6,):

    raise ValueError(
        "Expected y_test shape "
        "(6,), but obtained "
        f"{y_test.shape}"
    )


print(
    "[OK] Correct internal test set confirmed."
)


# ============================================================
# 2. DEFINE MODEL
# ============================================================

class HybridClassifier(
    nn.Module
):

    def __init__(
        self,
        input_dim=82,
        num_classes=3
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                32,
                num_classes
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 3. LOAD ORIGINAL MODEL
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model"

ORIGINAL_MODEL_FILE = (
    MODEL_DIR /
    "hybrid_classifier_final.pth"
)

CANDIDATE_MODEL_FILE = (
    MODEL_DIR /
    "hybrid_classifier_updated_candidate.pth"
)


def load_hybrid_model(
    model_path
):

    model = HybridClassifier(
        input_dim=82,
        num_classes=3
    ).to(device)


    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )


    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )


    model.eval()

    return model


# ============================================================
# 4. LOAD BOTH MODELS
# ============================================================

original_model = load_hybrid_model(
    ORIGINAL_MODEL_FILE
)

candidate_model = load_hybrid_model(
    CANDIDATE_MODEL_FILE
)


print(
    "\n[OK] Original model loaded."
)

print(
    "[OK] Candidate model loaded."
)


# ============================================================
# 5. CONVERT TEST DATA TO TENSOR
# ============================================================

X_test_tensor = torch.tensor(
    X_test_fused,
    dtype=torch.float32,
    device=device
)


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]


def evaluate_model(
    model,
    X,
    y
):

    model.eval()


    with torch.no_grad():

        logits = model(
            X
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )


    predictions = (
        predictions
        .cpu()
        .numpy()
    )


    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )


    accuracy = accuracy_score(
        y,
        predictions
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions
        )
    )


    macro_precision = (
        precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    macro_recall = (
        recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    macro_f1 = (
        f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    weighted_f1 = (
        f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )
    )


    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2]
    )


    return {

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "predictions":
            predictions,

        "probabilities":
            probabilities,

        "confusion_matrix":
            cm
    }


# ============================================================
# 7. EVALUATE ORIGINAL
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING ORIGINAL MODEL")
print("=" * 70)


original_results = evaluate_model(
    original_model,
    X_test_tensor,
    y_test
)


# ============================================================
# 8. EVALUATE CANDIDATE
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING CANDIDATE MODEL")
print("=" * 70)


candidate_results = evaluate_model(
    candidate_model,
    X_test_tensor,
    y_test
)


# ============================================================
# 9. DISPLAY ORIGINAL METRICS
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL MODEL RESULTS")
print("=" * 70)

print(
    f"Accuracy          : "
    f"{original_results['accuracy'] * 100:.2f}%"
)

print(
    f"Balanced Accuracy : "
    f"{original_results['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Macro Precision   : "
    f"{original_results['macro_precision'] * 100:.2f}%"
)

print(
    f"Macro Recall      : "
    f"{original_results['macro_recall'] * 100:.2f}%"
)

print(
    f"Macro F1          : "
    f"{original_results['macro_f1'] * 100:.2f}%"
)

print(
    f"Weighted F1       : "
    f"{original_results['weighted_f1'] * 100:.2f}%"
)


# ============================================================
# 10. DISPLAY CANDIDATE METRICS
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE MODEL RESULTS")
print("=" * 70)

print(
    f"Accuracy          : "
    f"{candidate_results['accuracy'] * 100:.2f}%"
)

print(
    f"Balanced Accuracy : "
    f"{candidate_results['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Macro Precision   : "
    f"{candidate_results['macro_precision'] * 100:.2f}%"
)

print(
    f"Macro Recall      : "
    f"{candidate_results['macro_recall'] * 100:.2f}%"
)

print(
    f"Macro F1          : "
    f"{candidate_results['macro_f1'] * 100:.2f}%"
)

print(
    f"Weighted F1       : "
    f"{candidate_results['weighted_f1'] * 100:.2f}%"
)


# ============================================================
# 11. METRIC COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL vs CANDIDATE")
print("=" * 70)

metrics = [

    (
        "Accuracy",
        "accuracy"
    ),

    (
        "Balanced Accuracy",
        "balanced_accuracy"
    ),

    (
        "Macro Precision",
        "macro_precision"
    ),

    (
        "Macro Recall",
        "macro_recall"
    ),

    (
        "Macro F1",
        "macro_f1"
    ),

    (
        "Weighted F1",
        "weighted_f1"
    )
]


print(
    f"\n{'Metric':<22}"
    f"{'Original':>12}"
    f"{'Candidate':>12}"
    f"{'Change':>12}"
)


print(
    "-" * 60
)


for display_name, key in metrics:

    original_value = (
        original_results[key]
    )

    candidate_value = (
        candidate_results[key]
    )

    change = (
        candidate_value
        -
        original_value
    )


    print(
        f"{display_name:<22}"
        f"{original_value * 100:>11.2f}%"
        f"{candidate_value * 100:>11.2f}%"
        f"{change * 100:>+11.2f}%"
    )


# ============================================================
# 12. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL MODEL CONFUSION MATRIX")
print("=" * 70)

print(
    original_results[
        "confusion_matrix"
    ]
)


print("\nRows = Actual")
print("Columns = Predicted")

print(
    "\nClasses:",
    CLASS_NAMES
)


print("\n" + "=" * 70)
print("CANDIDATE MODEL CONFUSION MATRIX")
print("=" * 70)

print(
    candidate_results[
        "confusion_matrix"
    ]
)


print("\nRows = Actual")
print("Columns = Predicted")

print(
    "\nClasses:",
    CLASS_NAMES
)


# ============================================================
# 13. CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL MODEL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        original_results[
            "predictions"
        ],
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


print("\n" + "=" * 70)
print("CANDIDATE MODEL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        candidate_results[
            "predictions"
        ],
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 14. PREDICTION-BY-PREDICTION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CASE-BY-CASE COMPARISON")
print("=" * 70)

for i in range(
    len(y_test)
):

    actual = CLASS_NAMES[
        y_test[i]
    ]

    original_pred = CLASS_NAMES[
        original_results[
            "predictions"
        ][i]
    ]

    candidate_pred = CLASS_NAMES[
        candidate_results[
            "predictions"
        ][i]
    ]


    original_correct = (
        original_pred == actual
    )

    candidate_correct = (
        candidate_pred == actual
    )


    print(
        f"\nCase {i + 1}"
    )

    print(
        f"Actual     : {actual}"
    )

    print(
        f"Original   : {original_pred} "
        f"({'Correct' if original_correct else 'Wrong'})"
    )

    print(
        f"Candidate  : {candidate_pred} "
        f"({'Correct' if candidate_correct else 'Wrong'})"
    )


# ============================================================
# 15. ACCEPT / REJECT DECISION
# ============================================================
#
# Primary criterion:
#       Balanced Accuracy
#
# Secondary criterion:
#       Macro F1
#
# Candidate must improve balanced accuracy.
# If balanced accuracy is equal, Macro F1 is considered.
# ============================================================


original_balanced = (
    original_results[
        "balanced_accuracy"
    ]
)

candidate_balanced = (
    candidate_results[
        "balanced_accuracy"
    ]
)


original_macro_f1 = (
    original_results[
        "macro_f1"
    ]
)

candidate_macro_f1 = (
    candidate_results[
        "macro_f1"
    ]
)


if (
    candidate_balanced
    >
    original_balanced
):

    accept_candidate = True

elif (
    np.isclose(
        candidate_balanced,
        original_balanced
    )
    and
    candidate_macro_f1
    >
    original_macro_f1
):

    accept_candidate = True

else:

    accept_candidate = False


# ============================================================
# 16. FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("MODEL UPDATE DECISION")
print("=" * 70)


if accept_candidate:

    print(
        "\n[OK] CANDIDATE MODEL SHOWS IMPROVEMENT."
    )

    print(
        "Candidate is eligible for acceptance."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The original model has still NOT been overwritten."
    )

    print(
        "\nCandidate file:"
    )

    print(
        CANDIDATE_MODEL_FILE
    )


else:

    print(
        "\n✗ CANDIDATE MODEL DOES NOT SHOW "
        "SUFFICIENT IMPROVEMENT."
    )

    print(
        "Candidate should be rejected."
    )

    print(
        "\nOriginal model remains the final model."
    )


# ============================================================
# 17. SAVE EVALUATION RESULTS
# ============================================================

evaluation_comparison = {

    "original": {

        "accuracy":
            float(
                original_results[
                    "accuracy"
                ]
            ),

        "balanced_accuracy":
            float(
                original_results[
                    "balanced_accuracy"
                ]
            ),

        "macro_precision":
            float(
                original_results[
                    "macro_precision"
                ]
            ),

        "macro_recall":
            float(
                original_results[
                    "macro_recall"
                ]
            ),

        "macro_f1":
            float(
                original_results[
                    "macro_f1"
                ]
            ),

        "weighted_f1":
            float(
                original_results[
                    "weighted_f1"
                ]
            )
    },


    "candidate": {

        "accuracy":
            float(
                candidate_results[
                    "accuracy"
                ]
            ),

        "balanced_accuracy":
            float(
                candidate_results[
                    "balanced_accuracy"
                ]
            ),

        "macro_precision":
            float(
                candidate_results[
                    "macro_precision"
                ]
            ),

        "macro_recall":
            float(
                candidate_results[
                    "macro_recall"
                ]
            ),

        "macro_f1":
            float(
                candidate_results[
                    "macro_f1"
                ]
            ),

        "weighted_f1":
            float(
                candidate_results[
                    "weighted_f1"
                ]
            )
    },


    "candidate_accepted":
        bool(
            accept_candidate
        )
}


EVALUATION_FILE = (
    FEEDBACK_DIR /
    "original_vs_candidate_evaluation.json"
)


with open(
    EVALUATION_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_comparison,
        f,
        indent=4
    )


print(
    "\nEvaluation saved to:"
)

print(
    EVALUATION_FILE
)


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)