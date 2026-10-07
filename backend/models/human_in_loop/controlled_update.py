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

# Exact latest controlled candidate update implementation.

# ============================================================
# SOURCE COLAB CELL 129
# ============================================================

# ============================================================
# CELL 50 — CONTROLLED MODEL UPDATE
# ============================================================
#
# Doctor-verified feedback
#        ↓
# 82-D fused features
#        ↓
# Candidate Hybrid Model
#        ↓
# Fine-tuning
#
# IMPORTANT:
# The original model is NEVER overwritten in this cell.
# ============================================================


import json
import joblib
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model"

FEEDBACK_DIR = (
    MODEL_DIR / "Doctor Feedback"
)

FEEDBACK_FILE = (
    FEEDBACK_DIR /
    "doctor_feedback.json"
)

ORIGINAL_MODEL_FILE = (
    MODEL_DIR /
    "hybrid_classifier_final.pth"
)

CANDIDATE_MODEL_FILE = (
    MODEL_DIR /
    "hybrid_classifier_updated_candidate.pth"
)


# ============================================================
# 2. CHECK FEEDBACK FILE
# ============================================================

if not FEEDBACK_FILE.exists():

    raise FileNotFoundError(
        f"Doctor feedback file not found:\n"
        f"{FEEDBACK_FILE}"
    )


# ============================================================
# 3. LOAD FEEDBACK
# ============================================================

with open(
    FEEDBACK_FILE,
    "r",
    encoding="utf-8"
) as f:

    feedback_data = json.load(f)


print("=" * 70)
print("DOCTOR FEEDBACK DATA")
print("=" * 70)

print(
    "Total feedback records:",
    len(feedback_data)
)


# ============================================================
# 4. FIND VERIFIED FEEDBACK WITH 82-D FEATURES
# ============================================================

verified_samples = []


for record in feedback_data:

    if (
        "fused_features" in record
        and
        record.get(
            "feature_dimension"
        ) == 82
        and
        record.get(
            "doctor_verified_class"
        ) in [
            "Normal",
            "Benign",
            "Malignant"
        ]
    ):

        verified_samples.append(
            record
        )


print(
    "Feedback records containing "
    "82-D features:",
    len(verified_samples)
)


# ============================================================
# 5. CHECK WHETHER THERE IS DATA
# ============================================================

if len(verified_samples) == 0:

    print(
        "\nNo usable doctor-verified "
        "82-D samples found."
    )

    print(
        "Model will NOT be updated."
    )

else:

    print(
        "\n[OK] Doctor-verified training "
        "samples are available."
    )


# ============================================================
# 6. SHOW VERIFIED SAMPLES
# ============================================================

for i, record in enumerate(
    verified_samples
):

    print(
        f"\nCase {i + 1}"
    )

    print(
        "Image       :",
        record.get(
            "image",
            "Unknown"
        )
    )

    print(
        "AI Prediction:",
        record.get(
            "ai_prediction"
        )
    )

    print(
        "Doctor Label :",
        record.get(
            "doctor_verified_class"
        )
    )

    print(
        "Reward       :",
        record.get(
            "reward"
        )
    )

    print(
        "Features     :",
        len(
            record[
                "fused_features"
            ]
        )
    )


# ============================================================
# 7. CREATE FEEDBACK DATASET
# ============================================================

X_feedback = []

y_feedback = []


label_mapping = {

    "Normal": 0,

    "Benign": 1,

    "Malignant": 2
}


for record in verified_samples:

    features = np.asarray(
        record[
            "fused_features"
        ],
        dtype=np.float32
    )


    # Verify dimension

    if len(features) != 82:

        continue


    label = label_mapping[
        record[
            "doctor_verified_class"
        ]
    ]


    X_feedback.append(
        features
    )

    y_feedback.append(
        label
    )


X_feedback = np.asarray(
    X_feedback,
    dtype=np.float32
)

y_feedback = np.asarray(
    y_feedback,
    dtype=np.int64
)


print("\n" + "=" * 70)
print("FEEDBACK DATASET")
print("=" * 70)

print(
    "X shape:",
    X_feedback.shape
)

print(
    "y shape:",
    y_feedback.shape
)


# ============================================================
# 8. VERIFY DATA
# ============================================================

if X_feedback.ndim != 2:

    raise ValueError(
        "Feedback feature matrix "
        "must be 2-dimensional."
    )


if X_feedback.shape[1] != 82:

    raise ValueError(
        "Feedback features must "
        "contain exactly 82 columns."
    )


print(
    "[OK] 82-D feedback matrix confirmed."
)


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

print(
    "\nDoctor-verified class distribution:"
)


for class_name, class_id in label_mapping.items():

    count = int(
        np.sum(
            y_feedback == class_id
        )
    )

    print(
        f"  {class_name:<10}: {count}"
    )


# ============================================================
# 10. DEFINE SAME HYBRID CLASSIFIER
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
# 11. LOAD ORIGINAL MODEL
# ============================================================

if not ORIGINAL_MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Original model not found:\n"
        f"{ORIGINAL_MODEL_FILE}"
    )


original_model = HybridClassifier(
    input_dim=82,
    num_classes=3
).to(device)


checkpoint = torch.load(
    ORIGINAL_MODEL_FILE,
    map_location=device,
    weights_only=False
)


if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    original_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    original_model.load_state_dict(
        checkpoint
    )


original_model.eval()


print(
    "\n[OK] Original Hybrid model loaded."
)


# ============================================================
# 12. CREATE CANDIDATE MODEL
# ============================================================

candidate_model = HybridClassifier(
    input_dim=82,
    num_classes=3
).to(device)


candidate_model.load_state_dict(
    original_model.state_dict()
)


print(
    "[OK] Candidate model created "
    "from original model."
)


# ============================================================
# 13. CONVERT FEEDBACK TO TENSORS
# ============================================================

X_tensor = torch.tensor(
    X_feedback,
    dtype=torch.float32,
    device=device
)

y_tensor = torch.tensor(
    y_feedback,
    dtype=torch.long,
    device=device
)


# ============================================================
# 14. FINE-TUNING SETUP
# ============================================================

candidate_model.train()


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    candidate_model.parameters(),
    lr=0.0001
)


epochs = 10


print(
    "\nStarting controlled model update..."
)

print(
    "Learning rate:",
    0.0001
)

print(
    "Epochs:",
    epochs
)


# ============================================================
# 15. FINE-TUNE CANDIDATE
# ============================================================

for epoch in range(
    epochs
):

    optimizer.zero_grad()


    outputs = candidate_model(
        X_tensor
    )


    loss = criterion(
        outputs,
        y_tensor
    )


    loss.backward()


    optimizer.step()


    print(
        f"Epoch "
        f"{epoch + 1:02d}/{epochs} "
        f"- Loss: "
        f"{loss.item():.6f}"
    )


# ============================================================
# 16. SAVE CANDIDATE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            candidate_model.state_dict(),

        "input_dim":
            82,

        "num_classes":
            3,

        "update_source":
            "Doctor-verified feedback",

        "feedback_samples":
            len(X_feedback)
    },

    CANDIDATE_MODEL_FILE
)


print(
    "\n[OK] Candidate updated model saved:"
)

print(
    CANDIDATE_MODEL_FILE
)


# ============================================================
# 17. IMPORTANT FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "CONTROLLED MODEL UPDATE COMPLETED"
)

print("=" * 70)

print(
    "\nOriginal model:"
)

print(
    "UNCHANGED [OK]"
)

print(
    "\nCandidate model:"
)

print(
    "Updated using doctor-verified feedback [OK]"
)

print(
    "\nNext step:"
)

print(
    "Evaluate ORIGINAL vs CANDIDATE "
    "on the SAME internal test set."
)

print(
    "\nThe candidate will only be accepted "
    "if the evaluation supports improvement."
)

print("=" * 70)