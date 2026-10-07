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
result = None

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

# Feedback storage logic is contained in the exact SARSA cell.

# ============================================================
# SOURCE COLAB CELL 128
# ============================================================

# ============================================================
# CELL 49 — DOCTOR FEEDBACK + SARSA
# USING VERIFIED 82-D FUSED FEATURES
# ============================================================

import json
import joblib
import numpy as np

from pathlib import Path
from datetime import datetime



# ============================================================
# 1. PATHS
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model"

FEEDBACK_DIR = (
    MODEL_DIR / "Doctor Feedback"
)

FEEDBACK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEEDBACK_FILE = (
    FEEDBACK_DIR /
    "doctor_feedback.json"
)

SARSA_FILE = (
    FEEDBACK_DIR /
    "sarsa_q_table.joblib"
)


# ============================================================
# 2. VERIFY CURRENT PREDICTION
# ============================================================

if "result" not in globals() or result is None:

    raise RuntimeError(
        "Prediction result is not available.\n"
        "Run Cell 47 first."
    )


# ============================================================
# 3. VERIFY 82-D FUSED FEATURES
# ============================================================

if "fused_features" not in globals():

    raise RuntimeError(
        "fused_features is not available.\n"
        "Run the ConvNeXt 82-D feature fusion cell first."
    )


fused_features = np.asarray(
    fused_features,
    dtype=np.float32
).reshape(-1)


print("=" * 70)
print("82-D FEATURE CHECK")
print("=" * 70)

print(
    "Feature dimension:",
    len(fused_features)
)


if len(fused_features) != 82:

    raise ValueError(
        f"Expected 82 features, "
        f"but found {len(fused_features)}."
    )


print(
    "[OK] Exact 82-D fused feature vector confirmed."
)


# ============================================================
# 4. GET AI PREDICTION
# ============================================================

predicted_class = result[
    "predicted_class"
]

probabilities = result[
    "probabilities"
]

confidence = float(
    probabilities[
        predicted_class
    ]
)


# ============================================================
# 5. DISPLAY AI RESULT
# ============================================================

print("\n" + "=" * 70)
print("AI PREDICTION FOR DOCTOR VERIFICATION")
print("=" * 70)

print(
    f"\nAI Prediction : {predicted_class}"
)

print(
    f"Confidence    : "
    f"{confidence * 100:.2f}%"
)

print(
    "\nModel-estimated probabilities:"
)

for class_name in [
    "Normal",
    "Benign",
    "Malignant"
]:

    print(
        f"  {class_name:<12}: "
        f"{probabilities[class_name] * 100:.2f}%"
    )


# ============================================================
# 6. CONFIDENCE STATE
# ============================================================

if confidence < 0.50:

    confidence_level = "Low"

elif confidence < 0.75:

    confidence_level = "Medium"

else:

    confidence_level = "High"


current_state = (
    predicted_class,
    confidence_level
)


print(
    "\nSARSA Current State:",
    current_state
)


# ============================================================
# 7. ASK DOCTOR
# ============================================================

print("\n" + "-" * 70)

doctor_response = input(
    "\nDoctor, is the AI prediction correct? "
    "(yes/no): "
).strip().lower()


while doctor_response not in [
    "yes",
    "y",
    "no",
    "n"
]:

    doctor_response = input(
        "Please enter 'yes' or 'no': "
    ).strip().lower()


# ============================================================
# 8. DETERMINE ACTION
# ============================================================

if doctor_response in [
    "yes",
    "y"
]:

    action = "accept_prediction"

else:

    action = "request_correction"


# ============================================================
# 9. GET DOCTOR VERIFIED CLASS
# ============================================================

doctor_label = predicted_class


if action == "request_correction":

    print("\n" + "-" * 70)

    print(
        "AI prediction marked as INCORRECT."
    )

    print(
        "\nDoctor, select the correct classification:"
    )

    print(
        "1. Normal"
    )

    print(
        "2. Benign"
    )

    print(
        "3. Malignant"
    )


    doctor_choice = input(
        "\nEnter 1, 2, or 3: "
    ).strip()


    while doctor_choice not in [
        "1",
        "2",
        "3"
    ]:

        doctor_choice = input(
            "Please enter 1, 2, or 3: "
        ).strip()


    label_mapping = {

        "1": "Normal",

        "2": "Benign",

        "3": "Malignant"
    }


    doctor_label = (
        label_mapping[
            doctor_choice
        ]
    )


# ============================================================
# 10. CALCULATE REWARD
# ============================================================

if doctor_label == predicted_class:

    reward = 1

else:

    reward = -1


# ============================================================
# 11. NEXT STATE
# ============================================================

next_state = (
    doctor_label,
    confidence_level
)


# ============================================================
# 12. NEXT ACTION
# ============================================================

if reward == 1:

    next_action = "accept_prediction"

else:

    next_action = "model_update"


# ============================================================
# 13. LOAD SARSA Q-TABLE
# ============================================================

if SARSA_FILE.exists():

    q_table = joblib.load(
        SARSA_FILE
    )

else:

    q_table = {}


# ============================================================
# 14. INITIALIZE CURRENT STATE
# ============================================================

if current_state not in q_table:

    q_table[current_state] = {}


if action not in q_table[
    current_state
]:

    q_table[
        current_state
    ][
        action
    ] = 0.0


# ============================================================
# 15. INITIALIZE NEXT STATE
# ============================================================

if next_state not in q_table:

    q_table[next_state] = {}


if next_action not in q_table[
    next_state
]:

    q_table[
        next_state
    ][
        next_action
    ] = 0.0


# ============================================================
# 16. SARSA PARAMETERS
# ============================================================

alpha = 0.10

gamma = 0.90


# ============================================================
# 17. CURRENT Q VALUE
# ============================================================

current_q = q_table[
    current_state
][
    action
]


# ============================================================
# 18. NEXT Q VALUE
# ============================================================

next_q = q_table[
    next_state
][
    next_action
]


# ============================================================
# 19. SARSA UPDATE
# ============================================================

updated_q = (
    current_q
    +
    alpha
    *
    (
        reward
        +
        gamma * next_q
        -
        current_q
    )
)


q_table[
    current_state
][
    action
] = updated_q


# ============================================================
# 20. SAVE SARSA TABLE
# ============================================================

joblib.dump(
    q_table,
    SARSA_FILE
)


# ============================================================
# 21. LOAD EXISTING FEEDBACK
# ============================================================

if FEEDBACK_FILE.exists():

    with open(
        FEEDBACK_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        feedback_data = json.load(f)

else:

    feedback_data = []


# ============================================================
# 22. CREATE FEEDBACK RECORD
# ============================================================

feedback_record = {

    "timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

    "image":
        result.get(
            "filename",
            "unknown"
        ),

    "ai_prediction":
        predicted_class,

    "ai_confidence":
        round(
            confidence * 100,
            2
        ),

    "normal_probability":
        round(
            float(
                probabilities["Normal"]
            ) * 100,
            2
        ),

    "benign_probability":
        round(
            float(
                probabilities["Benign"]
            ) * 100,
            2
        ),

    "malignant_probability":
        round(
            float(
                probabilities["Malignant"]
            ) * 100,
            2
        ),

    "doctor_verified_class":
        doctor_label,

    "prediction_correct":
        bool(
            reward == 1
        ),

    "reward":
        reward,

    "sarsa_state":
        list(
            current_state
        ),

    "sarsa_action":
        action,

    "next_state":
        list(
            next_state
        ),

    "next_action":
        next_action,

    "previous_q_value":
        float(
            current_q
        ),

    "updated_q_value":
        float(
            updated_q
        ),

    # ========================================================
    # CRITICAL
    # ========================================================

    "fused_features":
        fused_features.tolist(),

    "feature_dimension":
        82,

    "handcrafted_features":
        50,

    "convnext_pca_features":
        32,

    "model_update_required":
        bool(
            reward == -1
        )
}


# ============================================================
# 23. APPEND RECORD
# ============================================================

feedback_data.append(
    feedback_record
)


# ============================================================
# 24. SAVE FEEDBACK
# ============================================================

with open(
    FEEDBACK_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feedback_data,
        f,
        indent=4
    )


# ============================================================
# 25. UPDATE CURRENT RESULT
# ============================================================

result[
    "doctor_verified_class"
] = doctor_label

result[
    "doctor_feedback"
] = (
    "correct"
    if reward == 1
    else "incorrect"
)

result[
    "sarsa_reward"
] = reward

result[
    "sarsa_state"
] = list(
    current_state
)

result[
    "sarsa_action"
] = action

result[
    "fused_features"
] = fused_features.tolist()

result[
    "feature_dimension"
] = 82


# ============================================================
# 26. SAVE RESULT
# ============================================================

if "result_file" in result:

    joblib.dump(
        result,
        result["result_file"]
    )


# ============================================================
# 27. DISPLAY FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("DOCTOR FEEDBACK COMPLETED")
print("=" * 70)

print(
    f"\nAI Prediction      : "
    f"{predicted_class}"
)

print(
    f"Doctor Verified    : "
    f"{doctor_label}"
)

print(
    f"Prediction Correct : "
    f"{reward == 1}"
)

print(
    f"Reward             : "
    f"{reward}"
)

print(
    f"\nFeature Dimension  : "
    f"{len(fused_features)}"
)

print(
    f"SARSA State        : "
    f"{current_state}"
)

print(
    f"SARSA Action       : "
    f"{action}"
)

print(
    f"Previous Q-value   : "
    f"{current_q:.6f}"
)

print(
    f"Updated Q-value    : "
    f"{updated_q:.6f}"
)

print(
    "\nFeedback file:"
)

print(
    FEEDBACK_FILE
)

print(
    "\nSARSA Q-table:"
)

print(
    SARSA_FILE
)


# ============================================================
# 28. FINAL STATUS
# ============================================================

if reward == -1:

    print("\n" + "-" * 70)

    print(
        "✗ INCORRECT PREDICTION RECORDED"
    )

    print(
        f"AI prediction : {predicted_class}"
    )

    print(
        f"Correct class : {doctor_label}"
    )

    print(
        "Negative reward (-1) assigned."
    )

    print(
        "Exact 82-D feature vector stored."
    )

    print(
        "Case marked for controlled model update."
    )

    print(
        "\nOriginal model has NOT been modified."
    )

else:

    print("\n" + "-" * 70)

    print(
        "[OK] AI PREDICTION CONFIRMED BY DOCTOR"
    )

    print(
        "Positive reward (+1) assigned."
    )

    print(
        "Verified 82-D feature vector stored."
    )

print("\n" + "=" * 70)