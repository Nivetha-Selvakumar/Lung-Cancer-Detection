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

# Exact handcrafted feature selection and saved pipeline cells.

# ============================================================
# SOURCE COLAB CELL 24
# ============================================================

# ============================================================
# CELL 13A — FEATURE CLEANING
# ============================================================

# Work on copies
# X_train_clean = X_train_raw.copy()
# X_val_clean = X_val_raw.copy()
# X_test_clean = X_test_raw.copy()
# 
# print("Original feature count:", X_train_clean.shape[1])
# 
# ------------------------------------------------------------
# 1. Replace NaN / infinite values
# ------------------------------------------------------------
# 
# X_train_clean = np.nan_to_num(
#     X_train_clean,
#     nan=0.0,
#     posinf=0.0,
#     neginf=0.0
# )
# 
# X_val_clean = np.nan_to_num(
#     X_val_clean,
#     nan=0.0,
#     posinf=0.0,
#     neginf=0.0
# )
# 
# X_test_clean = np.nan_to_num(
#     X_test_clean,
#     nan=0.0,
#     posinf=0.0,
#     neginf=0.0
# )
# 
# ------------------------------------------------------------
# 2. Remove zero-variance features
# ------------------------------------------------------------
# 
# variance_selector = VarianceThreshold(
#     threshold=0.0
# )
# 
# X_train_var = variance_selector.fit_transform(
#     X_train_clean
# )
# 
# X_val_var = variance_selector.transform(
#     X_val_clean
# )
# 
# X_test_var = variance_selector.transform(
#     X_test_clean
# )
# 
# print(
#     "After zero-variance removal:",
#     X_train_var.shape[1]
# )
# 
# print(
#     "Features removed:",
#     X_train_clean.shape[1]
#     - X_train_var.shape[1]
# )
# 
# 
# ============================================================
# SOURCE COLAB CELL 25
# ============================================================
# 
# ============================================================
# CELL 13B — CORRELATION-BASED FEATURE REDUCTION
# ============================================================
# 
def remove_highly_correlated_features(
    X,
    threshold=0.95
):
    """
    Identify highly correlated features using
    training data only.
    """

    # Convert to DataFrame
    df_features = pd.DataFrame(
        X
    )

    # Correlation matrix
    correlation_matrix = df_features.corr(
        method="pearson"
    ).abs()

    # Upper triangle only
    upper = correlation_matrix.where(
        np.triu(
            np.ones(
                correlation_matrix.shape
            ),
            k=1
        ).astype(bool)
    )

    # Features to remove
    to_drop = [
        column
        for column in upper.columns
        if any(
            upper[column] > threshold
        )
    ]

    keep_indices = [
        i
        for i, column in enumerate(
            df_features.columns
        )
        if column not in to_drop
    ]

    return (
        keep_indices,
        to_drop
    )


# ------------------------------------------------------------
# Find redundant features ONLY using training data
# ------------------------------------------------------------

keep_indices, correlated_features = (
    remove_highly_correlated_features(
        X_train_var,
        threshold=0.95
    )
)

# Apply same feature selection to all sets
X_train_corr = X_train_var[
    :, keep_indices
]

X_val_corr = X_val_var[
    :, keep_indices
]

X_test_corr = X_test_var[
    :, keep_indices
]


print("==============================================")
print("CORRELATION-BASED FEATURE REDUCTION")
print("==============================================")

print(
    "Before:",
    X_train_var.shape[1]
)

print(
    "Removed:",
    len(correlated_features)
)

print(
    "After :",
    X_train_corr.shape[1]
)


# ============================================================
# SOURCE COLAB CELL 26
# ============================================================

# ============================================================
# CELL 13C — INFORMATIVE FEATURE SELECTION
# ============================================================

from sklearn.feature_selection import SelectKBest, f_classif

# ------------------------------------------------------------
# Number of features to retain
# ------------------------------------------------------------

K_FEATURES = 50

# Safety check
K_FEATURES = min(
    K_FEATURES,
    X_train_corr.shape[1]
)

print("Features before statistical selection:",
      X_train_corr.shape[1])

print("Selecting top", K_FEATURES,
      "features...")


# ------------------------------------------------------------
# Fit selector ONLY on training data
# ------------------------------------------------------------

feature_selector = SelectKBest(
    score_func=f_classif,
    k=K_FEATURES
)

X_train_selected = feature_selector.fit_transform(
    X_train_corr,
    y_train
)

# Apply the SAME fitted selector
# to validation and internal test
X_val_selected = feature_selector.transform(
    X_val_corr
)

X_test_selected = feature_selector.transform(
    X_test_corr
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n==============================================")
print("STATISTICAL FEATURE SELECTION")
print("==============================================")

print(
    "Before selection :",
    X_train_corr.shape[1]
)

print(
    "Selected         :",
    X_train_selected.shape[1]
)

print(
    "Removed          :",
    X_train_corr.shape[1]
    - X_train_selected.shape[1]
)

print("\nFinal feature matrices:")

print(
    "Training      :",
    X_train_selected.shape
)

print(
    "Validation    :",
    X_val_selected.shape
)

print(
    "Internal Test :",
    X_test_selected.shape
)


# ============================================================
# SOURCE COLAB CELL 28
# ============================================================

# ============================================================
# CELL 14 — FEATURE STANDARDIZATION
# ============================================================

from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Fit scaler ONLY on training data
# ------------------------------------------------------------

feature_scaler = StandardScaler()

X_train_final = feature_scaler.fit_transform(
    X_train_selected
)

# Apply same scaler
X_val_final = feature_scaler.transform(
    X_val_selected
)

X_test_final = feature_scaler.transform(
    X_test_selected
)

# ------------------------------------------------------------
# Convert to float32
# ------------------------------------------------------------

X_train_final = X_train_final.astype(np.float32)
X_val_final = X_val_final.astype(np.float32)
X_test_final = X_test_final.astype(np.float32)

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

print("==============================================")
print("FEATURE STANDARDIZATION")
print("==============================================")

print("Training      :", X_train_final.shape)
print("Validation    :", X_val_final.shape)
print("Internal Test :", X_test_final.shape)

print("\nTraining feature mean:")
print(f"{X_train_final.mean():.6f}")

print("\nTraining feature standard deviation:")
print(f"{X_train_final.std():.6f}")


# ============================================================
# SOURCE COLAB CELL 84
# ============================================================

# ============================================================
# CELL 21 — SAVE HANDCRAFTED FEATURE-SELECTION OBJECTS
# ============================================================

# ------------------------------------------------------------
# Check which feature-selection objects currently exist
# ------------------------------------------------------------

print("Checking handcrafted feature-selection objects...\n")

candidate_objects = [
    "variance_selector",
    "corr_selector",
    "correlation_selector",
    "anova_selector",
    "select_k_best",
    "handcrafted_scaler",
    "feature_scaler"
]

found_objects = {}

for object_name in candidate_objects:

    if object_name in globals():

        found_objects[object_name] = globals()[object_name]

        print(
            f"[OK] Found: {object_name}"
        )

    else:

        print(
            f"✗ Not found: {object_name}"
        )

# ------------------------------------------------------------
# Save the objects that exist
# ------------------------------------------------------------

handcrafted_objects_path = (
    MODEL_DIR / "handcrafted_feature_pipeline.joblib"
)

joblib.dump(
    found_objects,
    handcrafted_objects_path
)

# ------------------------------------------------------------
# Display result
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("HANDCRAFTED FEATURE PIPELINE SAVED")
print("=" * 60)

print(
    "Saved to:",
    handcrafted_objects_path
)

print(
    "\nObjects saved:",
    list(found_objects.keys())
)


# ============================================================
# SOURCE COLAB CELL 85
# ============================================================

# ============================================================
# CELL 22 — FIND HANDCRAFTED FEATURE-SELECTION OBJECTS
# ============================================================

print("=" * 60)
print("SEARCHING FOR HANDCRAFTED FEATURE OBJECTS")
print("=" * 60)

# Show relevant variables currently available in the notebook
keywords = [
    "variance",
    "corr",
    "correlation",
    "anova",
    "select",
    "kbest",
    "feature",
    "scaler",
    "mask"
]

found = []

for name in sorted(globals().keys()):

    name_lower = name.lower()

    if any(
        keyword in name_lower
        for keyword in keywords
    ):

        obj = globals()[name]

        # Ignore modules/functions/classes
        if not callable(obj):

            found.append(name)

            print(
                f"{name:<40} "
                f"{type(obj).__name__}"
            )

print("\n" + "=" * 60)
print("TOTAL CANDIDATE OBJECTS:", len(found))
print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 86
# ============================================================

# ============================================================
# CELL 23 — VERIFY EXISTING HANDCRAFTED FEATURE PIPELINE
# ============================================================

print("=" * 60)
print("HANDCRAFTED FEATURE PIPELINE VERIFICATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Variance selector
# ------------------------------------------------------------

print("\n1. Variance Selector")
print(
    "Type:",
    type(variance_selector).__name__
)

print(
    "Input features:",
    variance_selector.n_features_in_
)

print(
    "Output features:",
    variance_selector.get_support().sum()
)

# ------------------------------------------------------------
# 2. Correlation-removal information
# ------------------------------------------------------------

print("\n2. Correlation Removal")

print(
    "correlated_features type:",
    type(correlated_features).__name__
)

print(
    "Number of correlated features:",
    len(correlated_features)
)

print(
    "First 20 correlated feature entries:"
)

print(
    correlated_features[:20]
)

# ------------------------------------------------------------
# 3. Correlation-reduced matrices
# ------------------------------------------------------------

print("\n3. Correlation-Reduced Matrices")

print(
    "X_train_corr:",
    X_train_corr.shape
)

print(
    "X_val_corr:",
    X_val_corr.shape
)

print(
    "X_test_corr:",
    X_test_corr.shape
)

# ------------------------------------------------------------
# 4. ANOVA / SelectKBest
# ------------------------------------------------------------

print("\n4. SelectKBest")

print(
    "Type:",
    type(feature_selector).__name__
)

print(
    "Input features:",
    feature_selector.n_features_in_
)

print(
    "Selected features:",
    feature_selector.get_support().sum()
)

print(
    "K_FEATURES:",
    K_FEATURES
)

# ------------------------------------------------------------
# 5. Selected feature matrices
# ------------------------------------------------------------

print("\n5. Selected Feature Matrices")

print(
    "X_train_selected:",
    X_train_selected.shape
)

print(
    "X_val_selected:",
    X_val_selected.shape
)

print(
    "X_test_selected:",
    X_test_selected.shape
)

# ------------------------------------------------------------
# 6. Final handcrafted matrices
# ------------------------------------------------------------

print("\n6. Final Handcrafted Matrices")

print(
    "X_train_final:",
    X_train_final.shape
)

print(
    "X_val_final:",
    X_val_final.shape
)

print(
    "X_test_final:",
    X_test_final.shape
)

# ------------------------------------------------------------
# 7. Final scaler
# ------------------------------------------------------------

print("\n7. Feature Scaler")

print(
    "Type:",
    type(feature_scaler).__name__
)

print(
    "Scaler input features:",
    feature_scaler.n_features_in_
)

print("\n" + "=" * 60)
print("VERIFICATION COMPLETED")
print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 87
# ============================================================

# ============================================================
# CELL 24 — SAVE COMPLETE HANDCRAFTED FEATURE PIPELINE
# ============================================================

# ------------------------------------------------------------
# Create the complete handcrafted pipeline dictionary
# ------------------------------------------------------------

handcrafted_pipeline = {

    # Step 1: Remove zero-variance features
    "variance_selector": variance_selector,

    # Step 2: Remove highly correlated features
    "correlated_features": correlated_features,

    # Step 3: Select top 50 features using SelectKBest
    "feature_selector": feature_selector,

    # Step 4: Standardize the final 50 features
    "feature_scaler": feature_scaler,

    # Number of final features
    "n_final_features": 50,

    # Original feature count
    "n_original_features": 6146,

    # Feature count after variance filtering
    "n_after_variance": 5030,

    # Feature count after correlation filtering
    "n_after_correlation": 2643,

    # Feature selection method
    "feature_selection_method": "SelectKBest",

    # Final feature selection count
    "k_features": K_FEATURES
}

# ------------------------------------------------------------
# Save complete pipeline
# ------------------------------------------------------------

handcrafted_pipeline_path = (
    MODEL_DIR / "handcrafted_feature_pipeline_complete.joblib"
)

joblib.dump(
    handcrafted_pipeline,
    handcrafted_pipeline_path
)

# ------------------------------------------------------------
# Verify saved file
# ------------------------------------------------------------

print("=" * 60)
print("COMPLETE HANDCRAFTED PIPELINE SAVED")
print("=" * 60)

print(
    "\nSaved to:"
)

print(
    handcrafted_pipeline_path
)

print(
    "\nPipeline stages:"
)

print(
    "1. Original features:",
    handcrafted_pipeline["n_original_features"]
)

print(
    "2. After variance filtering:",
    handcrafted_pipeline["n_after_variance"]
)

print(
    "3. After correlation filtering:",
    handcrafted_pipeline["n_after_correlation"]
)

print(
    "4. After SelectKBest:",
    handcrafted_pipeline["k_features"]
)

print(
    "5. Final scaled features:",
    handcrafted_pipeline["n_final_features"]
)

print(
    "\nSaved objects:"
)

print(
    list(handcrafted_pipeline.keys())
)