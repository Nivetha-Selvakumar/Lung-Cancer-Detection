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

from pathlib import Path
import os

from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = MODELS_DIR.parent
WORKSPACE_ROOT = BACKEND_DIR.parent

# Hospital Raw CT Dataset path
_hosp_candidates = [
    WORKSPACE_ROOT / "Hospital Dataset" / "DATASET",
    WORKSPACE_ROOT / "Hopital Dataset" / "DATASET",
    WORKSPACE_ROOT / "DATASET"
]
_hosp_root = _hosp_candidates[0]
for c in _hosp_candidates:
    if c.exists():
        _hosp_root = c
        break

HOSPITAL_DATASET_ROOT = Path(
    os.environ.get(
        "HOSPITAL_DATASET_ROOT",
        str(_hosp_root)
    )
)

# IQ-OTH/NCCD Lung Cancer Dataset path
_iq_candidates = [
    WORKSPACE_ROOT / "archive" / "The IQ-OTHNCCD lung cancer dataset" / "The IQ-OTHNCCD lung cancer dataset",
    WORKSPACE_ROOT / "IQ Dataset" / "The IQ-OTHNCCD lung cancer dataset",
    WORKSPACE_ROOT / "archive"
]
_iq_root = _iq_candidates[0]
for c in _iq_candidates:
    if c.exists():
        _iq_root = c
        break

IQ_DATASET_ROOT = Path(
    os.environ.get(
        "IQ_DATASET_ROOT",
        str(_iq_root)
    )
)

DATASET_ROOT = HOSPITAL_DATASET_ROOT

def load_dataset_splits(dataset_type='hospital'):
    dataset_root = HOSPITAL_DATASET_ROOT if dataset_type.lower().startswith('hosp') else IQ_DATASET_ROOT
    dataset_root = Path(dataset_root)
    records = []
    
    if dataset_type.lower().startswith('hosp'):
        mapping = {'normal': 0, 'Normal': 0, 'benign': 1, 'Benign': 1, 'maligancy': 2, 'malignant': 2, 'Malignant': 2}
        class_names = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
        for fdir in dataset_root.iterdir():
            if fdir.is_dir() and fdir.name in mapping:
                label = mapping[fdir.name]
                for f in fdir.glob('*'):
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        records.append({'filepath': str(f), 'label': label, 'class_name': class_names[label]})
    else:
        mapping = {'Normal cases': 0, 'Normal case': 0, 'normal': 0, 'Benign cases': 1, 'Bengin cases': 1, 'benign': 1, 'Malignant cases': 2, 'Malignant case': 2, 'malignant': 2}
        class_names = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
        for fdir in dataset_root.iterdir():
            if fdir.is_dir() and fdir.name in mapping:
                label = mapping[fdir.name]
                for f in fdir.glob('*'):
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        records.append({'filepath': str(f), 'label': label, 'class_name': class_names[label]})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No images found in dataset directory: {dataset_root}")

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.33333, random_state=42, stratify=temp_df['label'])
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

