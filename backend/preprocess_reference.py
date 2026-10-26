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
    from shared.preprocess import segment_lung, preprocess_image  # type: ignore
except Exception:
    def segment_lung(x): return None, None, None
    def preprocess_image(x): return None

# Exact preprocessing/segmentation cells from Hybrid_LC_(1).ipynb

# ============================================================
# SOURCE COLAB CELL 0
# ============================================================

# ============================================================
# CELL 1 â€” INSTALL & IMPORT
# ============================================================

# !pip -q install scikit-image xgboost gplearn joblib

import os
import cv2
import json
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import ndimage as ndi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from skimage.feature import (
    hog,
    local_binary_pattern,
    graycomatrix,
    graycoprops,
    canny
)

from skimage.measure import regionprops, label as sk_label

from xgboost import XGBClassifier
from gplearn.genetic import SymbolicClassifier

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision import models, transforms

from PIL import Image

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Libraries imported successfully.")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# SOURCE COLAB CELL 7
# ============================================================

# ============================================================
# CELL 7 â€” IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {
            "valid": False,
            "height": 0,
            "width": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan
        }

    return {
        "valid": True,
        "height": img.shape[0],
        "width": img.shape[1],
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "min": int(np.min(img)),
        "max": int(np.max(img))
    }


# Check ALL 53 development images
# quality_records = []
# 
# if 'df' in globals():
#     for _, row in df.iterrows():
# 
#     result = check_image_quality(row["filepath"])
# 
#     quality_records.append({
#         "filepath": row["filepath"],