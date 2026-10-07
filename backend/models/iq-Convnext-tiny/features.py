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
DataLoader = None
test_df = None
transforms = None
Dataset = None
WeightedRandomSampler = None

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
# CELL 20: IMAGE TRANSFORMS\n# ============================================================

IMAGE_SIZE = 224

# ------------------------------------------------------------
# Training transformations
# ------------------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# Validation / Test transformations
# ------------------------------------------------------------

val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


print("Transforms created successfully.")

# ============================================================
# CELL 21: CUSTOM PYTORCH DATASET\n# ============================================================

class LungCTDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # Use the cached segmented lung ROI (segment_lung output).
        image_path = row["segmented_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("L")
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

# ============================================================
# CELL 22: CREATE DATASET OBJECTS\n# ============================================================

train_dataset = LungCTDataset(
    train_df,
    transform=train_transform
)

val_dataset = LungCTDataset(
    val_df,
    transform=val_test_transform
)

test_dataset = LungCTDataset(
    test_df,
    transform=val_test_transform
)


print("Training samples   :", len(train_dataset))
print("Validation samples :", len(val_dataset))
print("Testing samples    :", len(test_dataset))

# ============================================================
# CELL 23: HANDLE CLASS IMBALANCE\n# ============================================================

train_labels = train_df["label"].values

class_counts = np.bincount(
    train_labels,
    minlength=3
)

print("Training class counts:")
for class_name, count in zip(
    CLASS_NAMES,
    class_counts
):
    print(
        f"{class_name:10s}: {count}"
    )


# Weight for each class
class_weights = 1.0 / class_counts

print("\nClass weights:")
for class_name, weight in zip(
    CLASS_NAMES,
    class_weights
):
    print(
        f"{class_name:10s}: {weight:.6f}"
    )


# Weight for each individual image
sample_weights = np.array([
    class_weights[label]
    for label in train_labels
])


sample_weights = torch.DoubleTensor(
    sample_weights
)


train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


print("\nWeightedRandomSampler created successfully.")

# ============================================================
# CELL 24: CREATE DATALOADERS\n# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


print("DataLoaders created.")

print("\nNumber of batches:")
print("Train      :", len(train_loader))
print("Validation :", len(val_loader))
print("Test       :", len(test_loader))

# ============================================================
# CELL 25: VERIFY DATALOADER\n# ============================================================

images, labels = next(iter(train_loader))

print("Image batch shape :", images.shape)
print("Label shape       :", labels.shape)
print("Labels            :", labels[:10].tolist())

print("\nExpected image shape:")
print("(batch_size, 3, 224, 224)")