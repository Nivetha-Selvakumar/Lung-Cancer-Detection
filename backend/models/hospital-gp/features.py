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

# Exact shared handcrafted feature extraction.

# ============================================================
# SOURCE COLAB CELL 18
# ============================================================

# ============================================================
# CELL 11 — EXPLICIT FEATURE EXTRACTION
# ============================================================

from skimage.feature import (
    hog,
    local_binary_pattern,
    graycomatrix,
    graycoprops,
    canny
)

from skimage.measure import regionprops, label as sk_label


# ============================================================
# 1. INTENSITY FEATURES
# ============================================================

def extract_intensity_features(image, mask):

    lung_pixels = image[mask > 0].astype(np.float32)

    if len(lung_pixels) == 0:
        return np.zeros(10, dtype=np.float32)

    percentiles = np.percentile(
        lung_pixels,
        [10, 25, 50, 75, 90]
    )

    mean = np.mean(lung_pixels)
    std = np.std(lung_pixels)

    variance = np.var(lung_pixels)

    minimum = np.min(lung_pixels)
    maximum = np.max(lung_pixels)

    # Skewness
    if std > 1e-8:
        skewness = np.mean(
            ((lung_pixels - mean) / std) ** 3
        )

        # Kurtosis
        kurtosis = np.mean(
            ((lung_pixels - mean) / std) ** 4
        )
    else:
        skewness = 0.0
        kurtosis = 0.0

    features = [
        mean,
        std,
        variance,
        minimum,
        maximum,
        percentiles[0],
        percentiles[1],
        percentiles[2],
        percentiles[3],
        percentiles[4],
        skewness,
        kurtosis
    ]

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# 2. GLCM TEXTURE FEATURES
# ============================================================

def extract_glcm_features(image, mask):

    # Convert normalized image [0,1]
    # into 16 gray levels.
    quantized = np.floor(
        image * 16
    ).astype(np.uint8)

    quantized = np.clip(
        quantized,
        0,
        15
    )

    # Background = 0
    quantized[mask == 0] = 0

    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=16,
        symmetric=True,
        normed=True
    )

    features = []

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for prop in properties:

        values = graycoprops(
            glcm,
            prop
        )

        features.append(
            np.mean(values)
        )

        features.append(
            np.std(values)
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# 3. LBP TEXTURE FEATURES
# ============================================================

def extract_lbp_features(
    image,
    mask,
    radius=2
):

    points = 8 * radius

    lbp = local_binary_pattern(
        image,
        points,
        radius,
        method="uniform"
    )

    lung_pixels = lbp[
        mask > 0
    ]

    # Uniform LBP produces P + 2 bins
    n_bins = points + 2

    histogram, _ = np.histogram(
        lung_pixels,
        bins=np.arange(
            0,
            n_bins + 1
        ),
        range=(0, n_bins)
    )

    histogram = histogram.astype(
        np.float32
    )

    histogram /= (
        histogram.sum() + 1e-8
    )

    return histogram


# ============================================================
# 4. HOG FEATURES
# ============================================================

def extract_hog_features(
    image,
    mask
):

    roi = image.copy()

    roi[mask == 0] = 0

    hog_features = hog(
        roi,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return hog_features.astype(
        np.float32
    )


# ============================================================
# 5. SHAPE FEATURES
# ============================================================

def extract_shape_features(mask):

    binary_mask = (
        mask > 0
    ).astype(np.uint8)

    labeled = sk_label(
        binary_mask
    )

    regions = regionprops(
        labeled
    )

    # IMPORTANT:
    # Always return exactly 11 features
    # even if no region is detected.
    if len(regions) == 0:
        return np.zeros(
            11,
            dtype=np.float32
        )

    # Largest detected lung region
    largest_region = max(
        regions,
        key=lambda r: r.area
    )

    area = float(
        largest_region.area
    )

    perimeter = float(
        largest_region.perimeter
    )

    major_axis = float(
        largest_region.major_axis_length
    )

    minor_axis = float(
        largest_region.minor_axis_length
    )

    bbox_area = float(
        largest_region.bbox_area
    )

    circularity = (
        4 * np.pi * area /
        (perimeter ** 2 + 1e-8)
    )

    aspect_ratio = (
        major_axis /
        (minor_axis + 1e-8)
    )

    extent = float(
        largest_region.extent
    )

    solidity = float(
        largest_region.solidity
    )

    total_pixels = (
        binary_mask.shape[0] *
        binary_mask.shape[1]
    )

    area_ratio = (
        area /
        total_pixels
    )

    bbox_ratio = (
        bbox_area /
        total_pixels
    )

    features = [
        area,
        perimeter,
        major_axis,
        minor_axis,
        bbox_area,
        circularity,
        aspect_ratio,
        extent,
        solidity,
        area_ratio,
        bbox_ratio
    ]

    return np.asarray(
        features,
        dtype=np.float32
    )

# ============================================================
# 6. EDGE / GRADIENT FEATURES
# ============================================================

def extract_edge_features(
    image,
    mask
):

    # Sobel gradients
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        gx ** 2 +
        gy ** 2
    )

    lung_pixels = gradient_magnitude[
        mask > 0
    ]

    if len(lung_pixels) == 0:
        return np.zeros(
            11,
            dtype=np.float32
        )

    # Canny edge map
    edges = canny(
        image,
        sigma=1.0
    )

    edge_density = np.mean(
        edges[mask > 0]
    )

    mean_gradient = np.mean(
        lung_pixels
    )

    std_gradient = np.std(
        lung_pixels
    )

    max_gradient = np.max(
        lung_pixels
    )

    gradient_percentiles = np.percentile(
        lung_pixels,
        [10, 25, 50, 75, 90]
    )

    features = [
        edge_density,
        mean_gradient,
        std_gradient,
        max_gradient,
        gradient_percentiles[0],
        gradient_percentiles[1],
        gradient_percentiles[2],
        gradient_percentiles[3],
        gradient_percentiles[4]
    ]

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================

def extract_all_features(image_path):

    result = preprocess_image(
        image_path
    )

    image = result["normalized"]
    mask = result["mask"]

    # Extract individual feature groups
    intensity = extract_intensity_features(
        image,
        mask
    )

    glcm = extract_glcm_features(
        image,
        mask
    )

    lbp = extract_lbp_features(
        image,
        mask
    )

    hog_features = extract_hog_features(
        image,
        mask
    )

    shape = extract_shape_features(
        mask
    )

    edge = extract_edge_features(
        image,
        mask
    )

    # Combine all features
    feature_vector = np.concatenate([
        intensity,
        glcm,
        lbp,
        hog_features,
        shape,
        edge
    ])

    return feature_vector.astype(
        np.float32
    )


print("Feature extraction functions created successfully.")


# ============================================================
# SOURCE COLAB CELL 19
# ============================================================

# ============================================================
# CELL 11B — FEATURE COUNT
# ============================================================

# Use one training image only to determine
# the dimensionality of each feature group.

sample_path = train_df.iloc[0]["filepath"]

result = preprocess_image(
    sample_path
)

image = result["normalized"]
mask = result["mask"]


feature_groups = {

    "Intensity":
        len(
            extract_intensity_features(
                image, mask
            )
        ),

    "GLCM":
        len(
            extract_glcm_features(
                image, mask
            )
        ),

    "LBP":
        len(
            extract_lbp_features(
                image, mask
            )
        ),

    "HOG":
        len(
            extract_hog_features(
                image, mask
            )
        ),

    "Shape":
        len(
            extract_shape_features(
                mask
            )
        ),

    "Edge/Gradient":
        len(
            extract_edge_features(
                image, mask
            )
        )
}


feature_report = pd.DataFrame({
    "Feature Group":
        list(feature_groups.keys()),

    "Number of Features":
        list(feature_groups.values())
})


total_features = sum(
    feature_groups.values()
)


feature_report.loc[
    len(feature_report)
] = [
    "TOTAL",
    total_features
]


print("==============================================")
print("FEATURE EXTRACTION SUMMARY")
print("==============================================")

print(feature_report)

print(
    f"\nTotal explicitly extracted features: "
    f"{total_features}"
)


# ============================================================
# SOURCE COLAB CELL 20
# ============================================================

# ============================================================
# CELL 11C — FEATURE DIMENSION CONSISTENCY CHECK
# ============================================================

print("Checking feature dimensions for all development images...\n")

feature_dimensions = []

# if 'df' in globals():
#     for _, row in df.iterrows():
# 
#     vector = extract_all_features(
#         row["filepath"]
#     )
# 
#     feature_dimensions.append({
#         "filepath": row["filepath"],
#         "class": row["class"],
#         "feature_count": len(vector)
#     })
# 
# 
# feature_dimension_df = pd.DataFrame(
#     feature_dimensions
# )
# 
# print("==============================================")
# print("FEATURE DIMENSION CHECK")
# print("==============================================")
# 
# print(
#     feature_dimension_df[
#         "feature_count"
#     ].value_counts().sort_index()
# )
# 
# print("\nUnique feature counts:")
# 
# for count in sorted(
#     feature_dimension_df["feature_count"].unique()
# ):
#     print(
#         f"  {count} features : "
#         f"{(feature_dimension_df['feature_count'] == count).sum()} images"
#     )
# 
# 
# ============================================================
# SOURCE COLAB CELL 22
# ============================================================
# 
# ============================================================
# CELL 12 — EXTRACT FEATURES FOR ALL DATA SPLITS
# ============================================================
# 
# def build_feature_matrix(dataframe, dataset_name):
# 
#     features = []
#     labels = dataframe["label"].to_numpy()
# 
#     print(f"\nExtracting {dataset_name} features...")
#     print("-" * 50)
# 
#     for i, (_, row) in enumerate(dataframe.iterrows()):
# 
#         feature_vector = extract_all_features(
#             row["filepath"]
#         )
# 
#         features.append(
#             feature_vector
#         )
# 
#         if (i + 1) % 5 == 0 or (i + 1) == len(dataframe):
# 
#             print(
#                 f"{dataset_name}: "
#                 f"{i + 1}/{len(dataframe)}"
#             )
# 
#     X = np.vstack(features).astype(
#         np.float32
#     )
# 
#     y = labels.astype(
#         np.int64
#     )
# 
#     print(
#         f"{dataset_name} feature matrix: "
#         f"{X.shape}"
#     )
# 
#     return X, y
# 
# 
# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
# 
# X_train_raw, y_train = build_feature_matrix(
#     train_df,
#     "TRAIN"
# )
# 
# 
# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------
# 
# X_val_raw, y_val = build_feature_matrix(
#     val_df,
#     "VALIDATION"
# )
# 
# 
# ------------------------------------------------------------
# INTERNAL TEST
# ------------------------------------------------------------
# 
# X_test_raw, y_test = build_feature_matrix(
#     internal_test_df,
#     "INTERNAL TEST"
# )
# 
# 
# ============================================================
# FINAL SHAPE CHECK
# ============================================================
# 
# print("\n==============================================")
# print("FEATURE MATRIX SUMMARY")
# print("==============================================")
# 
# print(
#     "Training      :",
#     X_train_raw.shape
# )
# 
# print(
#     "Validation    :",
#     X_val_raw.shape
# )
# 
# print(
#     "Internal Test :",
#     X_test_raw.shape
# )
# 
# print("\nExpected feature count: 6146")