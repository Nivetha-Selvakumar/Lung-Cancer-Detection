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
ndi = None
hashlib = None
EXTRACT_PATH = None
tqdm = None
plt = None
e = None

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
# CELL 15: LUNG SEGMENTATION - OUTPUT DIRECTORIES\n# ============================================================
#
# This segmentation stage extracts the LUNG REGION (ROI) from
# each 2-D CT slice using classical image processing. It is
# lung-region extraction, NOT tumor/lesion segmentation.
#
# The SAME function (segment_lung, defined in the next cell) is
# used both here (to build the cached training/val/test ROI
# images) and later at inference time on a newly uploaded image.
# Using one single function everywhere removes the train/inference
# mismatch that was causing wrong predictions after segmentation.

SEGMENTED_DIR = os.path.join(EXTRACT_PATH, "segmented_roi")
MASK_DIR = os.path.join(EXTRACT_PATH, "segmented_mask")

os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

print("ROI output dir :", SEGMENTED_DIR)
print("Mask output dir:", MASK_DIR)


# ============================================================
# CELL 16: LUNG SEGMENTATION FUNCTION (CLASSICAL IMAGE PROCESSING)\n# ============================================================
#
# Why classical image processing instead of a pretrained
# HU-based model (e.g. LungMask) here:
#
#   - The IQ-OTH/NCCD images are already-windowed 8-bit JPG/PNG
#     slices, not raw Hounsfield-unit CT volumes. Pretrained
#     lung-segmentation networks like LungMask expect real HU
#     input; running them on JPGs needs a "noHU" workaround that
#     is explicitly documented as unreliable, and in practice it
#     produced a DIFFERENT mask each time depending on an
#     arbitrary intensity-rescaling guess. That inconsistency is
#     exactly what was causing correct predictions on the raw
#     image to flip to wrong predictions after segmentation.
#   - A deterministic Otsu-threshold + connected-components
#     pipeline gives the exact same mask for the exact same
#     image every time, and is the standard approach used in
#     the literature for lung-field extraction on 2-D CT slices.
#
# Algorithm:
#   1. Otsu-threshold the (blurred) slice, inverted, so that
#      dark regions (air: background + lungs) become foreground.
#   2. Remove any foreground blob touching the image border
#      (this removes the black background outside the body,
#      leaving only air pockets INSIDE the body -> the lungs).
#   3. Morphological opening to drop thin noise / bridges.
#   4. Keep the largest 1-2 components (left + right lung).
#   5. Fill holes inside the kept components, so vessels/nodules
#      (which are brighter than air) stay INSIDE the lung ROI
#      instead of being punched out as "holes".
#   6. Morphological closing + small dilation to smooth the
#      boundary and avoid clipping peripheral nodules.
#
# This is lung-region / ROI extraction. It is NOT tumor or
# lesion segmentation.

def segment_lung(image_path):
    """
    Deterministic lung-field segmentation for a 2-D grayscale
    CT slice.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    original : np.ndarray (H, W), uint8   grayscale image
    mask     : np.ndarray (H, W), uint8   binary lung mask {0,1}
    roi      : np.ndarray (H, W), uint8   original image with the
                                           mask applied (background
                                           zeroed out)
    """

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w = original.shape

    # -----------------------------------------------------
    # 1. Otsu threshold (inverted -> dark/air = foreground)
    # -----------------------------------------------------
    blurred = cv2.GaussianBlur(original, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = (binary > 0).astype(np.uint8)

    # -----------------------------------------------------
    # 2. Remove background air connected to the image border
    # -----------------------------------------------------
    labeled, _ = ndi.label(binary)

    border_labels = set(labeled[0, :].tolist())
    border_labels |= set(labeled[-1, :].tolist())
    border_labels |= set(labeled[:, 0].tolist())
    border_labels |= set(labeled[:, -1].tolist())
    border_labels.discard(0)

    if border_labels:
        border_cleared = np.where(
            np.isin(labeled, list(border_labels)),
            0,
            binary
        ).astype(np.uint8)
    else:
        border_cleared = binary

    # -----------------------------------------------------
    # 3. Morphological opening (remove thin noise / bridges)
    # -----------------------------------------------------
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened = cv2.morphologyEx(
        border_cleared,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1
    )

    # -----------------------------------------------------
    # 4. Keep the largest 1-2 components (left + right lung)
    # -----------------------------------------------------
    labeled2, num2 = ndi.label(opened)

    if num2 == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return original, empty_mask, original.copy()

    sizes = ndi.sum(opened, labeled2, range(1, num2 + 1))

    min_area = 0.003 * h * w  # ignore tiny specks

    candidate_labels = [
        i + 1 for i, s in enumerate(sizes) if s > min_area
    ]

    if len(candidate_labels) == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return original, empty_mask, original.copy()

    candidate_labels = sorted(
        candidate_labels,
        key=lambda lab: sizes[lab - 1],
        reverse=True
    )[:2]

    mask = np.isin(labeled2, candidate_labels).astype(np.uint8)

    # -----------------------------------------------------
    # 5. Fill holes (keep vessels / nodules inside the lung)
    # -----------------------------------------------------
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    # -----------------------------------------------------
    # 6. Smooth boundary + small dilation
    # -----------------------------------------------------
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2
    )

    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )

    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    roi = original.copy()
    roi[mask == 0] = 0

    return original, mask, roi


# ============================================================
# CELL 17: TEST SEGMENTATION ON ONE SAMPLE IMAGE\n# ============================================================

test_image_path = test_df.iloc[0]["image_path"]

original, lung_mask, lung_roi = segment_lung(test_image_path)

coverage = (np.sum(lung_mask > 0) / lung_mask.size) * 100

print("=" * 60)
print("SEGMENTATION RESULT")
print("=" * 60)
print("Image shape   :", original.shape)
print(f"Lung coverage : {coverage:.2f}%")

overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
overlay[lung_mask > 0] = [0, 255, 0]
overlay = cv2.addWeighted(
    cv2.cvtColor(original, cv2.COLOR_GRAY2RGB), 0.7, overlay, 0.3, 0
)

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(original, cmap="gray")
plt.title("Original CT")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(lung_mask, cmap="gray")
plt.title(f"Lung Mask\n{coverage:.2f}%")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(lung_roi, cmap="gray")
plt.title("Lung ROI")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(overlay)
plt.title("Lung Mask Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# CELL 18: SEGMENT AND CACHE ALL IMAGES\n# ============================================================

all_images_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

all_image_paths = (
    all_images_df["image_path"]
    .drop_duplicates()
    .tolist()
)

print("Total unique images:", len(all_image_paths))


def output_paths(image_path):

    stem = os.path.splitext(os.path.basename(image_path))[0]

    short_hash = hashlib.md5(
        image_path.encode("utf-8")
    ).hexdigest()[:10]

    filename = f"{stem}_{short_hash}.png"

    roi_path = os.path.join(SEGMENTED_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)

    return roi_path, mask_path


all_segmented_records = []
failed_segmentation = []

for image_path in tqdm(all_image_paths, desc="Generating lung ROIs"):

    roi_path, mask_path = output_paths(image_path)

    try:

        original, mask, roi = segment_lung(image_path)

        mask_ratio = np.count_nonzero(mask) / mask.size

        # Reject suspicious masks (too small / too large to be lungs).
        if mask_ratio < 0.02 or mask_ratio > 0.55:

            failed_segmentation.append(
                (image_path, f"Suspicious mask: {mask_ratio * 100:.2f}%")
            )
            continue

        cv2.imwrite(roi_path, roi)
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        all_segmented_records.append((image_path, roi_path, mask_path))

    except Exception as e:
        failed_segmentation.append((image_path, repr(e)))


print("\n" + "=" * 75)
print("SEGMENTATION SUMMARY")
print("=" * 75)
print("Total images      :", len(all_image_paths))
print("Successful images :", len(all_segmented_records))
print("Failed images     :", len(failed_segmentation))

if len(failed_segmentation) > 0:
    print("\nFirst 10 failed images:")
    for item in failed_segmentation[:10]:
        print(item[0], "->", item[1])

if len(all_segmented_records) == 0:
    raise RuntimeError("No valid lung ROIs were generated.")


# ============================================================
# CELL 19: ADD SEGMENTED PATHS TO TRAIN / VAL / TEST\n# ============================================================

segmented_map = {
    source: roi for source, roi, mask in all_segmented_records
}


def add_segmented_paths(dataframe):
    df = dataframe.copy()
    df["segmented_path"] = df["image_path"].map(segmented_map)
    return df


train_df = add_segmented_paths(train_df)
val_df = add_segmented_paths(val_df)
test_df = add_segmented_paths(test_df)

print("=" * 75)
print("SEGMENTED DATASET COVERAGE")
print("=" * 75)
print("Train:", train_df["segmented_path"].notna().sum(), "/", len(train_df))
print("Validation:", val_df["segmented_path"].notna().sum(), "/", len(val_df))
print("Test:", test_df["segmented_path"].notna().sum(), "/", len(test_df))

# ------------------------------------------------------------
# Drop any image for which segmentation failed / was rejected,
# instead of letting the Dataset crash later on a NaN path.
# ------------------------------------------------------------

before_counts = (len(train_df), len(val_df), len(test_df))

train_df = train_df[train_df["segmented_path"].notna()].reset_index(drop=True)
val_df = val_df[val_df["segmented_path"].notna()].reset_index(drop=True)
test_df = test_df[test_df["segmented_path"].notna()].reset_index(drop=True)

after_counts = (len(train_df), len(val_df), len(test_df))

print("\nDropped rows without a valid segmented ROI:")
print("Train:", before_counts[0] - after_counts[0])
print("Validation:", before_counts[1] - after_counts[1])
print("Test:", before_counts[2] - after_counts[2])

print("\nFinal dataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))