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


# ============================================================
# CELL 1: INSTALL REQUIRED LIBRARIES
# ============================================================

# !pip install -q torch torchvision torchaudio
# !pip install -q opencv-python scikit-learn seaborn pandas matplotlib pillow tqdm scipy


# ============================================================
# CELL 2: IMPORT LIBRARIES
# ============================================================

import os
import zipfile
import random
import time
import copy
import warnings
import hashlib

import numpy as np
import pandas as pd

import cv2
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision import transforms
from torchvision.models import (

    convnext_tiny,
    ConvNeXt_Tiny_Weights
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("PyTorch version :", torch.__version__)
print("Device          :", DEVICE)

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))


# ============================================================
# CELL 3: DATASET ACCESS (LOCAL / CONFIGURABLE)
# ============================================================
#
# Original notebook used Google Drive here.
# Only dataset access is changed for the standalone folder.
# The rest of the notebook logic is unchanged.
#
# Windows example:
# DATASET_ZIP_PATH = r"C:\Users\YourName\Downloads\archive.zip"
#
# Or set DATASET_ZIP_PATH as an environment variable.
# ============================================================

DATASET_ZIP_PATH = os.environ.get(
    "DATASET_ZIP_PATH",
    "data/archive.zip"
)

ZIP_PATH = DATASET_ZIP_PATH

EXTRACT_PATH = os.environ.get(
    "EXTRACT_PATH",
    "data/iqoth_nccd_convnext"
)

print("ZIP path:", ZIP_PATH)
print("Extract path:", EXTRACT_PATH)

if os.path.exists(ZIP_PATH):
    print("Dataset ZIP found successfully.")
else:
    raise FileNotFoundError(
        "Dataset ZIP was not found. "
        "Set DATASET_ZIP_PATH to your IQ-OTH/NCCD archive."
    )


# ============================================================
# CELL 5: EXTRACT DATASET\n# ============================================================

if not os.path.exists(EXTRACT_PATH):

    os.makedirs(EXTRACT_PATH, exist_ok=True)

    print("Extracting dataset...")

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zip_ref:

        zip_ref.extractall(EXTRACT_PATH)

    print("Extraction completed.")

else:

    print("Dataset already extracted.")


# ============================================================
# CELL 6: INSPECT DATASET STRUCTURE\n# ============================================================

for root, dirs, files in os.walk(EXTRACT_PATH):

    level = root.replace(
        EXTRACT_PATH,
        ""
    ).count(os.sep)

    indent = "    " * level

    print(
        f"{indent}{os.path.basename(root)}/"
    )

    if level >= 2:
        continue

    for file in files[:5]:

        print(
            f"{indent}    {file}"
        )


# ============================================================
# CELL 7: FIND CLASS DIRECTORIES\n# ============================================================

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]

class_directories = {}

for root, dirs, files in os.walk(EXTRACT_PATH):

    for directory in dirs:

        folder_name = directory.strip().lower()

        full_path = os.path.join(
            root,
            directory
        )

        # Normal
        if folder_name in [
            "normal",
            "normal cases"
        ]:

            class_directories["Normal"] = full_path

        # Benign / Bengin
        elif folder_name in [
            "benign",
            "benign cases",
            "bengin",
            "bengin cases"
        ]:

            class_directories["Benign"] = full_path

        # Malignant
        elif folder_name in [
            "malignant",
            "malignant cases"
        ]:

            class_directories["Malignant"] = full_path


print("\nDetected class directories:\n")

for class_name in CLASS_NAMES:

    if class_name in class_directories:

        print(
            f"{class_name:10s} -> "
            f"{class_directories[class_name]}"
        )

    else:

        print(
            f"{class_name:10s} -> NOT FOUND"
        )

if len(class_directories) != 3:

    raise RuntimeError(
        "One or more class directories could not be found."
    )

# ============================================================
# CELL 8: BUILD DATAFRAME\n# ============================================================

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)

LABEL_MAP = {
    "Normal": 0,
    "Benign": 1,
    "Malignant": 2
}

dataset_records = []

for class_name in CLASS_NAMES:

    class_path = class_directories[class_name]

    for root, dirs, files in os.walk(class_path):

        for file_name in files:

            if file_name.lower().endswith(IMAGE_EXTENSIONS):

                image_path = os.path.join(
                    root,
                    file_name
                )

                dataset_records.append({
                    "image_path": image_path,
                    "class_name": class_name,
                    "label": LABEL_MAP[class_name]
                })


dataset_df = pd.DataFrame(dataset_records)

# Shuffle
dataset_df = dataset_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


print("Total images:", len(dataset_df))
print("\nClass distribution:")
print(dataset_df["class_name"].value_counts())

print("\nFirst 5 records:")
print(dataset_df.head())

# ============================================================
# CELL 9: IMAGE QUALITY CHECK\n# ============================================================

quality_results = []

for index, row in dataset_df.iterrows():

    image_path = row["image_path"]

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        quality_results.append({
            "valid": False,
            "height": 0,
            "width": 0,
            "finite": False
        })

        continue

    height, width = image.shape

    finite_pixels = np.isfinite(image).all()

    valid = (
        height > 0
        and width > 0
        and finite_pixels
    )

    quality_results.append({
        "valid": valid,
        "height": height,
        "width": width,
        "finite": finite_pixels
    })


quality_df = pd.DataFrame(quality_results)

dataset_df["valid"] = quality_df["valid"]
dataset_df["height"] = quality_df["height"]
dataset_df["width"] = quality_df["width"]


print("Total images       :", len(dataset_df))
print("Valid images       :", dataset_df["valid"].sum())
print("Invalid images     :", (~dataset_df["valid"]).sum())

# ============================================================
# CELL 10: REMOVE INVALID IMAGES\n# ============================================================

clean_dataset_df = dataset_df[
    dataset_df["valid"] == True
].copy()

clean_dataset_df = clean_dataset_df.reset_index(
    drop=True
)

print("Clean dataset size:", len(clean_dataset_df))

print("\nFinal class distribution:")
print(
    clean_dataset_df["class_name"]
    .value_counts()
)

# ============================================================
# CELL 11: CLASS DISTRIBUTION\n# ============================================================

class_counts = (
    clean_dataset_df["class_name"]
    .value_counts()
    .reindex(CLASS_NAMES)
)

plt.figure(figsize=(8, 5))

class_counts.plot(
    kind="bar"
)

plt.title(
    "IQ-OTH/NCCD Dataset Class Distribution"
)

plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print(class_counts)

# ============================================================
# CELL 12: STRATIFIED DATASET SPLIT\n# ============================================================


# ------------------------------------------------------------
# Ensure Dataset Splits Loaded
# ------------------------------------------------------------
if 'train_df' not in globals() or globals().get('train_df') is None:
    try:
        import sys
        from pathlib import Path
        _shared_dir = str(Path(__file__).resolve().parent.parent / "shared")
        if _shared_dir not in sys.path:
            sys.path.insert(0, _shared_dir)
        from shared.dataset_config import load_dataset_splits
        dataset_type = 'hospital' if 'iq-Convnext-tiny'.startswith('hosp') else 'iq'
        train_df, val_df, internal_test_df = load_dataset_splits(dataset_type)
        print(f'Dataset splits loaded successfully for iq-Convnext-tiny: Train={len(train_df)}, Val={len(val_df)}, Test={len(internal_test_df)}')
    except Exception as e:
        print(f'Dataset split auto-load note: {e}')

train_df, temp_df = train_test_split(
    clean_dataset_df,
    test_size=0.30,
    stratify=clean_dataset_df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


print("Dataset Split")
print("=" * 50)

print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Testing    : {len(test_df)}")

print("\nTraining distribution:")
print(train_df["class_name"].value_counts())

print("\nValidation distribution:")
print(val_df["class_name"].value_counts())

print("\nTesting distribution:")
print(test_df["class_name"].value_counts())

# ============================================================
# CELL 13: DATA LEAKAGE CHECK\n# ============================================================

train_paths = set(train_df["image_path"])
val_paths = set(val_df["image_path"])
test_paths = set(test_df["image_path"])

train_val_overlap = train_paths.intersection(val_paths)
train_test_overlap = train_paths.intersection(test_paths)
val_test_overlap = val_paths.intersection(test_paths)

print("Train ↔ Validation overlap :", len(train_val_overlap))
print("Train ↔ Test overlap       :", len(train_test_overlap))
print("Validation ↔ Test overlap  :", len(val_test_overlap))

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print("\n[OK] No image-level data leakage detected.")

else:

    print("\n⚠️ Possible data leakage detected.")

# ============================================================
# CELL 14: SAMPLE IMAGES\n# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

for index, class_name in enumerate(CLASS_NAMES):

    sample_row = (
        clean_dataset_df[
            clean_dataset_df["class_name"] == class_name
        ]
        .sample(
            1,
            random_state=SEED
        )
        .iloc[0]
    )

    image = cv2.imread(
        sample_row["image_path"],
        cv2.IMREAD_GRAYSCALE
    )

    axes[index].imshow(
        image,
        cmap="gray"
    )

    axes[index].set_title(
        class_name
    )

    axes[index].axis("off")


plt.suptitle(
    "Sample CT Images from IQ-OTH/NCCD Dataset",
    fontsize=14
)

plt.tight_layout()
plt.show()

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

# ============================================================
# CELL 26: LOAD PRETRAINED CONVNEXT-TINY\n# ============================================================

# Load pretrained ConvNeXt-Tiny
weights = ConvNeXt_Tiny_Weights.DEFAULT

model = convnext_tiny(
    weights=weights
)

# ------------------------------------------------------------
# Replace the final classifier
# ------------------------------------------------------------

# ConvNeXt-Tiny's classifier:
# [LayerNorm, Flatten, Linear]

in_features = model.classifier[2].in_features

model.classifier[2] = nn.Linear(
    in_features,
    3
)

# Move model to GPU / CPU
model = model.to(DEVICE)

print(model.classifier)

print("\nModel loaded successfully.")
print("Number of output classes:", 3)
print("Classes:", CLASS_NAMES)

# ============================================================
# CELL 27: FREEZE BACKBONE\n# ============================================================

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False


# Unfreeze only the classifier
for param in model.classifier.parameters():
    param.requires_grad = True


# ------------------------------------------------------------
# Count parameters
# ------------------------------------------------------------

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total parameters     :", f"{total_parameters:,}")
print("Trainable parameters :", f"{trainable_parameters:,}")
print("Frozen parameters    :", f"{total_parameters - trainable_parameters:,}")

# ============================================================
# CELL 28: BENIGN-AWARE FOCAL LOSS
# ============================================================
#
# The segmentation, dataset, model architecture and input remain
# unchanged.
#
# The main training issue observed in the previous run was that
# the Benign class was frequently confused with the other classes.
# Focal loss gives more learning emphasis to hard/ambiguous
# training examples without changing the ConvNeXt-Tiny architecture.
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma=1.5,
        label_smoothing=0.02
    ):
        super().__init__()

        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):

        log_probs = torch.log_softmax(
            logits,
            dim=1
        )

        probs = torch.exp(log_probs)

        num_classes = logits.size(1)

        with torch.no_grad():

            smooth_targets = torch.full_like(
                logits,
                self.label_smoothing / num_classes
            )

            smooth_targets.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.label_smoothing
                + self.label_smoothing / num_classes
            )

        ce = -(
            smooth_targets * log_probs
        ).sum(dim=1)

        pt = (
            probs * smooth_targets
        ).sum(dim=1).clamp_min(1e-6)

        focal_factor = (
            1.0 - pt
        ).pow(self.gamma)

        loss = (
            focal_factor * ce
        ).mean()

        return loss


criterion = FocalLoss(
    gamma=1.5,
    label_smoothing=0.02
)

print("Loss function:", criterion)
print("Focal gamma   :", criterion.gamma)
print("Label smoothing:", criterion.label_smoothing)


# ============================================================
# CELL 29: STAGE-1 OPTIMIZER
# ============================================================

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("Optimizer:", optimizer)
print("Learning rate:", LEARNING_RATE)
print("Weight decay :", WEIGHT_DECAY)


# ============================================================
# CELL 30: STAGE-1 LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

print("Scheduler created successfully.")


# ============================================================
# CELL 31: TRAINING FUNCTION\n# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return (
        epoch_loss,
        epoch_accuracy
    )

# ============================================================
# CELL 32: VALIDATION FUNCTION\n# ============================================================

def validate_model(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    epoch_macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    epoch_macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "balanced_accuracy": epoch_balanced_accuracy,
        "macro_precision": epoch_macro_precision,
        "macro_recall": epoch_macro_recall,
        "macro_f1": epoch_macro_f1,
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probabilities)
    }

# ============================================================
# CELL 33: STAGE 1 TRAINING\n# ============================================================

STAGE1_EPOCHS = 8

best_val_macro_f1 = -1.0
best_stage1_state = None

stage1_history = []

print("=" * 70)
print("CONVNEXT-TINY — STAGE 1 TRANSFER LEARNING")
print("=" * 70)

for epoch in range(STAGE1_EPOCHS):

    start_time = time.time()

    # -------------------------------
    # Training
    # -------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    # -------------------------------
    # Validation
    # -------------------------------

    val_metrics = validate_model(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    # -------------------------------
    # Scheduler
    # -------------------------------

    scheduler.step(
        val_metrics["macro_f1"]
    )

    epoch_time = time.time() - start_time

    # Store history
    stage1_history.append({

        "Epoch": epoch + 1,

        "Train Loss": train_loss,

        "Train Accuracy": train_accuracy,

        "Val Loss": val_metrics["loss"],

        "Val Accuracy": val_metrics["accuracy"],

        "Val Balanced Accuracy":
            val_metrics["balanced_accuracy"],

        "Val Macro F1":
            val_metrics["macro_f1"]

    })

    # -------------------------------
    # Save best model
    # -------------------------------

    if (
        val_metrics["macro_f1"]
        > best_val_macro_f1
    ):

        best_val_macro_f1 = (
            val_metrics["macro_f1"]
        )

        best_stage1_state = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = epoch + 1

    # -------------------------------
    # Display
    # -------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{STAGE1_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Bal Acc: {val_metrics['balanced_accuracy']:.4f} | "
        f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
        f"Time: {epoch_time:.1f}s"
    )


print("\nBest Stage-1 Epoch:", best_epoch)
print(
    "Best Validation Macro F1:",
    f"{best_val_macro_f1:.4f}"
)

# ============================================================
# CELL 34: RESTORE BEST STAGE-1 MODEL\n# ============================================================

if best_stage1_state is not None:

    model.load_state_dict(
        best_stage1_state
    )

    print(
        f"Best Stage-1 model restored "
        f"from epoch {best_epoch}."
    )

else:

    print(
        "Warning: No best model state was saved."
    )

# ============================================================
# CELL 35: STAGE-1 TRAINING CURVES\n# ============================================================

stage1_history_df = pd.DataFrame(
    stage1_history
)

plt.figure(figsize=(8, 5))

plt.plot(
    stage1_history_df["Epoch"],
    stage1_history_df["Train Loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    stage1_history_df["Epoch"],
    stage1_history_df["Val Loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ConvNeXt-Tiny Stage-1 Loss"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    stage1_history_df["Epoch"],
    stage1_history_df["Val Macro F1"],
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Macro F1")

plt.title(
    "ConvNeXt-Tiny Stage-1 Validation Macro F1"
)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# CELL 36: STAGE 2 - UNFREEZE DEEPER LAYERS
# ============================================================
#
# Segmentation and ConvNeXt architecture are unchanged.
# We only make the fine-tuning more gradual so the pretrained
# representation is not destroyed by a large update.
# ============================================================

for param in model.parameters():
    param.requires_grad = False

# Fine-tune the two deepest feature stages.
for stage_index in [4, 6]:

    for param in model.features[stage_index].parameters():
        param.requires_grad = True

# Classification head
for param in model.classifier.parameters():
    param.requires_grad = True

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total parameters     :", f"{total_parameters:,}")
print("Trainable parameters :", f"{trainable_parameters:,}")
print("Frozen parameters    :", f"{total_parameters - trainable_parameters:,}")
print("Unfrozen feature stages: [4, 6]")


# ============================================================
# CELL 37: FINE-TUNING OPTIMIZER
# ============================================================
#
# Differential learning rates:
#   - Deep ConvNeXt features receive small updates.
#   - Classifier receives a larger update.
#
# This preserves pretrained visual features while adapting them
# to the segmented IQ-OTH/NCCD images.
# ============================================================

FINETUNE_WEIGHT_DECAY = 1e-4

feature_parameters = []
classifier_parameters = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if name.startswith("classifier"):
        classifier_parameters.append(parameter)

    else:
        feature_parameters.append(parameter)


finetune_optimizer = optim.AdamW(
    [
        {
            "params": feature_parameters,
            "lr": 1e-5
        },
        {
            "params": classifier_parameters,
            "lr": 1e-4
        }
    ],
    weight_decay=FINETUNE_WEIGHT_DECAY
)

print("Fine-tuning optimizer created.")
print("Feature learning rate   : 1e-5")
print("Classifier learning rate: 1e-4")
print("Weight decay             :", FINETUNE_WEIGHT_DECAY)


# ============================================================
# CELL 38: FINE-TUNING SCHEDULER
# ============================================================

FINETUNE_EPOCHS = 20

finetune_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    finetune_optimizer,
    T_max=FINETUNE_EPOCHS,
    eta_min=1e-7
)

print("Fine-tuning scheduler created.")
print("Fine-tuning epochs:", FINETUNE_EPOCHS)


# ============================================================
# CELL 39: STAGE 2 FINE-TUNING\n# ============================================================

FINETUNE_EPOCHS = 20

best_finetune_macro_f1 = best_val_macro_f1
best_finetune_state = copy.deepcopy(
    model.state_dict()
)

best_finetune_epoch = 0

finetune_history = []

print("=" * 80)
print("CONVNEXT-TINY — STAGE 2 FINE-TUNING")
print("=" * 80)

for epoch in range(FINETUNE_EPOCHS):

    start_time = time.time()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        finetune_optimizer,
        DEVICE
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_metrics = validate_model(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    finetune_scheduler.step(
        val_metrics["macro_f1"]
    )

    epoch_time = time.time() - start_time

    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    finetune_history.append({

        "Epoch": epoch + 1,

        "Train Loss": train_loss,

        "Train Accuracy": train_accuracy,

        "Val Loss": val_metrics["loss"],

        "Val Accuracy": val_metrics["accuracy"],

        "Val Balanced Accuracy":
            val_metrics["balanced_accuracy"],

        "Val Macro Precision":
            val_metrics["macro_precision"],

        "Val Macro Recall":
            val_metrics["macro_recall"],

        "Val Macro F1":
            val_metrics["macro_f1"]

    })

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if (
        val_metrics["macro_f1"]
        > best_finetune_macro_f1
    ):

        best_finetune_macro_f1 = (
            val_metrics["macro_f1"]
        )

        best_finetune_state = copy.deepcopy(
            model.state_dict()
        )

        best_finetune_epoch = epoch + 1

        improved = "[OK] BEST"

    else:

        improved = ""

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{FINETUNE_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Bal Acc: {val_metrics['balanced_accuracy']:.4f} | "
        f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
        f"{improved} | "
        f"Time: {epoch_time:.1f}s"
    )


print("\n" + "=" * 80)
print("Fine-tuning completed.")
print(
    "Best Validation Macro F1:",
    f"{best_finetune_macro_f1:.4f}"
)

if best_finetune_epoch > 0:
    print(
        "Best Fine-tuning Epoch:",
        best_finetune_epoch
    )
else:
    print(
        "Stage-1 model remained the best model."
    )

# ============================================================
# CELL 40: RESTORE BEST CONVNEXT MODEL\n# ============================================================

model.load_state_dict(
    best_finetune_state
)

print("Best ConvNeXt-Tiny model restored.")

print(
    "Best Validation Macro F1:",
    f"{best_finetune_macro_f1:.4f}"
)

# ============================================================
# CELL 41: FINE-TUNING CURVES\n# ============================================================

finetune_history_df = pd.DataFrame(
    finetune_history
)

# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    finetune_history_df["Epoch"],
    finetune_history_df["Train Loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    finetune_history_df["Epoch"],
    finetune_history_df["Val Loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ConvNeXt-Tiny Fine-Tuning Loss"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Macro F1
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    finetune_history_df["Epoch"],
    finetune_history_df["Val Macro F1"],
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Macro F1")

plt.title(
    "ConvNeXt-Tiny Validation Macro F1"
)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Balanced Accuracy
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    finetune_history_df["Epoch"],
    finetune_history_df["Val Balanced Accuracy"],
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Balanced Accuracy")

plt.title(
    "ConvNeXt-Tiny Validation Balanced Accuracy"
)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# CELL 42: FINAL VALIDATION EVALUATION\n# ============================================================

final_val_metrics = validate_model(
    model,
    val_loader,
    criterion,
    DEVICE
)

print("=" * 70)
print("FINAL CONVNEXT-TINY VALIDATION RESULTS")
print("=" * 70)

print(
    f"Accuracy           : "
    f"{final_val_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Balanced Accuracy  : "
    f"{final_val_metrics['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Macro Precision    : "
    f"{final_val_metrics['macro_precision'] * 100:.2f}%"
)

print(
    f"Macro Recall       : "
    f"{final_val_metrics['macro_recall'] * 100:.2f}%"
)

print(
    f"Macro F1           : "
    f"{final_val_metrics['macro_f1'] * 100:.2f}%"
)

print("=" * 70)

# ============================================================
# CELL 43: VALIDATION CLASSIFICATION REPORT\n# ============================================================

print(
    classification_report(
        final_val_metrics["labels"],
        final_val_metrics["predictions"],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# CELL 44: VALIDATION CONFUSION MATRIX\n# ============================================================

cm = confusion_matrix(
    final_val_metrics["labels"],
    final_val_metrics["predictions"]
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.title(
    "ConvNeXt-Tiny Validation Confusion Matrix"
)

plt.tight_layout()
plt.show()

# ============================================================
# CELL 45: STRONGER CONVNEXT-TINY FINE-TUNING
# ============================================================
#
# Same ConvNeXt-Tiny architecture and same segmented ROI input.
# We now adapt the full representation gradually.
# ============================================================

model.load_state_dict(best_finetune_state)

for param in model.parameters():
    param.requires_grad = False

# Unfreeze all four ConvNeXt feature stages for final adaptation.
# The learning rates below are deliberately small for the earlier
# pretrained stages.
for stage_index in [0, 2, 4, 6]:

    for param in model.features[stage_index].parameters():
        param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("=" * 70)
print("STRONGER FINE-TUNING CONFIGURATION")
print("=" * 70)

print(
    "Total parameters     :",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters :",
    f"{trainable_parameters:,}"
)

print(
    "Frozen parameters    :",
    f"{total_parameters - trainable_parameters:,}"
)

print("\nUnfrozen feature stages: [0, 2, 4, 6]")


# ============================================================
# CELL 46: FINAL FOCAL LOSS
# ============================================================
#
# Keep the same loss family used for training. The lower focal
# gamma avoids over-focusing on extremely hard/noisy samples.
# ============================================================

criterion_ft = FocalLoss(
    gamma=1.5,
    label_smoothing=0.02
)

print(criterion_ft)


# ============================================================
# CELL 47: FINAL FINE-TUNING OPTIMIZER
# ============================================================
#
# Differential learning rates preserve the pretrained
# representation while allowing the classifier and deepest
# features to adapt to the segmented CT ROI.
# ============================================================

FT2_WEIGHT_DECAY = 1e-4

parameter_groups = [
    {
        "params": model.features[0].parameters(),
        "lr": 2e-6
    },
    {
        "params": model.features[2].parameters(),
        "lr": 4e-6
    },
    {
        "params": model.features[4].parameters(),
        "lr": 8e-6
    },
    {
        "params": model.features[6].parameters(),
        "lr": 1.5e-5
    },
    {
        "params": model.classifier.parameters(),
        "lr": 8e-5
    }
]

optimizer_ft2 = optim.AdamW(
    parameter_groups,
    weight_decay=FT2_WEIGHT_DECAY
)

print("Final fine-tuning optimizer created.")
print("Feature-0 LR : 2e-6")
print("Feature-2 LR : 4e-6")
print("Feature-4 LR : 8e-6")
print("Feature-6 LR : 1.5e-5")
print("Classifier LR: 8e-5")
print("Weight decay :", FT2_WEIGHT_DECAY)


# ============================================================
# CELL 48: FINAL COSINE LEARNING RATE SCHEDULER
# ============================================================

FT2_EPOCHS = 30

scheduler_ft2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft2,
    T_max=FT2_EPOCHS,
    eta_min=5e-7
)

print("Cosine scheduler created.")
print("Epochs:", FT2_EPOCHS)


# ============================================================
# CELL 49: STRONG FINE-TUNING
# ============================================================

# Re-initialize the criterion to ensure it's defined
criterion_ft = FocalLoss(
    gamma=1.5,
    label_smoothing=0.02
)

best_ft2_macro_f1 = -1.0
best_ft2_accuracy = -1.0
best_ft2_balanced_accuracy = -1.0

best_ft2_state = None
best_ft2_epoch = 0

ft2_history = []

print("=" * 85)
print("CONVNEXT-TINY — STRONG FINE-TUNING")
print("=" * 85)

for epoch in range(FT2_EPOCHS):

    start_time = time.time()

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion_ft,
        optimizer_ft2,
        DEVICE
    )

    val_metrics = validate_model(
        model,
        val_loader,
        criterion_ft,
        DEVICE
    )

    scheduler_ft2.step()

    current_lr = optimizer_ft2.param_groups[-1]["lr"]

    epoch_time = time.time() - start_time

    ft2_history.append({

        "Epoch": epoch + 1,

        "Train Loss": train_loss,

        "Train Accuracy": train_accuracy,

        "Val Loss": val_metrics["loss"],

        "Val Accuracy": val_metrics["accuracy"],

        "Val Balanced Accuracy":
            val_metrics["balanced_accuracy"],

        "Val Macro Precision":
            val_metrics["macro_precision"],

        "Val Macro Recall":
            val_metrics["macro_recall"],

        "Val Macro F1":
            val_metrics["macro_f1"],

        "Learning Rate": current_lr
    })

    # Primary: Macro F1
    # Secondary: Balanced Accuracy
    # Tertiary: Accuracy
    is_better = False

    if val_metrics["macro_f1"] > best_ft2_macro_f1 + 1e-6:

        is_better = True

    elif abs(
        val_metrics["macro_f1"] - best_ft2_macro_f1
    ) <= 1e-6:

        if val_metrics["balanced_accuracy"] > best_ft2_balanced_accuracy + 1e-6:

            is_better = True

        elif abs(
            val_metrics["balanced_accuracy"]
            - best_ft2_balanced_accuracy
        ) <= 1e-6 and val_metrics["accuracy"] > best_ft2_accuracy:

            is_better = True

    if is_better:

        best_ft2_macro_f1 = val_metrics["macro_f1"]

        best_ft2_accuracy = val_metrics["accuracy"]

        best_ft2_balanced_accuracy = (
            val_metrics["balanced_accuracy"]
        )

        best_ft2_state = copy.deepcopy(
            model.state_dict()
        )

        best_ft2_epoch = epoch + 1

        marker = "[OK] BEST"

    else:

        marker = ""

    print(
        f"Epoch [{epoch+1:02d}/{FT2_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy*100:.2f}% | "
        f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
        f"Val Bal Acc: {val_metrics['balanced_accuracy']*100:.2f}% | "
        f"Val Macro F1: {val_metrics['macro_f1']*100:.2f}% | "
        f"LR: {current_lr:.2e} | "
        f"{marker} | "
        f"{epoch_time:.1f}s"
    )

print("\n" + "=" * 85)
print("STRONG FINE-TUNING COMPLETED")
print("=" * 85)

print(
    "Best Epoch              :",
    best_ft2_epoch
)

print(
    "Best Validation Accuracy:",
    f"{best_ft2_accuracy*100:.2f}%"
)

print(
    "Best Balanced Accuracy  :",
    f"{best_ft2_balanced_accuracy*100:.2f}%"
)

print(
    "Best Validation Macro F1:",
    f"{best_ft2_macro_f1*100:.2f}%"
)

# ============================================================
# CELL 50: RESTORE BEST STRONG FINE-TUNED MODEL\n# ============================================================

if best_ft2_state is not None:

    model.load_state_dict(
        best_ft2_state
    )

    print(
        "Best strongly fine-tuned ConvNeXt-Tiny restored."
    )

    print(
        "Best validation accuracy:",
        f"{best_ft2_accuracy * 100:.2f}%"
    )

    print(
        "Best validation Macro F1:",
        f"{best_ft2_macro_f1 * 100:.2f}%"
    )

else:

    print(
        "No best model was saved."
    )

# ============================================================
# CELL 51: FINAL VALIDATION EVALUATION (BEST MODEL)\n# ============================================================
# NOTE: this uses a DIFFERENT variable name from the
# "final_val_metrics" computed earlier (right after Stage-2,
# before strong fine-tuning). Keeping them separate matters:
#   - final_val_metrics      -> the Stage-2 checkpoint
#                                ("Previous ConvNeXt-Tiny" row
#                                in the comparison table below)
#   - best_ft2_val_metrics   -> the final, strong-fine-tuned
#                                checkpoint, used for the TTA
#                                check that follows
# Overwriting final_val_metrics here would make the comparison
# table show the same numbers twice.

best_ft2_val_metrics = validate_model(
    model,
    val_loader,
    criterion_ft,
    DEVICE
)

print("=" * 70)
print("FINAL CONVNEXT-TINY VALIDATION RESULTS (BEST MODEL)")
print("=" * 70)

print(f"Accuracy           : {best_ft2_val_metrics['accuracy'] * 100:.2f}%")
print(f"Balanced Accuracy  : {best_ft2_val_metrics['balanced_accuracy'] * 100:.2f}%")
print(f"Macro Precision    : {best_ft2_val_metrics['macro_precision'] * 100:.2f}%")
print(f"Macro Recall       : {best_ft2_val_metrics['macro_recall'] * 100:.2f}%")
print(f"Macro F1           : {best_ft2_val_metrics['macro_f1'] * 100:.2f}%")
print("=" * 70)


# ============================================================
# CELL 52: OPTIONAL TEST-TIME AUGMENTATION (TTA)\n# ============================================================
#
# TTA is evaluated on the VALIDATION set first.
# We only use it for final testing if it improves validation Macro F1.
# This avoids choosing a technique from the test set.

def predict_with_tta(model, images, device):
    """
    Average predictions from:
      1. original image
      2. horizontally flipped image

    Horizontal flip is consistent with the training augmentation
    and does not change the Normal/Benign/Malignant class.
    """
    model.eval()

    with torch.no_grad():

        logits_original = model(images)

        flipped_images = torch.flip(
            images,
            dims=[3]
        )

        logits_flipped = model(
            flipped_images
        )

        probabilities = (
            torch.softmax(
                logits_original,
                dim=1
            )
            +
            torch.softmax(
                logits_flipped,
                dim=1
            )
        ) / 2.0

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    return (
        predictions,
        probabilities
    )


def evaluate_tta(
    model,
    loader,
    device
):

    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            predictions, probabilities = predict_with_tta(
                model,
                images,
                device
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "labels": y_true,
        "predictions": y_pred,
        "probabilities": np.array(
            all_probabilities
        )
    }


tta_val_metrics = evaluate_tta(
    model,
    val_loader,
    DEVICE
)

print("=" * 75)
print("VALIDATION TTA CHECK")
print("=" * 75)

print(
    f"Standard Validation Accuracy : "
    f"{best_ft2_val_metrics['accuracy']*100:.2f}%"
)

print(
    f"TTA Validation Accuracy      : "
    f"{tta_val_metrics['accuracy']*100:.2f}%"
)

print(
    f"Standard Validation Macro F1 : "
    f"{best_ft2_val_metrics['macro_f1']*100:.2f}%"
)

print(
    f"TTA Validation Macro F1      : "
    f"{tta_val_metrics['macro_f1']*100:.2f}%"
)

USE_TTA = (
    tta_val_metrics["macro_f1"]
    >
    best_ft2_val_metrics["macro_f1"]
)

print(
    "\nTTA selected:",
    USE_TTA
)

if USE_TTA:
    print(
        "TTA will be used for final test evaluation."
    )
else:
    print(
        "Standard prediction will be used for final test evaluation."
    )

# ============================================================
# CELL 53: COMPARE CONVNEXT EXPERIMENTS
# ============================================================

comparison = pd.DataFrame({

    "Experiment": [
        "ConvNeXt-Tiny — Stage-2",
        "ConvNeXt-Tiny — Final Fine-Tuning"
    ],

    "Validation Accuracy (%)": [

        final_val_metrics["accuracy"] * 100,

        best_ft2_accuracy * 100
    ],

    "Balanced Accuracy (%)": [

        final_val_metrics["balanced_accuracy"] * 100,

        best_ft2_balanced_accuracy * 100
    ],

    "Macro F1 (%)": [

        final_val_metrics["macro_f1"] * 100,

        best_ft2_macro_f1 * 100
    ]
})

print(
    comparison.round(2)
)


# ============================================================
# CELL 54: FINAL CONVNEXT-TINY TEST EVALUATION\n# ============================================================

# Restore the BEST validation-selected model.
model.load_state_dict(best_ft2_state)

model = model.to(DEVICE)
model.eval()

print("Evaluating on the untouched test set...")

if USE_TTA:

    test_metrics = evaluate_tta(
        model,
        test_loader,
        DEVICE
    )

    print("\nEvaluation mode: Test-Time Augmentation (TTA)")

else:

    test_metrics = validate_model(
        model,
        test_loader,
        criterion_ft,
        DEVICE
    )

    print("\nEvaluation mode: Standard inference")

print("\n" + "=" * 75)
print("              FINAL CONVNEXT-TINY TEST PERFORMANCE")
print("=" * 75)

print(
    f"Accuracy           : "
    f"{test_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Balanced Accuracy  : "
    f"{test_metrics['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Macro Precision    : "
    f"{test_metrics['macro_precision'] * 100:.2f}%"
)

print(
    f"Macro Recall       : "
    f"{test_metrics['macro_recall'] * 100:.2f}%"
)

print(
    f"Macro F1           : "
    f"{test_metrics['macro_f1'] * 100:.2f}%"
)

test_weighted_f1 = f1_score(
    test_metrics["labels"],
    test_metrics["predictions"],
    average="weighted",
    zero_division=0
)

print(
    f"Weighted F1        : "
    f"{test_weighted_f1 * 100:.2f}%"
)

print("=" * 75)

# ============================================================
# CELL 55: FINAL TEST CLASSIFICATION REPORT\n# ============================================================

final_test_report = classification_report(
    test_metrics["labels"],
    test_metrics["predictions"],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print(
    "\nFINAL CONVNEXT-TINY TEST CLASSIFICATION REPORT\n"
)

print(final_test_report)

# ============================================================
# CELL 56: FINAL TEST CONFUSION MATRIX\n# ============================================================

test_cm = confusion_matrix(
    test_metrics["labels"],
    test_metrics["predictions"]
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    test_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.title(
    "Final ConvNeXt-Tiny Test Confusion Matrix"
)

plt.tight_layout()
plt.show()

# ============================================================
# CELL 57: TEST PREDICTION DISTRIBUTION\n# ============================================================

actual_counts = np.bincount(
    test_metrics["labels"],
    minlength=3
)

predicted_counts = np.bincount(
    test_metrics["predictions"],
    minlength=3
)

distribution_df = pd.DataFrame({

    "Class": CLASS_NAMES,

    "Actual": actual_counts,

    "Predicted": predicted_counts

})

print(distribution_df)

# ============================================================
# CELL 58: FINAL TEST RESULTS TABLE\n# ============================================================

final_test_results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1"
    ],

    "ConvNeXt-Tiny (%)": [

        test_metrics["accuracy"] * 100,

        test_metrics["balanced_accuracy"] * 100,

        test_metrics["macro_precision"] * 100,

        test_metrics["macro_recall"] * 100,

        test_metrics["macro_f1"] * 100,

        test_weighted_f1 * 100

    ]

})

print(
    final_test_results.round(2)
)