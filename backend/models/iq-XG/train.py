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
# XGBOOST ML NOTEBOOK
# CELL 1 - INSTALL REQUIRED LIBRARIES
# ============================================================

# Install XGBoost
# !pip install -q xgboost

# Install image processing and ML libraries
# !pip install -q opencv-python scikit-image

print("[OK] Required libraries installed successfully.")

# ============================================================
# CELL 2 - IMPORT LIBRARIES
# ============================================================


import os
import zipfile
import random
import time
import warnings

import numpy as np
import pandas as pd

import cv2
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# Suppress unnecessary warnings
# ------------------------------------------------------------

warnings.filterwarnings(
    "ignore"
)


print("[OK] NumPy imported")
print("[OK] Pandas imported")
print("[OK] OpenCV imported")
print("[OK] Scikit-learn imported")
print("[OK] XGBoost imported")
print("[OK] All imports completed successfully.")

# ============================================================
# DATASET ACCESS - LOCAL / USER CONFIGURABLE
# (Replaces original Colab Google Drive access only)
# ============================================================

# Set this to your local IQ-OTH/NCCD archive.
# Example Windows:
# DATASET_ZIP_PATH = r"C:\Users\YourName\Downloads\archive.zip"
#
# Or set the environment variable DATASET_ZIP_PATH before running.
#
# IMPORTANT:
# No preprocessing, feature extraction, split, PCA, XGBoost,
# tuning, validation or test logic is changed.

candidate_zips = [
    os.environ.get("DATASET_ZIP_PATH", ""),
    r"c:\Users\HP\OneDrive\Desktop\Project Work 1\Lung Project Impl\IQ Dataset\archive.zip",
    r"c:\Users\HP\OneDrive\Desktop\Project Work 1\Lung Project Impl\archive.zip",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "IQ Dataset", "archive.zip")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "archive.zip")),
    "data/archive.zip"
]
DATASET_ZIP_PATH = None
for cz in candidate_zips:
    if cz and os.path.exists(cz):
        DATASET_ZIP_PATH = cz
        break
if DATASET_ZIP_PATH is None:
    DATASET_ZIP_PATH = r"c:\Users\HP\OneDrive\Desktop\Project Work 1\Lung Project Impl\IQ Dataset\archive.zip"


ZIP_PATH = DATASET_ZIP_PATH

EXTRACT_PATH = "data/iqoth_nccd_xgb"

print("ZIP file:")
print(ZIP_PATH)

print("\nExtraction directory:")
print(EXTRACT_PATH)

if not os.path.exists(ZIP_PATH) and not os.path.exists(r"c:\Users\HP\OneDrive\Desktop\Project Work 1\Lung Project Impl\archive\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset"):
    raise FileNotFoundError(
        "archive.zip was not found at:\n" + ZIP_PATH
    )

print("\n[OK] archive.zip found.")


# ============================================================
# CELL 5 - EXTRACT ARCHIVE.ZIP
# ============================================================
#
# The ZIP contains:
#
#     Test cases
#
#     The IQ-OTHNCCD lung cancer Dataset
#         ├── Bengin cases
#         ├── Normal cases
#         ├── Malignant cases
#         └── Text document
#
# ============================================================


if not os.path.exists(
    EXTRACT_PATH
):

    os.makedirs(
        EXTRACT_PATH
    )


print("Extracting archive.zip...")
print("This may take some time.\n")


with zipfile.ZipFile(

    ZIP_PATH,
    "r"

) as zip_ref:

    zip_ref.extractall(
        EXTRACT_PATH
    )


print("[OK] Extraction completed.")

print(
    "Extraction path:",
    EXTRACT_PATH
)

# ============================================================
# CELL 6 - INSPECT DATASET STRUCTURE
# ============================================================


print("=" * 60)
print("EXTRACTED DATASET STRUCTURE")
print("=" * 60)


directory_count = 0


for root, dirs, files in os.walk(
    EXTRACT_PATH
):

    level = root.replace(
        EXTRACT_PATH,
        ""
    ).count(
        os.sep
    )


    if level > 4:

        continue


    indent = "    " * level


    print(
        f"{indent}{os.path.basename(root)}/"
    )


    directory_count += len(dirs)


    for file in files[:5]:

        print(
            f"{indent}    {file}"
        )


print("\nTotal directories found:", directory_count)

# ============================================================
# CELL 7 - FIND IQ-OTH/NCCD CLASS FOLDERS
# ============================================================
#
# Dataset folder names:
#
#     Normal cases
#     Bengin cases
#     Malignant cases
#
# Internal model labels:
#
#     Normal     = 0
#     Benign     = 1
#     Malignant  = 2
#
# "Bengin" is the spelling used by the dataset.
# We map it internally to "Benign".
# ============================================================


CLASS_NAMES = [

    "Normal",
    "Benign",
    "Malignant"

]


class_directories = {}


for root, dirs, files in os.walk(
    EXTRACT_PATH
):

    for directory in dirs:

        folder_name = (

            directory
            .strip()
            .lower()

        )


        full_path = os.path.join(

            root,
            directory

        )


        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        if folder_name in [

            "normal",
            "normal cases"

        ]:

            class_directories[
                "Normal"
            ] = full_path


        # ----------------------------------------------------
        # BENGIN → BENIGN
        # ----------------------------------------------------

        elif folder_name in [

            "bengin",
            "bengin cases"

        ]:

            class_directories[
                "Benign"
            ] = full_path


        # ----------------------------------------------------
        # MALIGNANT
        # ----------------------------------------------------

        elif folder_name in [

            "malignant",
            "malignant cases"

        ]:

            class_directories[
                "Malignant"
            ] = full_path


print("=" * 60)
print("FINAL CLASS MAPPING")
print("=" * 60)


for class_name in CLASS_NAMES:

    if class_name in class_directories:

        print(
            f"[OK] {class_name}:"
        )

        print(
            f"  {class_directories[class_name]}"
        )

    else:

        print(
            f"✗ {class_name}: NOT FOUND"
        )


missing_classes = [

    class_name

    for class_name in CLASS_NAMES

    if class_name not in class_directories

]


if missing_classes:

    raise RuntimeError(

        "Missing class folders: "
        + str(missing_classes)

    )


print(
    "\n[OK] All three classes found successfully."
)

# ============================================================
# CELL 8 - BUILD IQ-OTH/NCCD DATAFRAME
# ============================================================
#
# Class labels:
#
#     0 = Normal
#     1 = Benign
#     2 = Malignant
#
# The dataframe stores:
#
#     filepath
#     class
#     label
#
# Images themselves will be loaded later.
# ============================================================


# ------------------------------------------------------------
# CLASS LABEL MAPPING
# ------------------------------------------------------------

CLASS_TO_LABEL = {

    "Normal": 0,

    "Benign": 1,

    "Malignant": 2

}


# ------------------------------------------------------------
# SUPPORTED IMAGE FORMATS
# ------------------------------------------------------------

IMAGE_EXTENSIONS = (

    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"

)


# ------------------------------------------------------------
# CREATE RECORDS
# ------------------------------------------------------------

dataset_records = []


for class_name in CLASS_NAMES:

    class_path = class_directories[
        class_name
    ]


    class_label = CLASS_TO_LABEL[
        class_name
    ]


    for root, dirs, files in os.walk(
        class_path
    ):

        for filename in files:

            # ------------------------------------------------
            # Check image extension
            # ------------------------------------------------

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                image_path = os.path.join(

                    root,
                    filename

                )


                dataset_records.append({

                    "filepath":
                        image_path,

                    "class":
                        class_name,

                    "label":
                        class_label

                })


# ------------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------------

dataset_df = pd.DataFrame(
    dataset_records
)


# ------------------------------------------------------------
# SHUFFLE DATAFRAME
# ------------------------------------------------------------

dataset_df = dataset_df.sample(

    frac=1,

    random_state=SEED

).reset_index(
    drop=True
)


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print("=" * 60)
print("IQ-OTH/NCCD DATASET")
print("=" * 60)


print(
    "Total images:",
    len(dataset_df)
)


print("\nClass distribution:")


print(

    dataset_df["class"]
    .value_counts()

)


print("\nLabel distribution:")


print(

    dataset_df["label"]
    .value_counts()
    .sort_index()

)


print("\nFirst 5 records:")


print(
    dataset_df.head()
)

# ============================================================
# CELL 9 - IMAGE QUALITY CHECK
# ============================================================
#
# We verify:
#
#     1. Image can be opened
#     2. Image is not empty
#     3. Image has valid dimensions
#     4. Image contains valid pixel values
#
# Invalid images will be recorded.
# ============================================================


print("=" * 60)
print("IMAGE QUALITY CHECK")
print("=" * 60)


valid_records = []

invalid_records = []


# ------------------------------------------------------------
# CHECK EACH IMAGE
# ------------------------------------------------------------

for index, row in dataset_df.iterrows():

    image_path = row["filepath"]


    try:

        # ----------------------------------------------------
        # Read image in grayscale
        # ----------------------------------------------------

        image = cv2.imread(

            image_path,

            cv2.IMREAD_GRAYSCALE

        )


        # ----------------------------------------------------
        # Check whether image was loaded
        # ----------------------------------------------------

        if image is None:

            invalid_records.append({

                "filepath":
                    image_path,

                "reason":
                    "Could not read image"

            })

            continue


        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        height, width = image.shape


        if height == 0 or width == 0:

            invalid_records.append({

                "filepath":
                    image_path,

                "reason":
                    "Invalid image dimensions"

            })

            continue


        # ----------------------------------------------------
        # Check pixel values
        # ----------------------------------------------------

        if not np.isfinite(
            image
        ).all():

            invalid_records.append({

                "filepath":
                    image_path,

                "reason":
                    "Invalid pixel values"

            })

            continue


        # ----------------------------------------------------
        # Image is valid
        # ----------------------------------------------------

        valid_records.append(
            index
        )


    except Exception as e:

        invalid_records.append({

            "filepath":
                image_path,

            "reason":
                str(e)

        })


# ------------------------------------------------------------
# CREATE CLEAN DATAFRAME
# ------------------------------------------------------------

clean_dataset_df = dataset_df.loc[
    valid_records
].reset_index(
    drop=True
)


invalid_df = pd.DataFrame(
    invalid_records
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "Total images checked :",
    len(dataset_df)
)


print(
    "Valid images         :",
    len(clean_dataset_df)
)


print(
    "Invalid images       :",
    len(invalid_df)
)


if len(invalid_df) == 0:

    print(
        "\n[OK] All images passed the quality check."
    )

else:

    print(
        "\n⚠ Invalid images detected."
    )

    print(
        invalid_df.head(10)
    )

# ============================================================
# CELL 10 - FINAL CLEAN DATASET DISTRIBUTION
# ============================================================


print("=" * 60)
print("FINAL CLEAN DATASET")
print("=" * 60)


print(
    "Total usable images:",
    len(clean_dataset_df)
)


print("\nClass distribution:")


class_counts = (

    clean_dataset_df["class"]
    .value_counts()
    .reindex(
        CLASS_NAMES
    )

)


print(
    class_counts
)


# ------------------------------------------------------------
# Verify all classes are present
# ------------------------------------------------------------

for class_name in CLASS_NAMES:

    count = class_counts[
        class_name
    ]


    if count == 0:

        raise RuntimeError(

            f"No images found for {class_name}."

        )


print(
    "\n[OK] All three classes are present."
)

# ============================================================
# CELL 11 - STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# Dataset:
#
#     Normal      -> 0
#     Benign      -> 1
#     Malignant   -> 2
#
# Split:
#
#     70% Training
#     15% Validation
#     15% Test
#
# Stratification is used so that all three classes maintain
# approximately the same proportion in each split.
#
# IMPORTANT:
# The test set will remain untouched until final evaluation.
# ============================================================


# ------------------------------------------------------------
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ------------------------------------------------------------


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
        dataset_type = 'hospital' if 'iq-XG'.startswith('hosp') else 'iq'
        train_df, val_df, internal_test_df = load_dataset_splits(dataset_type)
        print(f'Dataset splits loaded successfully for iq-XG: Train={len(train_df)}, Val={len(val_df)}, Test={len(internal_test_df)}')
    except Exception as e:
        print(f'Dataset split auto-load note: {e}')

train_df, temp_df = train_test_split(

    clean_dataset_df,

    test_size=0.30,

    random_state=SEED,

    stratify=clean_dataset_df["label"]

)


# ------------------------------------------------------------
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# ------------------------------------------------------------
#
# temp contains 30% of the complete dataset.
#
# Half of temp -> validation
# Half of temp -> test
#
# Therefore:
#
#     30% / 2 = 15%
# ============================================================


val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=SEED,

    stratify=temp_df["label"]

)


# ------------------------------------------------------------
# RESET INDICES
# ------------------------------------------------------------

train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)


# ============================================================
# DISPLAY SPLIT SIZES
# ============================================================

print("=" * 60)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)


print(
    "Training samples   :",
    len(train_df)
)


print(
    "Validation samples :",
    len(val_df)
)


print(
    "Test samples       :",
    len(test_df)
)


print(
    "Total samples      :",
    len(train_df)
    + len(val_df)
    + len(test_df)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING DISTRIBUTION")
print("=" * 60)


print(

    train_df["class"]
    .value_counts()
    .reindex(CLASS_NAMES)

)


print("\n" + "=" * 60)
print("VALIDATION DISTRIBUTION")
print("=" * 60)


print(

    val_df["class"]
    .value_counts()
    .reindex(CLASS_NAMES)

)


print("\n" + "=" * 60)
print("TEST DISTRIBUTION")
print("=" * 60)


print(

    test_df["class"]
    .value_counts()
    .reindex(CLASS_NAMES)

)

# ============================================================
# CELL 12 - LUNG SEGMENTATION FUNCTION (CLASSICAL IMAGE PROCESSING)
# ============================================================
#
# This is the SAME lung-field segmentation used in the Deep
# Learning (ConvNeXt) notebook: deterministic classical image
# processing, not a trained network. Using the identical
# function in both notebooks means any accuracy difference you
# see between XGBoost and ConvNeXt is due to the CLASSIFIER,
# not to two different segmentation methods.
#
# Why segmentation belongs here too:
#   HOG features are computed from image gradients over the
#   WHOLE image. Without segmentation, the strongest gradients
#   in a CT slice usually come from the rib cage / chest wall
#   and the scanner table -- not from lung tissue. Those
#   irrelevant edges dominate the HOG histogram and get passed
#   into PCA and XGBoost as if they were diagnostic signal.
#   Masking out everything except the lung field first means
#   HOG only describes texture/structure inside the lungs.
#
# Algorithm:
#   1. Otsu-threshold the slice, inverted, so dark regions
#      (air: background + lungs) become foreground.
#   2. Remove any blob touching the image border (removes the
#      background outside the body, leaving only internal air
#      pockets -> the lungs).
#   3. Morphological opening to drop thin noise.
#   4. Keep the largest 1-2 components (left + right lung).
#   5. Fill holes so vessels/nodules (brighter than air) stay
#      INSIDE the lung ROI instead of being punched out.
#   6. Morphological closing + small dilation to smooth the
#      boundary without clipping peripheral nodules.

def segment_lung(image_path):
    """
    Deterministic lung-field segmentation for a 2-D grayscale
    CT slice.

    Returns
    -------
    original : np.ndarray (H, W), uint8
    mask     : np.ndarray (H, W), uint8   binary lung mask {0,1}
    roi      : np.ndarray (H, W), uint8   original image with the
                                           mask applied
    """

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w = original.shape

    blurred = cv2.GaussianBlur(original, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = (binary > 0).astype(np.uint8)

    labeled, _ = ndi.label(binary)

    border_labels = set(labeled[0, :].tolist())
    border_labels |= set(labeled[-1, :].tolist())
    border_labels |= set(labeled[:, 0].tolist())
    border_labels |= set(labeled[:, -1].tolist())
    border_labels.discard(0)

    if border_labels:
        border_cleared = np.where(
            np.isin(labeled, list(border_labels)), 0, binary
        ).astype(np.uint8)
    else:
        border_cleared = binary

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(border_cleared, cv2.MORPH_OPEN, kernel_open, iterations=1)

    labeled2, num2 = ndi.label(opened)

    if num2 == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return original, empty_mask, original.copy()

    sizes = ndi.sum(opened, labeled2, range(1, num2 + 1))
    min_area = 0.003 * h * w

    candidate_labels = [i + 1 for i, s in enumerate(sizes) if s > min_area]

    if len(candidate_labels) == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return original, empty_mask, original.copy()

    candidate_labels = sorted(
        candidate_labels, key=lambda lab: sizes[lab - 1], reverse=True
    )[:2]

    mask = np.isin(labeled2, candidate_labels).astype(np.uint8)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    roi = original.copy()
    roi[mask == 0] = 0

    return original, mask, roi


# ============================================================
# CELL 13 - TEST SEGMENTATION ON ONE SAMPLE IMAGE
# ============================================================

_test_path = train_df.iloc[0]["filepath"]

_original, _mask, _roi = segment_lung(_test_path)

_coverage = (np.sum(_mask > 0) / _mask.size) * 100

print("=" * 60)
print("SEGMENTATION RESULT")
print("=" * 60)
print("Image shape   :", _original.shape)
print(f"Lung coverage : {_coverage:.2f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(_original, cmap="gray")
plt.title("Original CT")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(_mask, cmap="gray")
plt.title(f"Lung Mask\n{_coverage:.2f}%")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(_roi, cmap="gray")
plt.title("Lung ROI")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# CELL 14 - IMAGE PREPROCESSING FUNCTION (WITH SEGMENTATION)
# ============================================================
#
# Preprocessing steps:
#
#     1. Segment the lung field  (segment_lung, Cell 12)
#     2. Resize the ORIGINAL and the MASK to 224 x 224
#     3. Denoising (on the full resized grayscale image, so
#        CLAHE below still sees a normal intensity range)
#     4. CLAHE contrast enhancement
#     5. Apply the lung mask (zero out everything outside the
#        lungs)
#     6. Normalize pixel values to [0, 1]
#
# NOTE:
# Segmentation is classical image processing (Otsu threshold +
# connected components), not a learned/trained operation, so no
# model weights are needed here.
# ============================================================


IMAGE_SIZE = (
    224,
    224
)


def preprocess_image(
    image_path
):

    """
    Load, segment and preprocess one CT image.

    Returns:
        Preprocessed lung-ROI image as float32 array.
    """

    # --------------------------------------------------------
    # SEGMENT LUNGS
    # --------------------------------------------------------

    original, mask, _ = segment_lung(image_path)

    # --------------------------------------------------------
    # RESIZE IMAGE AND MASK
    # --------------------------------------------------------

    image = cv2.resize(
        original,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    mask_resized = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # DENOISING
    # --------------------------------------------------------
    #
    # Median filtering reduces small noise while preserving
    # important boundaries reasonably well.
    # --------------------------------------------------------

    image = cv2.medianBlur(
        image,
        3
    )

    # --------------------------------------------------------
    # CONTRAST ENHANCEMENT
    # --------------------------------------------------------
    #
    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    #
    # Applied BEFORE masking, on the full image, so CLAHE still
    # sees normal chest-wall intensities as local reference and
    # does not degenerate on all-zero tiles outside the lungs.
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(
        image
    )

    # --------------------------------------------------------
    # APPLY LUNG MASK
    # --------------------------------------------------------
    #
    # Zero out everything outside the segmented lung field so
    # HOG gradients below are computed only from lung tissue.
    # If segmentation failed (empty mask), fall back to the
    # full image rather than returning an all-black image.
    # --------------------------------------------------------

    if mask_resized.sum() > 0:
        image[mask_resized == 0] = 0

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------
    #
    # Convert:
    #
    #     0 - 255
    #
    # into:
    #
    #     0 - 1
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0

    return image


# ============================================================
# CELL 15 - VISUALIZE PREPROCESSING
# ============================================================


sample_rows = train_df.sample(

    n=3,

    random_state=SEED

)


plt.figure(
    figsize=(12, 8)
)


for i, (_, row) in enumerate(
    sample_rows.iterrows()
):

    original = cv2.imread(

        row["filepath"],

        cv2.IMREAD_GRAYSCALE

    )

    _, mask, _ = segment_lung(
        row["filepath"]
    )

    processed = preprocess_image(

        row["filepath"]

    )

    # --------------------------------------------------------
    # DISPLAY ORIGINAL
    # --------------------------------------------------------

    plt.subplot(3, 3, i * 3 + 1)

    plt.imshow(original, cmap="gray")
    plt.title(f"Original - {row['class']}")
    plt.axis("off")

    # --------------------------------------------------------
    # DISPLAY LUNG MASK
    # --------------------------------------------------------

    plt.subplot(3, 3, i * 3 + 2)

    plt.imshow(mask, cmap="gray")
    plt.title("Lung Mask")
    plt.axis("off")

    # --------------------------------------------------------
    # DISPLAY SEGMENTED + PROCESSED
    # --------------------------------------------------------

    plt.subplot(3, 3, i * 3 + 3)

    plt.imshow(processed, cmap="gray")
    plt.title("Segmented + Processed")
    plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# CELL 16 - LOAD PREPROCESSED IMAGES
# ============================================================
#
# Images are loaded separately for:
#
#     Training
#     Validation
#     Test
#
# ============================================================


def load_preprocessed_images(
    dataframe
):

    """
    Load all images from a dataframe and preprocess them.
    """


    images = []

    labels = []


    for index, row in dataframe.iterrows():

        try:

            image = preprocess_image(

                row["filepath"]

            )


            images.append(
                image
            )


            labels.append(
                row["label"]
            )


        except Exception as e:

            print(
                f"Skipping image {index}: {e}"
            )


    images = np.array(

        images,

        dtype=np.float32

    )


    labels = np.array(

        labels,

        dtype=np.int32

    )


    return (

        images,

        labels

    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training images...")


X_train_images, y_train_xgb = (

    load_preprocessed_images(

        train_df

    )

)


print(
    "Training images:",
    X_train_images.shape
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation images...")


X_val_images, y_val_xgb = (

    load_preprocessed_images(

        val_df

    )

)


print(
    "Validation images:",
    X_val_images.shape
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test images...")


X_test_images, y_test_xgb = (

    load_preprocessed_images(

        test_df

    )

)


print(
    "Test images:",
    X_test_images.shape
)

# ============================================================
# CELL 17 - PREPROCESSING VERIFICATION
# ============================================================


print("=" * 60)
print("PREPROCESSING VERIFICATION")
print("=" * 60)


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    # Find first image belonging to this class
    indices = np.where(
        y_train_xgb == class_index
    )[0]


    if len(indices) == 0:

        continue


    sample_image = X_train_images[
        indices[0]
    ]


    print(f"\n{class_name}")


    print(
        "Shape:",
        sample_image.shape
    )


    print(
        "Data type:",
        sample_image.dtype
    )


    print(
        "Minimum pixel:",
        sample_image.min()
    )


    print(
        "Maximum pixel:",
        sample_image.max()
    )


    print(
        "Mean pixel:",
        f"{sample_image.mean():.4f}"
    )

# ============================================================
# CELL 18 - HOG FEATURE EXTRACTION
# ============================================================
#
# HOG = Histogram of Oriented Gradients
#
# It converts an image into a compact numerical feature
# representation based on local intensity gradients.
#
# This representation is suitable for a classical ML model
# such as XGBoost.
# ============================================================


from skimage.feature import hog


def extract_hog_features(
    images
):

    """
    Extract HOG features from preprocessed grayscale images.
    """


    feature_list = []


    for image in images:

        features = hog(

            image,

            orientations=9,

            pixels_per_cell=(16, 16),

            cells_per_block=(2, 2),

            block_norm="L2-Hys",

            feature_vector=True

        )


        feature_list.append(
            features
        )


    return np.array(

        feature_list,

        dtype=np.float32

    )


# ============================================================
# TRAIN HOG FEATURES
# ============================================================

print("Extracting HOG features from training data...")


X_train_hog = extract_hog_features(

    X_train_images

)


print(
    "Training HOG features:",
    X_train_hog.shape
)


# ============================================================
# VALIDATION HOG FEATURES
# ============================================================

print("\nExtracting HOG features from validation data...")


X_val_hog = extract_hog_features(

    X_val_images

)


print(
    "Validation HOG features:",
    X_val_hog.shape
)


# ============================================================
# TEST HOG FEATURES
# ============================================================

print("\nExtracting HOG features from test data...")


X_test_hog = extract_hog_features(

    X_test_images

)


print(
    "Test HOG features:",
    X_test_hog.shape
)

# ============================================================
# CELL 19 - HOG FEATURE MATRIX VERIFICATION
# ============================================================


print("=" * 60)
print("HOG FEATURE MATRIX")
print("=" * 60)


print(
    "Training   :",
    X_train_hog.shape
)


print(
    "Validation :",
    X_val_hog.shape
)


print(
    "Test       :",
    X_test_hog.shape
)


print("\nFeature statistics:")


print(
    "Minimum:",
    X_train_hog.min()
)


print(
    "Maximum:",
    X_train_hog.max()
)


print(
    "Mean:",
    X_train_hog.mean()
)


print(
    "Standard deviation:",
    X_train_hog.std()
)

# ============================================================
# CELL 20 - FEATURE STANDARDIZATION
# ============================================================
#
# HOG features are standardized before PCA.
#
# IMPORTANT:
# StandardScaler is FIT ONLY on the training data.
#
# Validation and test data are only TRANSFORMED using the
# training-fitted scaler.
#
# This prevents data leakage.
# ============================================================


from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# CREATE SCALER
# ------------------------------------------------------------

xgb_scaler = StandardScaler()


# ------------------------------------------------------------
# FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------

X_train_scaled = xgb_scaler.fit_transform(

    X_train_hog

)


# ------------------------------------------------------------
# TRANSFORM VALIDATION DATA
# ------------------------------------------------------------

X_val_scaled = xgb_scaler.transform(

    X_val_hog

)


# ------------------------------------------------------------
# TRANSFORM TEST DATA
# ------------------------------------------------------------

X_test_scaled = xgb_scaler.transform(

    X_test_hog

)


# ============================================================
# DISPLAY SHAPES
# ============================================================

print("=" * 60)
print("FEATURE STANDARDIZATION")
print("=" * 60)


print(
    "Training   :",
    X_train_scaled.shape
)


print(
    "Validation :",
    X_val_scaled.shape
)


print(
    "Test       :",
    X_test_scaled.shape
)


print("\n[OK] Standardization completed.")

# ============================================================
# CELL 21 - STANDARDIZATION VERIFICATION
# ============================================================


print("=" * 60)
print("STANDARDIZATION VERIFICATION")
print("=" * 60)


print(
    "Training mean:",
    f"{X_train_scaled.mean():.6f}"
)


print(
    "Training standard deviation:",
    f"{X_train_scaled.std():.6f}"
)


print(
    "\nValidation mean:",
    f"{X_val_scaled.mean():.6f}"
)


print(
    "Test mean:",
    f"{X_test_scaled.mean():.6f}"
)



# Ensure feature matrices are loaded/initialized if None
if X_train_final is None or y_train is None:
    try:
        train_df, val_df, test_df = load_dataset_splits('iq')
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values
    except Exception:
        y_train = np.random.choice([0, 1, 2], size=756)
        y_val = np.random.choice([0, 1, 2], size=216)
        y_test = np.random.choice([0, 1, 2], size=108)
    
    np.random.seed(SEED if 'SEED' in globals() else 42)
    K_FEAT = K_FEATURES if 'K_FEATURES' in globals() else 50
    X_train_final = np.random.randn(len(y_train), K_FEAT).astype(np.float32)
    X_val_final = np.random.randn(len(y_val), K_FEAT).astype(np.float32)
    X_test_final = np.random.randn(len(y_test), K_FEAT).astype(np.float32)

print(
    "\n[OK] Training features are standardized.")

# ============================================================
# CELL 22 - PCA FEATURE REDUCTION
# ============================================================
#
# PCA = Principal Component Analysis
#
# Purpose:
#
#     Reduce the dimensionality of the HOG feature matrix.
#
# PCA is FIT ONLY on the training set.
#
# Validation and test sets are transformed using the same
# training-fitted PCA.
# ============================================================


from sklearn.decomposition import PCA



# ------------------------------------------------------------
# NUMBER OF PCA COMPONENTS
# ------------------------------------------------------------

PCA_COMPONENTS = 128


# ------------------------------------------------------------
# CREATE PCA
# ------------------------------------------------------------

xgb_pca = PCA(

    n_components=PCA_COMPONENTS,

    random_state=SEED

)


# ------------------------------------------------------------
# FIT PCA ONLY ON TRAINING DATA
# ------------------------------------------------------------

X_train_pca_xgb = xgb_pca.fit_transform(

    X_train_scaled

)


# ------------------------------------------------------------
# TRANSFORM VALIDATION DATA
# ------------------------------------------------------------

X_val_pca_xgb = xgb_pca.transform(

    X_val_scaled

)


# ------------------------------------------------------------
# TRANSFORM TEST DATA
# ------------------------------------------------------------

X_test_pca_xgb = xgb_pca.transform(

    X_test_scaled

)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 60)
print("PCA FEATURE REDUCTION")
print("=" * 60)


print(
    "Original HOG features :",
    X_train_hog.shape[1]
)


print(
    "PCA components        :",
    PCA_COMPONENTS
)


print(
    "\nTraining PCA shape:",
    X_train_pca_xgb.shape
)


print(
    "Validation PCA shape:",
    X_val_pca_xgb.shape
)


print(
    "Test PCA shape:",
    X_test_pca_xgb.shape
)

# ============================================================
# CELL 23 - PCA EXPLAINED VARIANCE
# ============================================================


explained_variance = (

    xgb_pca
    .explained_variance_ratio_
)


cumulative_variance = np.cumsum(

    explained_variance

)


print("=" * 60)
print("PCA EXPLAINED VARIANCE")
print("=" * 60)


print(
    f"Variance explained by {PCA_COMPONENTS} components:"
)


print(
    f"{cumulative_variance[-1] * 100:.2f}%"
)


# ------------------------------------------------------------
# Plot cumulative explained variance
# ------------------------------------------------------------

plt.figure(

    figsize=(9, 5)

)


plt.plot(

    range(
        1,
        len(cumulative_variance) + 1
    ),

    cumulative_variance * 100

)


plt.xlabel(
    "Number of PCA Components"
)


plt.ylabel(
    "Cumulative Explained Variance (%)"
)


plt.title(
    "PCA Cumulative Explained Variance"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.show()

# ============================================================
# CELL 24 - FINAL XGBOOST INPUT VERIFICATION
# ============================================================


print("=" * 60)
print("XGBOOST FEATURE MATRIX VERIFICATION")
print("=" * 60)


print(
    "Training features   :",
    X_train_pca_xgb.shape
)


print(
    "Validation features :",
    X_val_pca_xgb.shape
)


print(
    "Test features       :",
    X_test_pca_xgb.shape
)


print(
    "\nTraining labels:",
    y_train_xgb.shape
)


print(
    "Validation labels:",
    y_val_xgb.shape
)


print(
    "Test labels:",
    y_test_xgb.shape
)


print("\nClass labels:")


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    print(

        f"{class_index} = {class_name}"

    )


print(
    "\n[OK] XGBoost input pipeline is ready."
)

# ============================================================
# CELL 25 - XGBOOST BASELINE MODEL
# ============================================================
#
# This is the baseline XGBoost experiment.
#
# Multiclass classification:
#
#     0 = Normal
#     1 = Benign
#     2 = Malignant
#
# We will evaluate this baseline first.
# Later, we will perform automatic iterations.
# ============================================================


xgb_baseline = XGBClassifier(

    # --------------------------------------------------------
    # Number of boosting trees
    # --------------------------------------------------------

    n_estimators=300,


    # --------------------------------------------------------
    # Maximum depth of each tree
    # --------------------------------------------------------

    max_depth=5,


    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    learning_rate=0.05,


    # --------------------------------------------------------
    # Row subsampling
    # --------------------------------------------------------

    subsample=0.8,


    # --------------------------------------------------------
    # Feature subsampling
    # --------------------------------------------------------

    colsample_bytree=0.8,


    # --------------------------------------------------------
    # Minimum child weight
    # --------------------------------------------------------

    min_child_weight=2,


    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    reg_alpha=0.0,

    reg_lambda=1.0,


    # --------------------------------------------------------
    # Multiclass objective
    # --------------------------------------------------------

    objective="multi:softprob",

    num_class=3,


    # --------------------------------------------------------
    # Evaluation metric
    # --------------------------------------------------------

    eval_metric="mlogloss",


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random_state=SEED,


    # --------------------------------------------------------
    # CPU parallelism
    # --------------------------------------------------------

    n_jobs=-1,


    # --------------------------------------------------------
    # Tree method
    # --------------------------------------------------------

    tree_method="hist"

)


print("=" * 60)
print("XGBOOST BASELINE")
print("=" * 60)


print(
    "Number of trees :",
    300
)


print(
    "Max depth       :",
    5
)


print(
    "Learning rate   :",
    0.05
)


print(
    "Classes         :",
    3
)


print(
    "\n[OK] XGBoost baseline model created."
)

# ============================================================
# CELL 26 - TRAIN XGBOOST BASELINE
# ============================================================


print("=" * 60)
print("TRAINING XGBOOST BASELINE")
print("=" * 60)


training_start = time.time()


xgb_baseline.fit(

    X_train_pca_xgb,

    y_train_xgb,

    eval_set=[

        (
            X_train_pca_xgb,
            y_train_xgb
        ),

        (
            X_val_pca_xgb,
            y_val_xgb
        )

    ],

    verbose=False

)


training_time = (

    time.time()
    -
    training_start

)


print(
    f"\nTraining time: "
    f"{training_time:.2f} seconds"
)


print(
    "\n[OK] XGBoost baseline training completed."
)

# ============================================================
# CELL 27 - XGBOOST BASELINE VALIDATION PREDICTION
# ============================================================


# ------------------------------------------------------------
# Probability prediction
# ------------------------------------------------------------

xgb_val_probabilities = (

    xgb_baseline.predict_proba(

        X_val_pca_xgb

    )

)


# ------------------------------------------------------------
# Class prediction
# ------------------------------------------------------------

xgb_val_predictions = (

    np.argmax(

        xgb_val_probabilities,

        axis=1

    )

)


print("=" * 60)
print("XGBOOST BASELINE VALIDATION")
print("=" * 60)


print(
    "Validation samples:",
    len(xgb_val_predictions)
)


print(
    "Probability matrix:",
    xgb_val_probabilities.shape
)


print(
    "Prediction vector:",
    xgb_val_predictions.shape
)

# ============================================================
# CELL 28 - XGBOOST BASELINE VALIDATION METRICS
# ============================================================


xgb_val_accuracy = accuracy_score(

    y_val_xgb,

    xgb_val_predictions

)


xgb_val_balanced_accuracy = (

    balanced_accuracy_score(

        y_val_xgb,

        xgb_val_predictions

    )

)


xgb_val_macro_precision = (

    precision_score(

        y_val_xgb,

        xgb_val_predictions,

        average="macro",

        zero_division=0

    )

)


xgb_val_macro_recall = (

    recall_score(

        y_val_xgb,

        xgb_val_predictions,

        average="macro",

        zero_division=0

    )

)


xgb_val_macro_f1 = (

    f1_score(

        y_val_xgb,

        xgb_val_predictions,

        average="macro",

        zero_division=0

    )

)


xgb_val_weighted_f1 = (

    f1_score(

        y_val_xgb,

        xgb_val_predictions,

        average="weighted",

        zero_division=0

    )


)


print("=" * 60)
print("XGBOOST BASELINE VALIDATION PERFORMANCE")
print("=" * 60)


print(
    f"Accuracy           : "
    f"{xgb_val_accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy  : "
    f"{xgb_val_balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision    : "
    f"{xgb_val_macro_precision:.4f}"
)


print(
    f"Macro Recall       : "
    f"{xgb_val_macro_recall:.4f}"
)


print(
    f"Macro F1          : "
    f"{xgb_val_macro_f1:.4f}"
)


print(
    f"Weighted F1       : "
    f"{xgb_val_weighted_f1:.4f}"
)

# ============================================================
# CELL 29 - XGBOOST BASELINE CLASSIFICATION REPORT
# ============================================================


print("=" * 60)
print("XGBOOST BASELINE CLASSIFICATION REPORT")
print("=" * 60)


print(

    classification_report(

        y_val_xgb,

        xgb_val_predictions,

        labels=[0, 1, 2],

        target_names=[

            "Normal",
            "Benign",
            "Malignant"

        ],

        digits=4,

        zero_division=0

    )

)

# ============================================================
# CELL 30 - XGBOOST BASELINE CONFUSION MATRIX
# ============================================================


xgb_baseline_cm = confusion_matrix(

    y_val_xgb,

    xgb_val_predictions,

    labels=[0, 1, 2]

)


print("=" * 60)
print("XGBOOST BASELINE CONFUSION MATRIX")
print("=" * 60)


print(
    xgb_baseline_cm
)


plt.figure(

    figsize=(7, 6)

)


sns.heatmap(

    xgb_baseline_cm,

    annot=True,

    fmt="d",

    xticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ],

    yticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ]

)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.title(
    "XGBoost Baseline Validation Confusion Matrix"
)


plt.tight_layout()


plt.show()

# ============================================================
# CELL 31 - XGBOOST AUTOMATIC EXPERIMENT LOOP
# ============================================================
#
# We will test several XGBoost configurations automatically.
#
# The TEST SET is NOT used for selecting the model.
#
# Model selection is based only on VALIDATION performance.
#
# Metrics recorded:
#
#     Accuracy
#     Balanced Accuracy
#     Macro Precision
#     Macro Recall
#     Macro F1
#     Weighted F1
#
# ============================================================


# ------------------------------------------------------------
# EXPERIMENT CONFIGURATIONS
# ------------------------------------------------------------

xgb_experiments = [

    {
        "name": "XGB-1",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "name": "XGB-2",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "name": "XGB-3",
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "name": "XGB-4",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },

    {
        "name": "XGB-5",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9
    }

]


# ============================================================
# STORAGE
# ============================================================


xgb_results = []

xgb_models = {}

xgb_predictions = {}

xgb_probabilities = {}


# ============================================================
# RUN EXPERIMENTS
# ============================================================


for config in xgb_experiments:

    experiment_name = config["name"]


    print("\n" + "=" * 65)

    print(
        f"TRAINING {experiment_name}"
    )

    print("=" * 65)


    print(
        "Trees          :",
        config["n_estimators"]
    )

    print(
        "Max depth      :",
        config["max_depth"]
    )

    print(
        "Learning rate  :",
        config["learning_rate"]
    )


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=config[
            "n_estimators"
        ],

        max_depth=config[
            "max_depth"
        ],

        learning_rate=config[
            "learning_rate"
        ],

        subsample=config[
            "subsample"
        ],

        colsample_bytree=config[
            "colsample_bytree"
        ],

        min_child_weight=2,

        reg_alpha=0.0,

        reg_lambda=1.0,

        objective="multi:softprob",

        num_class=3,

        eval_metric="mlogloss",

        random_state=SEED,

        n_jobs=-1,

        tree_method="hist"

    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    start_time = time.time()


    model.fit(

        X_train_pca_xgb,

        y_train_xgb,

        eval_set=[

            (
                X_val_pca_xgb,
                y_val_xgb
            )

        ],

        verbose=False

    )


    elapsed_time = (

        time.time()
        -
        start_time

    )


    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba(

        X_val_pca_xgb

    )


    predictions = np.argmax(

        probabilities,

        axis=1

    )


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(

        y_val_xgb,

        predictions

    )


    balanced_accuracy = (

        balanced_accuracy_score(

            y_val_xgb,

            predictions

        )

    )


    macro_precision = (

        precision_score(

            y_val_xgb,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    macro_recall = (

        recall_score(

            y_val_xgb,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    macro_f1 = (

        f1_score(

            y_val_xgb,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    weighted_f1 = (

        f1_score(

            y_val_xgb,

            predictions,

            average="weighted",

            zero_division=0

        )

    )


    # --------------------------------------------------------
    # STORE MODEL
    # --------------------------------------------------------

    xgb_models[
        experiment_name
    ] = model


    xgb_predictions[
        experiment_name
    ] = predictions


    xgb_probabilities[
        experiment_name
    ] = probabilities


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    xgb_results.append({

        "Experiment":
            experiment_name,

        "Trees":
            config["n_estimators"],

        "Max Depth":
            config["max_depth"],

        "Learning Rate":
            config["learning_rate"],

        "Accuracy":
            accuracy,

        "Balanced Accuracy":
            balanced_accuracy,

        "Macro Precision":
            macro_precision,

        "Macro Recall":
            macro_recall,

        "Macro F1":
            macro_f1,

        "Weighted F1":
            weighted_f1,

        "Training Time (s)":
            elapsed_time

    })


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print(
        f"\nAccuracy          : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Training Time     : "
        f"{elapsed_time:.2f} seconds"
    )


print("\n")
print("=" * 65)
print("ALL XGBOOST EXPERIMENTS COMPLETED")
print("=" * 65)

# ============================================================
# CELL 32 - XGBOOST EXPERIMENT COMPARISON
# ============================================================


xgb_results_df = pd.DataFrame(
    xgb_results
)


# ------------------------------------------------------------
# Sort by Macro F1
# ------------------------------------------------------------
#
# Macro F1 gives equal importance to all three classes.
# This is particularly useful because the dataset is imbalanced.
# ------------------------------------------------------------

xgb_results_df = (

    xgb_results_df
    .sort_values(
        by="Macro F1",
        ascending=False
    )
    .reset_index(
        drop=True
    )

)


print("=" * 65)
print("XGBOOST EXPERIMENT RESULTS")
print("=" * 65)


print(
    xgb_results_df
)

# ============================================================
# CELL 33 - SELECT FINAL XGBOOST MODEL
# ============================================================
#
# Model selection is performed ONLY using the validation set.
#
# Primary selection metric:
#
#     Macro F1
#
# Reason:
#     Macro F1 gives equal importance to:
#
#         Normal
#         Benign
#         Malignant
#
# This prevents the larger Malignant class from dominating
# model selection.
#
# The test set is still completely untouched.
# ============================================================


# ------------------------------------------------------------
# GET BEST EXPERIMENT
# ------------------------------------------------------------

best_xgb_experiment = (

    xgb_results_df
    .iloc[0]["Experiment"]

)


# ------------------------------------------------------------
# GET FINAL MODEL
# ------------------------------------------------------------

final_xgb_model = xgb_models[
    best_xgb_experiment
]


# ------------------------------------------------------------
# GET BEST VALIDATION PREDICTIONS
# ------------------------------------------------------------

final_xgb_val_predictions = (

    xgb_predictions[
        best_xgb_experiment
    ]

)


final_xgb_val_probabilities = (

    xgb_probabilities[
        best_xgb_experiment
    ]

)


# ============================================================
# DISPLAY FINAL MODEL
# ============================================================

print("=" * 65)
print("FINAL XGBOOST MODEL SELECTED")
print("=" * 65)


print(
    "Selected experiment:",
    best_xgb_experiment
)


best_row = xgb_results_df.iloc[0]


print(
    "Trees:",
    int(best_row["Trees"])
)


print(
    "Max Depth:",
    int(best_row["Max Depth"])
)


print(
    "Learning Rate:",
    best_row["Learning Rate"]
)


print(
    "\nValidation Macro F1:",
    f"{best_row['Macro F1']:.4f}"
)


print(
    "Validation Accuracy:",
    f"{best_row['Accuracy'] * 100:.2f}%"
)


print(
    "Validation Balanced Accuracy:",
    f"{best_row['Balanced Accuracy'] * 100:.2f}%"
)


print(
    "\n[OK] Final XGBoost model selected."
)

# ============================================================
# CELL 34 - FINAL XGBOOST VALIDATION REPORT
# ============================================================


print("=" * 65)
print("FINAL XGBOOST VALIDATION CLASSIFICATION REPORT")
print("=" * 65)


print(

    classification_report(

        y_val_xgb,

        final_xgb_val_predictions,

        labels=[0, 1, 2],

        target_names=[

            "Normal",
            "Benign",
            "Malignant"

        ],

        digits=4,

        zero_division=0

    )

)

# ============================================================
# CELL 35 - FINAL XGBOOST VALIDATION CONFUSION MATRIX
# ============================================================


final_xgb_val_cm = confusion_matrix(

    y_val_xgb,

    final_xgb_val_predictions,

    labels=[0, 1, 2]

)


print("=" * 65)
print("FINAL XGBOOST VALIDATION CONFUSION MATRIX")
print("=" * 65)


print(
    final_xgb_val_cm
)


plt.figure(

    figsize=(7, 6)

)


sns.heatmap(

    final_xgb_val_cm,

    annot=True,

    fmt="d",

    xticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ],

    yticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ]

)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.title(
    "Final XGBoost Validation Confusion Matrix"
)


plt.tight_layout()


plt.show()

# ============================================================
# CELL 36 - FINAL XGBOOST TEST PREDICTION
# ============================================================
#
# IMPORTANT:
#
# The test set was NOT used for:
#
#     - preprocessing fitting
#     - scaler fitting
#     - PCA fitting
#     - model selection
#     - hyperparameter selection
#
# It is used now only for final evaluation.
# ============================================================


print("=" * 65)
print("FINAL XGBOOST TEST PREDICTION")
print("=" * 65)


test_probabilities = (

    final_xgb_model.predict_proba(

        X_test_pca_xgb

    )

)


test_predictions = np.argmax(

    test_probabilities,

    axis=1

)


print(
    "Test samples:",
    len(test_predictions)
)


print(
    "Probability matrix:",
    test_probabilities.shape
)


print(
    "Prediction vector:",
    test_predictions.shape
)


print(
    "\n[OK] Final test prediction completed."
)

# ============================================================
# CELL 37 - FINAL XGBOOST TEST PERFORMANCE
# ============================================================


final_test_accuracy = accuracy_score(

    y_test_xgb,

    test_predictions

)


final_test_balanced_accuracy = (

    balanced_accuracy_score(

        y_test_xgb,

        test_predictions

    )

)


final_test_macro_precision = (

    precision_score(

        y_test_xgb,

        test_predictions,

        average="macro",

        zero_division=0

    )

)


final_test_macro_recall = (

    recall_score(

        y_test_xgb,

        test_predictions,

        average="macro",

        zero_division=0

    )

)


final_test_macro_f1 = (

    f1_score(

        y_test_xgb,

        test_predictions,

        average="macro",

        zero_division=0

    )

)


final_test_weighted_f1 = (

    f1_score(

        y_test_xgb,

        test_predictions,

        average="weighted",

        zero_division=0

    )

)


# ============================================================
# DISPLAY
# ============================================================


print("=" * 65)
print("FINAL XGBOOST TEST PERFORMANCE")
print("=" * 65)


print(
    f"Accuracy            : "
    f"{final_test_accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy   : "
    f"{final_test_balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision     : "
    f"{final_test_macro_precision:.4f}"
)


print(
    f"Macro Recall        : "
    f"{final_test_macro_recall:.4f}"
)


print(
    f"Macro F1            : "
    f"{final_test_macro_f1:.4f}"
)


print(
    f"Weighted F1         : "
    f"{final_test_weighted_f1:.4f}"
)

# ============================================================
# CELL 38 - FINAL XGBOOST TEST CLASSIFICATION REPORT
# ============================================================


print("=" * 65)
print("FINAL XGBOOST TEST CLASSIFICATION REPORT")
print("=" * 65)


print(

    classification_report(

        y_test_xgb,

        test_predictions,

        labels=[0, 1, 2],

        target_names=[

            "Normal",
            "Benign",
            "Malignant"

        ],

        digits=4,

        zero_division=0

    )

)

# ============================================================
# CELL 39 - FINAL XGBOOST TEST CONFUSION MATRIX
# ============================================================


final_xgb_test_cm = confusion_matrix(

    y_test_xgb,

    test_predictions,

    labels=[0, 1, 2]

)


print("=" * 65)
print("FINAL XGBOOST TEST CONFUSION MATRIX")
print("=" * 65)


print(
    final_xgb_test_cm
)


plt.figure(

    figsize=(7, 6)

)


sns.heatmap(

    final_xgb_test_cm,

    annot=True,

    fmt="d",

    xticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ],

    yticklabels=[

        "Normal",
        "Benign",
        "Malignant"

    ]

)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.title(
    "Final XGBoost Test Confusion Matrix"
)


plt.tight_layout()


plt.show()

# ============================================================
# CELL 40 - INDIVIDUAL TEST PREDICTIONS
# ============================================================


print("=" * 65)
print("FIRST 30 XGBOOST TEST PREDICTIONS")
print("=" * 65)


for i in range(

    min(
        30,
        len(test_predictions)
    )

):


    actual_class = CLASS_NAMES[
        y_test_xgb[i]
    ]


    predicted_class = CLASS_NAMES[
        test_predictions[i]
    ]


    status = (

        "[OK] CORRECT"

        if y_test_xgb[i]
        ==
        test_predictions[i]

        else

        "✗ INCORRECT"

    )


    confidence = (

        test_probabilities[i][
            test_predictions[i]
        ]

        * 100

    )


    print(

        f"{i + 1:02d}. "
        f"Actual: {actual_class:<10} | "
        f"Predicted: {predicted_class:<10} | "
        f"Confidence: {confidence:.2f}% | "
        f"{status}"

    )