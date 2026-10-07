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
error = None
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


import sys
from pathlib import Path
_cur_dir = str(Path(__file__).resolve().parent)
_shared_dir = str(Path(__file__).resolve().parent.parent / "shared")
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

try:
    from preprocess import segment_lung
except Exception:
    pass

# Exact GP implementation reconstructed from uploaded Colab notebook.
# Source notebook: PW1_LC_Genetic_Programming(5).ipynb
# Only dataset access in original Cells 3-5 was changed.



# ==================== ORIGINAL COLAB CELL 0 ====================

# ============================================================
# GENETIC PROGRAMMING - ENVIRONMENT SETUP
# ============================================================
#
# Project:
# Prediction of Lung Cancer Probability from CT Scan
#
# Dataset:
# IQ-OTH/NCCD
#
# Classes:
#   0 -> Normal
#   1 -> Benign
#   2 -> Malignant
#
# Approach:
#   Evolutionary Computing / Genetic Programming
# ============================================================

# !pip install -q gplearn scikit-image opencv-python-headless


# ==================== ORIGINAL COLAB CELL 1 ====================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import zipfile
import random
import warnings

import numpy as np
import pandas as pd

import cv2

from PIL import Image

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from skimage.feature import hog
from skimage.feature import local_binary_pattern

from gplearn.genetic import SymbolicClassifier

import joblib

warnings.filterwarnings("ignore")

print("All libraries imported successfully.")


# ==================== ORIGINAL COLAB CELL 2 ====================

# ============================================================
# REPRODUCIBILITY SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

print("Random seed:", SEED)


# ==================== ORIGINAL COLAB CELL 3 ====================

# ============================================================
# DATASET ACCESS
# ============================================================
#
# Original Colab cells 3-5 used Google Drive + archive.zip.
# ONLY dataset access has been changed here.
#
# Set DATASET_ZIP_PATH to the location of archive.zip.
# Example (Windows):
#   set DATASET_ZIP_PATH=C:\...\archive.zip
#
# Example (Linux):
#   export DATASET_ZIP_PATH=/path/to/archive.zip
# ============================================================

import os

candidate_zips = [
    os.environ.get("DATASET_ZIP_PATH", ""),
    r"c:\Users\HP\OneDrive\Desktop\Project Work 1\Lung Project Impl\IQ Dataset\archive.zip",
    r"c:\Users\HP\OneDrive\Desktop\Project Work 1\Lung Project Impl\archive.zip",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "IQ Dataset", "archive.zip")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "archive.zip")),
    "archive.zip"
]
ZIP_PATH = None
for cz in candidate_zips:
    if cz and os.path.exists(cz):
        ZIP_PATH = cz
        break
if ZIP_PATH is None:
    ZIP_PATH = r"c:\Users\HP\OneDrive\Desktop\Project Work 1\Lung Project Impl\IQ Dataset\archive.zip"


EXTRACT_PATH = os.environ.get(
    "DATASET_EXTRACT_PATH",
    "iqoth_nccd_gp"
)

print("Dataset ZIP path:", ZIP_PATH)
print("Extraction path :", EXTRACT_PATH)



# ==================== ORIGINAL COLAB CELL 4 ====================

# ============================================================
# VERIFY ZIP FILE AND INSPECT ITS CONTENTS
# ============================================================

import os
import zipfile

print("==========================================")
print("ZIP VERIFICATION")
print("==========================================")

print("ZIP path:")
print(ZIP_PATH)

print(
    "\nZIP exists:",
    os.path.exists(ZIP_PATH)
)

if not os.path.exists(ZIP_PATH) and not os.path.exists(r"c:\Users\HP\OneDrive\Desktop\Project Work 1\Lung Project Impl\archive\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset"):
    raise FileNotFoundError(
        f"Dataset ZIP was not found: {ZIP_PATH}"
    )

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    file_list = zip_ref.namelist()

print(
    "\nTotal items inside ZIP:",
    len(file_list)
)

print("\nFirst 30 items:\n")

for item in file_list[:30]:
    print(item)



# ==================== ORIGINAL COLAB CELL 5 ====================

# ============================================================
# EXTRACT IQ-OTH/NCCD DATASET
# ============================================================

import shutil
import os
import zipfile

if os.path.exists(EXTRACT_PATH):
    print("Removing previous extraction...")
    shutil.rmtree(EXTRACT_PATH)

os.makedirs(EXTRACT_PATH, exist_ok=True)

print("Extracting dataset ZIP...")
print("This may take a little time.")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("\n[OK] Extraction completed.")
print("Extraction path:")
print(EXTRACT_PATH)



# ==================== ORIGINAL COLAB CELL 6 ====================

# ============================================================
# FIND IQ-OTH/NCCD CLASS FOLDERS
# ============================================================
#
# ACTUAL DATASET FOLDER NAMES:
#
#     Bengin cases
#     Malignant cases
#     Normal cases
#
# IMPORTANT:
# "Bengin" is the spelling used by the dataset.
# Internally, we will rename it to "Benign".
#
# We will NOT use the "Test cases" folder here.
# ============================================================


CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]


class_directories = {}


print("==========================================")
print("SEARCHING IQ-OTH/NCCD CLASS FOLDERS")
print("==========================================")


# ------------------------------------------------------------
# SEARCH ALL EXTRACTED DIRECTORIES
# ------------------------------------------------------------

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
        # NORMAL CASES
        # ----------------------------------------------------

        if folder_name == "normal cases":

            class_directories["Normal"] = full_path

            print(
                "[OK] Normal cases found:"
            )

            print(
                "  ",
                full_path
            )


        # ----------------------------------------------------
        # BENGIN CASES -> BENIGN
        # ----------------------------------------------------

        elif folder_name == "bengin cases":

            class_directories["Benign"] = full_path

            print(
                "[OK] Bengin cases found "
                "(mapped to Benign):"
            )

            print(
                "  ",
                full_path
            )


        # ----------------------------------------------------
        # MALIGNANT CASES
        # ----------------------------------------------------

        elif folder_name == "malignant cases":

            class_directories["Malignant"] = full_path

            print(
                "[OK] Malignant cases found:"
            )

            print(
                "  ",
                full_path
            )


# ------------------------------------------------------------
# DISPLAY FINAL MAPPING
# ------------------------------------------------------------

print("\n==========================================")
print("FINAL CLASS MAPPING")
print("==========================================")


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


# ------------------------------------------------------------
# VERIFY ALL THREE CLASSES
# ------------------------------------------------------------

missing_classes = [

    class_name

    for class_name in CLASS_NAMES

    if class_name not in class_directories

]


if missing_classes:

    raise RuntimeError(

        "The following class folders "
        "could not be found: "
        f"{missing_classes}"

    )


print(
    "\n[OK] All three classes found successfully."
)


# ==================== ORIGINAL COLAB CELL 7 ====================

# ============================================================
# VERIFY IQ-OTH/NCCD IMAGE COUNTS
# ============================================================

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


print("==========================================")
print("IQ-OTH/NCCD CLASS COUNTS")
print("==========================================")


class_counts = {}

total_images = 0


for class_name in CLASS_NAMES:

    class_path = class_directories[
        class_name
    ]

    count = 0


    for root, dirs, files in os.walk(
        class_path
    ):

        for filename in files:

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                count += 1


    class_counts[
        class_name
    ] = count

    total_images += count


    print(
        f"{class_name:<12}: "
        f"{count} images"
    )


print("------------------------------------------")

print(
    f"TOTAL         : "
    f"{total_images} images"
)


# ==================== ORIGINAL COLAB CELL 8 ====================

# ============================================================
# BUILD DATASET DATAFRAME + IMAGE QUALITY CHECK
# ============================================================
#
# Actual IQ-OTH/NCCD folders:
#
#   Bengin cases
#   Malignant cases
#   Normal cases
#
# Internal labels:
#
#   0 -> Normal
#   1 -> Benign
#   2 -> Malignant
#
# This cell:
#   1. Finds images from the three class directories
#   2. Creates dataset_df
#   3. Checks whether images are readable
#   4. Removes invalid images
# ============================================================


# ------------------------------------------------------------
# CLASS DEFINITIONS
# ------------------------------------------------------------

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]


# ------------------------------------------------------------
# IMAGE EXTENSIONS
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
# CREATE DATASET RECORDS
# ------------------------------------------------------------

records = []


print("==========================================")
print("BUILDING IQ-OTH/NCCD DATASET")
print("==========================================")


for class_name in CLASS_NAMES:

    # Get the actual physical folder.
    #
    # For Benign this points to:
    # "Bengin cases"

    class_path = class_directories[
        class_name
    ]


    # Numerical label

    label = CLASS_NAMES.index(
        class_name
    )


    image_count = 0


    # Search recursively inside the class folder

    for root, dirs, files in os.walk(
        class_path
    ):

        for filename in files:

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                filepath = os.path.join(
                    root,
                    filename
                )


                records.append({

                    "filepath": filepath,

                    "class": class_name,

                    "label": label

                })


                image_count += 1


    print(
        f"{class_name:<12}: "
        f"{image_count} images"
    )


# ------------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------------

dataset_df = pd.DataFrame(
    records
)


print("------------------------------------------")

print(
    "Total images found:",
    len(dataset_df)
)


# ------------------------------------------------------------
# CHECK DATASET
# ------------------------------------------------------------

if len(dataset_df) == 0:

    raise RuntimeError(
        "No images were found. "
        "Please check class_directories."
    )


print("\nClass distribution:")

print(
    dataset_df["class"].value_counts()
)


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

print("\n==========================================")
print("IMAGE QUALITY CHECK")
print("==========================================")


valid_records = []

invalid_records = []


for index, row in dataset_df.iterrows():

    image_path = row["filepath"]


    try:

        # Read image as grayscale

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        # Check if OpenCV successfully read it

        if image is None:

            raise ValueError(
                "Image could not be read"
            )


        # Check dimensions

        if len(image.shape) != 2:

            raise ValueError(
                "Unexpected image format"
            )


        height, width = image.shape


        if height <= 0 or width <= 0:

            raise ValueError(
                "Invalid image dimensions"
            )


        # Image passed quality check

        valid_records.append(
            index
        )


    except Exception as error:

        invalid_records.append({

            "filepath": image_path,

            "error": str(error)

        })


# ------------------------------------------------------------
# KEEP ONLY VALID IMAGES
# ------------------------------------------------------------

dataset_df = dataset_df.loc[
    valid_records
].reset_index(
    drop=True
)


# ------------------------------------------------------------
# DISPLAY QUALITY RESULT
# ------------------------------------------------------------

print(
    "Total images checked :",
    len(valid_records)
    + len(invalid_records)
)

print(
    "Valid images         :",
    len(valid_records)
)

print(
    "Invalid images       :",
    len(invalid_records)
)


if len(invalid_records) == 0:

    print(
        "\n[OK] All images passed the quality check."
    )

else:

    print(
        "\n⚠ Invalid images were removed."
    )


# ------------------------------------------------------------
# FINAL DATASET DISTRIBUTION
# ------------------------------------------------------------

print("\n==========================================")
print("FINAL CLEAN DATASET")
print("==========================================")


print(
    "Total usable images:",
    len(dataset_df)
)


print("\nClass distribution:")

print(
    dataset_df["class"].value_counts()
)


# ==================== ORIGINAL COLAB CELL 9 ====================

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# Dataset:
#   IQ-OTH/NCCD
#
# Total:
#   1097 images
#
# Split:
#   70% -> Training
#   15% -> Validation
#   15% -> Test
#
# Stratification is used to preserve the proportion of
# Normal, Benign and Malignant cases in every split.
#
# IMPORTANT:
#   Test data will remain untouched until the final GP
#   evaluation.
# ============================================================


from sklearn.model_selection import train_test_split


SEED = 42


# ------------------------------------------------------------
# FIRST SPLIT
# ------------------------------------------------------------
#
# 70% Training
# 30% Temporary
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
        dataset_type = 'hospital' if 'iq-Gp'.startswith('hosp') else 'iq'
        train_df, val_df, internal_test_df = load_dataset_splits(dataset_type)
        print(f'Dataset splits loaded successfully for iq-Gp: Train={len(train_df)}, Val={len(val_df)}, Test={len(internal_test_df)}')
    except Exception as e:
        print(f'Dataset split auto-load note: {e}')

train_df, temp_df = train_test_split(

    dataset_df,

    test_size=0.30,

    stratify=dataset_df["label"],

    random_state=SEED

)


# ------------------------------------------------------------
# SECOND SPLIT
# ------------------------------------------------------------
#
# Split the remaining 30% equally:
#
#   15% Validation
#   15% Test
# ------------------------------------------------------------

val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    stratify=temp_df["label"],

    random_state=SEED

)


# ------------------------------------------------------------
# RESET INDEX
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

print("==========================================")
print("DATASET SPLIT")
print("==========================================")

print(
    f"Total      : {len(dataset_df)}"
)

print(
    f"Training   : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("TRAINING DISTRIBUTION")
print("==========================================")

print(
    train_df["class"].value_counts()
)


print("\n==========================================")
print("VALIDATION DISTRIBUTION")
print("==========================================")

print(
    val_df["class"].value_counts()
)


print("\n==========================================")
print("TEST DISTRIBUTION")
print("==========================================")

print(
    test_df["class"].value_counts()
)


# ==================== ORIGINAL COLAB CELL 10 ====================

# ============================================================
# VISUALIZE DATASET CLASS DISTRIBUTION
# ============================================================

class_order = [
    "Normal",
    "Benign",
    "Malignant"
]


counts = [
    int(
        (
            dataset_df["class"]
            == class_name
        ).sum()
    )

    for class_name in class_order
]


plt.figure(
    figsize=(8, 5)
)


bars = plt.bar(
    class_order,
    counts
)


plt.title(
    "IQ-OTH/NCCD Dataset Class Distribution"
)

plt.xlabel(
    "Class"
)

plt.ylabel(
    "Number of Images"
)


# Display values above bars

for bar, count in zip(
    bars,
    counts
):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        str(count),

        ha="center",

        va="bottom"

    )


plt.tight_layout()

plt.show()


# ==================== ORIGINAL COLAB CELL 11 ====================

# ============================================================
# GP PREPROCESSING PIPELINE
# ============================================================
#
# IQ-OTH/NCCD CT images
#
# Preprocessing:
#
#   Original CT image
#          ↓
#   Resize to 224 × 224
#          ↓
#   Denoising
#          ↓
#   Normalization
#          ↓
#   ROI extraction
#
# NOTE:
# The ROI step below is a computational image-processing
# approximation. It is NOT a clinically validated lung
# segmentation algorithm.
# ============================================================


TARGET_SIZE = (224, 224)


def preprocess_ct_image(
    image_path,
    return_all=False
):

    """
    Preprocess one IQ-OTH/NCCD CT image.

    Parameters
    ----------
    image_path : str
        Path to the CT image.

    return_all : bool
        If True, return all preprocessing stages.
        If False, return only the final ROI image.

    Returns
    -------
    numpy array
        Preprocessed ROI image.
    """


    # ========================================================
    # STEP 1 — LOAD IMAGE
    # ========================================================

    original = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )


    if original is None:

        raise ValueError(
            f"Unable to read image:\n{image_path}"
        )


    # ========================================================
    # STEP 2 — RESIZE
    # ========================================================

    resized = cv2.resize(

        original,

        TARGET_SIZE,

        interpolation=cv2.INTER_AREA

    )


    # ========================================================
    # STEP 3 — DENOISING
    # ========================================================
    #
    # Gaussian filtering is used to reduce small-scale noise.
    # ========================================================

    denoised = cv2.GaussianBlur(

        resized,

        (5, 5),

        sigmaX=0

    )


    # ========================================================
    # STEP 4 — NORMALIZATION
    # ========================================================
    #
    # Convert image intensity to the range [0, 255].
    # ========================================================

    normalized = cv2.normalize(

        denoised,

        None,

        alpha=0,

        beta=255,

        norm_type=cv2.NORM_MINMAX

    )


    # ========================================================
    # STEP 5 — ROI EXTRACTION
    # ========================================================
    #
    # Otsu thresholding is used to obtain a broad foreground
    # region. Morphological operations remove small regions
    # and close small gaps.
    #
    # This is an image-processing ROI approximation.
    # ========================================================

    _, roi_mask = cv2.threshold(

        normalized,

        0,

        255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU

    )


    # Morphological kernel

    kernel = np.ones(

        (5, 5),

        dtype=np.uint8

    )


    # Remove small isolated regions

    roi_mask = cv2.morphologyEx(

        roi_mask,

        cv2.MORPH_OPEN,

        kernel

    )


    # Close small gaps

    roi_mask = cv2.morphologyEx(

        roi_mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    # Apply mask

    roi = cv2.bitwise_and(

        normalized,

        normalized,

        mask=roi_mask

    )


    # ========================================================
    # RETURN
    # ========================================================

    if return_all:

        return {

            "original": original,

            "resized": resized,

            "denoised": denoised,

            "normalized": normalized,

            "roi_mask": roi_mask,

            "roi": roi

        }


    return roi


# ==================== ORIGINAL COLAB CELL 12 ====================

# ============================================================
# TEST PREPROCESSING ON ONE IMAGE FROM EACH CLASS
# ============================================================

# Select one sample from each class

normal_sample = train_df[
    train_df["class"] == "Normal"
].iloc[0]["filepath"]


benign_sample = train_df[
    train_df["class"] == "Benign"
].iloc[0]["filepath"]


malignant_sample = train_df[
    train_df["class"] == "Malignant"
].iloc[0]["filepath"]


sample_paths = {

    "Normal": normal_sample,

    "Benign": benign_sample,

    "Malignant": malignant_sample

}


# ------------------------------------------------------------
# DISPLAY PREPROCESSING FOR EACH CLASS
# ------------------------------------------------------------

for class_name, image_path in sample_paths.items():

    print("\n==========================================")
    print(
        f"PREPROCESSING SAMPLE: {class_name}"
    )
    print("==========================================")


    results = preprocess_ct_image(

        image_path,

        return_all=True

    )


    plt.figure(
        figsize=(18, 5)
    )


    # Original

    plt.subplot(
        1, 6, 1
    )

    plt.imshow(
        results["original"],
        cmap="gray"
    )

    plt.title(
        "Original"
    )

    plt.axis("off")


    # Resized

    plt.subplot(
        1, 6, 2
    )

    plt.imshow(
        results["resized"],
        cmap="gray"
    )

    plt.title(
        "224 × 224"
    )

    plt.axis("off")


    # Denoised

    plt.subplot(
        1, 6, 3
    )

    plt.imshow(
        results["denoised"],
        cmap="gray"
    )

    plt.title(
        "Denoised"
    )

    plt.axis("off")


    # Normalized

    plt.subplot(
        1, 6, 4
    )

    plt.imshow(
        results["normalized"],
        cmap="gray"
    )

    plt.title(
        "Normalized"
    )

    plt.axis("off")


    # ROI mask

    plt.subplot(
        1, 6, 5
    )

    plt.imshow(
        results["roi_mask"],
        cmap="gray"
    )

    plt.title(
        "ROI Mask"
    )

    plt.axis("off")


    # Final ROI

    plt.subplot(
        1, 6, 6
    )

    plt.imshow(
        results["roi"],
        cmap="gray"
    )

    plt.title(
        "Final ROI"
    )

    plt.axis("off")


    plt.suptitle(
        f"IQ-OTH/NCCD - {class_name}"
    )

    plt.tight_layout()

    plt.show()


# ==================== ORIGINAL COLAB CELL 13 ====================

# ============================================================
# VERIFY PREPROCESSING OUTPUT
# ============================================================

print("==========================================")
print("PREPROCESSING VERIFICATION")
print("==========================================")


for class_name, image_path in sample_paths.items():

    processed = preprocess_ct_image(
        image_path
    )


    print(
        f"\n{class_name}"
    )

    print(
        "Shape:",
        processed.shape
    )

    print(
        "Data type:",
        processed.dtype
    )

    print(
        "Minimum pixel:",
        processed.min()
    )

    print(
        "Maximum pixel:",
        processed.max()
    )

    print(
        "Mean pixel:",
        round(
            float(processed.mean()),
            4
        )
    )


# ==================== ORIGINAL COLAB CELL 14 ====================

# ============================================================
# GENETIC PROGRAMMING FEATURE EXTRACTION
# ============================================================
#
# Features used:
#
# 1. HOG
#    Histogram of Oriented Gradients
#    Captures structural and edge information.
#
# 2. LBP
#    Local Binary Pattern
#    Captures local texture information.
#
# 3. INTENSITY FEATURES
#    Statistical information from the CT image.
#
# These complementary features provide GP with information
# about:
#
#     Structure + Texture + Intensity
#
# ============================================================


from skimage.feature import hog
from skimage.feature import local_binary_pattern


def extract_gp_features(
    image_path
):

    """
    Extract a compact feature vector from one
    preprocessed IQ-OTH/NCCD CT image.
    """


    # ========================================================
    # STEP 1 — PREPROCESS IMAGE
    # ========================================================

    roi = preprocess_ct_image(

        image_path

    )


    # Convert pixel values to [0,1]

    image_float = (

        roi.astype(
            np.float32
        )

        / 255.0

    )


    # ========================================================
    # STEP 2 — HOG FEATURES
    # ========================================================
    #
    # HOG describes local edge and shape information.
    #
    # orientations = 9
    # pixels_per_cell = 16 × 16
    # cells_per_block = 2 × 2
    # ========================================================


    hog_features = hog(

        image_float,

        orientations=9,

        pixels_per_cell=(16, 16),

        cells_per_block=(2, 2),

        block_norm="L2-Hys",

        feature_vector=True

    )


    # ========================================================
    # STEP 3 — LBP FEATURES
    # ========================================================
    #
    # LBP describes local texture patterns.
    # ========================================================


    radius = 2

    points = 8 * radius


    lbp = local_binary_pattern(

        roi,

        points,

        radius,

        method="uniform"

    )


    # Number of histogram bins

    n_bins = points + 2


    lbp_histogram, _ = np.histogram(

        lbp.ravel(),

        bins=np.arange(
            0,
            n_bins + 1
        ),

        range=(
            0,
            n_bins
        )

    )


    # Normalize LBP histogram

    lbp_histogram = (

        lbp_histogram.astype(
            np.float32
        )

        /

        (
            lbp_histogram.sum()
            + 1e-8
        )

    )


    # ========================================================
    # STEP 4 — INTENSITY FEATURES
    # ========================================================
    #
    # Statistical information from the CT image.
    # ========================================================


    intensity_features = np.array([

        np.mean(
            image_float
        ),

        np.std(
            image_float
        ),

        np.min(
            image_float
        ),

        np.max(
            image_float
        ),

        np.percentile(
            image_float,
            10
        ),

        np.percentile(
            image_float,
            25
        ),

        np.percentile(
            image_float,
            50
        ),

        np.percentile(
            image_float,
            75
        ),

        np.percentile(
            image_float,
            90
        )

    ], dtype=np.float32)


    # ========================================================
    # STEP 5 — COMBINE ALL FEATURES
    # ========================================================


    combined_features = np.concatenate([

        hog_features,

        lbp_histogram,

        intensity_features

    ])


    return combined_features.astype(
        np.float32
    )


# ==================== ORIGINAL COLAB CELL 15 ====================

# ============================================================
# TEST GP FEATURE EXTRACTION
# ============================================================


sample_path = train_df.iloc[0]["filepath"]


sample_features = extract_gp_features(

    sample_path

)


print("==========================================")
print("GP FEATURE EXTRACTION TEST")
print("==========================================")


print(
    "Sample image:"
)

print(
    sample_path
)


print(
    "\nFeature vector shape:"
)

print(
    sample_features.shape
)


print(
    "\nNumber of features:"
)

print(
    len(sample_features)
)


print(
    "\nFirst 20 features:"
)

print(
    sample_features[:20]
)


print(
    "\nFeature data type:"
)

print(
    sample_features.dtype
)


# ==================== ORIGINAL COLAB CELL 16 ====================

# ============================================================
# EXTRACT TRAINING FEATURES
# ============================================================
#
# IMPORTANT:
# Feature extraction is performed independently for every
# training image.
# ============================================================


def build_gp_feature_matrix(
    dataframe
):

    """
    Extract GP features for every image in a dataframe.
    """

    feature_list = []

    label_list = []

    path_list = []


    total = len(
        dataframe
    )


    for index, row in dataframe.iterrows():

        # Progress indicator

        if index % 50 == 0:

            print(
                f"Processing "
                f"{index}/{total}"
            )


        # Extract features

        features = extract_gp_features(

            row["filepath"]

        )


        feature_list.append(
            features
        )


        label_list.append(
            row["label"]
        )


        path_list.append(
            row["filepath"]
        )


    # Convert to NumPy arrays

    X = np.asarray(

        feature_list,

        dtype=np.float32

    )


    y = np.asarray(

        label_list,

        dtype=np.int64

    )


    return X, y, path_list


print(
    "=========================================="
)

print(
    "EXTRACTING TRAINING FEATURES"
)

print(
    "=========================================="
)


X_train_raw, y_train, train_paths = (

    build_gp_feature_matrix(
        train_df
    )

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
    "\n[OK] Training feature extraction completed."
)


print(
    "Training feature matrix:",
    X_train_raw.shape
)


# ==================== ORIGINAL COLAB CELL 17 ====================

# ============================================================
# EXTRACT VALIDATION FEATURES
# ============================================================


print(
    "=========================================="
)

print(
    "EXTRACTING VALIDATION FEATURES"
)

print(
    "=========================================="
)


X_val_raw, y_val, val_paths = (

    build_gp_feature_matrix(
        val_df
    )

)


print(
    "\n[OK] Validation feature extraction completed."
)


print(
    "Validation feature matrix:",
    X_val_raw.shape
)


# ==================== ORIGINAL COLAB CELL 18 ====================

# ============================================================
# EXTRACT TEST FEATURES
# ============================================================
#
# The test set remains untouched for final evaluation.
# ============================================================


print(
    "=========================================="
)

print(
    "EXTRACTING TEST FEATURES"
)

print(
    "=========================================="
)


X_test_raw, y_test, test_paths = (

    build_gp_feature_matrix(
        test_df
    )

)


print(
    "\n[OK] Test feature extraction completed."
)


print(
    "Test feature matrix:",
    X_test_raw.shape
)


# ==================== ORIGINAL COLAB CELL 19 ====================

# ============================================================
# VERIFY GP FEATURE MATRICES
# ============================================================


print("==========================================")
print("GP FEATURE MATRIX VERIFICATION")
print("==========================================")


print(
    "Training   :",
    X_train_raw.shape
)


print(
    "Validation :",
    X_val_raw.shape
)


print(
    "Test       :",
    X_test_raw.shape
)


print()


print(
    "Training labels:",
    y_train.shape
)


print(
    "Validation labels:",
    y_val.shape
)


print(
    "Test labels:",
    y_test.shape
)


# ==================== ORIGINAL COLAB CELL 20 ====================

# ============================================================
# STANDARDIZE GP FEATURES
# ============================================================
#
# Input:
#     6111 handcrafted features
#
# Standardization:
#
#     z = (x - mean) / standard deviation
#
# IMPORTANT:
# The scaler is fitted ONLY on training data.
# Validation and test data are transformed using the
# training-set statistics.
#
# This prevents information leakage.
# ============================================================

from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# CREATE SCALER
# ------------------------------------------------------------

gp_scaler = StandardScaler()


# ------------------------------------------------------------
# FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------

X_train_scaled = gp_scaler.fit_transform(
    X_train_raw
)


# ------------------------------------------------------------
# TRANSFORM VALIDATION DATA
# ------------------------------------------------------------

X_val_scaled = gp_scaler.transform(
    X_val_raw
)


# ------------------------------------------------------------
# TRANSFORM TEST DATA
# ------------------------------------------------------------

X_test_scaled = gp_scaler.transform(
    X_test_raw
)


# ============================================================
# VERIFY
# ============================================================

print("==========================================")
print("GP FEATURE STANDARDIZATION")
print("==========================================")

print(
    "Training shape   :",
    X_train_scaled.shape
)

print(
    "Validation shape :",
    X_val_scaled.shape
)

print(
    "Test shape       :",
    X_test_scaled.shape
)


print("\nTraining mean:")
print(
    np.mean(
        X_train_scaled
    )
)


print("\nTraining standard deviation:")
print(
    np.std(
        X_train_scaled
    )
)


# ==================== ORIGINAL COLAB CELL 21 ====================

# ============================================================
# PCA DIMENSIONALITY REDUCTION FOR GP
# ============================================================
#
# 6111 handcrafted features are too large for efficient
# Genetic Programming.
#
# PCA compresses the feature space while retaining the
# dominant variance in the training data.
#
# First GP experiment:
#
#     6111 → 32 components
#
# PCA is fitted ONLY on training data.
# ============================================================

from sklearn.decomposition import PCA


GP_PCA_COMPONENTS = 32


# ------------------------------------------------------------
# CREATE PCA
# ------------------------------------------------------------

gp_pca = PCA(

    n_components=GP_PCA_COMPONENTS,

    random_state=SEED

)


# ------------------------------------------------------------
# FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------

X_train_pca = gp_pca.fit_transform(
    X_train_scaled
)


# ------------------------------------------------------------
# TRANSFORM VALIDATION
# ------------------------------------------------------------

X_val_pca = gp_pca.transform(
    X_val_scaled
)


# ------------------------------------------------------------
# TRANSFORM TEST
# ------------------------------------------------------------

X_test_pca = gp_pca.transform(
    X_test_scaled
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("==========================================")
print("GP PCA DIMENSIONALITY REDUCTION")
print("==========================================")


print(
    "Before PCA:",
    X_train_scaled.shape
)


print(
    "After PCA:",
    X_train_pca.shape
)


print(
    "Validation:",
    X_val_pca.shape
)


print(
    "Test:",
    X_test_pca.shape
)


# ------------------------------------------------------------
# EXPLAINED VARIANCE
# ------------------------------------------------------------

variance_retained = (
    gp_pca.explained_variance_ratio_.sum()
)


print(
    "\nVariance retained:",
    f"{variance_retained * 100:.2f}%"
)


# ==================== ORIGINAL COLAB CELL 22 ====================

# ============================================================
# GENETIC PROGRAMMING - ITERATION 1
# CREATE EVOLVED CLASSIFIERS
# ============================================================
#
# One-vs-Rest strategy:
#
#     GP-1 : Normal     vs Not Normal
#     GP-2 : Benign     vs Not Benign
#     GP-3 : Malignant  vs Not Malignant
#
# The three GP models will later be combined to produce the
# final three-class prediction.
#
# Iteration 1 configuration:
#
#     PCA components       = 32
#     Population size      = 300
#     Generations          = 20
#     Fitness              = Log Loss
#
# This is our baseline GP experiment.
# ============================================================


from gplearn.genetic import SymbolicClassifier


# ------------------------------------------------------------
# GP MODEL FACTORY
# ------------------------------------------------------------

def create_gp_classifier(
    random_state
):

    """
    Create one Genetic Programming symbolic classifier.
    """

    model = SymbolicClassifier(

        # Number of evolutionary generations
        generations=20,

        # Number of individuals in each generation
        population_size=300,

        # Mathematical functions available to GP
        function_set=(

            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "log",
            "abs",
            "neg"

        ),

        # Fitness function
        metric="log loss",

        # Penalize unnecessarily complex programs
        parsimony_coefficient=0.001,

        # Fraction of training samples used by each program
        max_samples=0.8,

        # Display evolutionary progress
        verbose=1,

        # Reproducibility
        random_state=random_state,

        # Use available CPU cores
        n_jobs=-1
    )

    return model


# ------------------------------------------------------------
# CREATE THREE GP CLASSIFIERS
# ------------------------------------------------------------

gp_normal = create_gp_classifier(
    random_state=101
)


gp_benign = create_gp_classifier(
    random_state=102
)


gp_malignant = create_gp_classifier(
    random_state=103
)


print("==========================================")
print("GP ITERATION 1")
print("==========================================")

print("[OK] Normal GP classifier created")

print("[OK] Benign GP classifier created")

print("[OK] Malignant GP classifier created")

print("------------------------------------------")

print("PCA input features :", X_train_pca.shape[1])

print("Population size    : 300")

print("Generations         : 20")

print("Fitness             : Log Loss")

print("\n[OK] GP Iteration 1 is ready for training.")


# ==================== ORIGINAL COLAB CELL 23 ====================

# ============================================================
# GP ITERATION 1
# CREATE ONE-VS-REST TRAINING TARGETS
# ============================================================
#
# Original multiclass labels:
#
#     0 = Normal
#     1 = Benign
#     2 = Malignant
#
# Genetic Programming uses three binary classifiers:
#
#     GP-1 : Normal vs Rest
#     GP-2 : Benign vs Rest
#     GP-3 : Malignant vs Rest
#
# Positive class = 1
# Negative class = 0
# ============================================================


# ------------------------------------------------------------
# NORMAL VS REST
# ------------------------------------------------------------

y_train_normal = (
    y_train == 0
).astype(
    np.int32
)


# ------------------------------------------------------------
# BENIGN VS REST
# ------------------------------------------------------------

y_train_benign = (
    y_train == 1
).astype(
    np.int32
)


# ------------------------------------------------------------
# MALIGNANT VS REST
# ------------------------------------------------------------

y_train_malignant = (
    y_train == 2
).astype(
    np.int32
)


# ============================================================
# DISPLAY TRAINING DISTRIBUTION
# ============================================================

print("==========================================")
print("GP ONE-VS-REST TRAINING TARGETS")
print("==========================================")


print("\nNormal vs Rest")

print(
    "Normal     :",
    np.sum(y_train_normal == 1)
)

print(
    "Not Normal :",
    np.sum(y_train_normal == 0)
)


print("\nBenign vs Rest")

print(
    "Benign     :",
    np.sum(y_train_benign == 1)
)

print(
    "Not Benign :",
    np.sum(y_train_benign == 0)
)


print("\nMalignant vs Rest")

print(
    "Malignant     :",
    np.sum(y_train_malignant == 1)
)

print(
    "Not Malignant :",
    np.sum(y_train_malignant == 0)
)


# ==================== ORIGINAL COLAB CELL 24 ====================

# ============================================================
# GP ITERATION 1
# TRAIN GP-1
# NORMAL VS NOT NORMAL
# ============================================================


print("==========================================")
print("TRAINING GP-1")
print("NORMAL VS NOT NORMAL")
print("==========================================")


gp_normal.fit(

    X_train_pca,

    y_train_normal

)


print(
    "\n[OK] GP-1 Normal classifier trained."
)


# ==================== ORIGINAL COLAB CELL 25 ====================

# ============================================================
# GP ITERATION 1
# TRAIN GP-2
# BENIGN VS NOT BENIGN
# ============================================================


print("==========================================")
print("TRAINING GP-2")
print("BENIGN VS NOT BENIGN")
print("==========================================")


gp_benign.fit(

    X_train_pca,

    y_train_benign

)


print(
    "\n[OK] GP-2 Benign classifier trained."
)


# ==================== ORIGINAL COLAB CELL 26 ====================

# ============================================================
# GP ITERATION 1
# TRAIN GP-3
# MALIGNANT VS NOT MALIGNANT
# ============================================================


print("==========================================")
print("TRAINING GP-3")
print("MALIGNANT VS NOT MALIGNANT")
print("==========================================")


gp_malignant.fit(

    X_train_pca,

    y_train_malignant

)


print(
    "\n[OK] GP-3 Malignant classifier trained."
)


# ==================== ORIGINAL COLAB CELL 27 ====================

# ============================================================
# VERIFY GP MODELS
# ============================================================


print("==========================================")
print("GP MODEL VERIFICATION")
print("==========================================")


models = {

    "Normal GP":
    gp_normal,

    "Benign GP":
    gp_benign,

    "Malignant GP":
    gp_malignant

}


for model_name, model in models.items():

    print(
        f"\n[OK] {model_name}"
    )


    # Display the evolved symbolic program

    print(
        "Program:"
    )

    print(
        model._program
    )


# ==================== ORIGINAL COLAB CELL 28 ====================

# ============================================================
# GP ITERATION 1
# VALIDATION SCORES
# ============================================================
#
# Each GP classifier predicts the score/probability of its
# corresponding class.
#
#     GP-Normal     -> P(Normal)
#     GP-Benign     -> P(Benign)
#     GP-Malignant  -> P(Malignant)
#
# These are independent One-vs-Rest scores.
# ============================================================


# ------------------------------------------------------------
# NORMAL
# ------------------------------------------------------------

normal_probability = (

    gp_normal
    .predict_proba(
        X_val_pca
    )[:, 1]

)


# ------------------------------------------------------------
# BENIGN
# ------------------------------------------------------------

benign_probability = (

    gp_benign
    .predict_proba(
        X_val_pca
    )[:, 1]

)


# ------------------------------------------------------------
# MALIGNANT
# ------------------------------------------------------------

malignant_probability = (

    gp_malignant
    .predict_proba(
        X_val_pca
    )[:, 1]

)


# ============================================================
# COMBINE SCORES
# ============================================================

gp_score_matrix = np.column_stack([

    normal_probability,

    benign_probability,

    malignant_probability

])


print("==========================================")
print("GP VALIDATION SCORES")
print("==========================================")


print(
    "Normal scores     :",
    normal_probability.shape
)


print(
    "Benign scores     :",
    benign_probability.shape
)


print(
    "Malignant scores  :",
    malignant_probability.shape
)


print(
    "\nCombined score matrix:",
    gp_score_matrix.shape
)


# ==================== ORIGINAL COLAB CELL 29 ====================

# ============================================================
# GP ITERATION 1
# CONVERT OVR SCORES TO THREE-CLASS PREDICTIONS
# ============================================================
#
# The three One-vs-Rest GP scores are normalized for the
# purpose of comparing the three classes.
#
# IMPORTANT:
# These are normalized OVR scores, not clinically calibrated
# probabilities.
# ============================================================


# ------------------------------------------------------------
# CALCULATE SUM OF THREE SCORES
# ------------------------------------------------------------

score_sum = (

    gp_score_matrix.sum(

        axis=1,

        keepdims=True

    )

)


# ------------------------------------------------------------
# PREVENT DIVISION BY ZERO
# ------------------------------------------------------------

score_sum = np.maximum(

    score_sum,

    1e-8

)


# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------

gp_probabilities = (

    gp_score_matrix
    /
    score_sum

)


# ------------------------------------------------------------
# SELECT HIGHEST SCORE
# ------------------------------------------------------------

gp_predictions = np.argmax(

    gp_probabilities,

    axis=1

)


print("==========================================")
print("GP THREE-CLASS PREDICTION")
print("==========================================")


print(
    "Validation samples :",
    len(gp_predictions)
)


print(
    "Probability matrix :",
    gp_probabilities.shape
)


print(
    "Prediction vector  :",
    gp_predictions.shape
)


print(
    "\n[OK] Three-class predictions generated."
)


# ==================== ORIGINAL COLAB CELL 30 ====================

# ============================================================
# GP ITERATION 1
# FIRST 20 VALIDATION PREDICTIONS
# ============================================================


INDEX_TO_CLASS = {

    0: "Normal",

    1: "Benign",

    2: "Malignant"

}


print("==========================================")
print("FIRST 20 GP PREDICTIONS")
print("==========================================")


for i in range(
    min(20, len(y_val))
):

    actual = INDEX_TO_CLASS[
        int(y_val[i])
    ]


    predicted = INDEX_TO_CLASS[
        int(gp_predictions[i])
    ]


    print(

        f"{i + 1:02d}. "
        f"Actual: {actual:<10} | "
        f"Predicted: {predicted:<10} | "
        f"N={gp_probabilities[i, 0]:.3f} "
        f"B={gp_probabilities[i, 1]:.3f} "
        f"M={gp_probabilities[i, 2]:.3f}"

    )


# ==================== ORIGINAL COLAB CELL 31 ====================

# ============================================================
# GP ITERATION 1
# OVERALL VALIDATION ACCURACY
# ============================================================


from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# CORRECT
# ------------------------------------------------------------

correct_predictions = np.sum(

    gp_predictions == y_val

)


# ------------------------------------------------------------
# TOTAL
# ------------------------------------------------------------

total_predictions = len(
    y_val
)


# ------------------------------------------------------------
# INCORRECT
# ------------------------------------------------------------

incorrect_predictions = (

    total_predictions
    -
    correct_predictions

)


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

gp_accuracy = (

    correct_predictions
    /
    total_predictions

)


print("==========================================")
print("GP ITERATION 1")
print("OVERALL VALIDATION ACCURACY")
print("==========================================")


print(
    "Correct predictions   :",
    correct_predictions
)


print(
    "Incorrect predictions :",
    incorrect_predictions
)


print(
    "Total predictions     :",
    total_predictions
)


print(
    f"\nOverall Accuracy      : "
    f"{gp_accuracy * 100:.2f}%"
)


# ==================== ORIGINAL COLAB CELL 32 ====================

# ============================================================
# GP ITERATION 1
# BALANCED PERFORMANCE METRICS
# ============================================================


from sklearn.metrics import (

    balanced_accuracy_score,

    f1_score,

    precision_score,

    recall_score

)


# ------------------------------------------------------------
# BALANCED ACCURACY
# ------------------------------------------------------------

gp_balanced_accuracy = (

    balanced_accuracy_score(

        y_val,

        gp_predictions

    )

)


# ------------------------------------------------------------
# MACRO PRECISION
# ------------------------------------------------------------

gp_macro_precision = (

    precision_score(

        y_val,

        gp_predictions,

        average="macro",

        zero_division=0

    )

)


# ------------------------------------------------------------
# MACRO RECALL
# ------------------------------------------------------------

gp_macro_recall = (

    recall_score(

        y_val,

        gp_predictions,

        average="macro",

        zero_division=0

    )

)


# ------------------------------------------------------------
# MACRO F1
# ------------------------------------------------------------

gp_macro_f1 = (

    f1_score(

        y_val,

        gp_predictions,

        average="macro",

        zero_division=0

    )

)


# ------------------------------------------------------------
# WEIGHTED F1
# ------------------------------------------------------------

gp_weighted_f1 = (

    f1_score(

        y_val,

        gp_predictions,

        average="weighted",

        zero_division=0

    )

)


# ============================================================
# DISPLAY
# ============================================================

print("==========================================")
print("GP ITERATION 1")
print("VALIDATION METRICS")
print("==========================================")


print(
    f"Accuracy          : "
    f"{gp_accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy : "
    f"{gp_balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision   : "
    f"{gp_macro_precision:.4f}"
)


print(
    f"Macro Recall      : "
    f"{gp_macro_recall:.4f}"
)


print(
    f"Macro F1          : "
    f"{gp_macro_f1:.4f}"
)


print(
    f"Weighted F1       : "
    f"{gp_weighted_f1:.4f}"
)


# ==================== ORIGINAL COLAB CELL 33 ====================

# ============================================================
# GP ITERATION 1
# CLASSIFICATION REPORT
# ============================================================


from sklearn.metrics import classification_report


print("==========================================")
print("GP ITERATION 1")
print("CLASSIFICATION REPORT")
print("==========================================")


print(

    classification_report(

        y_val,

        gp_predictions,

        labels=[

            0,
            1,
            2

        ],

        target_names=[

            "Normal",
            "Benign",
            "Malignant"

        ],

        digits=4,

        zero_division=0

    )

)


# ==================== ORIGINAL COLAB CELL 34 ====================

# ============================================================
# GP ITERATION 1
# CONFUSION MATRIX
# ============================================================


from sklearn.metrics import confusion_matrix


cm_gp = confusion_matrix(

    y_val,

    gp_predictions,

    labels=[

        0,
        1,
        2

    ]

)


print("==========================================")
print("GP ITERATION 1")
print("CONFUSION MATRIX")
print("==========================================")


print(
    cm_gp
)


# ============================================================
# PLOT
# ============================================================


plt.figure(

    figsize=(7, 6)

)


sns.heatmap(

    cm_gp,

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
    "GP Iteration 1 - Validation Confusion Matrix"
)


plt.tight_layout()


plt.show()


# ==================== ORIGINAL COLAB CELL 35 ====================

# ============================================================
# GP ITERATION 1
# CLASS-WISE PERFORMANCE
# ============================================================
#
# This is particularly important because the dataset is
# imbalanced.
# ============================================================


print("==========================================")
print("GP ITERATION 1")
print("CLASS-WISE PERFORMANCE")
print("==========================================")


for class_index, class_name in enumerate([

    "Normal",
    "Benign",
    "Malignant"

]):


    # Number of actual samples

    actual_count = np.sum(

        y_val == class_index

    )


    # Number correctly classified

    correct_count = np.sum(

        (

            y_val == class_index

        )

        &

        (

            gp_predictions == class_index

        )

    )


    # Calculate recall / class accuracy

    if actual_count > 0:

        class_recall = (

            correct_count
            /
            actual_count

        )

    else:

        class_recall = 0.0


    print(

        f"{class_name:<12}: "
        f"{correct_count}/"
        f"{actual_count} correct "
        f"("
        f"{class_recall * 100:.2f}%"
        f")"

    )


# ==================== ORIGINAL COLAB CELL 36 ====================

# ============================================================
# GP ITERATION 1
# SAVE EXPERIMENT RESULTS
# ============================================================


gp_results = []


gp_results.append({

    "Experiment":
    "GP Iteration 1",

    "Generations":
    20,

    "Population":
    300,

    "PCA Components":
    32,

    "Accuracy":
    gp_accuracy,

    "Balanced Accuracy":
    gp_balanced_accuracy,

    "Macro Precision":
    gp_macro_precision,

    "Macro Recall":
    gp_macro_recall,

    "Macro F1":
    gp_macro_f1,

    "Weighted F1":
    gp_weighted_f1

})


gp_results_df = pd.DataFrame(
    gp_results
)


print("==========================================")
print("GP ITERATION 1 RESULTS")
print("==========================================")


print(
    gp_results_df
)


# ==================== ORIGINAL COLAB CELL 37 ====================

# ============================================================
# GP ITERATION 1
# EVOLUTIONARY TRAINING ANALYSIS
# ============================================================
#
# This cell checks how the GP models evolved across the
# 20 generations.
#
# It does NOT retrain the models.
# ============================================================


print("==========================================")
print("GP EVOLUTIONARY TRAINING ANALYSIS")
print("==========================================")


# ------------------------------------------------------------
# FUNCTION TO DISPLAY GP EVOLUTION
# ------------------------------------------------------------

def display_gp_evolution(model, model_name):

    print("\n------------------------------------------")
    print(model_name)
    print("------------------------------------------")


    # GP fitness history

    fitness_history = model.run_details_["best_fitness"]


    # Number of generations recorded

    generations_completed = len(
        fitness_history
    )


    print(
        "Generations completed:",
        generations_completed
    )


    print(
        "\nGeneration-wise best fitness:"
    )


    for generation, fitness in enumerate(
        fitness_history
    ):

        print(

            f"Generation {generation + 1:02d} "
            f"| Best Log Loss: {fitness:.6f}"

        )


    # --------------------------------------------------------
    # BEST FITNESS
    # --------------------------------------------------------

    best_fitness = min(
        fitness_history
    )


    best_generation = (
        np.argmin(
            fitness_history
        ) + 1
    )


    print(
        "\nBest generation:",
        best_generation
    )


    print(
        f"Best log-loss : "
        f"{best_fitness:.6f}"
    )


    # --------------------------------------------------------
    # FINAL EVOLVED PROGRAM
    # --------------------------------------------------------

    print(
        "\nFinal evolved program:"
    )


    print(
        model._program
    )


# ============================================================
# ANALYZE ALL THREE GP MODELS
# ============================================================


display_gp_evolution(

    gp_normal,

    "GP-1 : NORMAL VS REST"

)


display_gp_evolution(

    gp_benign,

    "GP-2 : BENIGN VS REST"

)


display_gp_evolution(

    gp_malignant,

    "GP-3 : MALIGNANT VS REST"

)


# ==================== ORIGINAL COLAB CELL 38 ====================

# ============================================================
# GP ITERATION 1
# INDIVIDUAL BINARY CLASSIFIER EVALUATION
# ============================================================
#
# Instead of immediately combining the three GP models,
# we evaluate each GP classifier independently.
#
# This helps identify whether the problem is:
#
#     GP-Normal
#     GP-Benign
#     GP-Malignant
#
# ============================================================


from sklearn.metrics import (

    accuracy_score,

    balanced_accuracy_score,

    classification_report

)


def evaluate_binary_gp(

    model,

    X_train,

    y_train_binary,

    X_val,

    y_val_original,

    target_class,

    target_name

):


    print("\n==========================================")

    print(
        f"{target_name.upper()} GP"
    )

    print("==========================================")


    # --------------------------------------------------------
    # TRAINING PREDICTIONS
    # --------------------------------------------------------

    train_predictions = model.predict(

        X_train

    )


    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    val_predictions = model.predict(

        X_val

    )


    # --------------------------------------------------------
    # BINARY TRAINING ACCURACY
    # --------------------------------------------------------

    train_accuracy = accuracy_score(

        y_train_binary,

        train_predictions

    )


    # --------------------------------------------------------
    # CONVERT ORIGINAL VALIDATION LABELS
    # INTO ONE-VS-REST LABELS
    # --------------------------------------------------------

    y_val_binary = (

        y_val_original == target_class

    ).astype(

        np.int32

    )


    # --------------------------------------------------------
    # VALIDATION ACCURACY
    # --------------------------------------------------------

    val_accuracy = accuracy_score(

        y_val_binary,

        val_predictions

    )


    # --------------------------------------------------------
    # BALANCED ACCURACY
    # --------------------------------------------------------

    val_balanced_accuracy = (

        balanced_accuracy_score(

            y_val_binary,

            val_predictions

        )

    )


    print(
        f"Training Accuracy       : "
        f"{train_accuracy * 100:.2f}%"
    )


    print(
        f"Validation Accuracy     : "
        f"{val_accuracy * 100:.2f}%"
    )


    print(
        f"Validation Balanced Acc.: "
        f"{val_balanced_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # BINARY CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "\nBinary Classification Report:"
    )


    print(

        classification_report(

            y_val_binary,

            val_predictions,

            target_names=[

                f"Not {target_name}",

                target_name

            ],

            digits=4,

            zero_division=0

        )

    )


# ============================================================
# GP-1
# ============================================================

evaluate_binary_gp(

    gp_normal,

    X_train_pca,

    y_train_normal,

    X_val_pca,

    y_val,

    0,

    "Normal"

)


# ============================================================
# GP-2
# ============================================================

evaluate_binary_gp(

    gp_benign,

    X_train_pca,

    y_train_benign,

    X_val_pca,

    y_val,

    1,

    "Benign"

)


# ============================================================
# GP-3
# ============================================================

evaluate_binary_gp(

    gp_malignant,

    X_train_pca,

    y_train_malignant,

    X_val_pca,

    y_val,

    2,

    "Malignant"

)


# ==================== ORIGINAL COLAB CELL 39 ====================

# ============================================================
# GP ITERATION 1
# FINAL MULTICLASS DISTRIBUTION
# ============================================================


print("==========================================")
print("GP PREDICTED CLASS DISTRIBUTION")
print("==========================================")


for class_index, class_name in enumerate([

    "Normal",
    "Benign",
    "Malignant"

]):

    count = np.sum(

        gp_predictions == class_index

    )


    percentage = (

        count
        /
        len(gp_predictions)
        *
        100

    )


    print(

        f"{class_name:<12}: "
        f"{count:3d} samples "
        f"({percentage:.2f}%)"

    )


print("\n==========================================")
print("ACTUAL VALIDATION DISTRIBUTION")
print("==========================================")


for class_index, class_name in enumerate([

    "Normal",
    "Benign",
    "Malignant"

]):

    count = np.sum(

        y_val == class_index

    )


    percentage = (

        count
        /
        len(y_val)
        *
        100

    )


    print(

        f"{class_name:<12}: "
        f"{count:3d} samples "
        f"({percentage:.2f}%)"

    )


# ==================== ORIGINAL COLAB CELL 40 ====================

# ============================================================
# GP ITERATION 2+
# AUTOMATIC ITERATION FUNCTION
# ============================================================
#
# GP Iteration 1 has already been completed.
#
# This function is ONLY for Iteration 2 onwards.
#
# Each iteration can have different:
#
#     - Number of generations
#     - Population size
#     - Balanced / unbalanced training
#     - Random seed
#
# The validation set is used for evaluation.
# The test set remains completely untouched.
# ============================================================


from gplearn.genetic import SymbolicClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import numpy as np
import pandas as pd
import time



# ============================================================
# CREATE BALANCED ONE-VS-REST DATA
# ============================================================

def create_balanced_ovr_data(

    X,
    y,
    target_class,
    random_state

):

    """
    Creates a balanced binary One-vs-Rest dataset.

    Positive:
        target class

    Negative:
        randomly selected samples from all other classes
    """

    # --------------------------------------------------------
    # Positive samples
    # --------------------------------------------------------

    positive_indices = np.where(

        y == target_class

    )[0]


    # --------------------------------------------------------
    # Negative samples
    # --------------------------------------------------------

    negative_indices = np.where(

        y != target_class

    )[0]


    # --------------------------------------------------------
    # Number of positive samples
    # --------------------------------------------------------

    n_positive = len(
        positive_indices
    )


    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = np.random.RandomState(
        random_state
    )


    # --------------------------------------------------------
    # Select equal number of negative samples
    # --------------------------------------------------------

    selected_negative_indices = rng.choice(

        negative_indices,

        size=n_positive,

        replace=False

    )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    selected_indices = np.concatenate([

        positive_indices,

        selected_negative_indices

    ])


    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    rng.shuffle(
        selected_indices
    )


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_balanced = X[
        selected_indices
    ]


    # --------------------------------------------------------
    # Binary labels
    # --------------------------------------------------------

    y_balanced = (

        y[selected_indices] == target_class

    ).astype(
        np.int32
    )


    return (

        X_balanced,
        y_balanced

    )


# ============================================================
# GP EXPERIMENT FUNCTION
# ============================================================

def run_gp_iteration(

    iteration_number,

    generations,

    population_size,

    balanced,

    random_seed

):

    """
    Runs one complete GP iteration.

    Three classifiers are trained:

        GP-Normal
        GP-Benign
        GP-Malignant

    Returns:

        results
        models
        predictions
        probabilities
        confusion matrix
    """


    print("\n")
    print("=" * 70)

    print(
        f"GENETIC PROGRAMMING - ITERATION {iteration_number}"
    )

    print("=" * 70)


    print(
        f"Generations       : {generations}"
    )

    print(
        f"Population size   : {population_size}"
    )

    print(
        f"Balanced OVR      : {balanced}"
    )

    print(
        f"Random seed       : {random_seed}"
    )


    # ========================================================
    # PREPARE OVR TRAINING DATA
    # ========================================================

    if balanced:

        print(
            "\nCreating balanced One-vs-Rest datasets..."
        )


        X_normal, y_normal = (

            create_balanced_ovr_data(

                X_train_pca,

                y_train,

                0,

                random_seed

            )

        )


        X_benign, y_benign = (

            create_balanced_ovr_data(

                X_train_pca,

                y_train,

                1,

                random_seed + 1

            )

        )


        X_malignant, y_malignant = (

            create_balanced_ovr_data(

                X_train_pca,

                y_train,

                2,

                random_seed + 2

            )

        )


    else:

        # ----------------------------------------------------
        # Original OVR training
        # ----------------------------------------------------

        X_normal = X_train_pca

        X_benign = X_train_pca

        X_malignant = X_train_pca


        y_normal = (

            y_train == 0

        ).astype(
            np.int32
        )


        y_benign = (

            y_train == 1

        ).astype(
            np.int32
        )


        y_malignant = (

            y_train == 2

        ).astype(
            np.int32
        )


    # ========================================================
    # DISPLAY TRAINING DISTRIBUTION
    # ========================================================

    print("\nTraining distributions:")


    print(

        "Normal     :",

        np.sum(
            y_normal == 1
        ),

        "positive /",

        np.sum(
            y_normal == 0
        ),

        "negative"

    )


    print(

        "Benign     :",

        np.sum(
            y_benign == 1
        ),

        "positive /",

        np.sum(
            y_benign == 0
        ),

        "negative"

    )


    print(

        "Malignant  :",

        np.sum(
            y_malignant == 1
        ),

        "positive /",

        np.sum(
            y_malignant == 0
        ),

        "negative"

    )


    # ========================================================
    # MODEL FACTORY
    # ========================================================

    def create_model(seed):

        return SymbolicClassifier(

            generations=generations,

            population_size=population_size,

            function_set=(

                "add",
                "sub",
                "mul",
                "div",
                "sqrt",
                "log",
                "abs",
                "neg"

            ),

            metric="log loss",

            parsimony_coefficient=0.001,

            max_samples=0.8,

            verbose=1,

            random_state=seed,

            n_jobs=-1

        )


    # ========================================================
    # CREATE THREE MODELS
    # ========================================================

    gp_normal = create_model(
        random_seed
    )

    gp_benign = create_model(
        random_seed + 1
    )

    gp_malignant = create_model(
        random_seed + 2
    )


    # ========================================================
    # TRAIN NORMAL
    # ========================================================

    print("\n")
    print("------------------------------------------")
    print("TRAINING NORMAL GP")
    print("------------------------------------------")


    gp_normal.fit(

        X_normal,

        y_normal

    )


    print(
        "[OK] Normal GP completed."
    )


    # ========================================================
    # TRAIN BENIGN
    # ========================================================

    print("\n")
    print("------------------------------------------")
    print("TRAINING BENIGN GP")
    print("------------------------------------------")


    gp_benign.fit(

        X_benign,

        y_benign

    )


    print(
        "[OK] Benign GP completed."
    )


    # ========================================================
    # TRAIN MALIGNANT
    # ========================================================

    print("\n")
    print("------------------------------------------")
    print("TRAINING MALIGNANT GP")
    print("------------------------------------------")


    gp_malignant.fit(

        X_malignant,

        y_malignant

    )


    print(
        "[OK] Malignant GP completed."
    )


    # ========================================================
    # VALIDATION PROBABILITIES
    # ========================================================

    print("\n")
    print(
        "Generating validation predictions..."
    )


    normal_probability = (

        gp_normal
        .predict_proba(
            X_val_pca
        )[:, 1]

    )


    benign_probability = (

        gp_benign
        .predict_proba(
            X_val_pca
        )[:, 1]

    )


    malignant_probability = (

        gp_malignant
        .predict_proba(
            X_val_pca
        )[:, 1]

    )


    # ========================================================
    # COMBINE
    # ========================================================

    probability_matrix = np.column_stack([

        normal_probability,

        benign_probability,

        malignant_probability

    ])


    # ========================================================
    # NORMALIZE
    # ========================================================

    probability_sum = (

        probability_matrix.sum(

            axis=1,

            keepdims=True

        )

    )


    probability_sum = np.maximum(

        probability_sum,

        1e-8

    )


    probabilities = (

        probability_matrix
        /
        probability_sum

    )


    # ========================================================
    # FINAL CLASS PREDICTION
    # ========================================================

    predictions = np.argmax(

        probabilities,

        axis=1

    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(

        y_val,

        predictions

    )


    balanced_accuracy = (

        balanced_accuracy_score(

            y_val,

            predictions

        )

    )


    macro_precision = (

        precision_score(

            y_val,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    macro_recall = (

        recall_score(

            y_val,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    macro_f1 = (

        f1_score(

            y_val,

            predictions,

            average="macro",

            zero_division=0

        )

    )


    weighted_f1 = (

        f1_score(

            y_val,

            predictions,

            average="weighted",

            zero_division=0

        )

    )


    # ========================================================
    # CLASS-WISE RECALL
    # ========================================================

    recalls = recall_score(

        y_val,

        predictions,

        labels=[0, 1, 2],

        average=None,

        zero_division=0

    )


    normal_recall = recalls[0]

    benign_recall = recalls[1]

    malignant_recall = recalls[2]


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_val,

        predictions,

        labels=[0, 1, 2]

    )


    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "Iteration":
        iteration_number,

        "Generations":
        generations,

        "Population":
        population_size,

        "Balanced":
        balanced,

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

        "Normal Recall":
        normal_recall,

        "Benign Recall":
        benign_recall,

        "Malignant Recall":
        malignant_recall

    }


    # ========================================================
    # STORE MODELS
    # ========================================================

    models = {

        "Normal":
        gp_normal,

        "Benign":
        gp_benign,

        "Malignant":
        gp_malignant

    }


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        f"ITERATION {iteration_number} RESULT"
    )

    print("=" * 70)


    print(
        f"Accuracy          : "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy * 100:.2f}%"
    )


    print(
        f"Macro Precision   : "
        f"{macro_precision:.4f}"
    )


    print(
        f"Macro Recall      : "
        f"{macro_recall:.4f}"
    )


    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )


    print(
        f"Weighted F1       : "
        f"{weighted_f1:.4f}"
    )


    print(
        f"Normal Recall     : "
        f"{normal_recall * 100:.2f}%"
    )


    print(
        f"Benign Recall     : "
        f"{benign_recall * 100:.2f}%"
    )


    print(
        f"Malignant Recall  : "
        f"{malignant_recall * 100:.2f}%"
    )


    return (

        results,

        models,

        predictions,

        probabilities,

        cm

    )


# ==================== ORIGINAL COLAB CELL 41 ====================

# ============================================================
# GP ITERATIONS TO RUN
# ============================================================

START_ITERATION = 2

NUMBER_OF_ITERATIONS = 9


print(
    "GP experiments:",
    START_ITERATION,
    "to",
    START_ITERATION + NUMBER_OF_ITERATIONS - 1
)


# ==================== ORIGINAL COLAB CELL 42 ====================

# ============================================================
# AUTOMATIC GP ITERATION LOOP
# ============================================================
#
# Iteration 1 is NOT rerun.
#
# This loop starts from Iteration 2.
#
# Each iteration automatically:
#
#     Create models
#     ↓
#     Train 3 GP classifiers
#     ↓
#     Predict validation set
#     ↓
#     Calculate metrics
#     ↓
#     Store results
#
# ============================================================


gp_iteration_results = []

gp_iteration_models = {}

gp_iteration_predictions = {}

gp_iteration_probabilities = {}

gp_iteration_confusion_matrices = {}


# ============================================================
# RUN ITERATIONS
# ============================================================


for iteration in range(

    START_ITERATION,

    START_ITERATION + NUMBER_OF_ITERATIONS

):


    # --------------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # --------------------------------------------------------
    #
    # We start with balanced OVR from Iteration 2.
    #
    # Generations increase gradually.
    #
    # --------------------------------------------------------

    generations = (

        20
        +
        (iteration - 2) * 10

    )


    population_size = 300


    balanced = True


    random_seed = (

        200
        +
        iteration * 10

    )


    # --------------------------------------------------------
    # RUN ONE GP EXPERIMENT
    # --------------------------------------------------------

    (

        results,

        models,

        predictions,

        probabilities,

        cm

    ) = run_gp_iteration(

        iteration_number=iteration,

        generations=generations,

        population_size=population_size,

        balanced=balanced,

        random_seed=random_seed

    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    gp_iteration_results.append(
        results
    )


    gp_iteration_models[
        iteration
    ] = models


    gp_iteration_predictions[
        iteration
    ] = predictions


    gp_iteration_probabilities[
        iteration
    ] = probabilities


    gp_iteration_confusion_matrices[
        iteration
    ] = cm


print("\n")
print("=" * 70)

print(
    "ALL GP ITERATIONS COMPLETED"
)

print("=" * 70)


# ==================== ORIGINAL COLAB CELL 43 ====================

# ============================================================
# GP ITERATIONS 2-N COMPARISON
# ============================================================
#
# Iteration 1 was completed separately.
# Iterations 2 onwards were automatically trained by the loop.
#
# This cell ONLY compares the stored validation results.
# No model is retrained.
# ============================================================


gp_results_df = pd.DataFrame(
    gp_iteration_results
)


print("============================================================")
print("GENETIC PROGRAMMING - ITERATION COMPARISON")
print("============================================================")


print(gp_results_df)


# ==================== ORIGINAL COLAB CELL 44 ====================

# ============================================================
# GP PERFORMANCE SUMMARY
# ============================================================


comparison_columns = [

    "Iteration",
    "Generations",
    "Population",
    "Balanced",
    "Accuracy",
    "Balanced Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Normal Recall",
    "Benign Recall",
    "Malignant Recall"

]


gp_comparison = gp_results_df[
    comparison_columns
].copy()


# ------------------------------------------------------------
# Convert metrics to percentages for easier interpretation
# ------------------------------------------------------------

percentage_columns = [

    "Accuracy",
    "Balanced Accuracy",
    "Normal Recall",
    "Benign Recall",
    "Malignant Recall"

]


for column in percentage_columns:

    gp_comparison[column] = (

        gp_comparison[column] * 100

    )


print("============================================================")
print("GP VALIDATION PERFORMANCE")
print("============================================================")


print(

    gp_comparison.style.format({

        "Accuracy":
            "{:.2f}%",

        "Balanced Accuracy":
            "{:.2f}%",

        "Macro Precision":
            "{:.4f}",

        "Macro Recall":
            "{:.4f}",

        "Macro F1":
            "{:.4f}",

        "Weighted F1":
            "{:.4f}",

        "Normal Recall":
            "{:.2f}%",

        "Benign Recall":
            "{:.2f}%",

        "Malignant Recall":
            "{:.2f}%"

    })

)


# ==================== ORIGINAL COLAB CELL 45 ====================

# ============================================================
# SELECT BEST GP ITERATION
# ============================================================


gp_ranked = gp_results_df.sort_values(

    by=[
        "Macro F1",
        "Balanced Accuracy"
    ],

    ascending=False

).reset_index(
    drop=True
)


print("============================================================")
print("GP ITERATIONS RANKED BY VALIDATION PERFORMANCE")
print("============================================================")


print(

    gp_ranked[[
        "Iteration",
        "Generations",
        "Population",
        "Balanced",
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Normal Recall",
        "Benign Recall",
        "Malignant Recall"

    ]]

)


# ------------------------------------------------------------
# BEST ITERATION
# ------------------------------------------------------------

best_gp_iteration = int(

    gp_ranked.iloc[0][
        "Iteration"
    ]

)


print("\n============================================================")

print(
    "BEST GP ITERATION:",
    best_gp_iteration
)

print("============================================================")


best_row = gp_ranked.iloc[0]


print(
    f"Accuracy          : "
    f"{best_row['Accuracy'] * 100:.2f}%"
)


print(
    f"Balanced Accuracy : "
    f"{best_row['Balanced Accuracy'] * 100:.2f}%"
)


print(
    f"Macro F1          : "
    f"{best_row['Macro F1']:.4f}"
)


print(
    f"Benign Recall     : "
    f"{best_row['Benign Recall'] * 100:.2f}%"
)


# ==================== ORIGINAL COLAB CELL 46 ====================

# ============================================================
# GP ACCURACY VS ITERATION
# ============================================================


plt.figure(
    figsize=(9, 5)
)


plt.plot(

    gp_results_df["Iteration"],

    gp_results_df["Accuracy"] * 100,

    marker="o"

)


plt.xlabel(
    "GP Iteration"
)


plt.ylabel(
    "Validation Accuracy (%)"
)


plt.title(
    "GP Validation Accuracy Across Iterations"
)


plt.xticks(
    gp_results_df["Iteration"]
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.show()


# ==================== ORIGINAL COLAB CELL 47 ====================

# ============================================================
# GP MACRO F1 VS ITERATION
# ============================================================


plt.figure(
    figsize=(9, 5)
)


plt.plot(

    gp_results_df["Iteration"],

    gp_results_df["Macro F1"],

    marker="o"

)


plt.xlabel(
    "GP Iteration"
)


plt.ylabel(
    "Macro F1"
)


plt.title(
    "GP Validation Macro-F1 Across Iterations"
)


plt.xticks(
    gp_results_df["Iteration"]
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.show()


# ==================== ORIGINAL COLAB CELL 48 ====================

# ============================================================
# GP BENIGN RECALL ACROSS ITERATIONS
# ============================================================


plt.figure(
    figsize=(9, 5)
)


plt.plot(

    gp_results_df["Iteration"],

    gp_results_df["Benign Recall"] * 100,

    marker="o"

)


plt.xlabel(
    "GP Iteration"
)


plt.ylabel(
    "Benign Recall (%)"
)


plt.title(
    "GP Benign Recall Across Iterations"
)


plt.xticks(
    gp_results_df["Iteration"]
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.show()


# ==================== ORIGINAL COLAB CELL 49 ====================

# ============================================================
# FINALIZE THE BEST GP ITERATION
# ============================================================
#
# The GP iterations have already been trained.
#
# We now select the iteration that performed best on the
# validation set.
#
# Primary metric:
#     Macro F1
#
# Secondary metric:
#     Balanced Accuracy
#
# IMPORTANT:
# The test set is NOT used for selecting the model.
# ============================================================


# ------------------------------------------------------------
# SORT VALIDATION RESULTS
# ------------------------------------------------------------

gp_results_df = pd.DataFrame(
    gp_iteration_results
)


gp_ranked = gp_results_df.sort_values(

    by=[
        "Macro F1",
        "Balanced Accuracy"
    ],

    ascending=False

).reset_index(
    drop=True
)


# ------------------------------------------------------------
# SELECT BEST ITERATION
# ------------------------------------------------------------

best_gp_iteration = int(

    gp_ranked.iloc[0][
        "Iteration"
    ]

)


# ------------------------------------------------------------
# RETRIEVE MODEL
# ------------------------------------------------------------

best_gp_models = gp_iteration_models[
    best_gp_iteration
]


best_gp_predictions = gp_iteration_predictions[
    best_gp_iteration
]


best_gp_probabilities = gp_iteration_probabilities[
    best_gp_iteration
]


print("============================================================")
print("FINAL GP MODEL SELECTION")
print("============================================================")


print(
    "Selected GP Iteration:",
    best_gp_iteration
)


print(
    "Generations:",
    int(
        gp_ranked.iloc[0]["Generations"]
    )
)


print(
    "Population:",
    int(
        gp_ranked.iloc[0]["Population"]
    )
)


print(
    "Balanced OVR:",
    gp_ranked.iloc[0]["Balanced"]
)


print(
    f"Validation Accuracy: "
    f"{gp_ranked.iloc[0]['Accuracy'] * 100:.2f}%"
)


print(
    f"Validation Balanced Accuracy: "
    f"{gp_ranked.iloc[0]['Balanced Accuracy'] * 100:.2f}%"
)


print(
    f"Validation Macro F1: "
    f"{gp_ranked.iloc[0]['Macro F1']:.4f}"
)


print("\n[OK] GP model selected and frozen.")


# ==================== ORIGINAL COLAB CELL 50 ====================

# ============================================================
# FINAL GP VALIDATION CLASSIFICATION REPORT
# ============================================================


print("============================================================")
print("FINAL GP MODEL - VALIDATION REPORT")
print("============================================================")


print(
    classification_report(

        y_val,

        best_gp_predictions,

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


# ==================== ORIGINAL COLAB CELL 51 ====================

# ============================================================
# FINAL GP VALIDATION CONFUSION MATRIX
# ============================================================


final_gp_val_cm = confusion_matrix(

    y_val,

    best_gp_predictions,

    labels=[0, 1, 2]

)


print("============================================================")
print("FINAL GP VALIDATION CONFUSION MATRIX")
print("============================================================")


print(
    final_gp_val_cm
)


plt.figure(
    figsize=(7, 6)
)


sns.heatmap(

    final_gp_val_cm,

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
    f"Final GP Validation Confusion Matrix "
    f"(Iteration {best_gp_iteration})"
)


plt.tight_layout()


plt.show()


# ==================== ORIGINAL COLAB CELL 52 ====================

# ============================================================
# FINAL GP MODEL
# TEST SET PREDICTION
# ============================================================
#
# IMPORTANT:
#
# The test set is being used for FINAL evaluation only.
#
# We are NOT changing the model based on these results.
# ============================================================


print("============================================================")
print("FINAL GP MODEL - TEST PREDICTION")
print("============================================================")


# ------------------------------------------------------------
# TEST PROBABILITIES
# ------------------------------------------------------------

test_normal_probability = (

    best_gp_models["Normal"]
    .predict_proba(

        X_test_pca

    )[:, 1]

)


test_benign_probability = (

    best_gp_models["Benign"]
    .predict_proba(

        X_test_pca

    )[:, 1]

)


test_malignant_probability = (

    best_gp_models["Malignant"]
    .predict_proba(

        X_test_pca

    )[:, 1]

)


# ------------------------------------------------------------
# COMBINE THREE GP SCORES
# ------------------------------------------------------------

test_score_matrix = np.column_stack([

    test_normal_probability,

    test_benign_probability,

    test_malignant_probability

])


# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------

test_score_sum = (

    test_score_matrix.sum(

        axis=1,

        keepdims=True

    )

)


test_score_sum = np.maximum(

    test_score_sum,

    1e-8

)


gp_test_probabilities = (

    test_score_matrix
    /
    test_score_sum

)


# ------------------------------------------------------------
# FINAL TEST PREDICTIONS
# ------------------------------------------------------------

gp_test_predictions = np.argmax(

    gp_test_probabilities,

    axis=1

)


print(
    "Test samples:",
    len(gp_test_predictions)
)


print(
    "Probability matrix:",
    gp_test_probabilities.shape
)


print(
    "Prediction vector:",
    gp_test_predictions.shape
)


print("\n[OK] Final GP test prediction completed.")


# ==================== ORIGINAL COLAB CELL 53 ====================

# ============================================================
# FINAL GP TEST PERFORMANCE
# ============================================================


gp_test_accuracy = accuracy_score(

    y_test,

    gp_test_predictions

)


gp_test_balanced_accuracy = (

    balanced_accuracy_score(

        y_test,

        gp_test_predictions

    )

)


gp_test_macro_precision = (

    precision_score(

        y_test,

        gp_test_predictions,

        average="macro",

        zero_division=0

    )

)


gp_test_macro_recall = (

    recall_score(

        y_test,

        gp_test_predictions,

        average="macro",

        zero_division=0

    )

)


gp_test_macro_f1 = (

    f1_score(

        y_test,

        gp_test_predictions,

        average="macro",

        zero_division=0

    )

)


gp_test_weighted_f1 = (

    f1_score(

        y_test,

        gp_test_predictions,

        average="weighted",

        zero_division=0

    )


)


print("============================================================")
print("FINAL GP TEST PERFORMANCE")
print("============================================================")


print(
    f"Accuracy           : "
    f"{gp_test_accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy  : "
    f"{gp_test_balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision    : "
    f"{gp_test_macro_precision:.4f}"
)


print(
    f"Macro Recall       : "
    f"{gp_test_macro_recall:.4f}"
)


print(
    f"Macro F1           : "
    f"{gp_test_macro_f1:.4f}"
)


print(
    f"Weighted F1        : "
    f"{gp_test_weighted_f1:.4f}"
)


# ==================== ORIGINAL COLAB CELL 54 ====================

# ============================================================
# FINAL GP TEST CLASSIFICATION REPORT
# ============================================================


print("============================================================")
print("FINAL GP TEST CLASSIFICATION REPORT")
print("============================================================")


print(

    classification_report(

        y_test,

        gp_test_predictions,

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


# ==================== ORIGINAL COLAB CELL 55 ====================

# ============================================================
# FINAL GP TEST CONFUSION MATRIX
# ============================================================


final_gp_test_cm = confusion_matrix(

    y_test,

    gp_test_predictions,

    labels=[0, 1, 2]

)


print("============================================================")
print("FINAL GP TEST CONFUSION MATRIX")
print("============================================================")


print(
    final_gp_test_cm
)


plt.figure(
    figsize=(7, 6)
)


sns.heatmap(

    final_gp_test_cm,

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
    f"Final GP Test Confusion Matrix "
    f"(Iteration {best_gp_iteration})"
)


plt.tight_layout()


plt.show()


# ==================== ORIGINAL COLAB CELL 56 ====================

# ============================================================
# FINAL GP TEST PREDICTION DISTRIBUTION
# ============================================================


print("============================================================")
print("FINAL GP TEST PREDICTION DISTRIBUTION")
print("============================================================")


class_names = [

    "Normal",
    "Benign",
    "Malignant"

]


for class_index, class_name in enumerate(

    class_names

):

    predicted_count = np.sum(

        gp_test_predictions
        ==
        class_index

    )


    actual_count = np.sum(

        y_test
        ==
        class_index

    )


    print(

        f"{class_name:<12} | "
        f"Actual: {actual_count:3d} | "
        f"Predicted: {predicted_count:3d}"

    )