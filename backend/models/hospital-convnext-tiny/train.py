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

# Exact ConvNeXt/Hybrid training section from the Hospital Hybrid Colab.

# ============================================================
# SOURCE COLAB CELL 64
# ============================================================

# ============================================================
# CELL 1 — CLEAN SETUP & REPRODUCIBILITY
# ============================================================

import os
import random
import copy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from PIL import Image
from pathlib import Path

from torchvision import transforms
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")

print("PyTorch:", torch.__version__)
print("Device :", device)
print("Seed   :", SEED)


# ============================================================
# SOURCE COLAB CELL 65
# ============================================================

# ============================================================
# CELL 2 — CONVNEXT-TINY
# ============================================================

weights = ConvNeXt_Tiny_Weights.DEFAULT

convnext = convnext_tiny(
    weights=weights
)

# Remove original ImageNet classifier
convnext.classifier = nn.Identity()

convnext = convnext.to(device)
convnext.eval()

print("ConvNeXt-Tiny loaded successfully.")
print("Classifier:", convnext.classifier)


# ============================================================
# SOURCE COLAB CELL 66
# ============================================================

# ============================================================
# CELL 3 — VERIFY CONVNEXT OUTPUT
# ============================================================

dummy_input = torch.randn(
    1, 3, 224, 224
).to(device)

with torch.no_grad():

    dummy_output = convnext(
        dummy_input
    )

print(
    "Raw ConvNeXt output shape:",
    dummy_output.shape
)

dummy_features = torch.flatten(
    dummy_output,
    1
)

print(
    "Flattened feature shape:",
    dummy_features.shape
)


# ============================================================
# SOURCE COLAB CELL 67
# ============================================================

# ============================================================
# CELL 4 — CONVNEXT TRANSFORM
# ============================================================

convnext_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("ConvNeXt transform ready.")


# ============================================================
# SOURCE COLAB CELL 68
# ============================================================

# ============================================================
# CELL 5 — LUNG SEGMENTATION + PREPROCESSING FOR CONVNEXT
# ============================================================

def prepare_convnext_image(
    image_input,
    target_size=(224, 224)
):

    # --------------------------------------------------------
    # 1. LUNG-FIELD SEGMENTATION
    # --------------------------------------------------------

    segmentation_result = segment_lung(
        image_input
    )

    # segmentation_result[3]
    # = final lung ROI
    lung_roi = segmentation_result[3]

    lung_roi = np.asarray(
        lung_roi,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 2. RESIZE
    # --------------------------------------------------------

    image = cv2.resize(
        lung_roi,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 3. MEDIAN FILTERING
    # --------------------------------------------------------

    denoised = cv2.medianBlur(
        image,
        3
    )

    # --------------------------------------------------------
    # 4. CLAHE CONTRAST ENHANCEMENT
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoised
    )

    # --------------------------------------------------------
    # 5. NORMALIZATION
    # --------------------------------------------------------

    normalized = (
        enhanced.astype(np.float32) / 255.0
    )

    return normalized


print("ConvNeXt preprocessing function ready.")


# ============================================================
# SOURCE COLAB CELL 69
# ============================================================

# ============================================================
# CELL 6 — VERIFY CONVNEXT PREPROCESSING
# ============================================================

# Select one training image

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
        train_df, val_df, internal_test_df = load_dataset_splits('hospital')
        print(f"Dataset splits loaded successfully for hospital-convnext-tiny: Train={len(train_df)}, Val={len(val_df)}, Test={len(internal_test_df)}")
    except Exception as e:
        print(f"Dataset split auto-load note: {e}")
sample_image_path = train_df.iloc[0]["filepath"]

print("Sample image:")
print(sample_image_path)

# Read image in grayscale
sample_image = cv2.imread(
    str(sample_image_path),
    cv2.IMREAD_GRAYSCALE
)

if sample_image is None:
    raise ValueError(
        f"Could not read image: {sample_image_path}"
    )

# Apply finalized preprocessing
sample_processed = prepare_convnext_image(
    sample_image,
    target_size=(224, 224)
)

# ------------------------------------------------------------
# Display original and processed image
# ------------------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(
    sample_image,
    cmap="gray"
)
plt.title("Original CT Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    sample_processed,
    cmap="gray"
)
plt.title("Preprocessed Lung ROI")
plt.axis("off")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Verify shape and pixel range
# ------------------------------------------------------------

print("Original image shape:", sample_image.shape)
print("Processed image shape:", sample_processed.shape)
print("Processed image dtype:", sample_processed.dtype)
print("Minimum pixel value:", sample_processed.min())
print("Maximum pixel value:", sample_processed.max())


# ============================================================
# SOURCE COLAB CELL 70
# ============================================================

# ============================================================
# CELL 7 — CONVNEXT-TINY FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_convnext_features(image_paths):

    all_features = []

    # Keep ConvNeXt in evaluation mode
    convnext.eval()

    # Feature extraction does not require gradients
    with torch.no_grad():

        for image_path in image_paths:

            # ------------------------------------------------
            # 1. Read CT image
            # ------------------------------------------------

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                raise ValueError(
                    f"Could not read image: {image_path}"
                )

            # ------------------------------------------------
            # 2. Apply finalized lung preprocessing
            # ------------------------------------------------

            normalized = prepare_convnext_image(
                image,
                target_size=(224, 224)
            )

            # ------------------------------------------------
            # 3. Convert normalized image to uint8
            # ------------------------------------------------

            processed = (
                normalized * 255.0
            ).clip(
                0, 255
            ).astype(
                np.uint8
            )

            # ------------------------------------------------
            # 4. Convert grayscale CT to RGB
            # ------------------------------------------------

            rgb_image = cv2.cvtColor(
                processed,
                cv2.COLOR_GRAY2RGB
            )

            # ------------------------------------------------
            # 5. Convert NumPy array to PIL image
            # ------------------------------------------------

            pil_image = Image.fromarray(
                rgb_image
            )

            # ------------------------------------------------
            # 6. Apply ConvNeXt transformation
            # ------------------------------------------------

            tensor = convnext_transform(
                pil_image
            )

            # Add batch dimension
            tensor = tensor.unsqueeze(0)

            # Move tensor to CPU
            tensor = tensor.to(device)

            # ------------------------------------------------
            # 7. Extract ConvNeXt features
            # ------------------------------------------------

            features = convnext(tensor)

            # ConvNeXt output:
            # [1, 768, 1, 1]

            # Flatten:
            # [1, 768]

            features = torch.flatten(
                features,
                start_dim=1
            )

            # ------------------------------------------------
            # 8. Store the 768-dimensional feature vector
            # ------------------------------------------------

            all_features.append(
                features.cpu().numpy()[0]
            )

    # --------------------------------------------------------
    # 9. Convert all features into NumPy matrix
    # --------------------------------------------------------

    return np.asarray(
        all_features,
        dtype=np.float32
    )


print("ConvNeXt feature extraction function ready.")


# ============================================================
# SOURCE COLAB CELL 71
# ============================================================

# ============================================================
# CELL 8 — EXTRACT CONVNEXT FEATURES FROM TRAINING SET
# ============================================================

# Get training image paths from train_df
train_image_paths = train_df["filepath"].tolist()

print("Number of training images:", len(train_image_paths))

# ------------------------------------------------------------
# Extract ConvNeXt-Tiny features
# ------------------------------------------------------------

X_train_deep = extract_convnext_features(
    train_image_paths
)

# ------------------------------------------------------------
# Verify extracted feature matrix
# ------------------------------------------------------------

print("\nConvNeXt training feature extraction completed.")
print("X_train_deep shape:", X_train_deep.shape)
print("Feature data type:", X_train_deep.dtype)


# ============================================================
# SOURCE COLAB CELL 72
# ============================================================

# ============================================================
# CELL 9 — EXTRACT CONVNEXT FEATURES FROM VALIDATION SET
# ============================================================

# Get validation image paths from val_df
val_image_paths = val_df["filepath"].tolist()

print("Number of validation images:", len(val_image_paths))

# ------------------------------------------------------------
# Extract ConvNeXt-Tiny features
# ------------------------------------------------------------

X_val_deep = extract_convnext_features(
    val_image_paths
)

# ------------------------------------------------------------
# Verify extracted feature matrix
# ------------------------------------------------------------

print("\nConvNeXt validation feature extraction completed.")
print("X_val_deep shape:", X_val_deep.shape)
print("Feature data type:", X_val_deep.dtype)


# ============================================================
# SOURCE COLAB CELL 73
# ============================================================

# ============================================================
# CELL 10 — EXTRACT CONVNEXT FEATURES FROM INTERNAL TEST SET
# ============================================================

# Get internal test image paths from internal_test_df
test_image_paths = internal_test_df["filepath"].tolist()

print("Number of internal test images:", len(test_image_paths))

# ------------------------------------------------------------
# Extract ConvNeXt-Tiny features
# ------------------------------------------------------------

X_test_deep = extract_convnext_features(
    test_image_paths
)

# ------------------------------------------------------------
# Verify extracted feature matrix
# ------------------------------------------------------------

print("\nConvNeXt internal test feature extraction completed.")
print("X_test_deep shape:", X_test_deep.shape)
print("Feature data type:", X_test_deep.dtype)


# ============================================================
# SOURCE COLAB CELL 74
# ============================================================

# ============================================================
# CELL 11 — STANDARDIZE CONVNEXT DEEP FEATURES
# ============================================================

# Create scaler
deep_scaler = StandardScaler()

# ------------------------------------------------------------
# Fit ONLY on training features
# ------------------------------------------------------------

X_train_deep_scaled = deep_scaler.fit_transform(
    X_train_deep
)

# ------------------------------------------------------------
# Transform validation features
# ------------------------------------------------------------

X_val_deep_scaled = deep_scaler.transform(
    X_val_deep
)

# ------------------------------------------------------------
# Transform internal test features
# ------------------------------------------------------------

X_test_deep_scaled = deep_scaler.transform(
    X_test_deep
)

# ------------------------------------------------------------
# Verify shapes
# ------------------------------------------------------------

print("Deep feature standardization completed.\n")

print(
    "X_train_deep_scaled shape:",
    X_train_deep_scaled.shape
)

print(
    "X_val_deep_scaled shape:",
    X_val_deep_scaled.shape
)

print(
    "X_test_deep_scaled shape:",
    X_test_deep_scaled.shape
)

print(
    "\nTraining feature mean:",
    np.mean(X_train_deep_scaled)
)

print(
    "Training feature standard deviation:",
    np.std(X_train_deep_scaled)
)


# ============================================================
# SOURCE COLAB CELL 75
# ============================================================

# ============================================================
# CELL 12 — PCA FOR CONVNEXT DEEP FEATURES
# ============================================================

# Create PCA
deep_pca = PCA(
    n_components=32,
    random_state=SEED
)

# ------------------------------------------------------------
# Fit PCA ONLY on training data
# ------------------------------------------------------------

X_train_deep_reduced = deep_pca.fit_transform(
    X_train_deep_scaled
)

# ------------------------------------------------------------
# Transform validation data
# ------------------------------------------------------------

X_val_deep_reduced = deep_pca.transform(
    X_val_deep_scaled
)

# ------------------------------------------------------------
# Transform internal test data
# ------------------------------------------------------------

X_test_deep_reduced = deep_pca.transform(
    X_test_deep_scaled
)

# ------------------------------------------------------------
# Calculate explained variance
# ------------------------------------------------------------

explained_variance = (
    deep_pca.explained_variance_ratio_.sum()
    * 100
)

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("ConvNeXt PCA completed.\n")

print(
    "X_train_deep_reduced shape:",
    X_train_deep_reduced.shape
)

print(
    "X_val_deep_reduced shape:",
    X_val_deep_reduced.shape
)

print(
    "X_test_deep_reduced shape:",
    X_test_deep_reduced.shape
)

print(
    f"\nExplained variance by 32 PCA components: "
    f"{explained_variance:.2f}%"
)


# ============================================================
# SOURCE COLAB CELL 76
# ============================================================

# ============================================================
# CELL 13 — FUSE HANDCRAFTED + CONVNEXT FEATURES
# ============================================================

# ------------------------------------------------------------
# Verify handcrafted feature dimensions
# ------------------------------------------------------------

print("Handcrafted feature shapes:")
print(
    "Training:",
    X_train_final.shape
)

print(
    "Validation:",
    X_val_final.shape
)

print(
    "Internal Test:",
    X_test_final.shape
)

# ------------------------------------------------------------
# Verify ConvNeXt PCA feature dimensions
# ------------------------------------------------------------

print("\nConvNeXt PCA feature shapes:")
print(
    "Training:",
    X_train_deep_reduced.shape
)

print(
    "Validation:",
    X_val_deep_reduced.shape
)

print(
    "Internal Test:",
    X_test_deep_reduced.shape
)

# ------------------------------------------------------------
# Fuse the two feature branches
# ------------------------------------------------------------

X_train_fused = np.hstack([
    X_train_final,
    X_train_deep_reduced
])

X_val_fused = np.hstack([
    X_val_final,
    X_val_deep_reduced
])

X_test_fused = np.hstack([
    X_test_final,
    X_test_deep_reduced
])

# ------------------------------------------------------------
# Verify final fused feature matrices
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FUSION COMPLETED")
print("=" * 60)

print(
    "X_train_fused shape:",
    X_train_fused.shape
)

print(
    "X_val_fused shape:",
    X_val_fused.shape
)

print(
    "X_test_fused shape:",
    X_test_fused.shape
)

print(
    "\nTotal fused features:",
    X_train_fused.shape[1]
)


# ============================================================
# SOURCE COLAB CELL 77
# ============================================================

# ============================================================
# CELL 14 — PREPARE CLASS LABELS
# ============================================================

# ------------------------------------------------------------
# Class mapping
# ------------------------------------------------------------

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]

# ------------------------------------------------------------
# Extract labels from the corresponding DataFrames
# ------------------------------------------------------------

y_train = train_df["label"].to_numpy(dtype=np.int64)

y_val = val_df["label"].to_numpy(dtype=np.int64)

y_test = internal_test_df["label"].to_numpy(dtype=np.int64)

# ------------------------------------------------------------
# Verify label dimensions
# ------------------------------------------------------------

print("Label preparation completed.\n")

print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)

# ------------------------------------------------------------
# Display class distributions
# ------------------------------------------------------------

print("\nTraining class distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):

    count = np.sum(y_train == class_id)

    print(
        f"{class_name}: {count}"
    )


print("\nValidation class distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):

    count = np.sum(y_val == class_id)

    print(
        f"{class_name}: {count}"
    )


print("\nInternal Test class distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):

    count = np.sum(y_test == class_id)

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# SOURCE COLAB CELL 78
# ============================================================

# ============================================================
# CELL 15 — CREATE PYTORCH DATASETS AND DATALOADERS
# ============================================================

from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------
# Convert NumPy arrays to PyTorch tensors
# ------------------------------------------------------------

X_train_tensor = torch.tensor(
    X_train_fused,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_fused,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_fused,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)

# ------------------------------------------------------------
# Create TensorDatasets
# ------------------------------------------------------------

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

# ------------------------------------------------------------
# Create deterministic DataLoader generator
# ------------------------------------------------------------

loader_generator = torch.Generator()

loader_generator.manual_seed(SEED)

# ------------------------------------------------------------
# Create DataLoaders
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    generator=loader_generator,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

print("PyTorch datasets and DataLoaders created.\n")

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Internal test samples:", len(test_dataset))

print("\nFeature dimension:", X_train_tensor.shape[1])

print("\nBatch size:", 8)
print("Number of training batches:", len(train_loader))
print("Number of validation batches:", len(val_loader))
print("Number of test batches:", len(test_loader))


# ============================================================
# SOURCE COLAB CELL 79
# ============================================================

# ============================================================
# CELL 16 — DEFINE HYBRID CLASSIFIER
# ============================================================

class HybridClassifier(nn.Module):

    def __init__(
        self,
        input_dim=82,
        num_classes=3
    ):

        super().__init__()

        self.network = nn.Sequential(

            # ------------------------------------------------
            # Layer 1
            # 82 → 64
            # ------------------------------------------------
            nn.Linear(
                input_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.25
            ),

            # ------------------------------------------------
            # Layer 2
            # 64 → 32
            # ------------------------------------------------
            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            # ------------------------------------------------
            # Output Layer
            # 32 → 3
            # ------------------------------------------------
            nn.Linear(
                32,
                num_classes
            )
        )

    def forward(self, x):

        return self.network(x)


# ------------------------------------------------------------
# Create model
# ------------------------------------------------------------

model = HybridClassifier(
    input_dim=82,
    num_classes=3
).to(device)

# ------------------------------------------------------------
# Display architecture
# ------------------------------------------------------------

print(model)


# ============================================================
# SOURCE COLAB CELL 80
# ============================================================

# ============================================================
# CELL 17 — REPRODUCIBLE TRAINING SETUP
# ============================================================

# ------------------------------------------------------------
# Reset all random seeds before training
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# Create the model AFTER setting the seed
# ------------------------------------------------------------

model = HybridClassifier(
    input_dim=82,
    num_classes=3
).to(device)

# ------------------------------------------------------------
# Loss function
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

NUM_EPOCHS = 50

# ------------------------------------------------------------
# Best-model checkpoint variables
# ------------------------------------------------------------

best_val_accuracy = -1.0
best_epoch = 0
best_model_state = None

# ------------------------------------------------------------
# Training history
# ------------------------------------------------------------

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

# ------------------------------------------------------------
# Display configuration
# ------------------------------------------------------------

print("Reproducible training setup completed.\n")

print("Device:", device)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", 0.001)
print("Weight decay:", 0.01)
print("Batch size:", 8)
print("Loss function:", criterion.__class__.__name__)
print("Optimizer:", optimizer.__class__.__name__)
print("Random seed:", SEED)


# ============================================================
# SOURCE COLAB CELL 81
# ============================================================

# ============================================================
# CELL 18 — TRAIN HYBRID CLASSIFIER
# ============================================================

print("Starting Hybrid Classifier training...\n")

for epoch in range(NUM_EPOCHS):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_features, batch_labels in train_loader:

        # Move data to device
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_features)

        # Calculate loss
        loss = criterion(
            outputs,
            batch_labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # ----------------------------------------------------
        # Training statistics
        # ----------------------------------------------------

        running_train_loss += (
            loss.item() * batch_labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct_train += (
            (predictions == batch_labels)
            .sum()
            .item()
        )

        total_train += batch_labels.size(0)

    # Calculate epoch training metrics
    epoch_train_loss = (
        running_train_loss / total_train
    )

    epoch_train_accuracy = (
        correct_train / total_train
    ) * 100

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for batch_features, batch_labels in val_loader:

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = model(
                batch_features
            )

            # Validation loss
            loss = criterion(
                outputs,
                batch_labels
            )

            running_val_loss += (
                loss.item() * batch_labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_val += (
                (predictions == batch_labels)
                .sum()
                .item()
            )

            total_val += batch_labels.size(0)

    # Calculate validation metrics
    epoch_val_loss = (
        running_val_loss / total_val
    )

    epoch_val_accuracy = (
        correct_val / total_val
    ) * 100

    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    train_losses.append(
        epoch_train_loss
    )

    val_losses.append(
        epoch_val_loss
    )

    train_accuracies.append(
        epoch_train_accuracy
    )

    val_accuracies.append(
        epoch_val_accuracy
    )

    # ========================================================
    # SAVE BEST VALIDATION MODEL
    # ========================================================

    if epoch_val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            epoch_val_accuracy
        )

        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Train Acc: {epoch_train_accuracy:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Val Acc: {epoch_val_accuracy:.2f}%"
    )

# ============================================================
# RESTORE BEST VALIDATION MODEL
# ============================================================

if best_model_state is None:
    raise RuntimeError(
        "No best model checkpoint was created."
    )

model.load_state_dict(
    best_model_state
)

model.eval()

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Epoch: "
    f"{best_epoch}"
)

print(
    "\nBest validation model restored."
)


# ============================================================
# SOURCE COLAB CELL 82
# ============================================================

# ============================================================
# CELL 19 — EVALUATE BEST MODEL ON INTERNAL TEST SET
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# Make sure the best validation model is being used
# ------------------------------------------------------------

model.load_state_dict(
    best_model_state
)

model.eval()

# ------------------------------------------------------------
# Store predictions and actual labels
# ------------------------------------------------------------

test_predictions = []
test_actual = []

# ------------------------------------------------------------
# Evaluate WITHOUT gradients
# ------------------------------------------------------------

with torch.no_grad():

    for batch_features, batch_labels in test_loader:

        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # Forward pass
        outputs = model(
            batch_features
        )

        # Predicted class
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Store predictions
        test_predictions.extend(
            predictions.cpu().numpy()
        )

        # Store actual labels
        test_actual.extend(
            batch_labels.cpu().numpy()
        )

# Convert to NumPy arrays
test_predictions = np.asarray(
    test_predictions
)

test_actual = np.asarray(
    test_actual
)

# ============================================================
# CALCULATE METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_actual,
    test_predictions
) * 100

test_balanced_accuracy = (
    balanced_accuracy_score(
        test_actual,
        test_predictions
    ) * 100
)

test_macro_precision = (
    precision_score(
        test_actual,
        test_predictions,
        average="macro",
        zero_division=0
    ) * 100
)

test_macro_recall = (
    recall_score(
        test_actual,
        test_predictions,
        average="macro",
        zero_division=0
    ) * 100
)

test_macro_f1 = (
    f1_score(
        test_actual,
        test_predictions,
        average="macro",
        zero_division=0
    ) * 100
)

test_weighted_f1 = (
    f1_score(
        test_actual,
        test_predictions,
        average="weighted",
        zero_division=0
    ) * 100
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("INTERNAL TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy:             {test_accuracy:.2f}%"
)

print(
    f"Balanced Accuracy:    {test_balanced_accuracy:.2f}%"
)

print(
    f"Macro Precision:      {test_macro_precision:.2f}%"
)

print(
    f"Macro Recall:         {test_macro_recall:.2f}%"
)

print(
    f"Macro F1:             {test_macro_f1:.2f}%"
)

print(
    f"Weighted F1:          {test_weighted_f1:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        test_actual,
        test_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_actual,
    test_predictions,
    labels=[0, 1, 2]
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# ============================================================
# IMAGE-WISE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("IMAGE-WISE PREDICTIONS")
print("=" * 60)

for i, (actual, predicted) in enumerate(
    zip(test_actual, test_predictions)
):

    actual_name = CLASS_NAMES[actual]
    predicted_name = CLASS_NAMES[predicted]

    status = (
        "CORRECT"
        if actual == predicted
        else "WRONG"
    )

    print(
        f"{i + 1:02d}. "
        f"Actual: {actual_name:<10} | "
        f"Predicted: {predicted_name:<10} | "
        f"{status}"
    )


# ============================================================
# SOURCE COLAB CELL 83
# ============================================================

# ============================================================
# CELL 20 — SAVE FINAL FROZEN HYBRID MODEL
# ============================================================

import joblib
from pathlib import Path


# ------------------------------------------------------------
# Create model directory
# ------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Make sure the best validation model is loaded
# ------------------------------------------------------------

model.load_state_dict(
    best_model_state
)

model.eval()

# ------------------------------------------------------------
# Save Hybrid Classifier
# ------------------------------------------------------------

hybrid_model_path = (
    MODEL_DIR / "hybrid_classifier_final.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": 82,
        "num_classes": 3,
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "internal_test_accuracy": test_accuracy,
        "internal_test_balanced_accuracy": test_balanced_accuracy,
        "internal_test_macro_precision": test_macro_precision,
        "internal_test_macro_recall": test_macro_recall,
        "internal_test_macro_f1": test_macro_f1,
        "internal_test_weighted_f1": test_weighted_f1
    },
    hybrid_model_path
)

# ------------------------------------------------------------
# Save ConvNeXt feature scaler
# ------------------------------------------------------------

deep_scaler_path = (
    MODEL_DIR / "convnext_deep_scaler.joblib"
)

joblib.dump(
    deep_scaler,
    deep_scaler_path
)

# ------------------------------------------------------------
# Save ConvNeXt PCA
# ------------------------------------------------------------

deep_pca_path = (
    MODEL_DIR / "convnext_deep_pca.joblib"
)

joblib.dump(
    deep_pca,
    deep_pca_path
)

# ------------------------------------------------------------
# Save evaluation results
# ------------------------------------------------------------

evaluation_results = {
    "accuracy": test_accuracy,
    "balanced_accuracy": test_balanced_accuracy,
    "macro_precision": test_macro_precision,
    "macro_recall": test_macro_recall,
    "macro_f1": test_macro_f1,
    "weighted_f1": test_weighted_f1,
    "best_epoch": best_epoch,
    "best_validation_accuracy": best_val_accuracy
}

evaluation_path = (
    MODEL_DIR / "evaluation_results.joblib"
)

joblib.dump(
    evaluation_results,
    evaluation_path
)

# ------------------------------------------------------------
# Display saved files
# ------------------------------------------------------------

print("=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)

print("\nModel directory:")
print(MODEL_DIR)

print("\nSaved files:")

print(
    "1.",
    hybrid_model_path.name
)

print(
    "2.",
    deep_scaler_path.name
)

print(
    "3.",
    deep_pca_path.name
)

print(
    "4.",
    evaluation_path.name
)

print("\nFinal model information:")
print("Input features:", 82)
print("Classes:", CLASS_NAMES)
print("Best epoch:", best_epoch)
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)
print(
    f"Internal test accuracy: "
    f"{test_accuracy:.2f}%"
)


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


# ============================================================
# SOURCE COLAB CELL 88
# ============================================================

# ============================================================
# CELL 25 — INSPECT CONVNEXT FEATURE LAYERS FOR GRAD-CAM
# ============================================================

print("=" * 60)
print("CONVNEXT-TINY FEATURE LAYERS")
print("=" * 60)

for index, layer in enumerate(convnext.features):

    print(
        f"\nIndex {index}:"
    )

    print(
        layer
    )


# ============================================================
# SOURCE COLAB CELL 89
# ============================================================

# ============================================================
# CELL 26 — SELECT CONVNEXT LAYER FOR GRAD-CAM
# ============================================================

# ------------------------------------------------------------
# Select the final spatial ConvNeXt stage
# ------------------------------------------------------------

gradcam_target_layer = convnext.features[7]

print("=" * 60)
print("GRAD-CAM TARGET LAYER")
print("=" * 60)

print(
    "Target layer:"
)

print(
    gradcam_target_layer
)

print(
    "\nTarget layer index: convnext.features[7]"
)

print(
    "\nThis layer produces spatial feature maps "
    "before the final global pooling."
)