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

# Exact Hybrid training/evaluation cells.

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