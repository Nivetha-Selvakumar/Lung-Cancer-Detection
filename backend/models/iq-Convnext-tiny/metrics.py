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
criterion = None
optimizer = None
optim = None
scheduler = None
time = None
FocalLoss = None
plt = None
sns = None
DEVICE = None

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