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

# Exact Hospital ConvNeXt prediction, Grad-CAM, upload, Gemini, and result-saving cells.

# ============================================================
# SOURCE COLAB CELL 91
# ============================================================

# ============================================================
# GRAD-CAM SETUP
# ============================================================

# !pip install -q grad-cam


# ============================================================
# SOURCE COLAB CELL 92
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
import importlib

GradCAM = None
ClassifierOutputTarget = None
show_cam_on_image = None

try:
    _pgc = importlib.import_module("pytorch_grad_cam")
    GradCAM = getattr(_pgc, "GradCAM", None)
    _mt = importlib.import_module("pytorch_grad_cam.utils.model_targets")
    ClassifierOutputTarget = getattr(_mt, "ClassifierOutputTarget", None)
    _img = importlib.import_module("pytorch_grad_cam.utils.image")
    show_cam_on_image = getattr(_img, "show_cam_on_image", None)
except Exception:
    pass

print("Grad-CAM setup complete.")


# ============================================================
# SOURCE COLAB CELL 93
# ============================================================

# ============================================================
# CELL 27 — GRAD-CAM HYBRID WRAPPER
# ============================================================

class HybridGradCAMModel(nn.Module):

    def __init__(
        self,
        convnext_model,
        hybrid_model,
        deep_scaler,
        deep_pca,
        handcrafted_features
    ):

        super().__init__()

        # ----------------------------------------------------
        # Frozen ConvNeXt feature extractor
        # ----------------------------------------------------

        self.convnext = convnext_model

        # ----------------------------------------------------
        # Frozen final Hybrid classifier
        # ----------------------------------------------------

        self.hybrid = hybrid_model

        # ----------------------------------------------------
        # Pre-fitted ConvNeXt scaler and PCA
        # ----------------------------------------------------

        self.deep_scaler = deep_scaler
        self.deep_pca = deep_pca

        # ----------------------------------------------------
        # 50 handcrafted features for the selected image
        # ----------------------------------------------------

        self.handcrafted_features = handcrafted_features

    def forward(self, x):

        # ====================================================
        # 1. Pass through ConvNeXt feature extractor
        # ====================================================

        conv_features = self.convnext.features(x)

        # ----------------------------------------------------
        # Save spatial feature maps for Grad-CAM
        #
        # Shape:
        # [B, 768, H, W]
        # ----------------------------------------------------

        self.spatial_features = conv_features

        # ====================================================
        # 2. ConvNeXt final normalization
        # ====================================================

        conv_features = self.convnext.avgpool(
            conv_features
        )

        # ====================================================
        # 3. Flatten
        # ====================================================

        conv_features = torch.flatten(
            conv_features,
            1
        )

        # Shape:
        # [B, 768]

        # ====================================================
        # 4. Convert deep features to NumPy
        # ====================================================

        deep_numpy = (
            conv_features.detach()
            .cpu()
            .numpy()
        )

        # ====================================================
        # 5. Apply previously fitted StandardScaler
        # ====================================================

        deep_scaled = self.deep_scaler.transform(
            deep_numpy
        )

        # ====================================================
        # 6. Apply previously fitted PCA
        # ====================================================

        deep_reduced = self.deep_pca.transform(
            deep_scaled
        )

        # Shape:
        # [B, 32]

        # ====================================================
        # 7. Convert PCA features back to Torch
        # ====================================================

        deep_tensor = torch.tensor(
            deep_reduced,
            dtype=torch.float32,
            device=x.device
        )

        # ====================================================
        # 8. Get handcrafted 50 features
        # ====================================================

        handcrafted_tensor = (
            self.handcrafted_features
            .to(x.device)
        )

        # ----------------------------------------------------
        # Repeat handcrafted features for batch size
        # ----------------------------------------------------

        if handcrafted_tensor.shape[0] == 1:

            handcrafted_tensor = (
                handcrafted_tensor
                .repeat(x.shape[0], 1)
            )

        # ====================================================
        # 9. Fuse:
        #
        # 50 handcrafted + 32 ConvNeXt
        # = 82 features
        # ====================================================

        fused_features = torch.cat(
            [
                handcrafted_tensor,
                deep_tensor
            ],
            dim=1
        )

        # ====================================================
        # 10. Final Hybrid prediction
        # ====================================================

        output = self.hybrid(
            fused_features
        )

        return output


print("=" * 60)
print("GRAD-CAM HYBRID WRAPPER DEFINED")
print("=" * 60)

print(
    "ConvNeXt features: 768"
)

print(
    "PCA features: 32"
)

print(
    "Handcrafted features: 50"
)

print(
    "Fused features: 82"
)

print(
    "Output classes: 3"
)


# ============================================================
# SOURCE COLAB CELL 94
# ============================================================

# ============================================================
# CELL 28 — PREPARE ONE INTERNAL-TEST IMAGE FOR GRAD-CAM
# ============================================================

# ------------------------------------------------------------
# Select the first internal-test image
# ------------------------------------------------------------

gradcam_image_path = internal_test_df.iloc[0]["filepath"]

print("=" * 60)
print("GRAD-CAM IMAGE")
print("=" * 60)

print("Image path:")
print(gradcam_image_path)

# ------------------------------------------------------------
# Read original CT image
# ------------------------------------------------------------

gradcam_original = cv2.imread(
    str(gradcam_image_path),
    cv2.IMREAD_GRAYSCALE
)

if gradcam_original is None:

    raise ValueError(
        f"Could not read image: {gradcam_image_path}"
    )

print(
    "\nOriginal image shape:",
    gradcam_original.shape
)

# ------------------------------------------------------------
# Apply the SAME finalized ConvNeXt preprocessing
# ------------------------------------------------------------

gradcam_processed = prepare_convnext_image(
    gradcam_original,
    target_size=(224, 224)
)

print(
    "Processed image shape:",
    gradcam_processed.shape
)

print(
    "Processed image range:",
    gradcam_processed.min(),
    "to",
    gradcam_processed.max()
)

# ------------------------------------------------------------
# Convert processed image to uint8
# ------------------------------------------------------------

gradcam_uint8 = (
    gradcam_processed * 255.0
).clip(
    0,
    255
).astype(
    np.uint8
)

# ------------------------------------------------------------
# Convert grayscale → RGB
# ------------------------------------------------------------

gradcam_rgb = cv2.cvtColor(
    gradcam_uint8,
    cv2.COLOR_GRAY2RGB
)

# ------------------------------------------------------------
# Convert to PIL
# ------------------------------------------------------------

gradcam_pil = Image.fromarray(
    gradcam_rgb
)

# ------------------------------------------------------------
# Apply the SAME ConvNeXt transformation
# ------------------------------------------------------------

gradcam_tensor = convnext_transform(
    gradcam_pil
)

# Add batch dimension
gradcam_tensor = gradcam_tensor.unsqueeze(0)

# Move to CPU
gradcam_tensor = gradcam_tensor.to(device)

print(
    "\nConvNeXt input tensor shape:",
    gradcam_tensor.shape
)

print("=" * 60)
print("IMAGE PREPARATION COMPLETED")
print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 95
# ============================================================

# ============================================================
# CELL 29 — EXTRACT 50 HANDCRAFTED FEATURES FOR GRAD-CAM IMAGE
# =============================================================

print("=" * 60)
print("HANDCRAFTED FEATURE EXTRACTION FOR GRAD-CAM")
print("=" * 60)

# ------------------------------------------------------------
# 1. Prepare image and mask using the same preprocessing
# ------------------------------------------------------------

# Run the full preprocess_image pipeline to retrieve both the normalized image and the mask
preprocessing_result = preprocess_image(gradcam_original)
handcrafted_image = preprocessing_result["normalized"]
handcrafted_mask = preprocessing_result["mask"]

print(
    "Preprocessed image shape:",
    handcrafted_image.shape
)
print(
    "Preprocessed mask shape:",
    handcrafted_mask.shape
)

# ------------------------------------------------------------
# 2. Convert to uint8 for handcrafted feature extraction
# ------------------------------------------------------------

handcrafted_uint8 = (
    handcrafted_image * 255.0
).clip(
    0,
    255
).astype(
    np.uint8
)

# ------------------------------------------------------------
# 3. Extract the six handcrafted feature groups using image and mask
# ------------------------------------------------------------

intensity_features = extract_intensity_features(
    handcrafted_uint8,
    handcrafted_mask
)

glcm_features = extract_glcm_features(
    handcrafted_uint8,
    handcrafted_mask
)

lbp_features = extract_lbp_features(
    handcrafted_uint8,
    handcrafted_mask
)

hog_features = extract_hog_features(
    handcrafted_uint8,
    handcrafted_mask
)

shape_features = extract_shape_features(
    handcrafted_mask
)

edge_features = extract_edge_features(
    handcrafted_uint8,
    handcrafted_mask
)

# ------------------------------------------------------------
# 4. Combine all handcrafted features
# ------------------------------------------------------------

gradcam_handcrafted_raw = np.concatenate([
    intensity_features,
    glcm_features,
    lbp_features,
    hog_features,
    shape_features,
    edge_features
])

# ------------------------------------------------------------
# 5. Verify original feature count
# ------------------------------------------------------------

print(
    "\nOriginal handcrafted feature count:",
    len(gradcam_handcrafted_raw)
)

# ------------------------------------------------------------
# 6. Convert to 2D matrix
# ------------------------------------------------------------

gradcam_handcrafted_raw = (
    gradcam_handcrafted_raw
    .reshape(1, -1)
    .astype(np.float32)
)

print(
    "Raw feature matrix shape:",
    gradcam_handcrafted_raw.shape
)

# ------------------------------------------------------------
# 7. Handle NaN and Inf values
# ------------------------------------------------------------

gradcam_handcrafted_raw = np.nan_to_num(
    gradcam_handcrafted_raw,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# ------------------------------------------------------------
# 8. Apply the previously fitted variance selector
# ------------------------------------------------------------

gradcam_after_variance = (
    variance_selector.transform(
        gradcam_handcrafted_raw
    )
)

print(
    "After variance filtering:",
    gradcam_after_variance.shape
)

# ------------------------------------------------------------
# 9. Remove the same correlated feature indices
# ------------------------------------------------------------

gradcam_after_correlation = np.delete(
    gradcam_after_variance,
    correlated_features,
    axis=1
)

print(
    "After correlation filtering:",
    gradcam_after_correlation.shape
)

# ------------------------------------------------------------
# 10. Apply the previously fitted SelectKBest
# ------------------------------------------------------------

gradcam_selected = (
    feature_selector.transform(
        gradcam_after_correlation
    )
)

print(
    "After SelectKBest:",
    gradcam_selected.shape
)

# ------------------------------------------------------------
# 11. Apply the previously fitted StandardScaler
# ------------------------------------------------------------

gradcam_handcrafted_final = (
    feature_scaler.transform(
        gradcam_selected
    )
)

print(
    "Final handcrafted features:",
    gradcam_handcrafted_final.shape
)

print("=" * 60)
print("HANDCRAFTED FEATURES READY FOR GRAD-CAM")
print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 96
# ============================================================

# ============================================================
# CELL 30 — CONVERT HANDCRAFTED FEATURES TO PYTORCH TENSOR
# ============================================================

# Convert the final 50 handcrafted features to FloatTensor
gradcam_handcrafted_tensor = torch.tensor(
    gradcam_handcrafted_final,
    dtype=torch.float32
)

# Move to the same device as the model
gradcam_handcrafted_tensor = (
    gradcam_handcrafted_tensor.to(device)
)

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

print("=" * 60)
print("GRAD-CAM HANDCRAFTED FEATURE TENSOR")
print("=" * 60)

print(
    "Tensor shape:",
    gradcam_handcrafted_tensor.shape
)

print(
    "Tensor dtype:",
    gradcam_handcrafted_tensor.dtype
)

print(
    "Tensor device:",
    gradcam_handcrafted_tensor.device
)

print(
    "Expected shape:",
    "(1, 50)"
)

print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 97
# ============================================================

# ============================================================
# CELL 31 — CREATE GRAD-CAM MODEL WRAPPER
# ============================================================

# ------------------------------------------------------------
# Make sure the frozen Hybrid model is in evaluation mode
# ------------------------------------------------------------

model.eval()
convnext.eval()

# ------------------------------------------------------------
# Create Grad-CAM wrapper
# ------------------------------------------------------------

gradcam_model = HybridGradCAMModel(
    convnext_model=convnext,
    hybrid_model=model,
    deep_scaler=deep_scaler,
    deep_pca=deep_pca,
    handcrafted_features=gradcam_handcrafted_tensor
)

# Move wrapper to CPU
gradcam_model = gradcam_model.to(device)

# Set evaluation mode
gradcam_model.eval()

# ------------------------------------------------------------
# Verify wrapper
# ------------------------------------------------------------

print("=" * 60)
print("GRAD-CAM MODEL WRAPPER CREATED")
print("=" * 60)

print(
    "Device:",
    device
)

print(
    "Handcrafted features:",
    gradcam_handcrafted_tensor.shape
)

print(
    "ConvNeXt PCA components:",
    deep_pca.n_components_
)

print(
    "Hybrid input dimension:",
    82
)

print(
    "Number of output classes:",
    3
)

print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 98
# ============================================================

# ============================================================
# CELL 32 — VERIFY GRAD-CAM FORWARD PASS
# ============================================================

# ------------------------------------------------------------
# Clear any previous stored activation
# ------------------------------------------------------------

if hasattr(gradcam_model, "spatial_features"):
    del gradcam_model.spatial_features

# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

with torch.no_grad():

    gradcam_output = gradcam_model(
        gradcam_tensor
    )

# ------------------------------------------------------------
# Verify classifier output
# ------------------------------------------------------------

print("=" * 60)
print("GRAD-CAM FORWARD PASS VERIFICATION")
print("=" * 60)

print(
    "Classifier output shape:",
    gradcam_output.shape
)

print(
    "Classifier logits:",
    gradcam_output.cpu().numpy()
)

# ------------------------------------------------------------
# Verify spatial feature maps
# ------------------------------------------------------------

print(
    "\nSpatial feature map shape:",
    gradcam_model.spatial_features.shape
)

# ------------------------------------------------------------
# Convert logits to probabilities
# ------------------------------------------------------------

gradcam_probabilities = torch.softmax(
    gradcam_output,
    dim=1
)

print(
    "\nClass probabilities:"
)

for class_index, class_name in enumerate(CLASS_NAMES):

    probability = (
        gradcam_probabilities[0, class_index]
        .item()
        * 100
    )

    print(
        f"{class_name}: {probability:.2f}%"
    )

# ------------------------------------------------------------
# Predicted class
# ------------------------------------------------------------

gradcam_predicted_class = torch.argmax(
    gradcam_output,
    dim=1
).item()

print(
    "\nPredicted class:",
    CLASS_NAMES[gradcam_predicted_class]
)

print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 99
# ============================================================

# ============================================================
# CELL 33 — REGISTER GRAD-CAM HOOKS
# ============================================================

# ------------------------------------------------------------
# Storage for activations and gradients
# ------------------------------------------------------------

gradcam_activations = None
gradcam_gradients = None


# ------------------------------------------------------------
# Forward hook
# ------------------------------------------------------------

def gradcam_forward_hook(
    module,
    input,
    output
):

    global gradcam_activations

    gradcam_activations = output


# ------------------------------------------------------------
# Backward hook
# ------------------------------------------------------------

def gradcam_backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradcam_gradients

    # grad_output[0] contains the gradient
    # flowing out of the target layer

    gradcam_gradients = grad_output[0]


# ------------------------------------------------------------
# Register hooks on final spatial ConvNeXt stage
# ------------------------------------------------------------

forward_handle = (
    gradcam_target_layer.register_forward_hook(
        gradcam_forward_hook
    )
)

backward_handle = (
    gradcam_target_layer.register_full_backward_hook(
        gradcam_backward_hook
    )
)

# ------------------------------------------------------------
# Verify hooks
# ------------------------------------------------------------

print("=" * 60)
print("GRAD-CAM HOOKS REGISTERED")
print("=" * 60)

print(
    "Target layer: convnext.features[7]"
)

print(
    "Forward hook: registered"
)

print(
    "Backward hook: registered"
)

print(
    "Expected activation shape: [1, 768, 7, 7]"
)

print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 100
# ============================================================

# ============================================================
# CELL 34A — DIFFERENTIABLE HYBRID MODEL FOR GRAD-CAM
# ============================================================

class HybridGradCAMModel(nn.Module):

    def __init__(
        self,
        convnext_model,
        hybrid_model,
        deep_scaler,
        deep_pca,
        handcrafted_features
    ):
        super().__init__()

        self.convnext = convnext_model
        self.hybrid = hybrid_model
        self.handcrafted_features = handcrafted_features

        # ----------------------------------------------------
        # Convert sklearn StandardScaler parameters to tensors
        # ----------------------------------------------------

        self.register_buffer(
            "deep_mean",
            torch.tensor(
                deep_scaler.mean_,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "deep_scale",
            torch.tensor(
                deep_scaler.scale_,
                dtype=torch.float32
            )
        )

        # ----------------------------------------------------
        # Convert PCA parameters to tensors
        # ----------------------------------------------------

        self.register_buffer(
            "pca_mean",
            torch.tensor(
                deep_pca.mean_,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "pca_components",
            torch.tensor(
                deep_pca.components_,
                dtype=torch.float32
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # ConvNeXt spatial feature maps
        # ----------------------------------------------------

        spatial_features = self.convnext.features(x)

        # Store spatial feature maps
        self.spatial_features = spatial_features

        # Important:
        # Keep gradients for Grad-CAM
        spatial_features.retain_grad()

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        pooled_features = self.convnext.avgpool(
            spatial_features
        )

        deep_features = torch.flatten(
            pooled_features,
            1
        )

        # ----------------------------------------------------
        # Differentiable StandardScaler
        #
        # sklearn:
        # (X - mean) / scale
        # ----------------------------------------------------

        deep_scaled = (
            deep_features - self.deep_mean
        ) / self.deep_scale

        # ----------------------------------------------------
        # Differentiable PCA
        #
        # sklearn PCA transform:
        # (X - mean) @ components.T
        # ----------------------------------------------------

        deep_reduced = torch.matmul(
            deep_scaled - self.pca_mean,
            self.pca_components.T
        )

        # ----------------------------------------------------
        # Handcrafted features
        # ----------------------------------------------------

        handcrafted = self.handcrafted_features.to(
            x.device
        )

        if handcrafted.shape[0] == 1:
            handcrafted = handcrafted.expand(
                x.shape[0],
                -1
            )

        # ----------------------------------------------------
        # Feature fusion
        # 50 handcrafted + 32 deep = 82
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                handcrafted,
                deep_reduced
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Hybrid classifier
        # ----------------------------------------------------

        output = self.hybrid(
            fused_features
        )

        return output


# ============================================================
# SOURCE COLAB CELL 101
# ============================================================

# ============================================================
# CELL 34B — CREATE DIFFERENTIABLE GRAD-CAM MODEL
# ============================================================

gradcam_model = HybridGradCAMModel(
    convnext_model=convnext,
    hybrid_model=model,  # Fixed: changed 'hybrid_model' to 'model'
    deep_scaler=deep_scaler,
    deep_pca=deep_pca,
    handcrafted_features=gradcam_handcrafted_tensor
).to(device)

gradcam_model.eval()

print("Grad-CAM model created successfully.")
print("Hybrid input dimension:", 82)
print("Handcrafted features:", gradcam_handcrafted_tensor.shape)
print("PCA components:", deep_pca.n_components_)


# ============================================================
# SOURCE COLAB CELL 102
# ============================================================

# ============================================================
# CELL 34C — REMOVE OLD HOOKS
# ============================================================

try:
    forward_handle.remove()
    print("Old forward hook removed.")
except:
    print("No old forward hook to remove.")

try:
    backward_handle.remove()
    print("Old backward hook removed.")
except:
    print("No old backward hook to remove.")


# ============================================================
# SOURCE COLAB CELL 103
# ============================================================

# ============================================================
# CELL 34D — FORWARD + BACKWARD FOR GRAD-CAM
# ============================================================

gradcam_model.eval()

# Clear gradients
gradcam_model.zero_grad(set_to_none=True)

# Create fresh input
gradcam_input = gradcam_tensor.clone().detach()
gradcam_input.requires_grad_(True)

# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

gradcam_output = gradcam_model(
    gradcam_input
)

# ------------------------------------------------------------
# Predicted class
# ------------------------------------------------------------

gradcam_predicted_class = torch.argmax(
    gradcam_output,
    dim=1
).item()

predicted_class_name = CLASS_NAMES[
    gradcam_predicted_class
]

print("Predicted class:", predicted_class_name)

# ------------------------------------------------------------
# Target class score
# ------------------------------------------------------------

target_score = gradcam_output[
    0,
    gradcam_predicted_class
]

print(
    "Target class logit:",
    target_score.item()
)

# ------------------------------------------------------------
# Backward pass
# ------------------------------------------------------------

target_score.backward()

# ------------------------------------------------------------
# Get ConvNeXt spatial activations
# ------------------------------------------------------------

gradcam_activations = (
    gradcam_model.spatial_features
)

# Get gradients of the spatial feature maps
gradcam_gradients = (
    gradcam_model.spatial_features.grad
)

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("GRAD-CAM VERIFICATION")
print("=" * 60)

print(
    "Activation shape:",
    gradcam_activations.shape
)

if gradcam_gradients is None:
    raise RuntimeError(
        "Gradients are still None. "
        "The computational graph is not connected."
    )

print(
    "Gradient shape:",
    gradcam_gradients.shape
)

print(
    "Gradient absolute mean:",
    gradcam_gradients.abs().mean().item()
)

print(
    "Gradient maximum:",
    gradcam_gradients.abs().max().item()
)

print("=" * 60)
print("GRAD-CAM GRADIENTS SUCCESSFULLY CAPTURED")
print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 104
# ============================================================

# ============================================================
# CELL 35 — GENERATE GRAD-CAM HEATMAP
# ============================================================

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ------------------------------------------------------------
# Get activations and gradients
# ------------------------------------------------------------

activations = gradcam_activations
gradients = gradcam_gradients

# ------------------------------------------------------------
# Global average pooling of gradients
# Gives one importance weight per feature channel
# ------------------------------------------------------------

weights = gradients.mean(
    dim=(2, 3),
    keepdim=True
)

print("Channel weights shape:", weights.shape)

# ------------------------------------------------------------
# Weighted combination of activation maps
# ------------------------------------------------------------

cam = (
    weights * activations
).sum(
    dim=1,
    keepdim=True
)

print("Raw CAM shape:", cam.shape)

# ------------------------------------------------------------
# ReLU
# Keep only positive contributions
# ------------------------------------------------------------

cam = torch.relu(cam)

# ------------------------------------------------------------
# Resize from 7x7 → 224x224
# ------------------------------------------------------------

cam = F.interpolate(
    cam,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

# ------------------------------------------------------------
# Convert to NumPy
# ------------------------------------------------------------

cam = cam.squeeze().detach().cpu().numpy()

# ------------------------------------------------------------
# Normalize to 0–1
# ------------------------------------------------------------

cam_min = cam.min()
cam_max = cam.max()

cam = (
    cam - cam_min
) / (
    cam_max - cam_min + 1e-8
)

print("\n" + "=" * 60)
print("GRAD-CAM HEATMAP GENERATED")
print("=" * 60)

print("Heatmap shape:", cam.shape)
print("Minimum:", cam.min())
print("Maximum:", cam.max())


# ============================================================
# SOURCE COLAB CELL 105
# ============================================================

# ============================================================
# CELL 36 — VISUALIZE GRAD-CAM
# ============================================================

# Resize original CT to Grad-CAM size
gradcam_display = cv2.resize(
    gradcam_original,
    (224, 224),
    interpolation=cv2.INTER_AREA
)

# Convert grayscale to RGB
gradcam_display_rgb = cv2.cvtColor(
    gradcam_display,
    cv2.COLOR_GRAY2RGB
)

# ------------------------------------------------------------
# Create colored heatmap
# ------------------------------------------------------------

heatmap = np.uint8(255 * cam)

heatmap_color = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)

# ------------------------------------------------------------
# Overlay
# ------------------------------------------------------------

overlay = cv2.addWeighted(
    gradcam_display_rgb,
    0.55,
    heatmap_color,
    0.45,
    0
)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(
    gradcam_display,
    cmap="gray"
)
plt.title(
    f"CT Image\nPredicted: {predicted_class_name}"
)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(
    cam,
    cmap="jet"
)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(
    f"Grad-CAM Overlay\nPredicted: {predicted_class_name}"
)
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# SOURCE COLAB CELL 106
# ============================================================

# **UPLOAD AND CHECK THE DATA**


# ============================================================
# SOURCE COLAB CELL 107
# ============================================================

# ============================================================
# CELL 40 — COMPLETE IMAGE PREDICTION PIPELINE
# ============================================================

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def extract_handcrafted_for_prediction(image):

    # --------------------------------------------------------
    # Existing preprocessing
    # --------------------------------------------------------

    preprocessing_result = preprocess_image(image)

    normalized = preprocessing_result["normalized"]
    mask = preprocessing_result["mask"]

    processed_uint8 = (
        normalized * 255
    ).clip(0, 255).astype(np.uint8)

    # --------------------------------------------------------
    # Extract all handcrafted features
    # --------------------------------------------------------

    intensity_features = extract_intensity_features(
        processed_uint8,
        mask
    )

    glcm_features = extract_glcm_features(
        processed_uint8,
        mask
    )

    lbp_features = extract_lbp_features(
        processed_uint8,
        mask
    )

    hog_features = extract_hog_features(
        processed_uint8,
        mask
    )

    shape_features = extract_shape_features(
        mask
    )

    edge_features = extract_edge_features(
        processed_uint8,
        mask
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    raw_features = np.concatenate([
        intensity_features,
        glcm_features,
        lbp_features,
        hog_features,
        shape_features,
        edge_features
    ]).reshape(1, -1)

    # --------------------------------------------------------
    # Feature selection pipeline
    # --------------------------------------------------------

    X = variance_selector.transform(
        raw_features
    )

    X = np.delete(
        X,
        correlated_features,
        axis=1
    )

    X = feature_selector.transform(
        X
    )

    X = feature_scaler.transform(
        X
    )

    return X.astype(np.float32), normalized, mask


# ============================================================
# SOURCE COLAB CELL 108
# ============================================================

# ============================================================
# CELL 41 — CONVNEXT FEATURES FOR UPLOADED IMAGE
# ============================================================

def extract_deep_feature_for_prediction(image):

    # --------------------------------------------------------
    # Same preprocessing used during training
    # --------------------------------------------------------

    normalized = prepare_convnext_image(
        image,
        target_size=(224, 224)
    )

    processed = (
        normalized * 255
    ).clip(0, 255).astype(np.uint8)

    # --------------------------------------------------------
    # Grayscale → RGB
    # --------------------------------------------------------

    rgb_image = cv2.cvtColor(
        processed,
        cv2.COLOR_GRAY2RGB
    )

    pil_image = Image.fromarray(
        rgb_image
    )

    # --------------------------------------------------------
    # ConvNeXt transform
    # --------------------------------------------------------

    tensor = convnext_transform(
        pil_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Extract 768-D deep feature
    # --------------------------------------------------------

    convnext.eval()

    with torch.no_grad():

        features = convnext(
            tensor
        )

        features = torch.flatten(
            features,
            1
        )

    return features


# ============================================================
# SOURCE COLAB CELL 109
# ============================================================

# ============================================================
# CELL 42 — COMPLETE MODEL PREDICTION
# ============================================================

def predict_uploaded_image(image):

    # --------------------------------------------------------
    # Convert input to grayscale
    # --------------------------------------------------------

    if isinstance(image, Image.Image):

        image = np.array(
            image.convert("L")
        )

    elif len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    image = image.astype(np.uint8)

    # --------------------------------------------------------
    # Handcrafted branch
    # --------------------------------------------------------

    handcrafted_features, normalized, mask = (
        extract_handcrafted_for_prediction(
            image
        )
    )

    handcrafted_tensor = torch.tensor(
        handcrafted_features,
        dtype=torch.float32
    ).to(device)

    # --------------------------------------------------------
    # Deep branch
    # --------------------------------------------------------

    deep_features = (
        extract_deep_feature_for_prediction(
            image
        )
    )

    deep_features_numpy = (
        deep_features
        .cpu()
        .numpy()
    )

    deep_scaled = deep_scaler.transform(
        deep_features_numpy
    )

    deep_reduced = deep_pca.transform(
        deep_scaled
    )

    deep_tensor = torch.tensor(
        deep_reduced,
        dtype=torch.float32
    ).to(device)

    # --------------------------------------------------------
    # Fuse 50 + 32 = 82
    # --------------------------------------------------------

    fused_features = torch.cat(
        [
            handcrafted_tensor,
            deep_tensor
        ],
        dim=1
    )

    # --------------------------------------------------------
    # Hybrid classifier
    # --------------------------------------------------------

    hybrid_model.eval()

    with torch.no_grad():

        logits = hybrid_model(
            fused_features
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    probability_dict = {
        CLASS_NAMES[i]:
        float(probabilities[i].item())
        for i in range(len(CLASS_NAMES))
    }

    return {
        "original": image,
        "segmented": normalized,
        "mask": mask,
        "handcrafted_features": handcrafted_tensor,
        "deep_features": deep_tensor,
        "fused_features": fused_features,
        "logits": logits,
        "probabilities": probability_dict,
        "predicted_class": predicted_class,
        "predicted_index": predicted_index
    }


# ============================================================
# SOURCE COLAB CELL 110
# ============================================================

# ============================================================
# CELL 43 — GRAD-CAM FOR ANY UPLOADED IMAGE
# ============================================================

def generate_uploaded_gradcam(image):

    # --------------------------------------------------------
    # Convert image to grayscale
    # --------------------------------------------------------

    if isinstance(image, Image.Image):

        image = np.array(
            image.convert("L")
        )

    elif len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    image = image.astype(np.uint8)

    # --------------------------------------------------------
    # Extract handcrafted features
    # --------------------------------------------------------

    handcrafted_features, normalized, mask = (
        extract_handcrafted_for_prediction(
            image
        )
    )

    handcrafted_tensor = torch.tensor(
        handcrafted_features,
        dtype=torch.float32
    ).to(device)

    # --------------------------------------------------------
    # Create Grad-CAM model using THIS image's
    # handcrafted features. Explicitly using final_hybrid_model
    # --------------------------------------------------------

    model_gradcam = HybridGradCAMModel(
        convnext_model=convnext,
        hybrid_model=final_hybrid_model, # Fixed: explicitly using loaded final_hybrid_model to avoid namespace conflicts
        deep_scaler=deep_scaler,
        deep_pca=deep_pca,
        handcrafted_features=handcrafted_tensor
    ).to(device)

    model_gradcam.eval()

    # --------------------------------------------------------
    # Prepare ConvNeXt input
    # --------------------------------------------------------

    processed = (
        normalized * 255
    ).clip(0, 255).astype(np.uint8)

    rgb_image = cv2.cvtColor(
        processed,
        cv2.COLOR_GRAY2RGB
    )

    pil_image = Image.fromarray(
        rgb_image
    )

    tensor = convnext_transform(
        pil_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model_gradcam.zero_grad(set_to_none=True)

    output = model_gradcam(tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # --------------------------------------------------------
    # Backward pass for predicted class
    # --------------------------------------------------------

    target_score = output[
        0,
        predicted_index
    ]

    target_score.backward()

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    activations = model_gradcam.spatial_features
    gradients = model_gradcam.spatial_features.grad

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients were not captured."
        )

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(
        dim=1,
        keepdim=True
    )

    cam = torch.relu(cam)

    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    cam = (
        cam
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    # Normalize
    cam = (
        cam - cam.min()
    ) / (
        cam.max() - cam.min() + 1e-8
    )

    # --------------------------------------------------------
    # Prepare display image
    # --------------------------------------------------------

    display_image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    display_rgb = cv2.cvtColor(
        display_image,
        cv2.COLOR_GRAY2RGB
    )

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(
        cam * 255
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        display_rgb,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probability_dict = {
        CLASS_NAMES[i]:
        float(probabilities[i].item())
        for i in range(len(CLASS_NAMES))
    }

    return {
        "original": image,
        "segmented": normalized,
        "mask": mask,
        "gradcam": cam,
        "overlay": overlay,
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "probabilities": probability_dict
    }


# ============================================================
# SOURCE COLAB CELL 111
# ============================================================

# ============================================================
# CELL 44 — TEST COMPLETE UPLOAD PIPELINE
# ============================================================

test_image = cv2.imread(
    str(gradcam_image_path),
    cv2.IMREAD_GRAYSCALE
)

result = generate_uploaded_gradcam(
    test_image
)

print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    "Predicted class:",
    result["predicted_class"]
)

print("\nModel-estimated probabilities:")

for class_name, probability in result["probabilities"].items():

    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )

print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 112
# ============================================================

# ============================================================
# CELL 45 — COMPLETE VISUAL OUTPUT
# ============================================================

plt.figure(figsize=(16, 5))

# ------------------------------------------------------------
# Original
# ------------------------------------------------------------

plt.subplot(1, 3, 1)

plt.imshow(
    result["original"],
    cmap="gray"
)

plt.title(
    "Uploaded CT Image"
)

plt.axis("off")


# ------------------------------------------------------------
# Segmented lung
# ------------------------------------------------------------

plt.subplot(1, 3, 2)

plt.imshow(
    result["segmented"],
    cmap="gray"
)

plt.title(
    "Segmented Lung Region"
)

plt.axis("off")


# ------------------------------------------------------------
# Grad-CAM overlay
# ------------------------------------------------------------

plt.subplot(1, 3, 3)

plt.imshow(
    result["overlay"]
)

plt.title(
    f"Grad-CAM\n"
    f"Prediction: {result['predicted_class']}"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# SOURCE COLAB CELL 113
# ============================================================

# ============================================================
# FIX — RELOAD THE FINAL PYTORCH HYBRID CLASSIFIER
# ============================================================

import torch
import torch.nn as nn
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model"

MODEL_PATH = MODEL_DIR / "hybrid_classifier_final.pth"


# ------------------------------------------------------------
# Define the SAME HybridClassifier architecture
# ------------------------------------------------------------

class HybridClassifier(nn.Module):

    def __init__(
        self,
        input_dim=82,
        num_classes=3
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                32,
                num_classes
            )
        )

    def forward(self, x):

        return self.network(x)


# ------------------------------------------------------------
# Create a NEW variable
# Do NOT use hybrid_model
# ------------------------------------------------------------

final_hybrid_model = HybridClassifier(
    input_dim=82,
    num_classes=3
).to(device)


# ------------------------------------------------------------
# Load trained weights
# ------------------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)


if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    final_hybrid_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    final_hybrid_model.load_state_dict(
        checkpoint
    )


# ------------------------------------------------------------
# Evaluation mode
# ------------------------------------------------------------

final_hybrid_model.eval()


# Freeze model parameters
for parameter in final_hybrid_model.parameters():

    parameter.requires_grad = False


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

dummy_input = torch.randn(
    1,
    82,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    test_output = final_hybrid_model(
        dummy_input
    )


print("=" * 60)
print("FINAL HYBRID MODEL CHECK")
print("=" * 60)

print(
    "Model type:",
    type(final_hybrid_model)
)

print(
    "Output shape:",
    test_output.shape
)

print(
    "Expected shape: torch.Size([1, 3])"
)

print("=" * 60)
print("FINAL HYBRID MODEL LOADED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# SOURCE COLAB CELL 114
# ============================================================

# ============================================================
# CELL 47 — UPLOAD CT → CONVNEXT + HYBRID → GRAD-CAM
#             → DISPLAY RESULT → STORE RESULT
# ============================================================

# # from google.colab import files
from PIL import Image
# from IPython.display import display
from pathlib import Path
from datetime import datetime
import io
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. RESULT STORAGE DIRECTORY
# ============================================================

RESULT_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model/Prediction Results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Result directory:")
print(RESULT_DIR)


# ============================================================
# 2. UPLOAD IMAGE
# ============================================================

print("\n" + "=" * 70)
print("UPLOAD CT IMAGE FOR AI-ASSISTED PREDICTION")
print("=" * 70)

# uploaded = files.upload()
uploaded = {}

if not uploaded:

    print("No image selected.")

    current_result = None
    result = None

else:

    # --------------------------------------------------------
    # Get uploaded filename
    # --------------------------------------------------------

    filename = list(uploaded.keys())[0]

    print(f"\nUploaded: {filename}")


    # ========================================================
    # 3. READ IMAGE
    # ========================================================

    image_bytes = uploaded[filename]

    pil_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    image = np.array(
        pil_image
    ).astype(np.uint8)

    print(
        "Original image shape:",
        image.shape
    )


    # ========================================================
    # 4. RUN CONVNEXT + HYBRID + GRAD-CAM
    # ========================================================

    print("\nProcessing image...")

    print(
        "Using trained ConvNeXt-Tiny + "
        "handcrafted feature hybrid model."
    )

    print(
        "Generating Grad-CAM..."
    )

    current_result = generate_uploaded_gradcam(
        image
    )


    # ========================================================
    # 5. ADD UPLOAD INFORMATION TO RESULT
    # ========================================================

    current_result["filename"] = filename

    current_result["uploaded_image"] = image

    current_result["timestamp"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    current_result["model"] = (
        "ConvNeXt-Tiny + Handcrafted Features + Hybrid Classifier"
    )

    current_result["gradcam_available"] = True


    # ========================================================
    # 6. EXTRACT PREDICTION
    # ========================================================

    predicted_class = current_result[
        "predicted_class"
    ]

    probabilities = current_result[
        "probabilities"
    ]


    # ========================================================
    # 7. DISPLAY PREDICTION RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("AI-ASSISTED LUNG CT PREDICTION")
    print("=" * 70)

    print(
        f"\nUploaded Image : {filename}"
    )

    print(
        f"Prediction     : {predicted_class}"
    )

    print("\nModel-estimated probabilities:")

    for class_name in [
        "Normal",
        "Benign",
        "Malignant"
    ]:

        probability = probabilities[
            class_name
        ]

        print(
            f"  {class_name:<12}: "
            f"{probability * 100:.2f}%"
        )


    # ========================================================
    # 8. DISPLAY ORIGINAL + SEGMENTED + GRAD-CAM
    # ========================================================

    plt.figure(
        figsize=(18, 5)
    )


    # --------------------------------------------------------
    # Original CT
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        current_result["original"],
        cmap="gray"
    )

    plt.title(
        "Uploaded CT Image",
        fontsize=13
    )

    plt.axis("off")


    # --------------------------------------------------------
    # Segmented lung region
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        current_result["segmented"],
        cmap="gray"
    )

    plt.title(
        "Segmented Lung Region",
        fontsize=13
    )

    plt.axis("off")


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        current_result["overlay"]
    )

    plt.title(
        f"Grad-CAM\nPrediction: {predicted_class}",
        fontsize=13
    )

    plt.axis("off")


    plt.tight_layout()

    plt.show()


    # ========================================================
    # 9. CONFIDENCE
    # ========================================================

    confidence = (
        probabilities[predicted_class]
        * 100
    )


    # ========================================================
    # 10. USER-FRIENDLY RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("USER-FRIENDLY RESULT")
    print("=" * 70)

    print(
        f"""
The AI model predicts the image as:

        {predicted_class.upper()}

Estimated model probability:

        {confidence:.2f}%

The Grad-CAM visualization highlights the
regions of the CT image that contributed
to the model's prediction.

This is an AI-assisted preliminary prediction
and should not be considered a medical diagnosis.

Clinical evaluation by a qualified doctor
is recommended.
"""
    )

    print("=" * 70)


    # ========================================================
    # 11. STORE RESULT AS JOBLIB
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    result_filename = (
        f"prediction_result_{timestamp}.joblib"
    )

    result_path = (
        RESULT_DIR /
        result_filename
    )


    joblib.dump(
        current_result,
        result_path
    )


    # ========================================================
    # 12. ALSO SAVE VISUAL RESULTS
    # ========================================================

    # Original image
    original_path = (
        RESULT_DIR /
        f"{timestamp}_original.png"
    )

    Image.fromarray(
        current_result["original"].astype(np.uint8)
    ).save(
        original_path
    )


    # Segmented image
    segmented_path = (
        RESULT_DIR /
        f"{timestamp}_segmented.png"
    )

    Image.fromarray(
        current_result["segmented"].astype(np.uint8)
    ).save(
        segmented_path
    )


    # Grad-CAM overlay
    overlay_path = (
        RESULT_DIR /
        f"{timestamp}_gradcam.png"
    )

    overlay_image = current_result[
        "overlay"
    ]

    if overlay_image.dtype != np.uint8:

        overlay_image = (
            np.clip(
                overlay_image,
                0,
                1
            ) * 255
        ).astype(np.uint8)

    Image.fromarray(
        overlay_image
    ).save(
        overlay_path
    )


    # ========================================================
    # 13. STORE FILE PATHS IN RESULT
    # ========================================================

    current_result[
        "result_file"
    ] = str(result_path)

    current_result[
        "original_file"
    ] = str(original_path)

    current_result[
        "segmented_file"
    ] = str(segmented_path)

    current_result[
        "gradcam_file"
    ] = str(overlay_path)


    # Save updated result again
    joblib.dump(
        current_result,
        result_path
    )


    # ========================================================
    # 14. FINAL RESULT STATUS
    # ========================================================

    print("\n" + "=" * 70)
    print("RESULT STORED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nPrediction      : {predicted_class}"
    )

    print(
        f"Confidence      : {confidence:.2f}%"
    )

    print(
        f"\nResult file     : {result_path}"
    )

    print(
        f"Original image  : {original_path}"
    )

    print(
        f"Segmented image: {segmented_path}"
    )

    print(
        f"Grad-CAM image  : {overlay_path}"
    )

    print("\nThe complete result is available in:")

    print(
        RESULT_DIR
    )

    print("=" * 70)


    # ========================================================
    # 15. FINAL RESULT VARIABLE
    # ========================================================

    result = current_result

    print(
        "\nCurrent result is stored in variable: result"
    )


# ============================================================
# SOURCE COLAB CELL 115
# ============================================================

# **GEMINI MODEL TO CONNECT WITH LLM**


# ============================================================
# SOURCE COLAB CELL 116
# ============================================================

# ============================================================
# CELL 48A — CHECK AVAILABLE GEMINI MODELS
# ============================================================

from google import genai
# # from google.colab import userdata

GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in Colab Secrets."
    )

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


print("=" * 75)
print("AVAILABLE GEMINI MODELS")
print("=" * 75)


available_models = []

for model in gemini_client.models.list():

    model_name = model.name

    supported_methods = getattr(
        model,
        "supported_actions",
        None
    )

    print("\nModel:")
    print(model_name)

    if supported_methods:
        print(
            "Supported actions:",
            supported_methods
        )

    available_models.append(model)


print("\n" + "=" * 75)
print(
    f"Total models returned: {len(available_models)}"
)
print("=" * 75)


# ============================================================
# SOURCE COLAB CELL 117
# ============================================================

# ============================================================
# CELL 48B — FIND USABLE GEMINI MODELS
# ============================================================

print("=" * 75)
print("CHECKING GEMINI MODELS AVAILABLE FOR YOUR API KEY")
print("=" * 75)

usable_models = []

for model in gemini_client.models.list():

    model_name = model.name

    supported_actions = getattr(
        model,
        "supported_actions",
        []
    )

    # Check whether generateContent is supported
    if (
        "generateContent" in supported_actions
        or "generate_content" in supported_actions
    ):

        usable_models.append(
            model_name
        )

        print(
            f"\n[OK] {model_name}"
        )

        print(
            "  Supported:",
            supported_actions
        )


print("\n" + "=" * 75)

if usable_models:

    print(
        "USABLE GEMINI MODELS:"
    )

    for i, model_name in enumerate(
        usable_models,
        start=1
    ):

        print(
            f"{i}. {model_name}"
        )

else:

    print(
        "No models supporting generateContent "
        "were found for this API key."
    )

print("=" * 75)


# ============================================================
# SOURCE COLAB CELL 118
# ============================================================

# ============================================================
# CELL 48 — GEMINI LLM EXPLANATION
# SAFE MEDICAL WORDING VERSION
# ============================================================

from google import genai
from google.genai import types
# # from google.colab import userdata

from PIL import Image
import io
import joblib
from pathlib import Path


# ============================================================
# 1. GET GEMINI API KEY
# ============================================================

GEMINI_API_KEY = userdata.get(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in Colab Secrets."
    )


# ============================================================
# 2. INITIALIZE GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_MODEL = "gemini-3.5-flash"


print("=" * 70)
print("GEMINI INITIALIZED")
print("=" * 70)

print(
    "Gemini model:",
    GEMINI_MODEL
)

print("=" * 70)


# ============================================================
# 3. CHECK RESULT FROM CELL 47
# ============================================================

if result is None:

    raise ValueError(
        "No prediction result found. "
        "Please run Cell 47 first."
    )


# ============================================================
# 4. GET EXISTING MODEL OUTPUT
# ============================================================

predicted_class = result[
    "predicted_class"
]

probabilities = result[
    "probabilities"
]


# ============================================================
# 5. PROBABILITY VALUES
# ============================================================

normal_probability = (
    probabilities["Normal"] * 100
)

benign_probability = (
    probabilities["Benign"] * 100
)

malignant_probability = (
    probabilities["Malignant"] * 100
)


# ============================================================
# 6. DISPLAY EXISTING MODEL RESULT
# ============================================================

print("\n" + "=" * 70)
print("EXISTING AI MODEL OUTPUT")
print("=" * 70)

print(
    f"\nPredicted category: {predicted_class}"
)

print("\nAI model-estimated probabilities:")

print(
    f"Normal      : {normal_probability:.2f}%"
)

print(
    f"Benign      : {benign_probability:.2f}%"
)

print(
    f"Malignant   : {malignant_probability:.2f}%"
)


# ============================================================
# 7. CREATE SAFE GEMINI PROMPT
# ============================================================

llm_prompt = f"""
You are an AI-assisted explanation assistant for a
lung CT image classification research prototype.

A trained AI classification model has already processed
the uploaded CT image.

You are NOT the classification model.

Your task is ONLY to explain the existing AI model output
in simple and cautious language.

============================================================
EXISTING AI MODEL OUTPUT
============================================================

The AI model's predicted category is:

{predicted_class}

The AI model estimated the following probabilities:

Normal:
{normal_probability:.2f}%

Benign:
{benign_probability:.2f}%

Malignant:
{malignant_probability:.2f}%

============================================================
IMPORTANT MEDICAL SAFETY RULES
============================================================

Follow these rules strictly:

1. Do NOT change the supplied model prediction.

2. Do NOT override the model prediction.

3. Do NOT state that the patient definitely has or does
   not have cancer.

4. Do NOT use wording such as:
   "Cancer detected",
   "The patient has cancer",
   "No cancer",
   "Definitely normal",
   "Definitely benign",
   or "Definitely malignant".

5. Always describe the percentages as:
   "AI model-estimated probabilities" or
   "estimated probabilities from the AI model".

6. Clearly explain that an AI model can make incorrect
   predictions.

7. Clearly state that the AI result is a preliminary
   AI-assisted prediction and NOT a medical diagnosis.

8. Clearly recommend consultation with a qualified
   doctor/radiologist for confirmation and clinical
   interpretation.

9. Do not invent patient information.

10. Do not invent clinical findings.

============================================================
GRAD-CAM
============================================================

The attached image is the Grad-CAM visualization generated
by the trained AI model.

Explain that Grad-CAM highlights regions of the image that
contributed to the model's prediction.

IMPORTANT:

Do NOT say that the highlighted region is definitely a
tumor, cancer, nodule, lesion, or abnormality.

Grad-CAM is an explanation/visualization technique and does
not by itself confirm or clinically localize a disease.

============================================================
RESPONSE FORMAT
============================================================

Use exactly these sections:

### AI-Assisted Result

Explain that the AI model estimates the probability of
the uploaded CT image belonging to the predicted category.

State the predicted category exactly as supplied.

Use wording such as:

"The AI model estimates a probability of XX.XX% for
the [category] category."

Do not call this a confirmed diagnosis.

### Model-Estimated Probabilities

Briefly list:

Normal: XX.XX%
Benign: XX.XX%
Malignant: XX.XX%

Explain that these are probabilities estimated by the
AI model and should not be interpreted as confirmed
clinical probabilities.

### Grad-CAM Explanation

Explain in simple language that the highlighted regions
represent areas that contributed to the model's prediction.

Clearly state that the highlighted areas are not confirmed
lesions or cancerous regions.

### Interpretation

Explain the result cautiously.

Mention that the AI model has assigned the highest
estimated probability to the predicted category, but
the prediction may be incorrect.

### Important Clinical Note

Use clear wording similar to:

"This is an AI-assisted preliminary prediction and not
a medical diagnosis. AI models may sometimes produce
incorrect predictions. For confirmation and proper
clinical interpretation, please consult a qualified
doctor/radiologist."

Do not make the wording frightening or overly technical.

Keep the complete response concise, professional,
and suitable for displaying in a medical AI prototype.
"""


# ============================================================
# 8. PREPARE GRAD-CAM IMAGE
# ============================================================

gradcam_image = Image.fromarray(
    result["overlay"]
)

image_buffer = io.BytesIO()

gradcam_image.save(
    image_buffer,
    format="PNG"
)

gradcam_bytes = (
    image_buffer.getvalue()
)


# ============================================================
# 9. SEND MODEL OUTPUT + GRAD-CAM TO GEMINI
# ============================================================

print("\n" + "=" * 70)
print("SENDING MODEL RESULT + GRAD-CAM TO GEMINI...")
print("=" * 70)


try:

    response = gemini_client.models.generate_content(

        model=GEMINI_MODEL,

        contents=[
            llm_prompt,

            types.Part.from_bytes(
                data=gradcam_bytes,
                mime_type="image/png"
            )
        ],

        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=700
        )
    )


    gemini_explanation = response.text


    print(
        "\nGemini response received successfully."
    )


except Exception as e:

    print(
        "\nGemini request failed:"
    )

    print(
        str(e)
    )

    gemini_explanation = None


# ============================================================
# 10. DISPLAY GEMINI EXPLANATION
# ============================================================

print("\n" + "=" * 70)
print("AI-ASSISTED EXPLANATION")
print("=" * 70)


if gemini_explanation:

    print(
        gemini_explanation
    )

else:

    print(
        "Gemini explanation could not be generated."
    )


print("=" * 70)


# ============================================================
# 11. STORE GEMINI RESPONSE IN RESULT
# ============================================================

result[
    "gemini_model"
] = GEMINI_MODEL

result[
    "gemini_explanation"
] = gemini_explanation

result[
    "gemini_gradcam_input"
] = True


# ============================================================
# 12. STORE SAFETY INFORMATION
# ============================================================

result[
    "clinical_disclaimer"
] = (
    "This is an AI-assisted preliminary prediction "
    "and not a medical diagnosis. AI models may "
    "produce incorrect predictions. Consultation "
    "with a qualified doctor/radiologist is "
    "recommended for confirmation and clinical "
    "interpretation."
)


# ============================================================
# 13. UPDATE STORED RESULT FILE
# ============================================================

if "result_file" in result:

    result_path = Path(
        result["result_file"]
    )

    joblib.dump(
        result,
        result_path
    )

    print(
        "\nUpdated result saved to:"
    )

    print(
        result_path
    )


# ============================================================
# 14. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("COMPLETE AI PIPELINE")
print("=" * 70)

print(
    f"Prediction       : {predicted_class}"
)

print(
    f"Normal           : {normal_probability:.2f}%"
)

print(
    f"Benign           : {benign_probability:.2f}%"
)

print(
    f"Malignant        : {malignant_probability:.2f}%"
)

print(
    "Grad-CAM         : Generated"
)

print(
    f"Gemini Model     : {GEMINI_MODEL}"
)

if gemini_explanation:

    print(
        "Gemini           : Explanation generated"
    )

else:

    print(
        "Gemini           : Explanation failed"
    )

print(
    "Result Storage   : Updated"
)

print("=" * 70)


# ============================================================
# SOURCE COLAB CELL 119
# ============================================================

# ============================================================
# SAVE FINAL RESULT IN HUMAN-READABLE FORMAT
# ============================================================

import json
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model/Prediction Results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

timestamp = result["timestamp"].replace(
    ":", "-"
).replace(" ", "_")


# ============================================================
# 1. SAVE ORIGINAL IMAGE
# ============================================================

original_path = RESULT_DIR / f"{timestamp}_original.png"

Image.fromarray(
    result["original"].astype("uint8")
).save(
    original_path
)


# ============================================================
# 2. SAVE SEGMENTED IMAGE
# ============================================================

segmented_path = RESULT_DIR / f"{timestamp}_segmented.png"

Image.fromarray(
    result["segmented"].astype("uint8")
).save(
    segmented_path
)


# ============================================================
# 3. SAVE GRAD-CAM IMAGE
# ============================================================

gradcam_path = RESULT_DIR / f"{timestamp}_gradcam.png"

gradcam_image = result["overlay"]

if gradcam_image.dtype != np.uint8:

    gradcam_image = (
        np.clip(
            gradcam_image,
            0,
            1
        ) * 255
    ).astype(np.uint8)

Image.fromarray(
    gradcam_image
).save(
    gradcam_path
)


# ============================================================
# 4. CREATE HUMAN-READABLE RESULT
# ============================================================

readable_result = {

    "Uploaded Image":
        result["filename"],

    "Prediction":
        result["predicted_class"],

    "Normal Probability":
        f"{result['probabilities']['Normal'] * 100:.2f}%",

    "Benign Probability":
        f"{result['probabilities']['Benign'] * 100:.2f}%",

    "Malignant Probability":
        f"{result['probabilities']['Malignant'] * 100:.2f}%",

    "Model":
        "ConvNeXt-Tiny + Handcrafted Features + Hybrid Classifier",

    "Grad-CAM":
        "Generated",

    "Gemini Model":
        result.get(
            "gemini_model",
            "Not available"
        ),

    "Gemini Explanation":
        result.get(
            "gemini_explanation",
            "Not available"
        ),

    "Clinical Note":
        "This is an AI-assisted preliminary prediction "
        "and not a medical diagnosis. AI models may "
        "produce incorrect predictions. Consultation "
        "with a qualified doctor/radiologist is "
        "recommended for confirmation and clinical "
        "interpretation.",

    "Timestamp":
        result["timestamp"]
}


# ============================================================
# 5. SAVE JSON
# ============================================================

json_path = RESULT_DIR / f"{timestamp}_result.json"

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        readable_result,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 6. SAVE TXT REPORT
# ============================================================

txt_path = RESULT_DIR / f"{timestamp}_final_report.txt"

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        "AI-ASSISTED LUNG CT PREDICTION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Uploaded Image: {result['filename']}\n\n"
    )

    f.write(
        f"Prediction: {result['predicted_class']}\n\n"
    )

    f.write(
        "AI MODEL-ESTIMATED PROBABILITIES\n"
    )

    f.write(
        "-" * 45 + "\n"
    )

    f.write(
        f"Normal    : "
        f"{result['probabilities']['Normal'] * 100:.2f}%\n"
    )

    f.write(
        f"Benign    : "
        f"{result['probabilities']['Benign'] * 100:.2f}%\n"
    )

    f.write(
        f"Malignant : "
        f"{result['probabilities']['Malignant'] * 100:.2f}%\n\n"
    )

    f.write(
        "MODEL\n"
    )

    f.write(
        "-" * 45 + "\n"
    )

    f.write(
        "ConvNeXt-Tiny + Handcrafted Features "
        "+ Hybrid Classifier\n\n"
    )

    f.write(
        "GRAD-CAM\n"
    )

    f.write(
        "-" * 45 + "\n"
    )

    f.write(
        "Generated. The highlighted regions represent "
        "areas that contributed to the model prediction.\n\n"
    )

    f.write(
        "GEMINI AI-ASSISTED EXPLANATION\n"
    )

    f.write(
        "-" * 45 + "\n"
    )

    f.write(
        result.get(
            "gemini_explanation",
            "Not available"
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "CLINICAL NOTE\n"
    )

    f.write(
        "-" * 45 + "\n"
    )

    f.write(
        "This is an AI-assisted preliminary prediction "
        "and not a medical diagnosis. AI models may "
        "produce incorrect predictions. Consultation "
        "with a qualified doctor/radiologist is "
        "recommended for confirmation and clinical "
        "interpretation.\n\n"
    )

    f.write(
        f"Generated: {result['timestamp']}\n"
    )

    f.write(
        "=" * 70 + "\n"
    )


# ============================================================
# 7. SAVE PYTHON RESULT TOO
# ============================================================

joblib_path = RESULT_DIR / f"{timestamp}_result.joblib"

joblib.dump(
    result,
    joblib_path
)


# ============================================================
# 8. DISPLAY FILE LOCATIONS
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT FILES SAVED")
print("=" * 70)

print("\nHuman-readable files:")

print(
    "JSON Report :",
    json_path
)

print(
    "TXT Report  :",
    txt_path
)

print(
    "Original CT :",
    original_path
)

print(
    "Segmented   :",
    segmented_path
)

print(
    "Grad-CAM    :",
    gradcam_path
)

print(
    "\nPython backup:"
)

print(
    "JOBLIB      :",
    joblib_path
)

print("=" * 70)


# ============================================================
# SOURCE COLAB CELL 133
# ============================================================

# ============================================================
# CELL 52 — FINAL HOSPITAL TEST EVALUATION
# ============================================================

import os
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("FINAL HOSPITAL TEST EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model"

HOSPITAL_TEST_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Dataset/Hospital Raw Dataset/DATASET/Test"

RESULT_DIR = MODEL_DIR / "Final Evaluation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "hybrid_classifier_final.pth"

CLASS_NAMES = ["Normal", "Benign", "Malignant"]

# ------------------------------------------------------------
# 2. Load final original model
# ------------------------------------------------------------

class HybridClassifier(torch.nn.Module):

    def __init__(self, input_dim=82, num_classes=3):

        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),

            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


final_model = HybridClassifier(82, 3).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    final_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    final_model.load_state_dict(checkpoint)

final_model.eval()

for p in final_model.parameters():
    p.requires_grad = False

print("[OK] Final original model loaded")


# ============================================================
# SOURCE COLAB CELL 134
# ============================================================

# ============================================================
# CELL 53 — EXTRACT ZIP + LOAD FINAL HOSPITAL TEST SET
# CASE-INSENSITIVE
# ============================================================

from pathlib import Path
from PIL import Image
import numpy as np
import zipfile
import shutil

print("=" * 70)
print("EXTRACTING AND LOADING HOSPITAL TEST DATA")
print("=" * 70)

# ------------------------------------------------------------
# 1. ZIP location
# ------------------------------------------------------------

HOSPITAL_ROOT = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Dataset/Hospital Raw Dataset"

ZIP_PATH = HOSPITAL_ROOT / "DATASET.zip"

print("\nZIP file:")
print(ZIP_PATH)

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"DATASET.zip not found at:\n{ZIP_PATH}"
    )

print("[OK] DATASET.zip found")


# ------------------------------------------------------------
# 2. Extract ZIP
# ------------------------------------------------------------

EXTRACT_DIR = HOSPITAL_ROOT / "DATASET_EXTRACTED"

EXTRACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nExtracting ZIP...")

with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as zip_ref:

    zip_ref.extractall(
        EXTRACT_DIR
    )

print("[OK] ZIP extracted to:")
print(EXTRACT_DIR)


# ------------------------------------------------------------
# 3. Case-insensitive folder finder
# ------------------------------------------------------------

def find_folder_case_insensitive(
    parent,
    folder_name
):

    target = folder_name.lower()

    for item in parent.rglob("*"):

        if (
            item.is_dir()
            and item.name.lower() == target
        ):
            return item

    return None


# ------------------------------------------------------------
# 4. Find DATASET folder
# ------------------------------------------------------------

dataset_folder = find_folder_case_insensitive(
    EXTRACT_DIR,
    "DATASET"
)

if dataset_folder is None:

    print("\n⚠ DATASET folder not found.")

    print("\nExtracted folders:")

    for item in EXTRACT_DIR.rglob("*"):

        if item.is_dir():
            print(
                " -",
                item
            )

    raise FileNotFoundError(
        "DATASET folder not found after extraction."
    )

print("\n[OK] DATASET folder found:")
print(dataset_folder)


# ------------------------------------------------------------
# 5. Find TEST folder
# ------------------------------------------------------------

test_folder = find_folder_case_insensitive(
    dataset_folder,
    "Test"
)

if test_folder is None:

    print("\n⚠ Test folder not found.")

    print("\nAvailable directories:")

    for item in dataset_folder.rglob("*"):

        if item.is_dir():
            print(
                " -",
                item
            )

    raise FileNotFoundError(
        "Test folder not found."
    )

print("\n[OK] TEST folder found:")
print(test_folder)


# ------------------------------------------------------------
# 6. Class mapping
# ------------------------------------------------------------

label_map = {
    "Normal": 0,
    "Benign": 1,
    "Malignant": 2
}

test_images = []
test_labels = []
test_filenames = []


# ------------------------------------------------------------
# 7. Load test images
# ------------------------------------------------------------

for class_name, label in label_map.items():

    class_folder = find_folder_case_insensitive(
        test_folder,
        class_name
    )

    if class_folder is None:

        print(
            f"\n⚠ {class_name} folder not found"
        )

        continue

    print(
        f"\n[OK] {class_name} folder:"
    )

    print(
        f"  {class_folder}"
    )

    image_count = 0

    for file_path in sorted(
        class_folder.iterdir()
    ):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff"
        ]:
            continue

        try:

            image = Image.open(
                file_path
            ).convert("L")

            image = np.array(
                image
            ).astype(np.uint8)

            test_images.append(
                image
            )

            test_labels.append(
                label
            )

            test_filenames.append(
                file_path.name
            )

            image_count += 1

        except Exception as e:

            print(
                f"⚠ Error reading "
                f"{file_path.name}: {e}"
            )

    print(
        f"  Images loaded: {image_count}"
    )


# ------------------------------------------------------------
# 8. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL HOSPITAL TEST DATA SUMMARY")
print("=" * 70)

print(
    "\nTotal images:",
    len(test_images)
)

print("\nClass distribution:")

for class_name, label in label_map.items():

    count = sum(
        y == label
        for y in test_labels
    )

    print(
        f"{class_name:10s}: {count}"
    )


# ------------------------------------------------------------
# 9. Verify expected dataset
# ------------------------------------------------------------

expected_total = 17

if len(test_images) == expected_total:

    print(
        "\n[OK] ALL 17 HOSPITAL TEST CASES CONFIRMED"
    )

else:

    print(
        f"\n⚠ Expected {expected_total} images "
        f"but found {len(test_images)}."
    )

    print(
        "STOP HERE and check the output above."
    )


# ============================================================
# SOURCE COLAB CELL 135
# ============================================================

# ============================================================
# CELL 54 — FINAL HOSPITAL TEST FEATURE EXTRACTION
# CORRECTED: 224 x 224 BEFORE HANDCRAFTED FEATURES
# ============================================================

import joblib
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path

print("=" * 70)
print("GENERATING FINAL 82-D FEATURES")
print("=" * 70)

MODEL_DIR = Path(__file__).resolve().parent / "results" / "drive/MyDrive/Project work 1/Hospital Model"

# ------------------------------------------------------------
# 1. Load the SAME fitted handcrafted pipeline
# ------------------------------------------------------------

pipeline_path = (
    MODEL_DIR /
    "handcrafted_feature_pipeline_complete.joblib"
)

pipeline = joblib.load(
    pipeline_path
)

variance_selector = pipeline["variance_selector"]
correlated_features = pipeline["correlated_features"]
feature_selector = pipeline["feature_selector"]
feature_scaler = pipeline["feature_scaler"]

print("[OK] Handcrafted pipeline loaded")
print(
    "Expected original features:",
    variance_selector.n_features_in_
)

assert variance_selector.n_features_in_ == 6146

# ------------------------------------------------------------
# 2. Prepare ConvNeXt
# ------------------------------------------------------------

convnext.eval()

for p in convnext.parameters():
    p.requires_grad = False

# ------------------------------------------------------------
# 3. Generate final features
# ------------------------------------------------------------

final_test_features = []

for idx, image in enumerate(test_images):

    print(
        f"\nProcessing Case {idx + 1}/{len(test_images)}"
    )

    # ========================================================
    # PART A — SEGMENT LUNG
    # ========================================================

    segmentation_result = segment_lung(image)

    lung_mask = segmentation_result[1]
    lung_roi = segmentation_result[3]

    # --------------------------------------------------------
    # IMPORTANT:
    # Resize BOTH image and mask to 224 x 224.
    # This matches the feature-extraction pipeline used during
    # model development.
    # --------------------------------------------------------

    lung_roi_224 = cv2.resize(
        lung_roi,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    lung_mask_224 = cv2.resize(
        lung_mask,
        (224, 224),
        interpolation=cv2.INTER_NEAREST
    )

    lung_mask_224 = (
        lung_mask_224 > 0
    ).astype(np.uint8) * 255

    # ========================================================
    # PART B — HANDCRAFTED FEATURES
    # ========================================================

    intensity_features = extract_intensity_features(
        lung_roi_224,
        lung_mask_224
    )

    glcm_features = extract_glcm_features(
        lung_roi_224,
        lung_mask_224
    )

    lbp_features = extract_lbp_features(
        lung_roi_224,
        lung_mask_224,
        radius=2
    )

    hog_features = extract_hog_features(
        lung_roi_224,
        lung_mask_224
    )

    shape_features = extract_shape_features(
        lung_mask_224
    )

    edge_features = extract_edge_features(
        lung_roi_224,
        lung_mask_224
    )

    # --------------------------------------------------------
    # Combine all handcrafted features
    # --------------------------------------------------------

    handcrafted = np.concatenate([
        intensity_features,
        glcm_features,
        lbp_features,
        hog_features,
        shape_features,
        edge_features
    ]).astype(np.float32)

    handcrafted = handcrafted.reshape(
        1, -1
    )

    print(
        "  Handcrafted features:",
        handcrafted.shape
    )

    # --------------------------------------------------------
    # Confirm EXACT 6146 features
    # --------------------------------------------------------

    if handcrafted.shape[1] != 6146:

        raise ValueError(
            f"Expected 6146 handcrafted features, "
            f"but got {handcrafted.shape[1]} "
            f"for Case {idx + 1}."
        )

    # ========================================================
    # PART C — APPLY SAVED FEATURE SELECTION
    # ========================================================

    # Variance filtering
    X_var = variance_selector.transform(
        handcrafted
    )

    print(
        "  After variance filtering:",
        X_var.shape
    )

    # --------------------------------------------------------
    # Correlation filtering
    # --------------------------------------------------------

    keep_indices = np.ones(
        X_var.shape[1],
        dtype=bool
    )

    keep_indices[
        correlated_features
    ] = False

    X_corr = X_var[
        :,
        keep_indices
    ]

    print(
        "  After correlation filtering:",
        X_corr.shape
    )

    # --------------------------------------------------------
    # ANOVA SelectKBest
    # --------------------------------------------------------

    X_selected = feature_selector.transform(
        X_corr
    )

    print(
        "  After ANOVA selection:",
        X_selected.shape
    )

    # --------------------------------------------------------
    # StandardScaler
    # --------------------------------------------------------

    X_hand = feature_scaler.transform(
        X_selected
    )

    print(
        "  Final handcrafted:",
        X_hand.shape
    )

    assert X_hand.shape == (1, 50)

    # ========================================================
    # PART D — CONVNEXT-TINY DEEP FEATURES
    # ========================================================

    processed = prepare_convnext_image(
        image
    )

    processed_uint8 = np.clip(
        processed * 255,
        0,
        255
    ).astype(np.uint8)

    # Grayscale → RGB
    processed_rgb = np.stack(
        [
            processed_uint8,
            processed_uint8,
            processed_uint8
        ],
        axis=-1
    )

    processed_pil = Image.fromarray(
        processed_rgb,
        mode="RGB"
    )

    convnext_input = convnext_transform(
        processed_pil
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        deep = convnext(
            convnext_input
        )

        deep = torch.flatten(
            deep,
            1
        )

    deep = deep.cpu().numpy()

    print(
        "  ConvNeXt features:",
        deep.shape
    )

    assert deep.shape == (1, 768)

    # --------------------------------------------------------
    # Deep feature scaler
    # --------------------------------------------------------

    deep_scaled = deep_scaler.transform(
        deep
    )

    # --------------------------------------------------------
    # PCA: 768 → 32
    # --------------------------------------------------------

    deep_pca_features = deep_pca.transform(
        deep_scaled
    )

    print(
        "  ConvNeXt PCA:",
        deep_pca_features.shape
    )

    assert deep_pca_features.shape == (1, 32)

    # ========================================================
    # PART E — FUSION
    # ========================================================

    fused = np.concatenate(
        [
            X_hand,
            deep_pca_features
        ],
        axis=1
    )

    print(
        "  Fused features:",
        fused.shape
    )

    assert fused.shape == (1, 82)

    final_test_features.append(
        fused[0]
    )


# ============================================================
# FINAL MATRICES
# ============================================================

X_hospital_final = np.asarray(
    final_test_features,
    dtype=np.float32
)

y_hospital_final = np.asarray(
    test_labels,
    dtype=np.int64
)

print("\n" + "=" * 70)
print("FINAL FEATURE MATRIX")
print("=" * 70)

print(
    "X_hospital_final:",
    X_hospital_final.shape
)

print(
    "y_hospital_final:",
    y_hospital_final.shape
)

assert X_hospital_final.shape == (17, 82)
assert y_hospital_final.shape == (17,)

print(
    "\n[OK] 17 × 82 FINAL FEATURE MATRIX CONFIRMED"
)


# ============================================================
# SOURCE COLAB CELL 136
# ============================================================

# ============================================================
# CELL 55 — FINAL HOSPITAL TEST PREDICTION & EVALUATION
# ============================================================

from sklearn.metrics import (

    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("FINAL HOSPITAL TEST EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Convert final 82-D features to tensor
# ------------------------------------------------------------

X_final_tensor = torch.tensor(
    X_hospital_final,
    dtype=torch.float32
).to(device)

# ------------------------------------------------------------
# 2. Final prediction
# ------------------------------------------------------------

final_model.eval()

with torch.no_grad():

    logits = final_model(
        X_final_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    ).cpu().numpy()

# ------------------------------------------------------------
# 3. Calculate metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_hospital_final,
    predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_hospital_final,
    predictions
)

macro_precision = precision_score(
    y_hospital_final,
    predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_hospital_final,
    predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_hospital_final,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_hospital_final,
    predictions,
    average="weighted",
    zero_division=0
)

# ------------------------------------------------------------
# 4. Display final metrics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL HOSPITAL TEST METRICS")
print("=" * 70)

print(
    f"Accuracy          : {accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy : {balanced_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision   : {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall      : {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1          : {macro_f1 * 100:.2f}%"
)

print(
    f"Weighted F1       : {weighted_f1 * 100:.2f}%"
)

# ------------------------------------------------------------
# 5. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_hospital_final,
    predictions,
    labels=[0, 1, 2]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print("\nRows    = Actual")
print("Columns = Predicted")
print(
    "Classes =",
    CLASS_NAMES
)

# ------------------------------------------------------------
# 6. Classification Report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_hospital_final,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)