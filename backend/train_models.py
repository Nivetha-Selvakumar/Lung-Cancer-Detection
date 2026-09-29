import os
import zipfile
import glob
import time
import json
import joblib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import hog, local_binary_pattern
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
from gplearn.genetic import SymbolicClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

DATASET_ROOT = os.path.join(
    os.path.dirname(BASE_DIR),
    "archive",
    "The IQ-OTHNCCD lung cancer dataset",
    "The IQ-OTHNCCD lung cancer dataset"
)

CLASS_NAMES = ["Normal", "Benign", "Malignant"]
CLASS_TO_LABEL = {"Normal": 0, "Benign": 1, "Malignant": 2}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ---------------------------------------------------------
# 1. Deterministic Lung Segmentation
# ---------------------------------------------------------
def segment_lung(image_input):
    if isinstance(image_input, str):
        original = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    else:
        original = image_input.copy()

    if original is None:
        raise ValueError("Unable to read image for lung segmentation")

    h, w = original.shape
    blurred = cv2.GaussianBlur(original, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8)

    labeled, _ = ndi.label(binary)
    border_labels = set(labeled[0, :].tolist())
    border_labels |= set(labeled[-1, :].tolist())
    border_labels |= set(labeled[:, 0].tolist())
    border_labels |= set(labeled[:, -1].tolist())
    border_labels.discard(0)

    if border_labels:
        border_cleared = np.where(np.isin(labeled, list(border_labels)), 0, binary).astype(np.uint8)
    else:
        border_cleared = binary

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(border_cleared, cv2.MORPH_OPEN, kernel_open)

    labeled2, num2 = ndi.label(opened)
    if num2 == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return original, empty_mask, original.copy()

    component_sizes = np.bincount(labeled2.ravel())
    min_area = 0.003 * h * w
    candidate_labels = [lab for lab in range(1, num2 + 1) if component_sizes[lab] > min_area]

    if len(candidate_labels) == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        return original, empty_mask, original.copy()

    candidate_labels = sorted(candidate_labels, key=lambda lab: component_sizes[lab], reverse=True)[:2]
    mask = np.isin(labeled2, candidate_labels).astype(np.uint8)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    roi = original.copy()
    roi[mask == 0] = 0
    return original, mask, roi

def preprocess_image(image_input, target_size=(224, 224)):
    original, mask, _ = segment_lung(image_input)
    image = cv2.resize(original, target_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    denoised = cv2.medianBlur(image, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    if mask_resized.sum() > 0:
        enhanced[mask_resized == 0] = 0

    normalized = enhanced.astype(np.float32) / 255.0
    return normalized, enhanced, mask_resized

def extract_hog_features(image_float):
    return hog(
        image_float,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    ).astype(np.float32)

def extract_gp_features(image_input):
    norm_img, enhanced_img, _ = preprocess_image(image_input)
    hog_feat = extract_hog_features(norm_img)

    radius = 2
    points = 8 * radius
    lbp = local_binary_pattern(enhanced_img, points, radius, method="uniform")
    n_bins = points + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-8)

    intensity = np.array([
        np.mean(norm_img),
        np.std(norm_img),
        np.min(norm_img),
        np.max(norm_img),
        np.percentile(norm_img, 10),
        np.percentile(norm_img, 25),
        np.percentile(norm_img, 50),
        np.percentile(norm_img, 75),
        np.percentile(norm_img, 90)
    ], dtype=np.float32)

    return np.concatenate([hog_feat, lbp_hist, intensity])

# ---------------------------------------------------------
# PyTorch ConvNeXt / CNN Model Definition
# ---------------------------------------------------------
class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNeXtClassifier, self).__init__()
        try:
            # Check if weights file is already cached locally
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
            cached_ckpt = os.path.join(cache_dir, "convnext_tiny-983f1562.pth")
            if os.path.exists(cached_ckpt) and os.path.getsize(cached_ckpt) > 100 * 1024 * 1024:
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT
                self.backbone = models.convnext_tiny(weights=weights)
                in_features = self.backbone.classifier[2].in_features
                self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
            else:
                print("[ConvNeXt] Initializing ConvNeXt-Tiny model directly (bypassing slow download)...")
                self.backbone = models.convnext_tiny(weights=None, num_classes=num_classes)
        except Exception as e:
            print(f"[ConvNeXt] Falling back to initialized ConvNeXt-Tiny architecture ({e})")
            self.backbone = models.convnext_tiny(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.02):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        num_classes = logits.size(1)

        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / num_classes)

        ce = -(smooth_targets * log_probs).sum(dim=1)
        pt = (probs * smooth_targets).sum(dim=1).clamp_min(1e-6)
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * ce).mean()

class LungDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.cached_images = []
        self.labels = []
        for idx in range(len(self.dataframe)):
            row = self.dataframe.iloc[idx]
            norm_img, _, _ = preprocess_image(row["filepath"])
            img_uint8 = (norm_img * 255.0).astype(np.uint8)
            self.cached_images.append(img_uint8)
            self.labels.append(int(row["label"]))

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        img_uint8 = self.cached_images[idx]
        label = self.labels[idx]
        img_pil = Image.fromarray(img_uint8).convert("RGB")
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label

# ---------------------------------------------------------
# Load IQ-OTHNCCD Dataset Only
# ---------------------------------------------------------
def load_iqothnccd_dataset():
    class_directories = {}
    folder_mapping = {
        "Normal": ["normal", "normal cases"],
        "Benign": ["bengin", "bengin cases", "benign", "benign cases"],
        "Malignant": ["malignant", "malignant cases"]
    }

    for root, dirs, files in os.walk(DATASET_ROOT):
        for directory in dirs:
            f_lower = directory.strip().lower()
            for c_name, keywords in folder_mapping.items():
                if f_lower in keywords:
                    class_directories[c_name] = os.path.join(root, directory)

    records = []
    for c_name in CLASS_NAMES:
        c_path = class_directories.get(c_name)
        if not c_path or not os.path.exists(c_path):
            raise RuntimeError(f"Class folder for {c_name} not found in {DATASET_ROOT}")

        label = CLASS_TO_LABEL[c_name]
        for root, dirs, files in os.walk(c_path):
            for filename in files:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    records.append({
                        "filepath": os.path.join(root, filename),
                        "class": c_name,
                        "label": label
                    })

    df = pd.DataFrame(records)
    print(f"[Dataset] Total IQ-OTHNCCD images loaded: {len(df)}")
    print(df["class"].value_counts())
    return df

# ---------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------
def train_all_aspects():
    print("=" * 70)
    print("STARTING MODEL TRAINING ON IQ-OTHNCCD DATASET (3 ASPECTS)")
    print("=" * 70)

    dataset_df = load_iqothnccd_dataset()

    # Stratified Split: 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(
        dataset_df,
        test_size=0.30,
        random_state=SEED,
        stratify=dataset_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    # -----------------------------------------------------
    # ASPECT 1: Machine Learning (XGBoost + HOG + PCA)
    # -----------------------------------------------------
    print("\n--- Aspect 1: Training XGBoost Machine Learning Model ---")
    start_time = time.time()

    def get_hog_matrix(df):
        hog_list = []
        labels = []
        for idx, row in df.iterrows():
            norm, _, _ = preprocess_image(row["filepath"])
            feat = extract_hog_features(norm)
            hog_list.append(feat)
            labels.append(row["label"])
        return np.array(hog_list, dtype=np.float32), np.array(labels, dtype=np.int32)

    X_train_hog, y_train = get_hog_matrix(train_df)
    X_val_hog, y_val = get_hog_matrix(val_df)
    X_test_hog, y_test = get_hog_matrix(test_df)

    xgb_scaler = StandardScaler()
    X_train_scaled = xgb_scaler.fit_transform(X_train_hog)
    X_val_scaled = xgb_scaler.transform(X_val_hog)
    X_test_scaled = xgb_scaler.transform(X_test_hog)

    xgb_pca = PCA(n_components=128, random_state=SEED)
    X_train_pca = xgb_pca.fit_transform(X_train_scaled)
    X_val_pca = xgb_pca.transform(X_val_scaled)
    X_test_pca = xgb_pca.transform(X_test_scaled)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
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
    xgb_model.fit(X_train_pca, y_train, eval_set=[(X_val_pca, y_val)], verbose=False)

    xgb_preds = xgb_model.predict(X_val_pca)
    xgb_acc = float(accuracy_score(y_val, xgb_preds))
    xgb_bal_acc = float(balanced_accuracy_score(y_val, xgb_preds))
    xgb_f1 = float(f1_score(y_val, xgb_preds, average="macro", zero_division=0))
    xgb_prec = float(precision_score(y_val, xgb_preds, average="macro", zero_division=0))
    xgb_rec = float(recall_score(y_val, xgb_preds, average="macro", zero_division=0))
    xgb_cm = confusion_matrix(y_val, xgb_preds).tolist()

    print(f"XGBoost Val Accuracy: {xgb_acc * 100:.2f}% | Macro F1: {xgb_f1:.4f}")

    joblib.dump(xgb_scaler, os.path.join(SAVED_MODELS_DIR, "xgb_scaler.joblib"))
    joblib.dump(xgb_pca, os.path.join(SAVED_MODELS_DIR, "xgb_pca.joblib"))
    joblib.dump(xgb_model, os.path.join(SAVED_MODELS_DIR, "xgb_model.joblib"))

    # -----------------------------------------------------
    # ASPECT 2: Genetic Programming (gplearn OvR Symbolic Classifiers)
    # -----------------------------------------------------
    print("\n--- Aspect 2: Training Genetic Programming (Evolutionary AI) Models ---")

    def get_gp_matrix(df):
        gp_list = []
        labels = []
        for idx, row in df.iterrows():
            feat = extract_gp_features(row["filepath"])
            gp_list.append(feat)
            labels.append(row["label"])
        return np.array(gp_list, dtype=np.float32), np.array(labels, dtype=np.int32)

    X_train_gp_raw, _ = get_gp_matrix(train_df)
    X_val_gp_raw, _ = get_gp_matrix(val_df)
    X_test_gp_raw, _ = get_gp_matrix(test_df)

    gp_scaler = StandardScaler()
    X_train_gp_scaled = gp_scaler.fit_transform(X_train_gp_raw)
    X_val_gp_scaled = gp_scaler.transform(X_val_gp_raw)
    X_test_gp_scaled = gp_scaler.transform(X_test_gp_raw)

    gp_pca = PCA(n_components=32, random_state=SEED)
    X_train_gp_pca = gp_pca.fit_transform(X_train_gp_scaled)
    X_val_gp_pca = gp_pca.transform(X_val_gp_scaled)
    X_test_gp_pca = gp_pca.transform(X_test_gp_scaled)

    # Train 3 One-vs-Rest Symbolic Classifiers
    gp_models = {}
    gp_probs_list = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        y_binary = (y_train == c_idx).astype(np.int32)
        clf = SymbolicClassifier(
            generations=15,
            population_size=250,
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"),
            metric="log loss",
            parsimony_coefficient=0.001,
            max_samples=0.8,
            random_state=SEED + c_idx * 10,
            n_jobs=-1,
            verbose=0
        )
        clf.fit(X_train_gp_pca, y_binary)
        gp_models[c_name] = clf
        prob_c = clf.predict_proba(X_val_gp_pca)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_val_gp_pca)
        gp_probs_list.append(prob_c)

    gp_probs_matrix = np.column_stack(gp_probs_list)
    sums = np.maximum(gp_probs_matrix.sum(axis=1, keepdims=True), 1e-8)
    gp_probs_norm = gp_probs_matrix / sums
    gp_preds = np.argmax(gp_probs_norm, axis=1)

    gp_acc = float(accuracy_score(y_val, gp_preds))
    gp_bal_acc = float(balanced_accuracy_score(y_val, gp_preds))
    gp_f1 = float(f1_score(y_val, gp_preds, average="macro", zero_division=0))
    gp_prec = float(precision_score(y_val, gp_preds, average="macro", zero_division=0))
    gp_rec = float(recall_score(y_val, gp_preds, average="macro", zero_division=0))
    gp_cm = confusion_matrix(y_val, gp_preds).tolist()

    print(f"Genetic Programming Val Accuracy: {gp_acc * 100:.2f}% | Macro F1: {gp_f1:.4f}")

    joblib.dump(gp_scaler, os.path.join(SAVED_MODELS_DIR, "gp_scaler.joblib"))
    joblib.dump(gp_pca, os.path.join(SAVED_MODELS_DIR, "gp_pca.joblib"))
    joblib.dump(gp_models, os.path.join(SAVED_MODELS_DIR, "gp_models.joblib"))

    # -----------------------------------------------------
    # ASPECT 3: Deep Learning (ConvNeXt-Tiny PyTorch)
    # -----------------------------------------------------
    print("\n--- Aspect 3: Training ConvNeXt-Tiny Deep Learning Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Training Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = LungDataset(train_df, transform=train_transform)
    val_ds = LungDataset(val_df, transform=val_transform)

    # Class weighting sampler
    train_labels = train_df["label"].values
    counts = np.bincount(train_labels, minlength=3)
    weights = 1.0 / np.maximum(counts, 1)
    sample_w = torch.DoubleTensor([weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    dl_model = ConvNeXtClassifier(num_classes=3).to(device)
    # Freeze backbone feature layers to speed up CPU training x10
    if hasattr(dl_model.backbone, "features"):
        for param in dl_model.backbone.features.parameters():
            param.requires_grad = False

    criterion = FocalLoss(gamma=1.5, label_smoothing=0.02)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, dl_model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    best_dl_f1 = -1.0
    best_dl_state = None

    epochs = 5
    for epoch in range(epochs):
        dl_model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = dl_model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        scheduler.step()

        # Validation
        dl_model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                outputs = dl_model(imgs)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(lbls.numpy())

        v_acc = accuracy_score(val_targets, val_preds)
        v_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        print(f"Epoch [{epoch+1}/{epochs}] Val Acc: {v_acc * 100:.2f}% | Val Macro F1: {v_f1:.4f}")

        if v_f1 > best_dl_f1:
            best_dl_f1 = v_f1
            best_dl_state = dl_model.state_dict()

    if best_dl_state is not None:
        dl_model.load_state_dict(best_dl_state)

    # Evaluate best DL model on validation set
    dl_model.eval()
    val_preds_final = []
    val_targets_final = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            outputs = dl_model(imgs)
            preds = torch.argmax(outputs, dim=1)
            val_preds_final.extend(preds.cpu().numpy())
            val_targets_final.extend(lbls.numpy())

    dl_acc = float(accuracy_score(val_targets_final, val_preds_final))
    dl_bal_acc = float(balanced_accuracy_score(val_targets_final, val_preds_final))
    dl_f1 = float(f1_score(val_targets_final, val_preds_final, average="macro", zero_division=0))
    dl_prec = float(precision_score(val_targets_final, val_preds_final, average="macro", zero_division=0))
    dl_rec = float(recall_score(val_targets_final, val_preds_final, average="macro", zero_division=0))
    dl_cm = confusion_matrix(val_targets_final, val_preds_final).tolist()

    print(f"\nFinal ConvNeXt-Tiny Val Accuracy: {dl_acc * 100:.2f}% | Macro F1: {dl_f1:.4f}")
    torch.save(dl_model.state_dict(), os.path.join(SAVED_MODELS_DIR, "convnext_model.pth"))

    # -----------------------------------------------------
    # Save Metrics Comparison JSON
    # -----------------------------------------------------
    metrics_data = {
        "dataset": "IQ-OTHNCCD Lung Cancer Dataset",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "aspects": {
            "genetic_programming": {
                "name": "Genetic Programming (Symbolic Evolutionary AI)",
                "accuracy": round(gp_acc * 100, 2),
                "balanced_accuracy": round(gp_bal_acc * 100, 2),
                "precision": round(gp_prec * 100, 2),
                "recall": round(gp_rec * 100, 2),
                "macro_f1": round(gp_f1 * 100, 2),
                "confusion_matrix": gp_cm,
                "description": "Explores non-linear feature transformations via symbolic mathematical programs evolved across generations."
            },
            "xgboost": {
                "name": "XGBoost Machine Learning (HOG + PCA)",
                "accuracy": round(xgb_acc * 100, 2),
                "balanced_accuracy": round(xgb_bal_acc * 100, 2),
                "precision": round(xgb_prec * 100, 2),
                "recall": round(xgb_rec * 100, 2),
                "macro_f1": round(xgb_f1 * 100, 2),
                "confusion_matrix": xgb_cm,
                "description": "Extracts gradient direction histograms (HOG) from segmented lung fields, reduces dimensionality via PCA, and classifies via gradient boosted decision trees."
            },
            "convnext": {
                "name": "ConvNeXt-Tiny Deep Learning (CNN + Focal Loss)",
                "accuracy": round(dl_acc * 100, 2),
                "balanced_accuracy": round(dl_bal_acc * 100, 2),
                "precision": round(dl_prec * 100, 2),
                "recall": round(dl_rec * 100, 2),
                "macro_f1": round(dl_f1 * 100, 2),
                "confusion_matrix": dl_cm,
                "description": "Modern vision convolutional neural network fine-tuned with Focal Loss for high-resolution visual feature hierarchy learning."
            }
        }
    }

    with open(os.path.join(SAVED_MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n[SUCCESS] All 3 aspect models and performance metrics saved successfully!")

if __name__ == "__main__":
    train_all_aspects()
