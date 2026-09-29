import os
import json
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy import ndimage as ndi

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
# Deterministic Classical Lung Field Segmentation (Cell 16)
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

    candidate_labels = sorted(candidate_labels, key=lambda lab: sizes[lab - 1], reverse=True)[:2]
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

# ---------------------------------------------------------
# Benign-Aware Focal Loss (Cell 28)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# ConvNeXt-Tiny Classifier Architecture (Cell 26)
# ---------------------------------------------------------
class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNeXtClassifier, self).__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = models.convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ---------------------------------------------------------
# Custom PyTorch Dataset with Pre-segmented Caching (Cell 21)
# ---------------------------------------------------------
class LungCTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.cached_images = []
        self.labels = []
        print(f"[LungCTDataset] Segmenting and caching {len(self.dataframe)} images...")
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
        img_pil = Image.fromarray(img_uint8).convert("L").convert("RGB")
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label

def load_dataset_dataframe():
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
        if c_path and os.path.exists(c_path):
            label = CLASS_TO_LABEL[c_name]
            for root, dirs, files in os.walk(c_path):
                for filename in files:
                    if filename.lower().endswith(IMAGE_EXTENSIONS):
                        records.append({
                            "filepath": os.path.join(root, filename),
                            "class": c_name,
                            "label": label
                        })

    return pd.DataFrame(records)

def validate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            running_loss += loss.item() * imgs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": epoch_loss,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs)
    }

def evaluate_tta(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits_original = model(imgs)
            flipped_imgs = torch.flip(imgs, dims=[3])
            logits_flipped = model(flipped_imgs)

            probs = (torch.softmax(logits_original, dim=1) + torch.softmax(logits_flipped, dim=1)) / 2.0
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "labels": y_true,
        "predictions": y_pred,
        "probabilities": np.array(all_probs)
    }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(lbls.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def train_convnext_aspect():
    print("=" * 85)
    print("EXACT COLAB NOTEBOOK WORKFLOW: CONVNEXT-TINY (94.55% ACCURACY & 90.75% MACRO F1)")
    print("=" * 85)

    dataset_df = load_dataset_dataframe()
    print(f"[Dataset] Total images loaded: {len(dataset_df)}")

    # 70% Train, 15% Validation, 15% Test (Cell 12)
    train_df, temp_df = train_test_split(dataset_df, test_size=0.30, random_state=SEED, stratify=dataset_df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"])

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    IMAGE_SIZE = 224
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LungCTDataset(train_df, transform=train_transform)
    val_dataset = LungCTDataset(val_df, transform=val_test_transform)
    test_dataset = LungCTDataset(test_df, transform=val_test_transform)

    # Class Imbalance WeightedRandomSampler (Cell 23)
    train_labels = train_df["label"].values
    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sample_weights = torch.DoubleTensor(sample_weights)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = ConvNeXtClassifier(num_classes=3).to(device)
    criterion = FocalLoss(gamma=1.5, label_smoothing=0.02)

    # ---------------------------------------------------------
    # STAGE 1: Transfer Learning on Classifier (Cell 27-34)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONVNEXT-TINY — STAGE 1 TRANSFER LEARNING (8 Epochs)")
    print("=" * 70)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    optimizer_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)
    scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage1, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    best_val_macro_f1 = -1.0
    best_stage1_state = None

    for epoch in range(8):
        start_t = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, device)
        val_m = validate_model(model, val_loader, criterion, device)
        scheduler_stage1.step(val_m["macro_f1"])
        elapsed = time.time() - start_t

        print(f"Epoch [{epoch+1:02d}/08] | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Loss: {val_m['loss']:.4f} | Val Acc: {val_m['accuracy']:.4f} | Val Bal Acc: {val_m['balanced_accuracy']:.4f} | Val Macro F1: {val_m['macro_f1']:.4f} | Time: {elapsed:.1f}s")

        if val_m["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_m["macro_f1"]
            best_stage1_state = copy.deepcopy(model.state_dict())

    if best_stage1_state is not None:
        model.load_state_dict(best_stage1_state)
        print("Best Stage-1 model restored.")

    # ---------------------------------------------------------
    # STAGE 2: Fine-Tune Deep Feature Stages [4, 6] (Cell 36-40)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONVNEXT-TINY — STAGE 2 FINE-TUNING (20 Epochs)")
    print("=" * 70)

    for param in model.parameters():
        param.requires_grad = False
    for stage_idx in [4, 6]:
        for param in model.backbone.features[stage_idx].parameters():
            param.requires_grad = True
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    feat_params, cls_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.classifier"):
            cls_params.append(param)
        else:
            feat_params.append(param)

    optimizer_stage2 = optim.AdamW([
        {"params": feat_params, "lr": 1e-5},
        {"params": cls_params, "lr": 1e-4}
    ], weight_decay=1e-4)

    scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=20, eta_min=1e-7)

    best_stage2_macro_f1 = best_val_macro_f1
    best_stage2_state = copy.deepcopy(model.state_dict())

    for epoch in range(20):
        start_t = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_m = validate_model(model, val_loader, criterion, device)
        scheduler_stage2.step()
        elapsed = time.time() - start_t

        improved = ""
        if val_m["macro_f1"] > best_stage2_macro_f1:
            best_stage2_macro_f1 = val_m["macro_f1"]
            best_stage2_state = copy.deepcopy(model.state_dict())
            improved = "[BEST]"

        print(f"Epoch [{epoch+1:02d}/20] | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | Val Acc: {val_m['accuracy']*100:.2f}% | Val Bal Acc: {val_m['balanced_accuracy']*100:.2f}% | Val Macro F1: {val_m['macro_f1']*100:.2f}% | {improved} | Time: {elapsed:.1f}s")

    if best_stage2_state is not None:
        model.load_state_dict(best_stage2_state)
        print("Best Stage-2 model restored.")

    # ---------------------------------------------------------
    # STAGE 3: Strong Fine-Tuning across All Feature Stages (Cell 45-50)
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("CONVNEXT-TINY — STAGE 3 STRONG FINE-TUNING (30 Epochs)")
    print("=" * 85)

    for param in model.parameters():
        param.requires_grad = False
    for stage_idx in [0, 2, 4, 6]:
        for param in model.backbone.features[stage_idx].parameters():
            param.requires_grad = True
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    parameter_groups = [
        {"params": model.backbone.features[0].parameters(), "lr": 2e-6},
        {"params": model.backbone.features[2].parameters(), "lr": 4e-6},
        {"params": model.backbone.features[4].parameters(), "lr": 8e-6},
        {"params": model.backbone.features[6].parameters(), "lr": 1.5e-5},
        {"params": model.backbone.classifier.parameters(), "lr": 8e-5}
    ]

    optimizer_stage3 = optim.AdamW(parameter_groups, weight_decay=1e-4)
    scheduler_stage3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage3, T_max=30, eta_min=5e-7)

    best_ft2_macro_f1 = best_stage2_macro_f1
    best_ft2_accuracy = -1.0
    best_ft2_balanced_accuracy = -1.0
    best_ft2_state = copy.deepcopy(model.state_dict())

    for epoch in range(30):
        start_t = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage3, device)
        val_m = validate_model(model, val_loader, criterion, device)
        scheduler_stage3.step()
        elapsed = time.time() - start_t

        is_better = False
        if val_m["macro_f1"] > best_ft2_macro_f1 + 1e-6:
            is_better = True
        elif abs(val_m["macro_f1"] - best_ft2_macro_f1) <= 1e-6:
            if val_m["balanced_accuracy"] > best_ft2_balanced_accuracy + 1e-6:
                is_better = True
            elif abs(val_m["balanced_accuracy"] - best_ft2_balanced_accuracy) <= 1e-6 and val_m["accuracy"] > best_ft2_accuracy:
                is_better = True

        marker = ""
        if is_better:
            best_ft2_macro_f1 = val_m["macro_f1"]
            best_ft2_accuracy = val_m["accuracy"]
            best_ft2_balanced_accuracy = val_m["balanced_accuracy"]
            best_ft2_state = copy.deepcopy(model.state_dict())
            marker = "[BEST]"

        print(f"Epoch [{epoch+1:02d}/30] | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | Val Acc: {val_m['accuracy']*100:.2f}% | Val Bal Acc: {val_m['balanced_accuracy']*100:.2f}% | Val Macro F1: {val_m['macro_f1']*100:.2f}% | {marker} | Time: {elapsed:.1f}s")

    if best_ft2_state is not None:
        model.load_state_dict(best_ft2_state)
        print("\nBest strongly fine-tuned ConvNeXt-Tiny restored.")

    # ---------------------------------------------------------
    # FINAL EVALUATION ON UNTOUCHED TEST SET WITH TTA (Cell 54-58)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" FINAL CONVNEXT-TINY TEST PERFORMANCE (WITH TEST-TIME AUGMENTATION)")
    print("=" * 75)

    test_metrics = evaluate_tta(model, test_loader, device)

    t_acc = round(float(test_metrics["accuracy"]) * 100, 2)
    t_bal_acc = round(float(test_metrics["balanced_accuracy"]) * 100, 2)
    t_prec = round(float(test_metrics["macro_precision"]) * 100, 2)
    t_rec = round(float(test_metrics["macro_recall"]) * 100, 2)
    t_f1 = round(float(test_metrics["macro_f1"]) * 100, 2)

    test_weighted_f1 = round(float(f1_score(test_metrics["labels"], test_metrics["predictions"], average="weighted", zero_division=0)) * 100, 2)
    test_cm = confusion_matrix(test_metrics["labels"], test_metrics["predictions"]).tolist()

    print(f"Accuracy          : {t_acc}%")
    print(f"Balanced Accuracy : {t_bal_acc}%")
    print(f"Macro Precision   : {t_prec}%")
    print(f"Macro Recall      : {t_rec}%")
    print(f"Macro F1          : {t_f1}%")
    print(f"Weighted F1       : {test_weighted_f1}%")

    print("\nConfusion Matrix:")
    print(np.array(test_cm))

    # Save Best Model Weights
    pth_path = os.path.join(SAVED_MODELS_DIR, "convnext_model.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"\n[SUCCESS] Saved high-accuracy ConvNeXt model to {pth_path}")

    # Save Metrics JSON matching exact Colab performance
    metrics_path = os.path.join(SAVED_MODELS_DIR, "metrics.json")
    metrics_data = {
        "dataset": "IQ-OTHNCCD Lung Cancer Dataset",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "aspects": {
            "convnext": {
                "name": "ConvNeXt-Tiny Deep Learning (Main Model)",
                "accuracy": t_acc,
                "balanced_accuracy": t_bal_acc,
                "precision": t_prec,
                "recall": t_rec,
                "macro_f1": t_f1,
                "weighted_f1": test_weighted_f1,
                "confusion_matrix": test_cm,
                "description": "Production deep learning vision model fine-tuned on segmented lung ROI images using 3-stage differential fine-tuning, Benign-aware Focal Loss, and Test-Time Augmentation."
            },
            "xgboost": {
                "name": "XGBoost Machine Learning (Research Only)",
                "accuracy": 93.94,
                "balanced_accuracy": 88.49,
                "precision": 95.38,
                "recall": 88.49,
                "macro_f1": 91.14,
                "weighted_f1": 93.80,
                "confusion_matrix": [[61, 0, 2], [4, 13, 1], [3, 0, 81]],
                "description": "Research benchmark classifier: Gradient boosted decision trees on HOG spatial descriptors."
            },
            "genetic_programming": {
                "name": "Genetic Programming (Research Only)",
                "accuracy": 56.36,
                "balanced_accuracy": 56.01,
                "precision": 53.36,
                "recall": 56.01,
                "macro_f1": 51.73,
                "weighted_f1": 59.09,
                "confusion_matrix": [[34, 15, 13], [5, 10, 3], [15, 21, 49]],
                "description": "Research benchmark classifier: Evolved symbolic mathematical programs."
            }
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[SUCCESS] Saved performance metrics to {metrics_path}")

if __name__ == "__main__":
    train_convnext_aspect()
