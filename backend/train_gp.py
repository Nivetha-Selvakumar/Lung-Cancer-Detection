import os
import joblib
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from gplearn.genetic import SymbolicClassifier

SEED = 42
np.random.seed(SEED)

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

def train_gp_aspect():
    print("=" * 70)
    print("ASPECT 2: TRAINING GENETIC PROGRAMMING (SYMBOLIC EVOLUTIONARY AI)")
    print("=" * 70)

    dataset_df = load_iqothnccd_dataset()

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

    def get_gp_matrix(df):
        gp_list = []
        labels = []
        for idx, row in df.iterrows():
            feat = extract_gp_features(row["filepath"])
            gp_list.append(feat)
            labels.append(row["label"])
        return np.array(gp_list, dtype=np.float32), np.array(labels, dtype=np.int32)

    print("Extracting GP multi-feature representations (HOG + LBP + Intensity)...")
    X_train_gp_raw, y_train = get_gp_matrix(train_df)
    X_val_gp_raw, y_val = get_gp_matrix(val_df)
    X_test_gp_raw, y_test = get_gp_matrix(test_df)

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

    print("Evolving Symbolic Mathematical Programs across 3 OvR Classifiers...")
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
    preds = np.argmax(gp_probs_norm, axis=1)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"\n[Genetic Programming Result] Validation Accuracy: {acc * 100:.2f}% | Macro F1: {f1:.4f}")

    joblib.dump(gp_scaler, os.path.join(SAVED_MODELS_DIR, "gp_scaler.joblib"))
    joblib.dump(gp_pca, os.path.join(SAVED_MODELS_DIR, "gp_pca.joblib"))
    joblib.dump(gp_models, os.path.join(SAVED_MODELS_DIR, "gp_models.joblib"))
    print(f"[SUCCESS] Saved Genetic Programming artifacts to {SAVED_MODELS_DIR}")

if __name__ == "__main__":
    train_gp_aspect()
