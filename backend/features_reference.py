from typing import Literal
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog, canny
from skimage.measure import regionprops, label as sk_label

def preprocess_image(x):
    return x


# ============================================================
# 1. INTENSITY FEATURES
# ============================================================

def extract_intensity_features(image, mask):

    lung_pixels = image[mask > 0].astype(np.float32)

    if len(lung_pixels) == 0:
        return np.zeros(10, dtype=np.float32)

    percentiles = np.percentile(
        lung_pixels,
        [10, 25, 50, 75, 90]
    )

    mean = np.mean(lung_pixels)
    std = np.std(lung_pixels)

    variance = np.var(lung_pixels)

    minimum = np.min(lung_pixels)
    maximum = np.max(lung_pixels)

    # Skewness
    if std > 1e-8:
        skewness = np.mean(
            ((lung_pixels - mean) / std) ** 3
        )

        # Kurtosis
        kurtosis = np.mean(
            ((lung_pixels - mean) / std) ** 4
        )
    else:
        skewness = 0.0
        kurtosis = 0.0

    features = [
        mean,
        std,
        variance,
        minimum,
        maximum,
        percentiles[0],
        percentiles[1],
        percentiles[2],
        percentiles[3],
        percentiles[4],
        skewness,
        kurtosis
    ]

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# 2. GLCM TEXTURE FEATURES
# ============================================================

def extract_glcm_features(image, mask):

    # Convert normalized image [0,1]
    # into 16 gray levels.
    quantized = np.floor(
        image * 16
    ).astype(np.uint8)

    quantized = np.clip(
        quantized,
        0,
        15
    )

    # Background = 0
    quantized[mask == 0] = 0

    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=16,
        symmetric=True,
        normed=True
    )

    features = []

    GLCMProp = Literal[
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    properties: list[GLCMProp] = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for prop in properties:

        values = graycoprops(
            glcm,
            prop
        )

        features.append(
            np.mean(values)
        )

        features.append(
            np.std(values)
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# 3. LBP TEXTURE FEATURES
# ============================================================

def extract_lbp_features(
    image,
    mask,
    radius=2
):

    points = 8 * radius

    lbp = local_binary_pattern(
        image,
        points,
        radius,
        method="uniform"
    )

    lung_pixels = lbp[
        mask > 0
    ]

    # Uniform LBP produces P + 2 bins
    n_bins = points + 2

    histogram, _ = np.histogram(
        lung_pixels,
        bins=np.arange(
            0,
            n_bins + 1
        ),
        range=(0, n_bins)
    )

    histogram = histogram.astype(
        np.float32
    )

    histogram /= (
        histogram.sum() + 1e-8
    )

    return histogram


# ============================================================
# 4. HOG FEATURES
# ============================================================

def extract_hog_features(
    image,
    mask
):

    roi = image.copy()

    roi[mask == 0] = 0

    hog_features = hog(
        roi,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return hog_features.astype(
        np.float32
    )


# ============================================================
# 5. SHAPE FEATURES
# ============================================================

def extract_shape_features(mask):

    binary_mask = (
        mask > 0
    ).astype(np.uint8)

    labeled = sk_label(
        binary_mask
    )

    regions = regionprops(
        labeled
    )

    # IMPORTANT:
    # Always return exactly 11 features
    # even if no region is detected.
    if len(regions) == 0:
        return np.zeros(
            11,
            dtype=np.float32
        )

    # Largest detected lung region
    largest_region = max(
        regions,
        key=lambda r: r.area
    )

    area = float(
        largest_region.area
    )

    perimeter = float(
        largest_region.perimeter
    )

    major_axis = float(
        largest_region.major_axis_length
    )

    minor_axis = float(
        largest_region.minor_axis_length
    )

    bbox_area = float(
        largest_region.bbox_area
    )

    circularity = (
        4 * np.pi * area /
        (perimeter ** 2 + 1e-8)
    )

    aspect_ratio = (
        major_axis /
        (minor_axis + 1e-8)
    )

    extent = float(
        largest_region.extent
    )

    solidity = float(
        largest_region.solidity
    )

    total_pixels = (
        binary_mask.shape[0] *
        binary_mask.shape[1]
    )

    area_ratio = (
        area /
        total_pixels
    )

    bbox_ratio = (
        bbox_area /
        total_pixels
    )

    features = [
        area,
        perimeter,
        major_axis,
        minor_axis,
        bbox_area,
        circularity,
        aspect_ratio,
        extent,
        solidity,
        area_ratio,
        bbox_ratio
    ]

    return np.asarray(
        features,
        dtype=np.float32
    )

# ============================================================
# 6. EDGE / GRADIENT FEATURES
# ============================================================

def extract_edge_features(
    image,
    mask
):

    # Sobel gradients
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        gx ** 2 +
        gy ** 2
    )

    lung_pixels = gradient_magnitude[
        mask > 0
    ]

    if len(lung_pixels) == 0:
        return np.zeros(
            11,
            dtype=np.float32
        )

    # Canny edge map
    edges = canny(
        image,
        sigma=1.0
    )

    edge_density = np.mean(
        edges[mask > 0]
    )

    mean_gradient = np.mean(
        lung_pixels
    )

    std_gradient = np.std(
        lung_pixels
    )

    max_gradient = np.max(
        lung_pixels
    )

    gradient_percentiles = np.percentile(
        lung_pixels,
        [10, 25, 50, 75, 90]
    )

    features = [
        edge_density,
        mean_gradient,
        std_gradient,
        max_gradient,
        gradient_percentiles[0],
        gradient_percentiles[1],
        gradient_percentiles[2],
        gradient_percentiles[3],
        gradient_percentiles[4]
    ]

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================

def extract_all_features(image_path):

    result = preprocess_image(
        image_path
    )

    image = result["normalized"]
    mask = result["mask"]

    # Extract individual feature groups
    intensity = extract_intensity_features(
        image,
        mask
    )

    glcm = extract_glcm_features(
        image,
        mask
    )

    lbp = extract_lbp_features(
        image,
        mask
    )

    hog_features = extract_hog_features(
        image,
        mask
    )

    shape = extract_shape_features(
        mask
    )
    return np.hstack([intensity, glcm, lbp, hog_features, shape])