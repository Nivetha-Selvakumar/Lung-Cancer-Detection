# iq-Convnext-tiny — Exact Colab-derived folder

Source notebook:
`DL_Segemntation_Improved (1)(4).ipynb`

This folder is an extraction of the notebook. The methodology is not
replaced by a reconstructed pipeline.

## ONLY changed item

The original Google Drive dataset access in Cells 3-4 was changed to:

```python
DATASET_ZIP_PATH = os.environ.get(
    "DATASET_ZIP_PATH",
    "data/archive.zip"
)
ZIP_PATH = DATASET_ZIP_PATH
EXTRACT_PATH = os.environ.get(
    "EXTRACT_PATH",
    "data/iqoth_nccd_convnext"
)
```

The remaining notebook logic is preserved.

## Cell mapping

- `train.py`: Cells 1-3, 5-58 (Cell 4's original duplicate dataset path is omitted because Cell 3 now supplies the path)
- `preprocess.py`: Cells 15-19 — lung segmentation, sample segmentation, caching and segmented-path mapping
- `features.py`: Cells 20-25 — image transforms, PyTorch Dataset, class-imbalance sampler and DataLoaders
- `feature_selection.py`: No feature-selection algorithm exists in this ConvNeXt notebook
- `metrics.py`: Cells 31-58 — training/validation functions, Stage 1, Stage 2, stronger fine-tuning, TTA, comparison and final evaluation
- `predict.py`: Cells 59-66 — upload prediction, diagnostic input comparison, Grad-CAM, local XAI report, Gemini installation/test and Gemini explanation

## Exact segmentation

Cell 16 is the actual classical lung segmentation function:
Gaussian blur -> inverted Otsu -> remove border-connected background ->
morphological opening -> largest 1-2 components -> fill holes ->
morphological closing -> dilation -> fill holes -> lung ROI.

This is lung-field/ROI extraction, not tumor/lesion segmentation.

## Training

- ConvNeXt-Tiny pretrained ImageNet weights
- Stage 1 classifier-only training
- Stage 2 unfreezes feature stages [4, 6]
- Stronger fine-tuning unfreezes [0, 2, 4, 6]
- Benign-aware Focal Loss
- validation Macro F1 primary selection
- TTA selected only from validation Macro F1
- final test evaluation on untouched test split

## XAI

Cells 61-63 contain the notebook's Grad-CAM and natural-language XAI report.
Cells 64-66 contain Gemini installation/API test and the Gemini + Grad-CAM explanation.

No feature-selection/PCA stage is added because it is not present in this notebook.
