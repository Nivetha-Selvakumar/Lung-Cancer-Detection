# iq-Gp — Exact Colab-derived implementation

Source: `PW1_LC_Genetic_Programming(5).ipynb`

This folder is derived directly from the uploaded notebook.

## Cell mapping

- `train.py` -> Cells 0–56, in original order.
- `preprocess.py` -> Cell 11 (dependency imports included).
- `features.py` -> Cell 14 (dependency imports included).
- `feature_selection.py` -> Cells 20–21.
- `metrics.py` -> Cells 31–35, 50–51, 53–56.
- `predict.py` -> Cells 52 and 56 (the notebook's final test-set prediction workflow).
- `config.json` -> exact settings/cell map.
- `requirements.txt` -> packages imported/used by the notebook.

## ONLY intentional modification

Original Cells 3–5 used Google Colab Google Drive and:
`/content/drive/MyDrive/Project work 1/Dataset/archive.zip`

In `train.py`, only dataset access was changed to:
`DATASET_ZIP_PATH` environment variable, defaulting to `archive.zip`.

No GP preprocessing, feature extraction, standardization, PCA, GP tuning,
iteration logic, model selection, or evaluation algorithm was replaced.

## Run

Install:
`pip install -r requirements.txt`

Windows:
`set DATASET_ZIP_PATH=C:\path\to\archive.zip`
`python train.py`

Linux/macOS:
`export DATASET_ZIP_PATH=/path/to/archive.zip`
`python train.py`
