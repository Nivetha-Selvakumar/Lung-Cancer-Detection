# iq-XG — Exact from Colab

Source:
`PW1_LC_Machine_Learning_SEGMENTATION(3).ipynb`

This folder is derived from the uploaded notebook. The model methodology
has not been replaced with a reconstructed implementation.

## Cell mapping

- `train.py` -> Cells 1–40 (original notebook Cells 0–39), in the same order.
- `preprocess.py` -> Cell 12 + Cell 14 (original notebook Cells 11 + 13).
- `features.py` -> Cell 18 (original notebook Cell 17).
- `feature_selection.py` -> Cell 20 + Cell 22 (original notebook Cells 19 + 21).
- `metrics.py` -> Validation/Test metric and report cells (original Cells 27–39).
- `predict.py` -> Original upload/prediction Cell 41 (original Cell 40).
- `config.json` -> Dataset/model configuration from the notebook.
- `requirements.txt` -> Libraries used by the notebook.

## ONLY change made

The original Google Colab Drive dataset access was changed so the dataset
can be supplied locally:

`DATASET_ZIP_PATH=data/archive.zip`

or by setting the `DATASET_ZIP_PATH` environment variable.

No XGBoost parameters, segmentation algorithm, preprocessing, HOG extraction,
standardization, PCA, experiment configurations, validation selection,
test evaluation, or prediction logic was intentionally changed.

## Original pipeline

Lung segmentation
-> 224x224 preprocessing
-> median filtering
-> CLAHE
-> lung masking
-> normalization
-> HOG
-> StandardScaler
-> PCA (128)
-> XGBoost
-> validation-based experiment selection by Macro F1
-> final test evaluation

The notebook uses:
- 70% training
- 15% validation
- 15% test
- seed 42

The separate test set is not used for model selection.

## Run

Place `archive.zip` in `data/`, or set `DATASET_ZIP_PATH`.

Then:

`python train.py`

Note: the original notebook is written for Colab and contains display/plot
calls. This folder preserves those notebook operations rather than silently
rewriting the experiment.
