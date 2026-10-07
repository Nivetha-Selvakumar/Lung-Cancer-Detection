# Lung AI — Six Required Model Folders

Required folders:

1. iq-Gp
2. iq-XG
3. iq-Convnext-tiny
4. Hospital-gp
5. hospital-xg
6. hospital-convnext-tiny

Each folder is an independent backend training package.

All six use the same CT preprocessing and the same class order:
Normal, Benign, Malignant.

GP and XGBoost use explicit handcrafted features:
6146 -> variance -> correlation >0.95 -> ANOVA 50 -> StandardScaler.

Standalone ConvNeXt-Tiny uses the same preprocessed lung ROI as the CNN input.
It is kept separate because its learned CNN representation is not the same
thing as handcrafted feature extraction.

Hospital training uses the 70/20/10 split in the supplied scripts and ignores
the separate Test folder. IQ training is kept separate from Hospital training.

Actual accuracies are produced only after running each train.py and are stored
inside that folder's results/metrics.json.
