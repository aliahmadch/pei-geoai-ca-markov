# Trained classifiers

Nine classifiers x five years (2021-2025) = 45 models, one per classifier and year, trained on the five-class reference
pixels of that year. File names: `<year>_<classifier>.<extension>`.

| Classifier | Input | File | Load with |
|---|---|---|---|
| CatBoost | 13 features | `.cbm` | `catboost.CatBoostClassifier().load_model(path)` |
| LightGBM | 13 features | `.txt` | `lightgbm.Booster(model_file=path)` |
| XGBoost | 13 features | `.json` | `xgboost.Booster().load_model(path)` |
| DeepMLP | 13 features | `.npz` (weights, scaler; plain arrays) and `.joblib` (scikit-learn objects) | `numpy.load`, `joblib.load` |
| FT-Transformer | 13 features | `.pt` | `torch.load(path, weights_only=False)` and `nets.make_net` |
| VCHNet, Swin-GNNet, FMTUNet, TimeSenCLIP | 13 x 9 x 9 window | `.pt` | as FT-Transformer |

`predict_example.py` loads any of the models and classifies the blocked test pixels of its year; `nets.py` defines the
network architectures. The `.pt` files hold a dictionary: `name`, `hp` (configuration), `state` (PyTorch state dict) and
`mean`, `std` (per-feature or per-band standardisation from the training pixels).

## Input

Per pixel, 13 features in this order: B2, B3, B4, B8, B11, B12 (Sentinel-2 surface reflectance, 0-1), NDVI, NDWI, NDBI,
BSI, elev (NASADEM elevation / 300 m), slope (slope / 45 degrees), gsw_occ (JRC Global Surface Water occurrence / 100);
non-finite values are set to 0. The patch networks take the 13 x 9 x 9 window centred on the pixel (`data/patches`).
DeepMLP and the networks standardise their input internally (scaler or `mean`/`std` stored with the model); the tree
ensembles use the features as they are.

## Output

Class codes 1 Forest, 2 Agricultural land (cropland and grassland), 3 Wetland, 4 Settlement, 5 Otherland. The tree
ensembles, DeepMLP and the networks return 0-4 (add 1).

## Training

Configuration: the one selected for each classifier by spatially blocked cross-validation (`tuning/selected_configurations.csv`).
Each model was fitted to the training partition (seed 42, 80% of the 3 km blocks) of its year with a fixed seed (42) on one
CPU thread (Intel Xeon 6972P). Training software: Python 3.11, numpy 1.26.4, scikit-learn 1.8.0, PyTorch 2.14.1,
XGBoost 3.2.0, LightGBM 4.7.0, CatBoost 1.2.7. Reloading every model with newer versions (scikit-learn 1.9.1, PyTorch 2.15.0,
XGBoost 3.4.1, CatBoost 1.2.10) reproduced its stored test predictions (`results/test_predictions/tuned`) for all
test pixels. Accuracy by year: `results/classifiers/classifier_by_year.csv`.

The annual maps were produced with the XGBoost models (`code/mapping/classify_tiles.py`).
