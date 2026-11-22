# Code

Folder names used below (`data/`, `models/`, `maps/`, `land_change/`, ...) are those of the original data set; the main
`README.md` maps them to the folders of this repository and shows how to link them. Packages: `environment/requirements.txt`.

## 1. Sampled pixels - `sample_extraction/extract_samples.py`

    python extract_samples.py --root <data set folder> --out <output folder>

Reference polygons (`data/samples`, `data/boundary`) and feature tiles (`data/feature_tiles_v4`) -> `train_<year>.csv`
(year, class, x, y, block_id, tile, row, col and the 13 features). The deposited tables (`data/tables`) add the columns
cover_type, class5, class5_name, partition_seed42, partition_seed1 (from `classifiers/common.py: spatial_mask` with seed
42 and 1) and has_9x9_window.

## 2. Image patches - `mapping/extract_patches.py`

    python extract_patches.py --tables <data set folder>/data/tables --tiles <data set folder>/data/feature_tiles_v4 --out <output folder>

## 3. Hyperparameter tuning and classifiers - `classifiers/`

Reads `data/tables` and `data/patches` of the data set (or the folder in the environment variable PEI_DATA) and writes
to `classifiers/results/`.

    python tune.py <classifier> <trial>        # one configuration (0-29) of one classifier; 270 in total
    python select_best.py                      # selected configuration per classifier
    python final.py <classifier> tuned         # final models of 2021-2025 and their test predictions
    python final.py <classifier> default       # the same with the configuration fixed before tuning
    python final.py <classifier> alt           # alternative partition (seed 1): predictions and training time

`tune.sbatch`, `select.sbatch` and `final.sbatch` run the same steps as SLURM job arrays (CPU only). Classifiers:
catboost, lightgbm, xgboost, deepmlp, ft_transformer, vchnet, swingnnet, fmtunet, timesenclip. Search space:
`configs.py` (also `tuning/search_space.json`). The deposited outputs of this step are in `tuning/`, `models/` and
`results/test_predictions/`.

## 4. Annual maps - `mapping/classify_tiles.py`

    python classify_tiles.py --year 2021 --tiles <data set folder>/data/feature_tiles_v4/2021 \
        --model <data set folder>/models/2021_xgboost.json --land-mask <data set folder>/land_change/land_mask_10m.tif --out <output folder>

## 5. Land-change model - `land_change/`

The 10 m driver rasters (not deposited; about 0.8 GB) are rebuilt from `data/drivers/source` by the first command.

    python build_drivers.py --inputs <data set folder>/data/drivers/source --land-mask <data set folder>/land_change/land_mask_10m.tif --out <data set folder>/data/drivers/10m
    python prepare_inputs.py --root <data set folder> --work <working folder>
    PEI_WORK=<working folder> python ca_model.py hindcast
    PEI_WORK=<working folder> python ca_scenario_2029.py
    python write_geotiffs.py --work <working folder> --out <output folder>

`ca_model.py hindcast` simulates 2025 from 2023 (calibration 2021-2023) and saves the transition-potential MLP;
`ca_scenario_2029.py` runs the conditional 2029 scenario from the temporally consistent 2025 baseline.

## Checks made with the deposited files

- Every saved classifier, reloaded, reproduced its stored test predictions for all test pixels.
- `classify_tiles.py` and `extract_patches.py` reproduced the deposited 2021 maps and patches on test tiles.
- `extract_samples.py` reproduced the polygon counts of the paper (3,356 cleaned polygons) and, on test tiles, the class,
  coordinates, block and features of every sampled pixel.
- `build_drivers.py` reproduced exactly the 10 m drivers used for the land-change model; `prepare_inputs.py`,
  `ca_model.py`, `ca_scenario_2029.py` and `write_geotiffs.py` reproduced the deposited land-change results exactly.
