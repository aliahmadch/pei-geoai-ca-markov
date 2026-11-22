# Spatially Validated GeoAI and ANN-CA-Markov Modelling for Land-Change Scenarios in Prince Edward Island

Code, trained models and data for:

Ahmad, A., Khan, M.S., Hussain, N., Rafiq, M.M., Yaseen, E., Farooque, A.A. *Spatially Validated GeoAI and ANN-CA-Markov
Modelling for Land-Change Scenarios in Prince Edward Island.* (submitted)

Full archive (this repository plus the large files): Zenodo, https://doi.org/10.5281/zenodo.24238874

## Overview

Five land-cover classes (forest, agricultural land, wetland, settlement, otherland) were mapped across Prince Edward
Island (PEI), Canada, at 10 m for each year from 2021 to 2025 from June-September Sentinel-2 composites. Agricultural land
combines cropland and grassland, which are rotation phases of the same land use.

- Nine GeoAI classifiers (CatBoost, LightGBM, XGBoost, DeepMLP, FT-Transformer, VCHNet, Swin-GNNet, FMTUNet, TimeSenCLIP)
  were tuned by spatially blocked cross-validation and evaluated on a 3 km spatial-block holdout (20% of blocks, seed 42).
- The annual maps were produced with XGBoost and a 3 x 3 majority filter.
- An ANN-CA-Markov land-change model was tested with an equal-interval hindcast (calibrated on 2021-2023, simulated
  2023-2025) and used for a conditional 2029 scenario.

## What is here and what is on Zenodo

| Folder | Contents | GitHub | Zenodo |
|---|---|:-:|:-:|
| `1_Trained_models/` | 45 trained classifiers (9 classifiers x 5 years), model card, loading example | yes | yes |
| `2_Fine_tune_settings/` | Search space, all 270 tuning trials, selected configurations | yes | yes |
| `3_Data/1_Reference_polygons_and_boundary/` | Reference polygons 2021-2025 (by year and cover type) and the island boundary | yes | yes |
| `3_Data/2_Sampled_pixels/` | Sampled pixels per year with the 13 features, class labels and block partitions | yes | yes |
| `3_Data/3_Image_patches_9x9/` | 13 x 9 x 9 windows used by the patch networks (152 MB) | - | yes |
| `3_Data/4_Maps_2021-2025/` | Annual XGBoost maps, raw and 3 x 3 majority (43 MB) | - | yes |
| `3_Data/5_Land_change_model_hindcast_2025_scenario_2029/` | Driver inputs, land mask, transition-potential model, 2025 hindcast and 2029 scenario rasters (124 MB) | - | yes |
| `3_Data/6_Result_tables/` | Accuracy, bootstrap, leakage, separability, map and class-area tables; test predictions | yes | yes |
| `3_Data/7_Sentinel-2_feature_tiles/` | Sentinel-2 feature tiles (950 tiles x 13 bands per year, about 2.1 GB per year) | note only | yes |
| `4_Code/` | Python code: sample extraction, tuning and classifiers, mapping, land-change model | yes | yes |

On Zenodo each folder is one zip (for example `3_Data_4_Maps_2021-2025.zip`); the feature tiles are
`feature_tiles_v4_2021.zip` ... `feature_tiles_v4_2025.zip`. Unzip everything into one folder to get the layout above.

## Data description

- **Features (13, in this order):** B2, B3, B4, B8, B11, B12 (Sentinel-2 Level-2A surface reflectance, median of
  1 June-30 September), NDVI, NDWI, NDBI, BSI, elevation / 300 m (NASADEM), slope / 45 degrees, JRC Global Surface Water
  occurrence / 100.
- **Classes:** 1 Forest, 2 Agricultural land, 3 Wetland, 4 Settlement, 5 Otherland (0 = no data / ocean in the maps).
- **Grid:** 10 m, NAD83(CSRS) / Prince Edward Island Stereographic (EPSG:2954).
- **Reference polygons:** digitised separately for each year by visual interpretation in Google Earth; 3,356 cleaned
  polygons in total.

## Running the code

Packages and versions: `4_Code/environment/requirements.txt`. The steps and commands are in `4_Code/README.md`; the
trained models are described in `1_Trained_models/MODEL_CARD.md`.

The scripts use the folder names of the original data set (`data/tables`, `data/patches`, `models`, `maps`,
`land_change`, ...). This table maps them to the folders of this repository:

| Name used in the code and in `4_Code/README.md` | Folder here |
|---|---|
| `data/samples` | `3_Data/1_Reference_polygons_and_boundary/reference_polygons` |
| `data/boundary` | `3_Data/1_Reference_polygons_and_boundary/island_boundary` |
| `data/tables` | `3_Data/2_Sampled_pixels` |
| `data/patches` | `3_Data/3_Image_patches_9x9` |
| `data/feature_tiles_v4` | created by unzipping the feature-tile zips in the top folder |
| `data/drivers/source` | `3_Data/5_Land_change_model_hindcast_2025_scenario_2029/driver_inputs` |
| `models` | `1_Trained_models` |
| `maps` | `3_Data/4_Maps_2021-2025` |
| `land_change` | `3_Data/5_Land_change_model_hindcast_2025_scenario_2029` |
| `tuning` | `2_Fine_tune_settings` |
| `results` | `3_Data/6_Result_tables` |
| `code` | `4_Code` |

To use the top folder as the "data set folder" of the code, create these links once (macOS/Linux, from the top folder,
after unzipping the Zenodo files there):

```bash
mkdir -p data/drivers
ln -s ../3_Data/1_Reference_polygons_and_boundary/reference_polygons data/samples
ln -s ../3_Data/1_Reference_polygons_and_boundary/island_boundary data/boundary
ln -s ../3_Data/2_Sampled_pixels data/tables
ln -s ../3_Data/3_Image_patches_9x9 data/patches
ln -s ../../3_Data/5_Land_change_model_hindcast_2025_scenario_2029/driver_inputs data/drivers/source
ln -s 1_Trained_models models
ln -s 3_Data/4_Maps_2021-2025 maps
ln -s 3_Data/5_Land_change_model_hindcast_2025_scenario_2029 land_change
```

Then pass the top folder wherever the commands ask for `<data set folder>`. The classifier scripts find `data/` on their
own (or set `PEI_DATA=<top folder>/data`).

## Sources and attribution

Contains modified Copernicus Sentinel data (2021-2025). NASADEM (NASA) and JRC Global Surface Water accessed through
Google Earth Engine. Roads and waterways from OpenStreetMap (Open Database License).

## Licence

Code: MIT (`LICENSE`). Data: CC BY 4.0.

## Citation

Please cite the paper above and the Zenodo archive (https://doi.org/10.5281/zenodo.24238874). See `CITATION.cff`.
