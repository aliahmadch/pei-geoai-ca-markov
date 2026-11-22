"""Constants and the two functions used for training: the 3 km block partition (as in the sample extraction) and the
feature matrix."""
import numpy as np, pandas as pd
CLASS_NAMES = {1: "forest", 2: "cropland", 3: "grassland", 4: "wetland", 5: "settlement", 6: "otherland"}
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI", "BSI", "elev", "slope", "gsw_occ"]
BLOCK_M = 3000
SEED = 42
TEST_FRAC = 0.2


def spatial_mask(df: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    blocks = df["block_id"].unique()
    rng.shuffle(blocks)
    n_test = max(1, int(round(len(blocks) * TEST_FRAC)))
    test_blocks = set(blocks[:n_test])
    train = ~df["block_id"].isin(test_blocks)
    # Keep every class in both sides when possible.
    for attempt in range(8):
        tr_cls = set(df.loc[train, "class"])
        te_cls = set(df.loc[~train, "class"])
        if tr_cls == te_cls == set(CLASS_NAMES):
            break
        rng.shuffle(blocks)
        test_blocks = set(blocks[:n_test])
        train = ~df["block_id"].isin(test_blocks)
    return train.to_numpy()


def feature_matrix(df: pd.DataFrame):
    X = df[BANDS].to_numpy(dtype=np.float32)
    y = df["class"].to_numpy(dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y
