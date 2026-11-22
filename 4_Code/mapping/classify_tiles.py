"""Annual five-class map of PEI from the feature tiles of one year with the saved XGBoost model of that year.

    python classify_tiles.py --year 2021 --tiles data/feature_tiles_v4/2021 --model models/2021_xgboost.json \
                             --land-mask land_change/land_mask_10m.tif --out maps

Each tile (13 bands, float32; B2, B3, B4, B8, B11, B12, NDVI, NDWI, NDBI, BSI, elev, slope, gsw_occ) is classified pixel by
pixel: non-finite values are set to 0, and pixels whose 13 features are all 0 (outside the image) are left unclassified.
Tiles are mosaicked in the order of their names on the 10 m grid (EPSG:2954; origin 291810 E, 779250 N; 18,725 x 12,392
pixels); where tiles overlap, a classified pixel of a later tile replaces the earlier value. The mosaic is masked to land.
A 3 x 3 majority filter (the centre pixel included; ties go to the lowest class code; unclassified pixels stay 0) gives
the filtered map. Outputs (uint8, 0 = no data): PEI_LULC_<year>_XGBoost_5class_raw.tif and ..._3x3majority.tif."""
import argparse, glob, os
import numpy as np, rasterio, xgboost as xgb
from rasterio.transform import from_origin

LEFT, TOP, RES, H, W = 291810.0, 779250.0, 10.0, 12392, 18725
LEGEND = '0 no data, 1 Forest, 2 Agricultural land, 3 Wetland, 4 Settlement, 5 Otherland'


def classify(booster, X):
    p = booster.inplace_predict(X)
    return (p.argmax(1) + 1 if p.ndim == 2 else p.astype(int) + 1).astype(np.uint8)


def majority(raw):
    p = np.pad(raw, 1); best = np.zeros_like(raw); bestn = np.zeros(raw.shape, np.uint8)
    for c in range(1, 6):
        cnt = np.zeros(raw.shape, np.uint8)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2): cnt += (p[dy:dy + raw.shape[0], dx:dx + raw.shape[1]] == c)
        upd = cnt > bestn; best[upd] = c; bestn[upd] = cnt[upd]
    best[raw == 0] = 0
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True); ap.add_argument('--tiles', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--land-mask', required=True); ap.add_argument('--out', required=True); ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    booster = xgb.Booster(); booster.load_model(a.model); booster.set_param({'nthread': a.threads})
    mos = np.zeros((H, W), np.uint8)
    tiles = sorted(glob.glob(os.path.join(a.tiles, 'feat_*.tif'))); assert tiles, 'no tiles found'
    for k, f in enumerate(tiles):
        with rasterio.open(f) as s:
            arr = s.read(); tr = s.transform
        n, h, w = arr.shape; assert n == 13
        r0 = int(round((TOP - tr.f) / RES)); c0 = int(round((tr.c - LEFT) / RES))
        X = np.nan_to_num(arr.reshape(n, -1).T, nan=0.0, posinf=0.0, neginf=0.0)
        valid = np.any(np.abs(X) > 1e-8, axis=1); pred = np.zeros(len(X), np.uint8)
        if valid.any(): pred[valid] = classify(booster, np.ascontiguousarray(X[valid]))
        pred = pred.reshape(h, w); dest = mos[r0:r0 + h, c0:c0 + w]
        mos[r0:r0 + h, c0:c0 + w] = np.where(pred > 0, pred, dest)
        if (k + 1) % 100 == 0: print(a.year, 'tiles', k + 1, 'of', len(tiles), flush=True)
    with rasterio.open(a.land_mask) as d: land = d.read(1).astype(bool)
    raw = mos * land; best = majority(raw)
    prof = dict(driver='GTiff', height=H, width=W, count=1, dtype='uint8', crs='EPSG:2954', transform=from_origin(LEFT, TOP, RES, RES),
                nodata=0, compress='deflate', tiled=True, blockxsize=512, blockysize=512)
    for kind, arr, desc in (('raw', raw, 'unfiltered'), ('3x3majority', best, '3 x 3 majority filter')):
        with rasterio.open(os.path.join(a.out, f'PEI_LULC_{a.year}_XGBoost_5class_{kind}.tif'), 'w', **prof) as d:
            d.write(arr, 1); d.update_tags(DESCRIPTION=f'XGBoost (tuned, five classes) annual map {a.year}, {desc}', LEGEND=LEGEND)
    print(a.year, 'land pixels without a class:', int(((raw == 0) & land).sum()), '| class counts:', np.bincount(raw[land], minlength=6)[1:].tolist())


if __name__ == '__main__':
    main()
