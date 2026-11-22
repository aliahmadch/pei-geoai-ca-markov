"""Prepare the working folder of the land-change model from the files of this data set.

    python prepare_inputs.py --root <data set folder> --work <working folder>

Writes to the working folder (land vectors follow the row-major order of the land pixels):
  land_mask.npy        land pixels of the 10 m grid (EPSG:2954; origin 291810 E, 779250 N; 18,725 x 12,392 pixels)
  block_land.npy       3 km block of each land pixel (rank of its block_id among the blocks with land)
  maj_land.npy         class of each land pixel in the annual maps after the 3 x 3 majority filter, 2021-2025 (5 x n)
  raw_land.npy         the same for the unfiltered annual maps
  drivers_std_f16.npy  the five drivers of each land pixel, standardised (n x 5, float16): elevation, slope,
                       log(1 + population), log(1 + distance to roads), log(1 + distance to waterways), each as z-scores
                       over the land pixels (mean and standard deviation in drivers_standardisation.json)
  test_<year>.csv      blocked test pixels of each year (x, y, class, block_id), for accuracy at independent pixels
  train_<year>.csv     training pixels of each year (x, y, class, block_id)"""
import argparse, json, os
import numpy as np, pandas as pd, rasterio

YEARS = range(2021, 2026)
DRIVERS = [('elevation', False), ('slope', False), ('population', True), ('dist_roads', True), ('dist_water', True)]


def read(path):
    with rasterio.open(path) as d:
        assert d.crs.to_epsg() == 2954 and (d.transform.c, d.transform.f, d.transform.a, d.transform.e) == (291810.0, 779250.0, 10.0, -10.0)
        return d.read(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--work', required=True); a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True); W = lambda f: os.path.join(a.work, f)
    land = read(os.path.join(a.root, 'land_change', 'land_mask_10m.tif')).astype(bool); np.save(W('land_mask.npy'), land)
    # 3 km blocks: block_id = floor(x / 3000) * 10^7 + floor(y / 3000) at the pixel centre, as for the sampled pixels
    r, c = np.nonzero(land); r = r.astype(np.int32); c = c.astype(np.int32)
    bx = (291815 + 10 * c) // 3000; by = (779245 - 10 * r) // 3000; del r, c
    key = bx.astype(np.int64) * 10_000_000 + by; del bx, by
    np.save(W('block_land.npy'), np.unique(key, return_inverse=True)[1].astype(np.int32).ravel()); del key
    for kind, suffix in (('maj', '3x3majority'), ('raw', 'raw')):
        S = np.stack([read(os.path.join(a.root, 'maps', f'PEI_LULC_{y}_XGBoost_5class_{suffix}.tif'))[land] for y in YEARS])
        assert S.min() >= 1 and S.max() <= 5; np.save(W(f'{kind}_land.npy'), S); del S
    D = np.empty((int(land.sum()), len(DRIVERS)), np.float16); st = {}
    for j, (name, log) in enumerate(DRIVERS):
        v = read(os.path.join(a.root, 'data', 'drivers', '10m', f'{name}_10m.tif'))[land].astype(np.float64)
        if log: v = np.log1p(v)
        mu, sd = v.mean(), v.std(); D[:, j] = ((v - mu) / sd).astype(np.float16); st[name] = dict(log1p=log, mean=float(mu), std=float(sd)); del v
    np.save(W('drivers_std_f16.npy'), D); json.dump(st, open(W('drivers_standardisation.json'), 'w'), indent=1)
    for y in YEARS:
        t = pd.read_csv(os.path.join(a.root, 'data', 'tables', f'train_{y}.csv'))
        for part in ('test', 'train'):
            t[t.partition_seed42 == part][['x', 'y', 'class5', 'block_id']].rename(columns={'class5': 'class'}).to_csv(W(f'{part}_{y}.csv'), index=False)
    print('land pixels', int(land.sum()), '| working folder ready:', a.work)


if __name__ == '__main__':
    main()
