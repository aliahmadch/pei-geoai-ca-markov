"""13 x 9 x 9 windows around every sampled pixel, read from the feature tile of the pixel's year (input of the four patch
networks). A window is kept only if it lies entirely inside the tile; otherwise ok is False and the pixel is not used by
the patch networks.

    python extract_patches.py --tables data/tables --tiles data/feature_tiles_v4 --out data/patches --years 2021 2022 2023 2024 2025

Output per year: patches_<year>.npz with P (n x 13 x 9 x 9, float32; rows in the order of train_<year>.csv) and ok (n, bool)."""
import argparse, os
import numpy as np, pandas as pd, rasterio

HALF = 4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tables', required=True); ap.add_argument('--tiles', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--years', type=int, nargs='+', default=[2021, 2022, 2023, 2024, 2025]); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for y in a.years:
        df = pd.read_csv(os.path.join(a.tables, f'train_{y}.csv'), usecols=['tile', 'row', 'col'])
        P = np.zeros((len(df), 13, 9, 9), np.float32); ok = np.zeros(len(df), bool)
        for tile, sub in df.groupby('tile'):
            with rasterio.open(os.path.join(a.tiles, str(y), tile)) as s: arr = s.read()
            hh, ww = arr.shape[1:]
            for i, r, c in zip(sub.index, sub.row, sub.col):
                if r - HALF < 0 or c - HALF < 0 or r + HALF >= hh or c + HALF >= ww: continue
                P[i] = arr[:, r - HALF:r + HALF + 1, c - HALF:c + HALF + 1]; ok[i] = True
        np.savez_compressed(os.path.join(a.out, f'patches_{y}.npz'), P=P, ok=ok); print(y, len(df), 'windows inside a tile:', int(ok.sum()))


if __name__ == '__main__':
    main()
