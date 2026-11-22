"""GeoTIFFs of the land-change results (10 m, EPSG:2954, grid of the annual maps; 0 = no data).

    python write_geotiffs.py --work <working folder> --out <output folder>

hindcast_2025/: the 2025 map simulated from 2023 (calibration 2021-2023) and the random-allocation map of the same
quantities; scenario_2029/: the temporally consistent 2025 baseline, the 2029 scenario and the 2025 -> 2029 transitions
(10 x baseline class + 2029 class; 11, 22, 33, 44, 55 = unchanged)."""
import argparse, os
import numpy as np, rasterio
from rasterio.transform import from_origin

NAMES = ['Forest', 'Agricultural land', 'Wetland', 'Settlement', 'Otherland']
RGB = {1: (27, 120, 55), 2: (230, 171, 2), 3: (33, 102, 172), 4: (214, 96, 77), 5: (150, 150, 150)}
CLS = {'class_%d' % (k + 1): n for k, n in enumerate(NAMES)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    land = np.load(os.path.join(a.work, 'land_mask.npy')); H, W = land.shape
    res = lambda f: np.load(os.path.join(a.work, 'res', f))

    def full(vec):
        f = np.zeros((H, W), np.uint8); f[land] = vec; return f

    def write(sub, name, arr, cmap, band, desc):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True); path = os.path.join(a.out, sub, name)
        prof = dict(driver='GTiff', height=H, width=W, count=1, dtype='uint8', crs='EPSG:2954', transform=from_origin(291810, 779250, 10, 10),
                    nodata=0, compress='deflate', tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(path, 'w', **prof) as d:
            d.write(arr, 1); d.write_colormap(1, cmap); d.set_band_description(1, band); d.update_tags(**CLS, description=desc)
        with rasterio.open(path) as d: assert (d.read(1) == arr).all()
        print('written', path)

    cm5 = {0: (255, 255, 255, 0), **{k: v + (255,) for k, v in RGB.items()}}
    write('hindcast_2025', 'PEI_LULC_2025_hindcast_ANN-CA-Markov_5class.tif', full(res('sim_hindcast_2021_2023_to_2025.npy')), cm5, '2025 simulated from 2023 (5 classes)',
          'ANN-CA-Markov hindcast: 2025 simulated from the 2023 map with Markov quantities and transition potentials calibrated on 2021-2023. 0 = no data.')
    write('hindcast_2025', 'PEI_LULC_2025_hindcast_random_allocation_5class.tif', full(res('rand_hindcast_2021_2023_to_2025.npy')), cm5, '2025 random allocation from 2023 (5 classes)',
          'Null model of the hindcast: the same Markov quantities allocated at random within each source class. 0 = no data.')
    base, sim = res('baseline_2025_consistent.npy'), res('sim_scenario_2029.npy')
    write('scenario_2029', 'PEI_LULC_2025_baseline_temporally_consistent_5class.tif', full(base), cm5, '2025 baseline (5 classes)',
          '2025 class where the 2024 and 2025 maps (3 x 3 majority) agree; otherwise the most frequent class of 2021-2025 (ties: most recent year). 0 = no data.')
    write('scenario_2029', 'PEI_LULC_2029_scenario_ANN-CA-Markov_5class.tif', full(sim), cm5, '2029 scenario (5 classes)',
          'Conditional 2029 scenario: ANN-CA-Markov with transition probabilities from the 2021-2025 transitions of temporally consistent pixels and the transition potentials of the hindcast model, one four-year step from the 2025 baseline. 0 = no data.')
    T = np.zeros((H, W), np.uint8); T[land] = 10 * base + sim
    cmt = {0: (255, 255, 255, 0)}
    for i in range(1, 6):
        for j in range(1, 6): cmt[10 * i + j] = (225, 225, 225, 255) if i == j else RGB[j] + (255,)
    write('scenario_2029', 'PEI_LULC_transitions_2025_2029_scenario.tif', T, cmt, 'transition code 10 x 2025 baseline class + 2029 class',
          '10 x 2025 baseline class + 2029 scenario class (11, 22, 33, 44, 55 = unchanged); colour: class gained; 0 = no data.')


if __name__ == '__main__':
    main()
