"""Land-change drivers on the 10 m EPSG:2954 classification grid, written for land pixels (no data -9999).

    python build_drivers.py --inputs <data/drivers/source> --land-mask <land_change/land_mask_10m.tif> --out <data/drivers/10m>

- elevation: pei_dem.tif (EPSG:4326, 1 arc-second) -> 30 m EPSG:2954 (bilinear) -> 10 m (bilinear)
- slope: Horn slope (degrees; 3 x 3 Sobel operator) on the 30 m projected DEM, then bilinear to 10 m
- population: pei_population.tif (30 arc-second); no-data cells are filled with the nearest valid cell, then bilinear to 10 m
- distance to roads / waterways: OpenStreetMap lines rasterised on the 10 m grid (all touched; footways, paths, cycleways,
  steps, pedestrian ways and bridleways are not counted as roads), Euclidean distance (m)."""
import argparse, os
import numpy as np, rasterio, geopandas as gpd, cv2
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt

CRS2954 = CRS.from_epsg(2954)
H, W = 12392, 18725
T10 = from_origin(291810, 779250, 10, 10)
T30 = from_origin(291810, 779250, 30, 30); H30, W30 = int(np.ceil(H / 3)), int(np.ceil(W / 3))
EXCL = {'footway', 'path', 'cycleway', 'steps', 'pedestrian', 'bridleway'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', required=True); ap.add_argument('--land-mask', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); I = lambda f: os.path.join(a.inputs, f); os.makedirs(a.out, exist_ok=True)
    with rasterio.open(a.land_mask) as d: land = d.read(1).astype(bool)
    assert land.shape == (H, W)

    def save(name, arr):
        out = np.where(land, arr, -9999).astype(np.float32)
        with rasterio.open(os.path.join(a.out, f'{name}_10m.tif'), 'w', driver='GTiff', height=H, width=W, count=1, dtype='float32', crs=CRS2954,
                           transform=T10, nodata=-9999, compress='lzw', tiled=True, predictor=3) as d:
            d.write(out, 1)
        v = arr[land]; print(name, 'min %.2f median %.2f max %.2f' % (np.nanmin(v), np.nanmedian(v), np.nanmax(v)), flush=True)

    with rasterio.open(I('pei_dem.tif')) as s:
        dem = s.read(1).astype(np.float32)
        dem30 = np.full((H30, W30), np.nan, np.float32)
        reproject(dem, dem30, src_transform=s.transform, src_crs=s.crs, dst_transform=T30, dst_crs=CRS2954, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    # the source no-data value 0 is not applied: 0 m is a valid coastal elevation, and the ocean is removed by the land mask
    k = np.nan_to_num(dem30, nan=0.0)
    dzdx = cv2.Sobel(k, cv2.CV_32F, 1, 0, ksize=3) / (8 * 30.0); dzdy = cv2.Sobel(k, cv2.CV_32F, 0, 1, ksize=3) / (8 * 30.0)
    slope30 = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)
    for name, a30 in (('elevation', dem30), ('slope', slope30)):
        a10 = np.full((H, W), np.nan, np.float32)
        reproject(a30, a10, src_transform=T30, src_crs=CRS2954, dst_transform=T10, dst_crs=CRS2954, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        save(name, a10); del a10
    with rasterio.open(I('pei_population.tif')) as s:
        p = s.read(1).astype(np.float32); p[p == s.nodata] = np.nan
        nn = distance_transform_edt(np.isnan(p), return_distances=False, return_indices=True); p = p[tuple(nn)]
        p10 = np.full((H, W), np.nan, np.float32)
        reproject(p, p10, src_transform=s.transform, src_crs=s.crs, dst_transform=T10, dst_crs=CRS2954, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        save('population', p10); del p10
    for name, f, keep in (('dist_roads', 'gis_osm_roads_free_1.shp', lambda g: g[~g.fclass.isin(EXCL)]), ('dist_water', 'gis_osm_waterways_free_1.shp', lambda g: g)):
        g = keep(gpd.read_file(I(f))).to_crs(CRS2954)
        r = rasterize(((geom, 1) for geom in g.geometry), out_shape=(H, W), transform=T10, fill=0, dtype=np.uint8, all_touched=True)
        src = np.where(r > 0, 0, 255).astype(np.uint8); del r
        d = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * 10.0; del src
        save(name, d); del d


if __name__ == '__main__':
    main()
