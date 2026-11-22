"""Sampled reference pixels of each year from the reference polygons and the feature tiles.

    python extract_samples.py --root <data set folder> --out <output folder> [--years 2021 2022 2023 2024 2025]

The reference polygons of a year (one shapefile per cover type; the class comes from the folder name) are repaired
(make_valid), reduced to 2D, reprojected to the PEI stereographic projection (EPSG:2954) and clipped to the island
boundary. They are rasterised onto every same-year feature tile they intersect; in each tile, up to 400 pixels per cover
type are drawn at random; pixels whose 13 features are all 0 (outside the image) are dropped; then up to 2,500 pixels per
cover type and year are kept. Each pixel is assigned to a 3 km block, block_id = floor(x / 3000) * 10^7 + floor(y / 3000)
at the pixel centre. Output: train_<year>.csv (year, class, x, y, block_id, tile, row, col and the 13 features)."""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from shapely import force_2d, make_valid
from shapely.geometry import box
from shapely.ops import unary_union

YEARS = (2021, 2022, 2023, 2024, 2025)
CLASS_NAMES = {1: 'forest', 2: 'cropland', 3: 'grassland', 4: 'wetland', 5: 'settlement', 6: 'otherland'}
BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI', 'BSI', 'elev', 'slope', 'gsw_occ']
PEI_WKT = ('PROJCS["NAD_1983_CSRS_Prince_Edward_Island",GEOGCS["GCS_North_American_1983_CSRS",'
           'DATUM["D_North_American_1983_CSRS",SPHEROID["GRS_1980",6378137.0,298.257222101]],'
           'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Double_Stereographic"],'
           'PARAMETER["False_Easting",400000.0],PARAMETER["False_Northing",800000.0],PARAMETER["Central_Meridian",-63.0],'
           'PARAMETER["Scale_Factor",0.999912],PARAMETER["Latitude_Of_Origin",47.25],UNIT["Meter",1.0]]')
BLOCK_M = 3000
SEED = 42
PER_TILE_CAP = 400
MAX_PIXELS_PER_CLASS = 2500


def load_boundary(data):
    gdf = gpd.read_file(data / 'boundary' / 'pei.shp')
    if gdf.crs is None: gdf.set_crs(PEI_WKT, inplace=True)
    return gdf.to_crs(PEI_WKT)


def clean_geometry(geom):
    if geom is None or geom.is_empty: return None
    geom = force_2d(make_valid(geom))
    if geom.is_empty: return None
    if geom.geom_type in ('Polygon', 'MultiPolygon'): return geom
    if geom.geom_type == 'GeometryCollection':
        parts = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        return unary_union(parts) if parts else None
    return None


def load_year_samples(data, year):
    boundary = load_boundary(data); frames = []
    for cid, name in CLASS_NAMES.items():
        gdf = gpd.read_file(data / 'samples' / str(year) / name / f'{name}.shp')
        if gdf.crs is None: gdf.set_crs(4326, inplace=True)
        gdf['geometry'] = gdf.geometry.map(clean_geometry)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        gdf = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
        gdf = gdf.to_crs(PEI_WKT).clip(boundary)
        gdf = gdf[['geometry']].copy(); gdf['class'] = cid; frames.append(gdf)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry='geometry', crs=PEI_WKT)


def tile_index(data):
    """Bounds of the 5 km tiles (the same grid every year): tile_index.csv, or read from the 2021 tiles."""
    f = data / 'feature_tiles_v4' / 'tile_index.csv'
    if f.exists(): df = pd.read_csv(f)
    else:
        rows = []
        for p in sorted((data / 'feature_tiles_v4' / '2021').glob('feat_*.tif')):
            with rasterio.open(p) as s: b = s.bounds; rows.append(dict(tile=p.name, left=b.left, bottom=b.bottom, right=b.right, top=b.top, width=s.width, height=s.height, res_x=s.res[0], res_y=s.res[1]))
        df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry=[box(x1, y1, x2, y2) for x1, y1, x2, y2 in df[['left', 'bottom', 'right', 'top']].to_numpy()], crs=PEI_WKT)


def block_id(x, y):
    return np.floor(x / BLOCK_M).astype(np.int64) * 10_000_000 + np.floor(y / BLOCK_M).astype(np.int64)


def extract(data, year, index):
    samples = load_year_samples(data, year)
    hit = gpd.sjoin(samples, index[['tile', 'geometry']], predicate='intersects', how='inner')
    chunks = []; rng = np.random.RandomState(SEED + year)
    for tile_name, grp in hit.groupby('tile'):
        with rasterio.open(data / 'feature_tiles_v4' / str(year) / tile_name) as src:
            arr = src.read(); transform = src.transform; h, w = src.height, src.width
        if not np.any(arr): continue
        lab = rasterize([(g, int(c)) for g, c in zip(grp.geometry, grp['class'])], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
        rr, cc = np.nonzero(lab)
        if len(rr) == 0: continue
        keep = np.zeros(len(rr), dtype=bool); cls = lab[rr, cc]
        for cid in CLASS_NAMES:
            idx = np.flatnonzero(cls == cid)
            if len(idx) > PER_TILE_CAP: idx = rng.choice(idx, PER_TILE_CAP, replace=False)
            keep[idx] = True
        rr, cc = rr[keep], cc[keep]
        xs, ys = rasterio.transform.xy(transform, rr, cc, offset='center')
        chunk = pd.DataFrame(arr[:, rr, cc].T, columns=BANDS)
        chunk.insert(0, 'col', cc); chunk.insert(0, 'row', rr); chunk.insert(0, 'tile', tile_name)
        chunk.insert(0, 'block_id', block_id(np.asarray(xs), np.asarray(ys))); chunk.insert(0, 'y', np.asarray(ys)); chunk.insert(0, 'x', np.asarray(xs))
        chunk.insert(0, 'class', lab[rr, cc]); chunk.insert(0, 'year', year); chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df = df[(df[BANDS].abs() > 1e-8).any(axis=1)]
    df = pd.concat([g.sample(min(len(g), MAX_PIXELS_PER_CLASS), random_state=rng) for _, g in df.groupby('class')], ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True); ap.add_argument('--years', type=int, nargs='+', default=list(YEARS))
    a = ap.parse_args(); data = Path(a.root) / 'data'; os.makedirs(a.out, exist_ok=True); index = tile_index(data)
    for year in a.years:
        df = extract(data, year, index); df.to_csv(os.path.join(a.out, f'train_{year}.csv'), index=False)
        print(year, 'pixels', len(df), 'per cover type', df['class'].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
