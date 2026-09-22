"""
Compute a flood mask from pre/post Sentinel-1 scenes and label OSM roads.

Outputs:
    data/interim/flood_mask_ulysses_2020.tif       (GeoTIFF)
    data/processed/roads_with_flood_labels.parquet (roads + `flooded` column)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import median_filter

from src.utils import config

PRE_PATH = config.raw_path("sentinel1/pre_ulysses_2020.tif")
POST_PATH = config.raw_path("sentinel1/post_ulysses_2020.tif")
MASK_PATH = config.interim_path("flood_mask_ulysses_2020.tif")
ROADS_IN = config.processed_path(config.PROCESSED_ROAD_FEATURES)
ROADS_OUT = config.processed_path("roads_with_flood_labels.parquet")

# VV water threshold in dB (typical for calm water)
WATER_DB_THRESHOLD = -15.0
# Speckle filter window
SPECKLE_WINDOW = 5


def read_vv_db(path) -> tuple[np.ndarray, dict]:
    """Read band 1 (VV), convert to dB, apply speckle filter. Returns (array, meta)."""
    with rasterio.open(path) as src:
        # GRDH COG band 1 is VV; already calibrated (sigma0 linear or dB — check)
        vv = src.read(1).astype(np.float32)
        meta = src.meta.copy()

        # If values are >0 and look linear (not dB), convert
        # dB values are typically negative; linear sigma0 is 0..1 or higher
        if np.nanmedian(vv) > 0:
            vv = 10 * np.log10(np.maximum(vv, 1e-6))

        # Speckle filter
        vv = median_filter(vv, size=SPECKLE_WINDOW)
        return vv, meta


def main() -> None:
    print("Reading pre-event scene...")
    pre, meta = read_vv_db(PRE_PATH)
    print("Reading post-event scene...")
    post, _ = read_vv_db(POST_PATH)

    # Reproject post to pre grid if shapes differ
    if pre.shape != post.shape:
        print(f"Resampling post from {post.shape} to {pre.shape}")
        post_rs = np.empty_like(pre)
        reproject(
            source=post,
            destination=post_rs,
            src_transform=rasterio.open(POST_PATH).transform,
            src_crs=rasterio.open(POST_PATH).crs,
            dst_transform=meta["transform"],
            dst_crs=meta["crs"],
            resampling=Resampling.bilinear,
        )
        post = post_rs

    # Water detection: low VV backscatter on the post scene
    post_water = post < WATER_DB_THRESHOLD
    pre_water = pre < WATER_DB_THRESHOLD

    # Flood = new water (post is water, pre was not)
    flood = post_water & ~pre_water
    print(f"Flood mask: {flood.sum():,} water pixels ({100*flood.mean():.2f}% of scene)")

    # Save mask
    MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(MASK_PATH, "w", **meta) as dst:
        dst.write(flood.astype(np.uint8), 1)
    print(f"Wrote {MASK_PATH}")

    # Overlay on roads
    print("Labeling roads...")
    roads = pd.read_parquet(ROADS_IN)
    with rasterio.open(MASK_PATH) as mask:
        coords = list(zip(roads["mid_lon"], roads["mid_lat"]))
        samples = np.array([v[0] for v in mask.sample(coords)])
    roads["flooded_ulysses_2020"] = samples.astype(bool)
    roads.to_parquet(ROADS_OUT, index=False)
    print(
        f"Wrote {ROADS_OUT} — "
        f"{roads['flooded_ulysses_2020'].sum():,} / {len(roads):,} roads flooded "
        f"({100 * roads['flooded_ulysses_2020'].mean():.2f}%)"
    )


if __name__ == "__main__":
    main()