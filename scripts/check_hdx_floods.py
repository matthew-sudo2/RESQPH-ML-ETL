"""
Download a set of UNOSAT flood extents from HDX, unzip, and check whether
each one covers Metro Manila.

Does not modify any state — just downloads to data/raw/flood_events/ and prints
a summary so we can pick the most useful one.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

from src.utils import config

# Metro Manila bbox (WGS84)
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = 120.90, 14.35, 121.15, 14.80
MM_BBOX = (MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)

CANDIDATES = [
    ("unosat_2026_09_07", "https://data.humdata.org/dataset/74e07aaa-629a-4b40-a2fd-6ca3540ea1fa/resource/09f449c1-4535-4bcc-a727-f2cb736ff894/download/20260907_1807_fld_s1_shp.zip"),
    ("unosat_2026_08_31", "https://data.humdata.org/dataset/67127c8f-46f3-40d5-8e33-d357de4afe33/resource/08773e8b-9c43-4312-9cab-05b9d3bdd022/download/20260831_0545_fld_s1_shp.zip.zip"),
    ("unosat_2026_08_13", "https://data.humdata.org/dataset/7076d253-00b5-41da-80cd-23268f0b1f1c/resource/ea41f584-c4a5-486b-a8a7-87db146921ed/download/20260813_0546_fld_s1_shp.zip"),
    ("unosat_2026_08_30", "https://data.humdata.org/dataset/7f292782-165b-4af1-85b1-fd1af534658e/resource/4a28d0df-c5e9-4c06-b518-bb4ec1ef50f7/download/20260830_0554_fld_s1_shp..zip.zip"),
    ("unosat_2025_07_20", "https://data.humdata.org/dataset/83892eb0-210d-40c0-a0a6-390e60c7f6e6/resource/f06bdb75-149d-4e0d-b618-557798defdd5/download/20250720_0530_fld_s1_shp.zip"),
    ("unosat_2025_05_21", "https://data.humdata.org/dataset/1e799688-8f17-468c-b21c-b80bdae226bb/resource/6380ee8f-62f1-49b0-b74d-15b6a516503b/download/20250521_0532_fld_s1_shp.zip"),
]


def bbox_overlaps(a, b) -> bool:
    """Check if two (minx, miny, maxx, maxy) bboxes overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def main() -> None:
    out_dir = config.raw_path("flood_events")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Metro Manila bbox: {MM_BBOX}\n")

    for label, url in CANDIDATES:
        print(f"=== {label} ===")
        try:
            r = requests.get(url, timeout=120)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} — skipped\n")
                continue

            size_mb = len(r.content) / 1e6
            print(f"  Downloaded {size_mb:.1f} MB")

            # Unzip in memory
            z = zipfile.ZipFile(io.BytesIO(r.content))
            shp_names = [n for n in z.namelist() if n.lower().endswith(".shp")]
            if not shp_names:
                print("  No .shp found in archive — skipped\n")
                continue

            # Extract to a subfolder
            sub = out_dir / label
            sub.mkdir(parents=True, exist_ok=True)
            z.extractall(sub)

            for shp in shp_names:
                shp_path = sub / shp
                try:
                    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
                except Exception as e:
                    print(f"  Failed to read {shp}: {e}")
                    continue

                bounds = tuple(gdf.total_bounds)   # (minx, miny, maxx, maxy)
                overlap = bbox_overlaps(bounds, MM_BBOX)
                n_features = len(gdf)

                print(f"  {shp}")
                print(f"    features: {n_features:,}")
                print(f"    bounds:   ({bounds[0]:.3f}, {bounds[1]:.3f}) → ({bounds[2]:.3f}, {bounds[3]:.3f})")
                print(f"    overlaps Metro Manila: {'YES' if overlap else 'no'}")
                if "geometry" in gdf.columns or gdf.geometry.name:
                    print(f"    geometry types: {gdf.geom_type.value_counts().to_dict()}")
            print()
        except Exception as e:
            print(f"  Error: {e}\n")

    print(f"All downloaded files are in {out_dir}")


if __name__ == "__main__":
    main()