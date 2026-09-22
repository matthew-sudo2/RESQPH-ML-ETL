"""
Overlay the GFD Ondoy 2009 flood extent onto the OSM road network.

Produces data/processed/roads_with_flood_labels.parquet with a
`flooded_ondoy_2009` binary column for each road segment.

Run with: python -m scripts.build_flood_labels
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import rasterio

from src.utils import config

FLOOD_RASTER = config.raw_path("flood_events/gfd_ondoy_2009/gfd_ondoy_2009_mm.tif")


def main() -> None:
    print("Loading roads...")
    roads = pd.read_parquet(config.processed_path(config.PROCESSED_ROAD_FEATURES))
    print(f"  {len(roads):,} roads")

    print(f"\nLoading flood raster: {FLOOD_RASTER.name}")
    with rasterio.open(FLOOD_RASTER) as src:
        # Sample the flood raster at each road midpoint
        coords = list(zip(roads["mid_lon"], roads["mid_lat"]))
        samples = np.array([v[0] for v in src.sample(coords)])

    roads["flooded_ondoy_2009"] = (samples > 0).astype(int)

    n_flooded = int(roads["flooded_ondoy_2009"].sum())
    rate = 100 * roads["flooded_ondoy_2009"].mean()
    print(f"\n  → {n_flooded:,} / {len(roads):,} roads flooded ({rate:.2f}%)")

    if n_flooded > 0:
        print("\n  Flooded by road class:")
        by_class = roads.groupby("road_class")["flooded_ondoy_2009"].agg(["sum", "count"])
        by_class["pct"] = (100 * by_class["sum"] / by_class["count"]).round(2)
        by_class = by_class.sort_values("sum", ascending=False).head(10)
        print(by_class.to_string())

    out_path = config.processed_path("roads_with_flood_labels.parquet")
    roads.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()