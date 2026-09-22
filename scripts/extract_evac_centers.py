"""
Extract real evacuation centers from OpenStreetMap for Metro Manila.

Queries Overpass API for shelter-related tags, saves as GeoJSON.

Run with: python -m scripts.extract_evac_centers
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import osmnx as ox

from src.utils import config

# Tags that indicate a shelter / evacuation facility
TAGS = {
    "amenity": ["evacuation_centre", "shelter", "community_centre", "school"],
    "social_facility": ["shelter", "outreach"],
    "emergency": ["evacuation_centre", "shelter"],
    "building": ["school", "community_centre"],
}


def main() -> None:
    print("Querying OSM for evacuation-related facilities in Metro Manila...")
    print("(This may take 1-3 minutes)")

    gdf = ox.features_from_place("Metro Manila, Philippines", tags=TAGS)
    print(f"  Found {len(gdf):,} candidate features")

    # Filter to those that look like real shelters
    # Prefer explicit evacuation tags; fall back to community/school buildings
    keep = (
        gdf["amenity"].isin(["evacuation_centre", "shelter"])
        | gdf["social_facility"].isin(["shelter"])
        | gdf["emergency"].isin(["evacuation_centre", "shelter"])
    )
    evac = gdf[keep].copy()
    print(f"  Explicit evacuation/shelter tags: {len(evac):,}")

    if len(evac) < 10:
        print("  Too few explicit tags — including community centers and schools")
        evac = gdf.copy()

    # Convert polygons to centroids
    evac["geometry"] = evac.geometry.centroid
    evac = evac.to_crs("EPSG:4326")

    # Keep minimal columns
    out = evac[["geometry"]].copy()
    out["name"] = evac.get("name", "unnamed").fillna("unnamed")
    out["lat"] = evac.geometry.y
    out["lon"] = evac.geometry.x

    out_path = config.raw_path("evacuation_centers.geojson")
    out.to_file(out_path, driver="GeoJSON")
    print(f"\nSaved {len(out):,} evacuation centers → {out_path}")
    print(out.head(10)[["name", "lat", "lon"]].to_string())


if __name__ == "__main__":
    main()