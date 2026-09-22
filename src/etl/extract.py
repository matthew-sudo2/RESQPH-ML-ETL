"""
Data extraction routines for ResQPH ML/ETL.

Loads raw OpenStreetMap roads, flood hazard rasters/vectors, elevation data,
evacuation centers, and citizen reports. If raw files are not present in
data/raw/, high-fidelity synthetic seed datasets within the Metro Manila
study area are automatically generated for pipeline execution and testing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils import config

logger = config.get_logger(__name__)


def generate_synthetic_roads(n_segments: int = 350) -> pd.DataFrame:
    """Generate synthetic road segments within the study area bounds."""
    min_lat, min_lon, max_lat, max_lon = config.STUDY_AREA_BOUNDS
    rng = np.random.default_rng(config.RANDOM_SEED)

    road_classes = list(config.ROAD_CLASS_CODES.keys())
    # Weight residential, tertiary, secondary more commonly
    weights = [0.05, 0.05, 0.15, 0.20, 0.20, 0.10, 0.20, 0.05]

    start_lats = rng.uniform(min_lat, max_lat, n_segments)
    start_lons = rng.uniform(min_lon, max_lon, n_segments)
    # Segments are between 50m and ~1500m long (roughly 0.0005 to 0.015 degrees)
    delta_lats = rng.uniform(-0.01, 0.01, n_segments)
    delta_lons = rng.uniform(-0.01, 0.01, n_segments)

    end_lats = np.clip(start_lats + delta_lats, min_lat, max_lat)
    end_lons = np.clip(start_lons + delta_lons, min_lon, max_lon)
    classes = rng.choice(road_classes, size=n_segments, p=weights)

    return pd.DataFrame(
        {
            "u": [f"node_{i}" for i in range(n_segments)],
            "v": [f"node_{i + 1000}" for i in range(n_segments)],
            "start_lat": start_lats,
            "start_lon": start_lons,
            "end_lat": end_lats,
            "end_lon": end_lons,
            "road_class": classes,
        }
    )


def generate_synthetic_flood(n_points: int = 150) -> pd.DataFrame:
    """Generate synthetic flood observation points."""
    min_lat, min_lon, max_lat, max_lon = config.STUDY_AREA_BOUNDS
    rng = np.random.default_rng(config.RANDOM_SEED + 1)

    lats = rng.uniform(min_lat, max_lat, n_points)
    lons = rng.uniform(min_lon, max_lon, n_points)
    # Flood depth between 0.0m and 2.5m
    depths = rng.exponential(scale=0.6, size=n_points).round(2)
    # Align with NOAH classification:
    #   0 = no flood (< 0.05m), 1 = low (< 0.5m), 2 = medium (< 1.5m), 3 = high
    hazard_classes = np.select(
        [depths < 0.05, depths < 0.50, depths < 1.50],
        [0, 1, 2],
        default=3,
    )

    return pd.DataFrame(
        {
            "lat": lats,
            "lon": lons,
            "flood_depth_m": depths,
            "flood_hazard_class": hazard_classes,
        }
    )


def generate_synthetic_elevation(n_points: int = 150) -> pd.DataFrame:
    """Generate synthetic elevation grid points."""
    min_lat, min_lon, max_lat, max_lon = config.STUDY_AREA_BOUNDS
    rng = np.random.default_rng(config.RANDOM_SEED + 2)

    lats = rng.uniform(min_lat, max_lat, n_points)
    lons = rng.uniform(min_lon, max_lon, n_points)
    # Elevation from 2m (coastal/riverbanks) to 45m (higher ground)
    elevations = rng.uniform(2.0, 45.0, n_points).round(1)

    return pd.DataFrame(
        {
            "lat": lats,
            "lon": lons,
            "elevation_m": elevations,
        }
    )


def generate_synthetic_evac_centers(n_centers: int = 25) -> pd.DataFrame:
    """Generate synthetic evacuation center locations."""
    min_lat, min_lon, max_lat, max_lon = config.STUDY_AREA_BOUNDS
    rng = np.random.default_rng(config.RANDOM_SEED + 3)

    lats = rng.uniform(min_lat, max_lat, n_centers)
    lons = rng.uniform(min_lon, max_lon, n_centers)
    names = [f"Evacuation Center {i+1} - District {i%5 + 1}" for i in range(n_centers)]

    return pd.DataFrame(
        {
            "name": names,
            "lat": lats,
            "lon": lons,
        }
    )


def generate_synthetic_reports(n_reports: int = 200) -> pd.DataFrame:
    """Generate synthetic citizen distress reports with emergency entities and priority labels."""
    rng = np.random.default_rng(config.RANDOM_SEED + 4)
    min_lat, min_lon, max_lat, max_lon = config.STUDY_AREA_BOUNDS

    sample_texts_by_priority = {
        "critical": [
            "Trapped on roof, chest-deep water, grandmother needs insulin immediately!",
            "Water reaching ceiling on 2nd floor, dialysis patient stranded with infant baby.",
            "Chest-deep flood water, trapped on rooftop with elderly person having stroke symptoms.",
            "Water up to neck, pregnant mother in labor trapped on roof of house.",
            "Submerged ground floor, family trapped in attic, oxygen tank running out for elderly father.",
        ],
        "high": [
            "Waist-deep flood entering living room, 80 year old grandmother cannot walk.",
            "Water waist-deep and rising fast, 3 kids and disabled father need evacuation.",
            "House flooded waist deep, need boat rescue for elderly neighbor.",
            "Rapidly rising water waist high, small children and pregnant woman stranded.",
            "Chest-deep current outside, front door blocked, need rescue boat.",
        ],
        "medium": [
            "Knee-deep water in street, cannot get out to buy food or medicine.",
            "Road is impassable due to knee deep flood, power lines down on street.",
            "Water knee-deep inside garage, power is out and drinking water running low.",
            "Flood rising to knee level, vehicles stalled on the avenue.",
            "Need help transporting supplies, road flooded knee deep.",
        ],
        "low": [
            "Ankle-deep water on street, requesting update on barangay flood gates.",
            "Light rain and ankle deep gutter overflow in front of market.",
            "Inquiry on evacuation center availability near high school.",
            "Puddles forming near corner, street still passable to light vehicles.",
            "Minor gutter flooding, asking for sandbags if rain continues.",
        ],
    }

    texts = []
    labels = []
    priorities = list(sample_texts_by_priority.keys())
    weights = [0.25, 0.35, 0.25, 0.15]

    for _ in range(n_reports):
        p = rng.choice(priorities, p=weights)
        text_template = rng.choice(sample_texts_by_priority[p])
        # Add slight variation
        variation_id = rng.integers(100, 999)
        texts.append(f"{text_template} [Ref #{variation_id}]")
        labels.append(p)

    lats = rng.uniform(min_lat, max_lat, n_reports)
    lons = rng.uniform(min_lon, max_lon, n_reports)

    return pd.DataFrame(
        {
            "text": texts,
            "priority_label": labels,
            "lat": lats,
            "lon": lons,
        }
    )


def extract_roads(raw_path: Optional[Path] = None) -> Any:
    """Extract road network from file or generate synthetic segments."""
    path = raw_path or config.raw_path(config.RAW_OSM_GRAPH)
    if path.exists() and path.stat().st_size > 0:
        logger.info("Loading road network from %s", path)
        try:
            import osmnx as ox

            return ox.load_graphml(path)
        except Exception as exc:
            logger.warning("Failed to load %s (%s). Using fallback roads.", path, exc)

    logger.info("Generating synthetic road network for study area %s", config.STUDY_AREA_NAME)
    return generate_synthetic_roads(n_segments=350)


def extract_flood(raw_path: Optional[Path] = None) -> Any:
    """
    Extract flood hazard polygons or points.

    Priority:
      1. UP NOAH 100-year flood shapefile at data/raw/flood/MetroManila_Flood_100year.shp
      2. Any GeoJSON at data/raw/flood_hazard.geojson (config.RAW_FLOOD_LAYER)
      3. Synthetic point fallback

    When the NOAH shapefile is used, returns a GeoDataFrame with columns:
      geometry,
      flood_hazard_class (0/1/2/3)  where 0 = no flood, 1 = low, 2 = medium, 3 = high,
      flood_depth_m (0.3 / 1.0 / 2.0),
      hazard_label ("low" / "medium" / "high")
    """
    # 1) Try the UP NOAH shapefile first
    noah_path = config.raw_path(config.RAW_NOAH_SHAPEFILE)
    if noah_path.exists() and noah_path.stat().st_size > 0:
        try:
            import geopandas as gpd

            logger.info("Loading UP NOAH flood shapefile from %s", noah_path)
            gdf = gpd.read_file(noah_path)

            # Ensure WGS84 for consistent downstream joins
            if gdf.crs is None:
                gdf = gdf.set_crs(config.PROJECTED_CRS)
            gdf = gdf.to_crs(config.WGS84_CRS)

            # Map Var (1/2/3) → hazard_class (1/2/3) + depth_m
            def _map_var(v: Any, key: str) -> Any:
                if v not in config.NOAH_HAZARD_MAP:
                    return None
                return config.NOAH_HAZARD_MAP[v][key]

            gdf["flood_hazard_class"] = gdf["Var"].map(lambda v: _map_var(v, "hazard_class"))
            gdf["flood_depth_m"] = gdf["Var"].map(lambda v: _map_var(v, "depth_m"))
            gdf["hazard_label"] = gdf["Var"].map(lambda v: _map_var(v, "label"))

            # Drop rows we couldn't map (shouldn't happen with Var=1/2/3)
            gdf = gdf.dropna(subset=["flood_hazard_class", "flood_depth_m"]).copy()
            gdf["flood_hazard_class"] = gdf["flood_hazard_class"].astype(int)
            gdf["flood_depth_m"] = gdf["flood_depth_m"].astype(float)

            out = gdf[
                ["geometry", "flood_hazard_class", "flood_depth_m", "hazard_label"]
            ].copy()

            logger.info(
                "Loaded %d flood polygons | hazard distribution: %s",
                len(out),
                out["hazard_label"].value_counts().to_dict(),
            )
            return out

        except Exception as exc:
            logger.warning(
                "NOAH shapefile load failed (%s). Trying GeoJSON fallback.", exc
            )

    # 2) Try the generic GeoJSON path
    geojson_path = raw_path or config.raw_path(config.RAW_FLOOD_LAYER)
    if geojson_path.exists() and geojson_path.stat().st_size > 0:
        logger.info("Loading flood data from %s", geojson_path)
        try:
            import geopandas as gpd

            gdf = gpd.read_file(geojson_path)
            if "lat" not in gdf.columns:
                gdf["lat"] = gdf.geometry.centroid.y
                gdf["lon"] = gdf.geometry.centroid.x
            return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        except Exception as exc:
            logger.warning(
                "Failed to load %s (%s). Using fallback flood data.", geojson_path, exc
            )

    # 3) Synthetic fallback
    logger.info("Generating synthetic flood data points")
    return generate_synthetic_flood(n_points=150)


def extract_elevation(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Extract elevation samples from the Copernicus DEM raster.

    Samples the raster at a coarse grid across the study area so that
    downstream nearest-point lookup in transform.py has coverage over
    every road segment. Falls back to synthetic elevation if the raster
    is missing or unreadable.
    """
    path = raw_path or config.raw_path(config.RAW_ELEVATION)

    if path.exists() and path.stat().st_size > 0:
        logger.info("Loading elevation raster from %s", path)
        try:
            import rasterio

            with rasterio.open(path) as src:
                crs = src.crs
                bounds = src.bounds

                # Build a grid of sample points across the study area
                min_lat, min_lon, max_lat, max_lon = config.STUDY_AREA_BOUNDS

                # Clip sample grid to raster bounds
                lat_lo = max(min_lat, bounds.bottom)
                lat_hi = min(max_lat, bounds.top)
                lon_lo = max(min_lon, bounds.left)
                lon_hi = min(max_lon, bounds.right)

                # ~500 x 500 samples over the study area (~60m spacing)
                n = 500
                lats = np.linspace(lat_lo, lat_hi, n)
                lons = np.linspace(lon_lo, lon_hi, n)
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                sample_lats = lat_grid.ravel()
                sample_lons = lon_grid.ravel()

                # Reproject sample points to raster CRS if needed
                if crs is not None and str(crs).upper() != config.WGS84_CRS:
                    import pyproj

                    transformer = pyproj.Transformer.from_crs(
                        config.WGS84_CRS, crs, always_xy=True
                    )
                    xs, ys = transformer.transform(sample_lons, sample_lats)
                else:
                    xs, ys = sample_lons, sample_lats

                # Sample the raster at each point
                coords = list(zip(xs, ys))
                elevations = np.array([val[0] for val in src.sample(coords)], dtype=float)

                # Mask nodata if the raster defines it
                nodata = src.nodata
                if nodata is not None:
                    elevations = np.where(elevations == nodata, np.nan, elevations)

            df = pd.DataFrame(
                {
                    "lat": sample_lats,
                    "lon": sample_lons,
                    "elevation_m": elevations,
                }
            ).dropna(subset=["elevation_m"])

            logger.info(
                "Sampled %d elevation points | min=%.1fm, max=%.1fm, mean=%.1fm",
                len(df),
                df["elevation_m"].min(),
                df["elevation_m"].max(),
                df["elevation_m"].mean(),
            )
            return df

        except Exception as exc:
            logger.warning(
                "Failed to sample raster %s (%s). Falling back to synthetic.",
                path,
                exc,
            )

    logger.info("Generating synthetic elevation points")
    return generate_synthetic_elevation(n_points=150)


def extract_evac_centers(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """Extract evacuation centers."""
    path = raw_path or config.raw_path(config.RAW_EVAC_CENTERS)
    if path.exists() and path.stat().st_size > 0:
        logger.info("Loading evacuation centers from %s", path)
        try:
            import geopandas as gpd

            gdf = gpd.read_file(path)
            if "lat" not in gdf.columns:
                gdf["lat"] = gdf.geometry.centroid.y
                gdf["lon"] = gdf.geometry.centroid.x
            return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        except Exception as exc:
            logger.warning("Failed to load %s (%s). Using fallback evac centers.", path, exc)

    logger.info("Generating synthetic evacuation centers")
    return generate_synthetic_evac_centers(n_centers=30)


def extract_reports(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """Extract citizen incident reports."""
    path = raw_path or config.raw_path(config.RAW_REPORTS)
    if path.exists() and path.stat().st_size > 0:
        logger.info("Loading citizen reports from %s", path)
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning("Failed to read %s (%s). Using fallback reports.", path, exc)

    logger.info("Generating synthetic citizen reports")
    return generate_synthetic_reports(n_reports=250)


def extract_all() -> Dict[str, Any]:
    """
    Extract all raw datasets needed for ResQPH ML/ETL transformation.

    Returns:
        dict with keys 'roads', 'flood', 'elevation', 'evac_centers', 'reports'.
    """
    logger.info("Starting extract_all()")
    return {
        "roads": extract_roads(),
        "flood": extract_flood(),
        "elevation": extract_elevation(),
        "evac_centers": extract_evac_centers(),
        "reports": extract_reports(),
    }