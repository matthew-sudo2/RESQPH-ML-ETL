"""Geospatial helper functions shared across ETL and ML components."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.config import (
    PROJECTED_CRS,
    STUDY_AREA_BOUNDS,
    WGS84_CRS,
    get_logger,
)

logger = get_logger(__name__)

EARTH_RADIUS_M: float = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def point_in_study_area(lat: float, lon: float) -> bool:
    """Return True if the point is inside STUDY_AREA_BOUNDS."""
    min_lat, min_lon, max_lat, max_lon = STUDY_AREA_BOUNDS
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def filter_to_study_area(
    df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon"
) -> pd.DataFrame:
    """Drop points outside the study area."""
    mask = df.apply(lambda r: point_in_study_area(r[lat_col], r[lon_col]), axis=1)
    return df.loc[mask].reset_index(drop=True)


def _require_geopandas():
    try:
        import geopandas as gpd  # noqa: F401

        return gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for this operation. Install with `pip install geopandas`."
        ) from exc


def to_geodataframe(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = WGS84_CRS,
):
    """Build a GeoDataFrame from a DataFrame with lat/lon columns."""
    gpd = _require_geopandas()
    from shapely.geometry import Point

    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)


def project_gdf(gdf, crs: str = PROJECTED_CRS):
    """Reproject a GeoDataFrame to the projected CRS used for distances."""
    return gdf.to_crs(crs)


def nearest_distance_m(
    src_lat: float,
    src_lon: float,
    targets_lat: Iterable[float],
    targets_lon: Iterable[float],
) -> float:
    """Return minimum Haversine distance (m) from a point to a set of points."""
    dmin = math.inf
    for tlat, tlon in zip(targets_lat, targets_lon):
        d = haversine_m(src_lat, src_lon, tlat, tlon)
        if d < dmin:
            dmin = d
    return float(dmin) if dmin < math.inf else float("nan")


def clip_to_bounds(
    df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon"
) -> pd.DataFrame:
    """Clip a DataFrame to the study-area bounding box."""
    min_lat, min_lon, max_lat, max_lon = STUDY_AREA_BOUNDS
    return df[
        (df[lat_col].between(min_lat, max_lat)) & (df[lon_col].between(min_lon, max_lon))
    ].reset_index(drop=True)


def safe_numeric(series: pd.Series, fill: float = 0.0) -> pd.Series:
    """Coerce to numeric, replace inf, fill NaN."""
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill)
    )