"""
Transform raw ResQPH datasets into ML-ready features.

Consumes the dict produced by `extract_all()` and returns processed
DataFrames for road-risk modeling and NLP triage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import config
from src.utils.geospatial import haversine_m, nearest_distance_m, safe_numeric

logger = config.get_logger(__name__)
rng = np.random.default_rng(config.RANDOM_SEED)


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------
def build_road_segments(raw_roads) -> pd.DataFrame:
    """Normalize roads into a DataFrame of segments with start/end coords."""
    if isinstance(raw_roads, pd.DataFrame):
        df = raw_roads.copy()
    else:
        import osmnx as ox

        edges = ox.graph_to_gdfs(raw_roads, nodes=False, edges=True).reset_index()
        df = pd.DataFrame(
            {
                "u": edges["u"].values,
                "v": edges["v"].values,
                "start_lat": edges.geometry.apply(lambda g: g.coords[0][1]).values,
                "start_lon": edges.geometry.apply(lambda g: g.coords[0][0]).values,
                "end_lat": edges.geometry.apply(lambda g: g.coords[-1][1]).values,
                "end_lon": edges.geometry.apply(lambda g: g.coords[-1][0]).values,
                "road_class": edges["highway"].astype(str).values,
            }
        )

    df["road_class"] = (
        df["road_class"]
        .astype(str)
        .str.replace(r"[\[\]']", "", regex=True)
        .str.split(",")
        .str[0]
        .str.strip()
    )
    df.loc[~df["road_class"].isin(config.ROAD_CLASS_CODES), "road_class"] = "unclassified"
    df["road_class_code"] = df["road_class"].map(config.ROAD_CLASS_CODES).fillna(6).astype(int)

    df["road_length_m"] = [
        haversine_m(slat, slon, elat, elon)
        for slat, slon, elat, elon in zip(
            df["start_lat"], df["start_lon"], df["end_lat"], df["end_lon"]
        )
    ]
    df["road_length_m"] = safe_numeric(df["road_length_m"], fill=0.0).clip(lower=1.0)

    df["speed_kph"] = df["road_class"].map(config.DEFAULT_SPEED_KPH).fillna(30.0)

    df["mid_lat"] = (df["start_lat"] + df["end_lat"]) / 2.0
    df["mid_lon"] = (df["start_lon"] + df["end_lon"]) / 2.0
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Flood + elevation attach
# ---------------------------------------------------------------------------
def attach_flood_and_elevation(
    roads: pd.DataFrame, flood, elevation: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach flood hazard + elevation to each road segment.

    - If `flood` is a GeoDataFrame (real NOAH polygons with a `geometry`
      column and `crs`), use a proper spatial join on road midpoints.
    - If `flood` is a plain DataFrame with `lat`/`lon` columns (synthetic
      fallback), use nearest-point Haversine lookup.

    Elevation is attached via a vectorized nearest-point lookup using
    scipy.spatial.cKDTree on a local tangent-plane approximation.

    Hazard class scale: 0 = no flood, 1 = low, 2 = medium, 3 = high.
    """
    out = roads.copy()

    # ---------- Flood: GeoDataFrame path (real NOAH shapefile) ----------
    if hasattr(flood, "geometry") and hasattr(flood, "crs"):
        import geopandas as gpd
        from shapely.geometry import Point

        roads_gdf = gpd.GeoDataFrame(
            out,
            geometry=[Point(xy) for xy in zip(out["mid_lon"], out["mid_lat"])],
            crs=config.WGS84_CRS,
        )
        # Match CRS before joining
        if flood.crs is not None and flood.crs != roads_gdf.crs:
            flood = flood.to_crs(roads_gdf.crs)

        joined = gpd.sjoin(
            roads_gdf,
            flood[["flood_depth_m", "flood_hazard_class", "geometry"]],
            how="left",
            predicate="within",
        )
        # Roads outside any flood polygon → hazard 0, depth 0
        joined["flood_depth_m"] = joined["flood_depth_m"].fillna(0.0)
        joined["flood_hazard_class"] = joined["flood_hazard_class"].fillna(0).astype(int)

        # Deduplicate in case a midpoint landed inside overlapping polygons
        joined = joined[~joined.index.duplicated(keep="first")].sort_index()

        out["flood_depth_m"] = joined["flood_depth_m"].values
        out["flood_hazard_class"] = joined["flood_hazard_class"].values

        logger.info(
            "Flood spatial join: %d/%d roads inside a flood polygon",
            int((out["flood_depth_m"] > 0).sum()),
            len(out),
        )

    # ---------- Flood: plain DataFrame path (synthetic points) ----------
    elif isinstance(flood, pd.DataFrame) and {"lat", "lon"} <= set(flood.columns):
        f_lat = flood["lat"].to_numpy()
        f_lon = flood["lon"].to_numpy()
        f_dep = (
            flood["flood_depth_m"].to_numpy()
            if "flood_depth_m" in flood
            else np.zeros(len(flood))
        )
        f_haz = (
            flood["flood_hazard_class"].to_numpy()
            if "flood_hazard_class" in flood
            else np.zeros(len(flood))
        )

        depths, hazs = [], []
        for lat, lon in zip(out["mid_lat"], out["mid_lon"]):
            d = np.array([haversine_m(lat, lon, la, lo) for la, lo in zip(f_lat, f_lon)])
            idx = int(np.argmin(d))
            depths.append(float(f_dep[idx]))
            hazs.append(int(f_haz[idx]))
        out["flood_depth_m"] = depths
        out["flood_hazard_class"] = hazs
        logger.info("Flood nearest-point join applied (synthetic fallback)")

    else:
        out["flood_depth_m"] = 0.0
        out["flood_hazard_class"] = 0
        logger.warning("Flood data unusable — defaulting to zero flood on all roads")

    # ---------- Elevation (vectorized nearest-point via scipy cKDTree) ----------
    if isinstance(elevation, pd.DataFrame) and {"lat", "lon", "elevation_m"} <= set(
        elevation.columns
    ):
        from scipy.spatial import cKDTree

        # Approximate the local tangent plane so we can use a KDTree
        # in 2D meters instead of Haversine per pair.
        lat0 = float(out["mid_lat"].mean())
        cos_lat = float(np.cos(np.radians(lat0)))
        R = 6_371_000.0  # Earth radius in meters

        # Query points (road midpoints)
        qx = np.radians(out["mid_lon"].to_numpy()) * R * cos_lat
        qy = np.radians(out["mid_lat"].to_numpy()) * R

        # Reference points (elevation samples)
        rx = np.radians(elevation["lon"].to_numpy()) * R * cos_lat
        ry = np.radians(elevation["lat"].to_numpy()) * R
        r_elev = elevation["elevation_m"].to_numpy()

        tree = cKDTree(np.column_stack([rx, ry]))
        _, idx = tree.query(np.column_stack([qx, qy]), k=1)
        out["elevation_m"] = r_elev[idx]

        logger.info(
            "Elevation nearest-point join: min=%.1fm, max=%.1fm, mean=%.1fm",
            float(np.min(r_elev[idx])),
            float(np.max(r_elev[idx])),
            float(np.mean(r_elev[idx])),
        )
    else:
        out["elevation_m"] = 0.0

    return out


# ---------------------------------------------------------------------------
# Distance to evacuation center
# ---------------------------------------------------------------------------
def attach_distance_to_evac(roads: pd.DataFrame, evac: pd.DataFrame) -> pd.DataFrame:
    out = roads.copy()
    if isinstance(evac, pd.DataFrame) and {"lat", "lon"} <= set(evac.columns):
        e_lat = evac["lat"].to_numpy()
        e_lon = evac["lon"].to_numpy()
        out["distance_to_evac_m"] = [
            nearest_distance_m(lat, lon, e_lat, e_lon)
            for lat, lon in zip(out["mid_lat"], out["mid_lon"])
        ]
    else:
        out["distance_to_evac_m"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Risk label / target
# ---------------------------------------------------------------------------
def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic, explainable risk score in [0, 1].

    Used as the supervised target when real labels are unavailable.
    Combines flood depth, hazard class, low elevation, and distance to evac.

    Hazard class scale: 0 = no flood, 1 = low, 2 = medium, 3 = high.
    Elevation scale: 0m = max risk, 10m+ = min risk (Metro Manila is a
    flat coastal delta, so 10m is effectively "high ground" for the city).
    """
    out = df.copy()
    depth = safe_numeric(out["flood_depth_m"], 0.0)
    hazard = safe_numeric(out["flood_hazard_class"], 0.0)
    elev = safe_numeric(out["elevation_m"], 0.0)
    dist = safe_numeric(out["distance_to_evac_m"], 1000.0)

    depth_n = np.clip(depth / config.FLOOD_IMPASSABLE_DEPTH_M, 0, 1)
    hazard_n = np.clip(hazard / 3.0, 0, 1)
    elev_n = np.clip(1.0 - elev / 10.0, 0, 1)
    dist_n = np.clip(dist / 5000.0, 0, 1)

    score = 0.50 * depth_n + 0.25 * hazard_n + 0.15 * elev_n + 0.10 * dist_n
    out[config.ROAD_RISK_TARGET] = np.clip(score, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Full road feature pipeline
# ---------------------------------------------------------------------------
def build_road_features(raw: dict) -> pd.DataFrame:
    logger.info("Building road features")
    roads = build_road_segments(raw["roads"])
    roads = attach_flood_and_elevation(roads, raw["flood"], raw["elevation"])
    roads = attach_distance_to_evac(roads, raw["evac_centers"])
    roads = compute_risk_score(roads)

    keep = [
        "u", "v",
        "start_lat", "start_lon", "end_lat", "end_lon",
        "mid_lat", "mid_lon",
        "road_class",
        *config.ROAD_RISK_FEATURES,
        config.ROAD_RISK_TARGET,
    ]
    missing = [c for c in keep if c not in roads.columns]
    if missing:
        raise ValueError(f"Missing expected columns after transform: {missing}")

    out = roads[keep].copy()
    logger.info("Road features ready: %d rows, %d cols", *out.shape)
    return out


# ---------------------------------------------------------------------------
# NLP data transform
# ---------------------------------------------------------------------------
def build_nlp_data(raw_reports: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning for the citizen-report corpus."""
    df = raw_reports.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"]).reset_index(drop=True)
    if "priority_label" in df.columns:
        df["priority_label"] = df["priority_label"].astype(str).str.lower()
        df = df[df["priority_label"].isin(config.NLP_TRIAGE_LABELS)].reset_index(drop=True)
    logger.info("NLP data ready: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Top-level transform
# ---------------------------------------------------------------------------
def transform_all(raw: dict) -> dict:
    logger.info("Starting transform")
    return {
        "road_features": build_road_features(raw),
        "nlp_data": build_nlp_data(raw["reports"]),
    }