"""
Global configuration for ResQPH ML/ETL.

Every component (ETL, ML, tests, notebooks) imports from this module to
guarantee consistent paths, seeds, CRS, feature definitions, and
hyperparameters.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = ROOT_DIR / "models"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
TESTS_DIR: Path = ROOT_DIR / "tests"

for _p in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("RESQPH_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that respects LOG_LEVEL."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Geospatial
# ---------------------------------------------------------------------------
STUDY_AREA_NAME: str = "Metro Manila"

# (min_lat, min_lon, max_lat, max_lon) in WGS84
STUDY_AREA_BOUNDS: Tuple[float, float, float, float] = (14.35, 120.90, 14.80, 121.15)

WGS84_EPSG: int = 4326
WGS84_CRS: str = f"EPSG:{WGS84_EPSG}"
PROJECTED_EPSG: int = 32651            # UTM zone 51N (covers the Philippines)
PROJECTED_CRS: str = f"EPSG:{PROJECTED_EPSG}"

OSM_NETWORK_TYPE: str = "drive"
OSM_HIGHWAY_FILTER: List[str] = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service",
]

# Road-class -> numeric code (used by both ETL and ML)
ROAD_CLASS_CODES: Dict[str, int] = {
    "motorway": 1, "trunk": 2, "primary": 3, "secondary": 4,
    "tertiary": 5, "unclassified": 6, "residential": 7, "service": 8,
}

# Default speed limits (kph) by class — used when OSM lacks maxspeed
DEFAULT_SPEED_KPH: Dict[str, float] = {
    "motorway": 80.0, "trunk": 60.0, "primary": 50.0, "secondary": 40.0,
    "tertiary": 40.0, "unclassified": 30.0, "residential": 25.0, "service": 20.0,
}

# ---------------------------------------------------------------------------
# UP NOAH flood hazard shapefile
# ---------------------------------------------------------------------------
RAW_NOAH_SHAPEFILE: str = "flood/MetroManila_Flood_100year.shp"

# Hazard class scale (0 = no flood, 1 = low, 2 = medium, 3 = high)
# UP NOAH Var codes → hazard_class + representative depth
# Reference: UP NOAH Flood Hazard Map classification
#   Var=1 → Low    : depth < 0.5 m
#   Var=2 → Medium : depth 0.5 – 1.5 m
#   Var=3 → High   : depth > 1.5 m
NOAH_HAZARD_MAP: Dict[float, Dict[str, object]] = {
    1.0: {"hazard_class": 1, "depth_m": 0.30, "label": "low"},
    2.0: {"hazard_class": 2, "depth_m": 1.00, "label": "medium"},
    3.0: {"hazard_class": 3, "depth_m": 2.00, "label": "high"},
}

# ---------------------------------------------------------------------------
# Flood simulation scenarios
# ---------------------------------------------------------------------------
FLOOD_SCENARIOS: Dict[str, Dict[str, float]] = {
    "low":      {"depth_m": 0.15, "hazard_class": 1},
    "moderate": {"depth_m": 0.50, "hazard_class": 2},
    "severe":   {"depth_m": 1.20, "hazard_class": 3},
}
FLOOD_IMPASSABLE_DEPTH_M: float = 0.80   # depth above which a road is impassable

# ---------------------------------------------------------------------------
# Global Flood Database (real flood event labels)
# ---------------------------------------------------------------------------
# GEE asset: GLOBAL_FLOOD_DB/MODIS_EVENTS/V1
# Event ID 3552 = Typhoon Ondoy (Ketsana) flood extent, 2009-09-30
GFD_EE_COLLECTION: str = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
GFD_ONDOY_EVENT_ID: int = 3552
GFD_ONDOY_RASTER: str = "flood_events/gfd_ondoy_2009/gfd_ondoy_2009_mm.tif"

# ---------------------------------------------------------------------------
# Feature definitions (must match training + inference)
# ---------------------------------------------------------------------------
ROAD_RISK_FEATURES: List[str] = [
    "road_length_m",
    "speed_kph",
    "elevation_m",
    "flood_depth_m",
    "flood_hazard_class",
    "road_class_code",
    "distance_to_evac_m",
]

ROAD_RISK_TARGET: str = "risk_score"          # continuous in [0, 1] — surrogate target
ROAD_FLOOD_TARGET: str = "flooded_ondoy_2009"  # binary — real-label target

NLP_TRIAGE_LABELS: List[str] = ["low", "medium", "high", "critical"]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.10   # fraction carved from the training split

MODEL_PARAMS: Dict[str, dict] = {
    "logreg": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 3,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_SEED,
        "eval_metric": "logloss",
    },
}

# Guardrails before training a supervised model
MIN_SAMPLES_FOR_ML: int = 200
MIN_POSITIVE_CLASS_RATIO: float = 0.05

# ---------------------------------------------------------------------------
# Input file names (raw + processed)
# ---------------------------------------------------------------------------
RAW_OSM_GRAPH: str = "osm_roads.graphml"
RAW_FLOOD_LAYER: str = "flood_hazard.geojson"
RAW_ELEVATION: str = "elevation/Copernicus_DSM_COG_10_N14_00_E120_00_DEM.tif"
RAW_EVAC_CENTERS: str = "evacuation_centers.geojson"
RAW_REPORTS: str = "citizen_reports.csv"

INTERIM_ROADS: str = "roads_segments.parquet"
PROCESSED_ROAD_FEATURES: str = "road_features.parquet"
PROCESSED_ROAD_FLOOD_LABELS: str = "roads_with_flood_labels.parquet"
PROCESSED_NLP_DATA: str = "citizen_reports_clean.csv"

# ---------------------------------------------------------------------------
# Output model file names
# ---------------------------------------------------------------------------
# Surrogate: learned replica of the deterministic risk rule (fast inference).
MODEL_ROAD_RISK: str = "road_risk_surrogate.pkl"

# Classifier: real-label model trained on Typhoon Ondoy 2009 flood extent.
MODEL_ROAD_FLOOD_CLASSIFIER: str = "road_flood_classifier.pkl"

# Metadata sidecars
MODEL_FEATURE_META: str = "feature_metadata.json"
MODEL_FLOOD_CLASSIFIER_META: str = "flood_classifier_metadata.json"

# NLP triage model directory
MODEL_NLP_DIR: str = "nlp_triage_model"


# Convenience path builders
def raw_path(name: str) -> Path:
    return RAW_DIR / name


def interim_path(name: str) -> Path:
    return INTERIM_DIR / name


def processed_path(name: str) -> Path:
    return PROCESSED_DIR / name


def model_path(name: str) -> Path:
    return MODELS_DIR / name