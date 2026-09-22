"""
Persistence routines for processed ResQPH datasets.

Saves transformed road features and cleaned NLP datasets to data/processed/
in Parquet and CSV formats for downstream model training and evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils import config

logger = config.get_logger(__name__)


def save_processed_roads(df: pd.DataFrame) -> Path:
    """Save processed road features to Parquet and CSV."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = config.processed_path(config.PROCESSED_ROAD_FEATURES)
    csv_path = config.PROCESSED_DIR / "road_features.csv"

    logger.info("Saving road features to %s (%d rows)", parquet_path, len(df))
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return parquet_path


def save_processed_nlp(df: pd.DataFrame) -> Path:
    """Save cleaned NLP citizen reports to CSV."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.processed_path(config.PROCESSED_NLP_DATA)

    logger.info("Saving cleaned NLP data to %s (%d rows)", csv_path, len(df))
    df.to_csv(csv_path, index=False)
    return csv_path


def load_processed(processed: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
    """
    Persist all processed datasets to data/processed/.

    Args:
        processed: dict with keys 'road_features' and 'nlp_data'.

    Returns:
        dict mapping dataset names to saved file paths.
    """
    logger.info("Starting load_processed()")
    saved_paths: Dict[str, Path] = {}

    if "road_features" in processed:
        saved_paths["road_features"] = save_processed_roads(processed["road_features"])
    if "nlp_data" in processed:
        saved_paths["nlp_data"] = save_processed_nlp(processed["nlp_data"])

    logger.info("load_processed completed. Files written: %s", list(saved_paths.values()))
    return saved_paths


def get_processed_roads() -> pd.DataFrame:
    """Load the processed road features DataFrame from disk."""
    parquet_path = config.processed_path(config.PROCESSED_ROAD_FEATURES)
    if not parquet_path.exists():
        csv_path = config.PROCESSED_DIR / "road_features.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"Processed road features not found at {parquet_path}")
    return pd.read_parquet(parquet_path)


def get_processed_nlp() -> pd.DataFrame:
    """Load the cleaned citizen reports DataFrame from disk."""
    csv_path = config.processed_path(config.PROCESSED_NLP_DATA)
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed NLP data not found at {csv_path}")
    return pd.read_csv(csv_path)
