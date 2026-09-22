"""
Full ETL Pipeline runner for ResQPH.

Orchestrates extraction of raw geospatial and incident data, transformation
and alignment of road segments and flood hazard features, and persistence
to data/processed/ in Parquet and CSV format.
"""
from __future__ import annotations

import sys
from typing import Dict

from src.etl.extract import extract_all
from src.etl.load import load_processed
from src.etl.transform import transform_all
from src.utils import config

logger = config.get_logger(__name__)


def run_pipeline() -> Dict[str, str]:
    """Execute the end-to-end ResQPH ETL pipeline."""
    logger.info("==========================================")
    logger.info("Starting ResQPH ETL Pipeline")
    logger.info("==========================================")

    config.set_global_seed(config.RANDOM_SEED)

    # 1. Extraction
    logger.info("Step 1/3: Extracting raw data sources...")
    raw = extract_all()

    # 2. Transformation
    logger.info("Step 2/3: Transforming & aligning features...")
    processed = transform_all(raw)

    # 3. Loading / Persistence
    logger.info("Step 3/3: Saving processed datasets...")
    saved_paths = load_processed(processed)

    road_features = processed.get("road_features")
    nlp_data = processed.get("nlp_data")

    logger.info("==========================================")
    logger.info("ETL Pipeline Completed Successfully!")
    if road_features is not None:
        logger.info("  Road Features: %d segments, %d features", len(road_features), road_features.shape[1])
    if nlp_data is not None:
        logger.info("  Cleaned Reports: %d incident records", len(nlp_data))
    for name, path in saved_paths.items():
        logger.info("  Saved [%s] -> %s", name, path)
    logger.info("==========================================")

    return {name: str(path) for name, path in saved_paths.items()}

# Backwards-compatible alias
run_etl = run_pipeline


if __name__ == "__main__":
    try:
        run_pipeline()
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed with error: %s", exc)
        sys.exit(1)
