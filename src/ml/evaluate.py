"""
Model evaluation module for ResQPH ML components.

Evaluates road-risk regression/classification models and citizen distress
NLP triage models, printing comprehensive performance metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.etl.load import get_processed_nlp, get_processed_roads
from src.etl.pipeline import run_pipeline
from src.ml.nlp_triage import evaluate_nlp
from src.ml.road_risk import train
from src.utils import config
from src.utils.geospatial import safe_numeric

logger = config.get_logger(__name__)


def evaluate_road_risk(
    model_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate road-risk model against processed road features."""
    logger.info("==========================================")
    logger.info("Evaluating Road-Risk Model")
    logger.info("==========================================")

    # 1. Ensure data exists
    try:
        df = get_processed_roads() if data_path is None else pd.read_parquet(data_path)
    except Exception:
        logger.info("Road features not found. Running ETL pipeline...")
        run_pipeline()
        df = get_processed_roads()

    # 2. Ensure model exists
    target_model_path = model_path or config.model_path(config.MODEL_ROAD_RISK)
    if not target_model_path.exists():
        logger.info("Trained model not found at %s. Training default model...", target_model_path)
        train(model_name="random_forest", export=True)

    model = joblib.load(target_model_path)

    X = df[config.ROAD_RISK_FEATURES].copy()
    y = np.clip(safe_numeric(df[config.ROAD_RISK_TARGET], 0.0), 0.0, 1.0)

    # Use holdout test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    print("\n--- ROAD-RISK MODEL EVALUATION RESULTS ---")
    print(f"Test Samples: {len(X_test)}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Root Mean Squared (RMSE):  {rmse:.4f}")
    print(f"R-squared (R2 Score):      {r2:.4f}")
    print("------------------------------------------\n")

    return metrics


def evaluate_nlp_triage(data_path: Optional[Path] = None) -> Dict[str, Any]:
    """Evaluate NLP triage entity extraction and priority classification."""
    logger.info("==========================================")
    logger.info("Evaluating NLP Citizen Distress Triage")
    logger.info("==========================================")

    # 1. Ensure NLP data exists
    try:
        df = get_processed_nlp() if data_path is None else pd.read_csv(data_path)
    except Exception:
        logger.info("Cleaned citizen reports not found. Running ETL pipeline...")
        run_pipeline()
        df = get_processed_nlp()

    results = evaluate_nlp(df, target_col="priority_label")

    print("\n--- NLP CITIZEN TRIAGE EVALUATION RESULTS ---")
    print(f"Total Evaluated Reports: {results.get('sample_count', 0)}")
    if "accuracy" in results:
        print(f"Overall Accuracy: {results['accuracy']:.2%}")
        print("\nPer-Class Breakdown:")
        report = results.get("report", {})
        for label in config.NLP_TRIAGE_LABELS:
            if label in report:
                metrics = report[label]
                print(
                    f"  [{label.upper():>8}] Precision: {metrics.get('precision', 0.0):.2f} | "
                    f"Recall: {metrics.get('recall', 0.0):.2f} | "
                    f"F1-Score: {metrics.get('f1-score', 0.0):.2f} | "
                    f"Support: {metrics.get('support', 0)}"
                )
    print("---------------------------------------------\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="ResQPH Model Evaluation Utility")
    parser.add_argument(
        "--road-risk",
        action="store_true",
        help="Evaluate the road-risk passability model",
    )
    parser.add_argument(
        "--nlp",
        action="store_true",
        help="Evaluate the NLP citizen distress triage model",
    )
    args = parser.parse_args()

    # If neither flag passed, evaluate both by default
    eval_road = args.road_risk or (not args.road_risk and not args.nlp)
    eval_nlp = args.nlp or (not args.road_risk and not args.nlp)

    if eval_road:
        evaluate_road_risk()

    if eval_nlp:
        evaluate_nlp_triage()


if __name__ == "__main__":
    main()
