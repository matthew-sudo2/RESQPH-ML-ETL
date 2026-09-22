"""
Road-risk and passability predictive modeling for ResQPH.

Combines flood hazard layers, road classification, elevation, and distance to
evacuation centers to generate a probabilistic passability risk score in [0, 1].
Supports multiple ML models (Linear Regression, Decision Tree, Random Forest,
XGBoost, KNN, Logistic Regression) with an automatic rule-based fallback when
training samples are insufficient.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.etl.load import get_processed_roads
from src.etl.pipeline import run_pipeline
from src.utils import config
from src.utils.geospatial import safe_numeric

logger = config.get_logger(__name__)


class RuleBasedRisk:
    """
    Deterministic rule-based risk heuristic in [0, 1].

    Used as an operational fallback or explainable baseline.
    Hazard class scale: 0 = no flood, 1 = low, 2 = medium, 3 = high.
    Elevation scale: 0m = max risk, 10m+ = min risk (Metro Manila is a
    flat coastal delta, so 10m is effectively "high ground" for the city).
    """

    def fit(self, X=None, y=None):
        """Fit stub to adhere to standard estimator interface."""
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        depth = safe_numeric(df["flood_depth_m"], 0.0)
        hazard = safe_numeric(df["flood_hazard_class"], 0.0)

        if "elevation_m" in df.columns:
            elev = safe_numeric(df["elevation_m"], 0.0)
        else:
            elev = pd.Series(0.0, index=df.index)

        if "distance_to_evac_m" in df.columns:
            dist = safe_numeric(df["distance_to_evac_m"], 1000.0)
        else:
            dist = pd.Series(1000.0, index=df.index)

        depth_n = np.clip(depth / config.FLOOD_IMPASSABLE_DEPTH_M, 0.0, 1.0)
        hazard_n = np.clip(hazard / 3.0, 0.0, 1.0)
        elev_n = np.clip(1.0 - elev / 10.0, 0.0, 1.0)
        dist_n = np.clip(dist / 5000.0, 0.0, 1.0)

        score = 0.50 * depth_n + 0.25 * hazard_n + 0.15 * elev_n + 0.10 * dist_n
        return np.clip(score.to_numpy(), 0.0, 1.0)


class ProbabilisticClassifierWrapper:
    """Wrapper around a classifier to return positive class probability as risk score."""

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)
            return np.clip(probs[:, 1], 0.0, 1.0)
        preds = self.classifier.predict(X)
        return np.clip(preds, 0.0, 1.0)


def build_model(model_name: str):
    """
    Instantiate the ML estimator with global hyperparameters.

    Supported models:
      - linear         : Linear Regression (baseline; target is linear)
      - decision_tree  : single Decision Tree
      - random_forest  : Random Forest Regressor
      - xgboost        : XGBoost Regressor
      - knn            : K-Nearest Neighbors Regressor
      - logreg         : Logistic Regression (wrapped as probability output)
    """
    params = config.MODEL_PARAMS.get(model_name, {})

    if model_name == "linear":
        from sklearn.linear_model import LinearRegression

        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

    if model_name == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=3,
            random_state=config.RANDOM_SEED,
        )

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**dict(params))

    if model_name == "xgboost":
        import xgboost as xgb

        xgb_params = dict(params)
        # For continuous risk target [0, 1], squared error regression is standard
        if "eval_metric" in xgb_params and xgb_params["eval_metric"] == "logloss":
            xgb_params["eval_metric"] = "rmse"
        return xgb.XGBRegressor(**xgb_params)

    if model_name == "knn":
        from sklearn.neighbors import KNeighborsRegressor

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
            ]
        )

    if model_name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return ProbabilisticClassifierWrapper(LogisticRegression(**params))

    raise ValueError(
        f"Unsupported model: {model_name}. "
        "Use linear, decision_tree, random_forest, xgboost, knn, or logreg."
    )


def train(
    model_name: str = "random_forest",
    export: bool = True,
    data_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train a road-risk model or instantiate fallback.

    Args:
        model_name: Name of model algorithm. See `build_model` for options.
        export: Whether to save model and feature metadata to models/.
        data_path: Optional custom path to processed road features.

    Returns:
        Dictionary containing model mode, trained model, metrics, and artifact paths.
    """
    config.set_global_seed(config.RANDOM_SEED)

    # 1. Load data or run pipeline if needed
    try:
        if data_path:
            df = (
                pd.read_parquet(data_path)
                if data_path.suffix == ".parquet"
                else pd.read_csv(data_path)
            )
        else:
            df = get_processed_roads()
    except FileNotFoundError:
        logger.info("Processed road features not found. Running ETL pipeline first...")
        run_pipeline()
        df = get_processed_roads()

    logger.info("Loaded road features for training: %d rows", len(df))

    # Validate features & target
    feature_cols = [c for c in config.ROAD_RISK_FEATURES if c in df.columns]
    target_col = config.ROAD_RISK_TARGET

    if len(feature_cols) != len(config.ROAD_RISK_FEATURES):
        missing = set(config.ROAD_RISK_FEATURES) - set(feature_cols)
        raise ValueError(f"Processed dataset missing required features: {missing}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Clean any inf or nan
    X = X.fillna(X.median()).clip(-1e9, 1e9)
    y = np.clip(safe_numeric(y, 0.0), 0.0, 1.0)

    # 2. Check sample threshold for ML
    model: Any
    if len(df) < config.MIN_SAMPLES_FOR_ML:
        logger.warning(
            "Sample count (%d) is below MIN_SAMPLES_FOR_ML (%d). Using rule-based fallback.",
            len(df),
            config.MIN_SAMPLES_FOR_ML,
        )
        mode = "rule_based"
        model = RuleBasedRisk()
        y_pred = model.predict(df)
        metrics = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "mse": float(mean_squared_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
        }
    else:
        mode = "ml"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
        )

        model = build_model(model_name)
        logger.info("Training %s (%s) on %d samples...", model_name, mode, len(X_train))

        if model_name == "logreg":
            # For logistic regression, train on binarized threshold
            y_train_bin = (y_train >= 0.5).astype(int)
            model.fit(X_train, y_train_bin)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    logger.info("Model metrics (%s): %s", mode, metrics)

    # 3. Model Export
    model_file = config.model_path(config.MODEL_ROAD_RISK)
    meta_file = config.model_path(config.MODEL_FEATURE_META)

    if export:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_file)
        logger.info("Saved model artifact to %s", model_file)

        metadata = {
            "model_name": model_name,
            "mode": mode,
            "features": config.ROAD_RISK_FEATURES,
            "target": config.ROAD_RISK_TARGET,
            "metrics": metrics,
            "sample_count": len(df),
            "study_area": config.STUDY_AREA_NAME,
            "random_seed": config.RANDOM_SEED,
            "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved feature metadata to %s", meta_file)

    return {
        "mode": mode,
        "model": model,
        "metrics": metrics,
        "model_path": model_file,
        "metadata_path": meta_file,
    }


def main():
    parser = argparse.ArgumentParser(description="ResQPH Road-Risk Modeling")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the road-risk prediction model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["linear", "decision_tree", "random_forest", "xgboost", "knn", "logreg"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not export model artifacts to models/",
    )
    args = parser.parse_args()

    if args.train:
        result = train(model_name=args.model, export=not args.no_export)
        logger.info(
            "Training complete. Result: mode=%s, metrics=%s",
            result["mode"],
            result["metrics"],
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()