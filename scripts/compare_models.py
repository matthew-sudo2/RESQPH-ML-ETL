"""
Train every supported model on the same train/test split and compare them:
  - R², MAE, RMSE
  - Training time
  - Inference time
  - Model file size on disk

Writes:
  models/comparison_results.json

Run with: python -m scripts.compare_models
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.etl.load import get_processed_roads
from src.ml.road_risk import build_model
from src.utils import config

MODELS = ["linear", "decision_tree", "random_forest", "xgboost", "knn"]


def human_size(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes = n_bytes / 1024
    return f"{n_bytes:.1f} TB"


def main() -> None:
    config.set_global_seed(config.RANDOM_SEED)
    df = get_processed_roads()
    X = df[config.ROAD_RISK_FEATURES]
    y = df[config.ROAD_RISK_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}\n")

    tmp_model = config.model_path("_tmp_compare.pkl")
    rows = []

    for name in MODELS:
        print(f"=== {name} ===")
        try:
            # Train
            model = build_model(name)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_s = time.perf_counter() - t0

            # Predict
            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            infer_s = time.perf_counter() - t0

            # Metrics
            metrics = {
                "model": name,
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "train_seconds": round(train_s, 2),
                "inference_seconds": round(infer_s, 3),
            }

            # Size
            joblib.dump(model, tmp_model)
            size_bytes = tmp_model.stat().st_size
            metrics["size_bytes"] = size_bytes
            metrics["size_human"] = human_size(size_bytes)

            print(
                f"  R²={metrics['r2']:.6f}  "
                f"MAE={metrics['mae']:.6f}  "
                f"train={train_s:.1f}s  "
                f"infer={infer_s:.2f}s  "
                f"size={metrics['size_human']}"
            )
            rows.append(metrics)
        except Exception as e:
            print(f"  FAILED: {e}")

    if tmp_model.exists():
        tmp_model.unlink()

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'model':<16}{'R²':>10}{'MAE':>12}{'train(s)':>10}{'infer(s)':>10}{'size':>14}")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: -x["r2"]):
        print(
            f"{r['model']:<16}"
            f"{r['r2']:>10.6f}"
            f"{r['mae']:>12.6f}"
            f"{r['train_seconds']:>10.2f}"
            f"{r['inference_seconds']:>10.3f}"
            f"{r['size_human']:>14}"
        )

    # Save
    out = config.model_path("comparison_results.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()