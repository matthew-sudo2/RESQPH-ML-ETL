"""
Train a road-flood classifier using real Ondoy 2009 labels from the
Global Flood Database.

Binary classification:
  target = flooded_ondoy_2009  (0 = dry, 1 = flooded)
  models = logreg, random_forest, xgboost

Reports accuracy, AUC, confusion matrix, and feature importances.
Exports the best model to models/road_flood_classifier.pkl.

Model selection: prefers XGBoost when its AUC is within 0.01 of the best,
because XGBoost achieves higher recall (safety-critical for rescue routing)
at ~66x smaller model size than Random Forest.

Run with: python -m scripts.train_flood_classifier
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import config

TARGET_COL = "flooded_ondoy_2009"


def build_classifier(name: str):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
            )),
        ])

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        )

    if name == "xgboost":
        from xgboost import XGBClassifier
        # scale_pos_weight handles class imbalance
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=200,   # ~1:200 imbalance
            eval_metric="logloss",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier: {name}")


def main() -> None:
    config.set_global_seed(config.RANDOM_SEED)

    # Load labeled data
    data_path = config.processed_path("roads_with_flood_labels.parquet")
    if not data_path.exists():
        raise SystemExit(
            f"Missing {data_path}. Run scripts.build_flood_labels first."
        )

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} roads from {data_path.name}")

    if TARGET_COL not in df.columns:
        raise SystemExit(f"Target column '{TARGET_COL}' not found in parquet.")

    # Class balance
    n_pos = int(df[TARGET_COL].sum())
    n_neg = len(df) - n_pos
    pos_rate = n_pos / len(df)
    print(f"\nTarget: {TARGET_COL}")
    print(f"  positives: {n_pos:,}  ({100 * pos_rate:.3f}%)")
    print(f"  negatives: {n_neg:,}  ({100 * (1 - pos_rate):.3f}%)")

    if n_pos < 50:
        raise SystemExit(f"Only {n_pos} positives. Not enough to train reliably.")

    X = df[config.ROAD_RISK_FEATURES].copy()
    y = df[TARGET_COL].astype(int).to_numpy()

    # Clean features
    X = X.fillna(X.median()).clip(-1e9, 1e9)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )
    print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}")

    results = {}
    models_by_name = {}
    auc_by_name = {}

    for name in ["logreg", "random_forest", "xgboost"]:
        print(f"\n=== {name} ===")
        try:
            clf = build_classifier(name)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy":  float(accuracy_score(y_test, y_pred)),
                "auc":       float(roc_auc_score(y_test, y_prob)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
                "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
                "confusion": confusion_matrix(y_test, y_pred).tolist(),
            }
            print(f"  accuracy:  {metrics['accuracy']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
            print(f"  precision: {metrics['precision']:.4f}")
            print(f"  recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  confusion: {metrics['confusion']}")
            print()
            print(classification_report(
                y_test, y_pred,
                target_names=["dry", "flooded"],
                zero_division=0,
            ))

            results[name] = metrics
            models_by_name[name] = clf
            auc_by_name[name] = metrics["auc"]

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    # Summary
    print("\n" + "=" * 72)
    print(f"{'model':<16}{'accuracy':>10}{'AUC':>10}{'precision':>12}{'recall':>10}{'F1':>10}")
    print("-" * 72)
    for name, m in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        print(
            f"{name:<16}"
            f"{m['accuracy']:>10.4f}"
            f"{m['auc']:>10.4f}"
            f"{m['precision']:>12.4f}"
            f"{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}"
        )

    # ----------------------------------------------------------------
    # Model selection
    #
    # Pick XGBoost when its AUC is within 0.01 of the best model.
    # Rationale: XGBoost achieves higher recall (safety-critical for
    # rescue routing) at ~66x smaller model size than Random Forest.
    # Otherwise, fall back to the highest-AUC model.
    # ----------------------------------------------------------------
    best_model = None
    best_name = None
    best_auc = -1.0

    if "xgboost" in auc_by_name:
        max_auc = max(auc_by_name.values())
        if auc_by_name["xgboost"] >= max_auc - 0.01:
            best_name = "xgboost"
            best_auc = auc_by_name["xgboost"]
            best_model = models_by_name["xgboost"]

    if best_model is None and results:
        # Fall back to highest-AUC model
        best_name = max(auc_by_name, key=lambda k: auc_by_name[k])
        best_auc = auc_by_name[best_name]
        best_model = models_by_name[best_name]

    if best_model is not None:
        out_path = config.model_path("road_flood_classifier.pkl")
        joblib.dump(best_model, out_path)
        print(f"\nSelected: {best_name} (AUC={best_auc:.4f})")
        print(f"Saved → {out_path}")

        meta = {
            "best_model":      best_name,
            "target":          TARGET_COL,
            "features":        config.ROAD_RISK_FEATURES,
            "metrics":         results,
            "sample_count":    int(len(df)),
            "positive_count":  n_pos,
            "positive_rate":   float(pos_rate),
            "random_seed":     config.RANDOM_SEED,
            "selection_note":  (
                "XGBoost preferred when its AUC is within 0.01 of the best "
                "model due to higher recall and smaller model size."
            ),
        }
        meta_path = config.model_path("flood_classifier_metadata.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata → {meta_path}")

        # Feature importances
        if best_name in ("random_forest", "xgboost"):
            importances = best_model.feature_importances_
            print("\nFeature importances:")
            for feat, imp in sorted(
                zip(config.ROAD_RISK_FEATURES, importances),
                key=lambda x: -x[1],
            ):
                bar = "█" * int(imp * 50)
                print(f"  {feat:25s} {imp:.4f}  {bar}")
        elif best_name == "logreg":
            coefs = best_model.named_steps["model"].coef_[0]
            print("\nLogistic regression coefficients (signed):")
            for feat, c in sorted(
                zip(config.ROAD_RISK_FEATURES, coefs),
                key=lambda x: -abs(x[1]),
            ):
                print(f"  {feat:25s} {c:+.4f}")


if __name__ == "__main__":
    main()