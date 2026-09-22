from pathlib import Path

import pandas as pd

from src.etl.extract import (
    extract_all,
    generate_synthetic_roads,
)
from src.etl.load import load_processed
from src.etl.pipeline import run_pipeline
from src.etl.transform import build_road_segments, safe_numeric, transform_all
from src.utils import config


def test_safe_numeric():
    s = pd.Series(["1.5", "nan", "inf", "-inf", "invalid", 42])
    result = safe_numeric(s, fill=0.0)
    assert result.iloc[0] == 1.5
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == 0.0
    assert result.iloc[3] == 0.0
    assert result.iloc[4] == 0.0
    assert result.iloc[5] == 42.0


def test_extract_all_structure():
    raw = extract_all()
    assert "roads" in raw
    assert "flood" in raw
    assert "elevation" in raw
    assert "evac_centers" in raw
    assert "reports" in raw
    assert len(raw["roads"]) >= config.MIN_SAMPLES_FOR_ML
    assert len(raw["reports"]) > 0


def test_build_road_segments():
    synthetic_roads = generate_synthetic_roads(n_segments=10)
    df = build_road_segments(synthetic_roads)
    assert "road_length_m" in df.columns
    assert "speed_kph" in df.columns
    assert "road_class_code" in df.columns
    assert "mid_lat" in df.columns
    assert "mid_lon" in df.columns
    assert (df["road_length_m"] > 0).all()


def test_transform_all_and_risk_score_bounds():
    raw = extract_all()
    processed = transform_all(raw)
    roads = processed["road_features"]
    nlp = processed["nlp_data"]

    assert len(roads) >= config.MIN_SAMPLES_FOR_ML
    assert config.ROAD_RISK_TARGET in roads.columns
    assert (roads[config.ROAD_RISK_TARGET] >= 0.0).all()
    assert (roads[config.ROAD_RISK_TARGET] <= 1.0).all()

    for col in config.ROAD_RISK_FEATURES:
        assert col in roads.columns

    assert "text" in nlp.columns
    assert len(nlp) > 0


def test_load_processed_persists_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)

    raw = extract_all()
    processed = transform_all(raw)
    saved = load_processed(processed)

    assert "road_features" in saved
    assert "nlp_data" in saved
    assert saved["road_features"].exists()
    assert saved["nlp_data"].exists()


def test_pipeline_runner():
    res = run_pipeline()
    assert "road_features" in res
    assert "nlp_data" in res
    assert Path(res["road_features"]).exists()
    assert Path(res["nlp_data"]).exists()
