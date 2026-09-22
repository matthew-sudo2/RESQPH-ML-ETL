"""Tests for global configuration and environment settings."""

from src.utils import config


def test_paths_exist_and_are_directories():
    assert config.ROOT_DIR.exists()
    assert config.DATA_DIR.is_dir()
    assert config.RAW_DIR.is_dir()
    assert config.INTERIM_DIR.is_dir()
    assert config.PROCESSED_DIR.is_dir()
    assert config.MODELS_DIR.is_dir()


def test_path_builders():
    raw_p = config.raw_path("test.csv")
    assert raw_p == config.RAW_DIR / "test.csv"

    interim_p = config.interim_path("test.parquet")
    assert interim_p == config.INTERIM_DIR / "test.parquet"

    proc_p = config.processed_path("test.parquet")
    assert proc_p == config.PROCESSED_DIR / "test.parquet"

    model_p = config.model_path("model.pkl")
    assert model_p == config.MODELS_DIR / "model.pkl"


def test_study_area_bounds():
    min_lat, min_lon, max_lat, max_lon = config.STUDY_AREA_BOUNDS
    assert min_lat < max_lat
    assert min_lon < max_lon
    # Metro Manila coordinates
    assert 14.0 <= min_lat <= 15.0
    assert 120.0 <= min_lon <= 122.0


def test_features_and_labels_definitions():
    assert "road_length_m" in config.ROAD_RISK_FEATURES
    assert "flood_depth_m" in config.ROAD_RISK_FEATURES
    assert config.ROAD_RISK_TARGET == "risk_score"
    assert set(config.NLP_TRIAGE_LABELS) == {"low", "medium", "high", "critical"}


def test_seed_reproducibility():
    import numpy as np

    config.set_global_seed(123)
    val1 = np.random.rand()
    config.set_global_seed(123)
    val2 = np.random.rand()
    assert val1 == val2
