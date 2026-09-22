"""Global pytest fixtures and configuration for ResQPH ML/ETL."""
import pytest

from src.utils import config


@pytest.fixture(autouse=True)
def setup_seed():
    """Ensure consistent random seed across test execution."""
    config.set_global_seed(config.RANDOM_SEED)
