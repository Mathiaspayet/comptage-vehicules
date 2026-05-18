"""Shared pytest fixtures for all test modules."""
import pytest

from src.database import Database
from src.config import Config, DEFAULTS, _deep_merge


@pytest.fixture
def tmp_db(tmp_path):
    """Return a Database instance backed by a temp file."""
    db_file = tmp_path / "test_vehicles.db"
    return Database(db_file)


@pytest.fixture
def base_config():
    """Return a Config built from DEFAULTS only (no file)."""
    return Config(DEFAULTS.copy())


@pytest.fixture
def tmp_config_file(tmp_path):
    """Write a minimal config.yaml to tmp_path and return its Path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "detector:\n"
        "  imgsz: 640\n"
        "  model_name: yolo11n\n",
        encoding="utf-8",
    )
    return config_file
