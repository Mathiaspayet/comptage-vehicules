"""Configuration loading from YAML file."""
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULTS = {
    "video_folder": "/video",
    "filename_datetime_format": "Doorbell_%Y%m%d_%H%M%S.mp4",
    "timezone": "Europe/Paris",
    "ingestion": {
        "scan_interval_seconds": 60,
        "file_stable_delay_seconds": 120,
    },
    "motion_filter": {
        "sample_fps": 2,
        "motion_threshold": 25,
        "min_motion_area": 500,
        "segment_padding_seconds": 1.0,
    },
    "detector": {
        "sample_fps": 2,
        "confidence_threshold": 0.35,
        "vehicle_classes": ["car", "motorcycle", "bus", "truck"],
        "count_direction": False,
        "model_dir": "/app/data/models",
    },
    "counting": {
        "line_p1": [0, 300],
        "line_p2": [1280, 300],
        "roi_polygon": [[0, 200], [1280, 200], [1280, 500], [0, 500]],
    },
    "database": {
        "path": "/app/data/vehicles.db",
    },
    "dashboard": {
        "port": 8080,
        "default_days": 30,
    },
    "calibration": {
        "port": 8081,
    },
    "logging": {
        "level": "INFO",
        "max_size_mb": 10,
        "backup_count": 3,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    def __init__(self, data: dict):
        self._data = data

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # Shortcuts for frequently accessed values
    @property
    def video_folder(self) -> Path:
        return Path(self.get("video_folder"))

    @property
    def filename_datetime_format(self) -> str:
        return self.get("filename_datetime_format")

    @property
    def timezone(self) -> str:
        return self.get("timezone")

    @property
    def db_path(self) -> Path:
        return Path(self.get("database", "path"))

    @property
    def dashboard_port(self) -> int:
        return int(self.get("dashboard", "port"))

    @property
    def calibration_port(self) -> int:
        return int(self.get("calibration", "port"))

    @property
    def scan_interval(self) -> int:
        return int(self.get("ingestion", "scan_interval_seconds"))

    @property
    def file_stable_delay(self) -> int:
        return int(self.get("ingestion", "file_stable_delay_seconds"))

    @property
    def motion_sample_fps(self) -> float:
        return float(self.get("motion_filter", "sample_fps"))

    @property
    def motion_threshold(self) -> int:
        return int(self.get("motion_filter", "motion_threshold"))

    @property
    def min_motion_area(self) -> int:
        return int(self.get("motion_filter", "min_motion_area"))

    @property
    def segment_padding(self) -> float:
        return float(self.get("motion_filter", "segment_padding_seconds"))

    @property
    def detector_sample_fps(self) -> float:
        return float(self.get("detector", "sample_fps"))

    @property
    def confidence_threshold(self) -> float:
        return float(self.get("detector", "confidence_threshold"))

    @property
    def vehicle_classes(self) -> list:
        return self.get("detector", "vehicle_classes", default=[])

    @property
    def count_direction(self) -> bool:
        return bool(self.get("detector", "count_direction"))

    @property
    def model_dir(self) -> Path:
        return Path(self.get("detector", "model_dir"))

    @property
    def line_p1(self) -> tuple:
        p = self.get("counting", "line_p1")
        return tuple(p)

    @property
    def line_p2(self) -> tuple:
        p = self.get("counting", "line_p2")
        return tuple(p)

    @property
    def roi_polygon(self) -> list:
        return self.get("counting", "roi_polygon", default=[])

    @property
    def log_level(self) -> str:
        return self.get("logging", "level", default="INFO")

    @property
    def default_days(self) -> int:
        return int(self.get("dashboard", "default_days", default=30))

    def raw(self) -> dict:
        return self._data


def load_config(config_path: str | Path | None = None) -> Config:
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "/app/data/config.yaml")

    config_path = Path(config_path)
    data = DEFAULTS.copy()

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, user_data)
        logger.info("Configuration chargée depuis %s", config_path)
    else:
        logger.warning(
            "Fichier de configuration introuvable : %s. Utilisation des valeurs par défaut.",
            config_path,
        )

    return Config(data)
