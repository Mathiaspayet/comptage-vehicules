"""Configuration loading from YAML file."""
import hashlib
import json
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
        "max_recent_files": 0,
    },
    "motion_filter": {
        "sample_fps": 2,
        "motion_threshold": 25,
        "min_motion_area": 500,
        "segment_padding_seconds": 1.0,
    },
    "detector": {
        "sample_fps": 2,
        "imgsz": 320,
        "model_name": "yolo11n",
        "confidence_threshold": 0.35,
        "vehicle_classes": ["car", "motorcycle", "bus", "truck"],
        "count_direction": False,
        "model_dir": "/app/data/models",
        "night_confidence_threshold": 0.18,
        "night_enhance": True,
        "roi_crop": True,
    },
    "location": {
        "latitude": 43.67,
        "longitude": 1.42,
        "timezone_offset": 1,
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
        "username": "",
        "password": "",
    },
    "calibration": {
        "port": 8081,
    },
    "logging": {
        "level": "INFO",
        "max_size_mb": 10,
        "backup_count": 3,
    },
    "performance": {
        "prefetch_motion": True,
        "backlog_throttle_threshold": 10,
        "backlog_fps_factor": 0.5,
    },
    "data": {
        "retention_days": 0,  # 0 = keep forever
    },
    "audio_filter": {
        "enabled": True,
        "window_sec": 0.5,          # taille de la fenêtre RMS (secondes)
        "calibration_files": 10,    # fichiers requis avant d'utiliser le filtre
        "sigma_factor": 2.5,        # seuil = p10_moyen + sigma × std_moyen
        "segment_padding": 2.0,     # secondes ajoutées autour des segments
        "min_energy_db": -55.0,     # seuil absolu minimum (dBFS) — jour
        "night_min_energy_db": -65.0,  # seuil absolu minimum la nuit (plus permissif)
        "night_calibration": True,  # utiliser les fichiers de nuit comme référence
        "night_start_hour": 22,     # début plage nuit (heure locale)
        "night_end_hour": 6,        # fin plage nuit (heure locale)
    },
    "night_detection": {
        "enabled": True,
        "brightness_threshold": 50,    # luminosité médiane < seuil → mode nuit (phares)
        "twilight_threshold": 100,     # luminosité entre night et twilight → les deux détecteurs
        "sample_fps": 5,               # échantillonnage luminosité (peu coûteux)
        "flash_sigma": 3.0,            # pic = baseline + sigma × std
        "min_flash_sep_sec": 1.5,      # fusionne les pics plus proches que ça
        "merge_window_sec": 4.0,       # fenêtre de fusion doublon crépuscule (secondes)
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
    def __init__(self, data: dict, config_path: "Path | None" = None):
        self._data = data
        self._config_path = config_path
        self._backlog_size: int = 0  # set by processing_loop for adaptive fps

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
    def max_recent_files(self) -> int:
        """Max number of recent files to process per scan (0 = unlimited)."""
        return int(self.get("ingestion", "max_recent_files", default=0))

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
    def imgsz(self) -> int:
        return int(self.get("detector", "imgsz", default=320))

    @property
    def roi_crop(self) -> bool:
        return bool(self.get("detector", "roi_crop", default=True))

    @property
    def night_confidence_threshold(self) -> float:
        return float(self.get("detector", "night_confidence_threshold"))

    @property
    def night_enhance(self) -> bool:
        return bool(self.get("detector", "night_enhance"))

    @property
    def latitude(self) -> float:
        return float(self.get("location", "latitude"))

    @property
    def longitude(self) -> float:
        return float(self.get("location", "longitude"))

    @property
    def timezone_offset(self) -> int:
        return int(self.get("location", "timezone_offset"))

    @property
    def vehicle_classes(self) -> list:
        return self.get("detector", "vehicle_classes", default=[])

    @property
    def count_direction(self) -> bool:
        return bool(self.get("detector", "count_direction"))

    @property
    def model_name(self) -> str:
        return self.get("detector", "model_name", default="yolo11n")

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

    @property
    def dashboard_username(self) -> str:
        return self.get("dashboard", "username", default="") or ""

    @property
    def dashboard_password(self) -> str:
        return self.get("dashboard", "password", default="") or ""

    @property
    def data_retention_days(self) -> int:
        return int(self.get("data", "retention_days", default=0))

    @property
    def prefetch_motion(self) -> bool:
        return bool(self.get("performance", "prefetch_motion", default=True))

    @property
    def backlog_throttle_threshold(self) -> int:
        return int(self.get("performance", "backlog_throttle_threshold", default=10))

    @property
    def backlog_fps_factor(self) -> float:
        return float(self.get("performance", "backlog_fps_factor", default=0.5))

    @property
    def effective_motion_fps(self) -> float:
        base = self.motion_sample_fps
        t = self.backlog_throttle_threshold
        if t > 0 and self._backlog_size > t:
            return max(0.5, base * self.backlog_fps_factor)
        return base

    @property
    def effective_detector_fps(self) -> float:
        base = self.detector_sample_fps
        t = self.backlog_throttle_threshold
        if t > 0 and self._backlog_size > t:
            return max(1.0, base * self.backlog_fps_factor)
        return base

    @property
    def audio_enabled(self) -> bool:
        return bool(self.get("audio_filter", "enabled", default=True))

    @property
    def audio_window_sec(self) -> float:
        return float(self.get("audio_filter", "window_sec", default=0.5))

    @property
    def audio_calibration_files(self) -> int:
        return int(self.get("audio_filter", "calibration_files", default=20))

    @property
    def audio_sigma_factor(self) -> float:
        return float(self.get("audio_filter", "sigma_factor", default=2.5))

    @property
    def audio_segment_padding(self) -> float:
        return float(self.get("audio_filter", "segment_padding", default=2.0))

    @property
    def audio_min_energy_db(self) -> float:
        return float(self.get("audio_filter", "min_energy_db", default=-55.0))

    @property
    def audio_night_min_energy_db(self) -> float:
        return float(self.get("audio_filter", "night_min_energy_db", default=-65.0))

    @property
    def audio_night_calibration(self) -> bool:
        return bool(self.get("audio_filter", "night_calibration", default=True))

    @property
    def audio_night_start_hour(self) -> int:
        return int(self.get("audio_filter", "night_start_hour", default=22))

    @property
    def audio_night_end_hour(self) -> int:
        return int(self.get("audio_filter", "night_end_hour", default=6))

    @property
    def night_detection_enabled(self) -> bool:
        return bool(self.get("night_detection", "enabled", default=True))

    @property
    def night_brightness_threshold(self) -> float:
        return float(self.get("night_detection", "brightness_threshold", default=50))

    @property
    def night_sample_fps(self) -> float:
        return float(self.get("night_detection", "sample_fps", default=5))

    @property
    def night_flash_sigma(self) -> float:
        return float(self.get("night_detection", "flash_sigma", default=3.0))

    @property
    def night_twilight_threshold(self) -> float:
        return float(self.get("night_detection", "twilight_threshold", default=100))

    @property
    def night_min_flash_sep_sec(self) -> float:
        return float(self.get("night_detection", "min_flash_sep_sec", default=1.5))

    @property
    def night_merge_window_sec(self) -> float:
        return float(self.get("night_detection", "merge_window_sec", default=4.0))

    def motion_fingerprint(self) -> str:
        """MD5 of parameters that affect only motion detection (not AI inference)."""
        relevant = {
            "motion_threshold": self.motion_threshold,
            "min_motion_area": self.min_motion_area,
            "motion_sample_fps": self.motion_sample_fps,
            "segment_padding": self.segment_padding,
            "roi_polygon": self.roi_polygon,
        }
        blob = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()

    def reload(self) -> set:
        """Re-reads the config file and updates _data in place.
        Returns the set of top-level keys whose values changed."""
        if self._config_path is None or not self._config_path.exists():
            return set()
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                user_data = yaml.safe_load(f) or {}
            new_data = _deep_merge(DEFAULTS, user_data)
            changed = {
                k for k in set(self._data) | set(new_data)
                if self._data.get(k) != new_data.get(k)
            }
            self._data = new_data
            if changed:
                logger.info("Configuration rechargée — sections modifiées : %s", changed)
                for w in validate_config(new_data):
                    logger.warning("Config : %s", w)
            return changed
        except Exception as e:
            logger.error("Erreur rechargement config : %s", e)
            return set()

    def detection_fingerprint(self) -> str:
        """MD5 of the main detection parameters — informational only.

        A change here is logged but NEVER triggers an automatic re-processing
        of the history. New parameters apply to future files only; re-processing
        old files is a manual action ("Remettre en attente" in the dashboard).
        """
        relevant = {
            "line_p1": list(self.line_p1),
            "line_p2": list(self.line_p2),
            "roi_polygon": self.roi_polygon,
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "vehicle_classes": sorted(self.vehicle_classes),
            "count_direction": self.count_direction,
            "audio_enabled": self.audio_enabled,
            "audio_sigma_factor": self.audio_sigma_factor,
            "audio_min_energy_db": self.audio_min_energy_db,
        }
        blob = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()

    def raw(self) -> dict:
        return self._data

    def validate(self) -> list[str]:
        return validate_config(self._data)


def validate_config(data: dict) -> list[str]:
    """Returns a list of human-readable warnings for invalid config values."""
    warnings: list[str] = []
    counting = data.get("counting", {})

    for key in ("line_p1", "line_p2"):
        val = counting.get(key)
        if val is not None and (not isinstance(val, (list, tuple)) or len(val) != 2):
            warnings.append(f"counting.{key} doit être [x, y] — valeur actuelle : {val!r}")

    roi = counting.get("roi_polygon")
    if roi is not None:
        if not isinstance(roi, list) or len(roi) < 3:
            n = len(roi) if isinstance(roi, list) else "invalide"
            warnings.append(f"counting.roi_polygon doit avoir au moins 3 points (actuel : {n})")
        else:
            for i, pt in enumerate(roi):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    warnings.append(f"counting.roi_polygon[{i}] doit être [x, y] — valeur : {pt!r}")

    detector = data.get("detector", {})
    for key in ("confidence_threshold", "night_confidence_threshold"):
        val = detector.get(key)
        if val is not None:
            try:
                v = float(val)
                if not 0.0 <= v <= 1.0:
                    warnings.append(f"detector.{key} doit être entre 0 et 1 (actuel : {v})")
            except (ValueError, TypeError):
                warnings.append(f"detector.{key} doit être un nombre (actuel : {val!r})")

    for section, key in [("motion_filter", "sample_fps"), ("detector", "sample_fps")]:
        val = data.get(section, {}).get(key)
        if val is not None:
            try:
                if float(val) <= 0:
                    warnings.append(f"{section}.{key} doit être > 0 (actuel : {val})")
            except (ValueError, TypeError):
                warnings.append(f"{section}.{key} doit être un nombre (actuel : {val!r})")

    valid_classes = {"car", "motorcycle", "bus", "truck"}
    classes = detector.get("vehicle_classes", [])
    if isinstance(classes, list):
        unknown = [c for c in classes if c not in valid_classes]
        if unknown:
            warnings.append(f"detector.vehicle_classes contient des valeurs inconnues : {unknown}")

    return warnings


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

    # Environment variable overrides (highest priority — override config file)
    _env_overrides = [
        ("DASHBOARD_PORT",        ("dashboard", "port"),          int),
        ("VIDEO_FOLDER",          ("video_folder", None),         str),
        ("DB_PATH",               ("database", "path"),           str),
        ("LOG_LEVEL",             ("logging", "level"),           str),
        ("DASHBOARD_USERNAME",    ("dashboard", "username"),      str),
        ("DASHBOARD_PASSWORD",    ("dashboard", "password"),      str),
        ("DATA_RETENTION_DAYS",   ("data", "retention_days"),     int),
    ]
    for env_key, (section, key), cast in _env_overrides:
        val = os.environ.get(env_key)
        if val is None:
            continue
        try:
            typed = cast(val)
        except (ValueError, TypeError) as exc:
            logger.warning("Variable d'env %s invalide (%r) : %s", env_key, val, exc)
            continue
        if key is None:
            data[section] = typed
        else:
            if section not in data or not isinstance(data[section], dict):
                data[section] = {}
            data[section][key] = typed
        logger.info("Override env %s → %s.%s = %r", env_key, section, key, typed)

    for w in validate_config(data):
        logger.warning("Config : %s", w)

    return Config(data, config_path=config_path)
