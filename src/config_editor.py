"""Blueprint de configuration — lecture et écriture de config.yaml via l'interface web."""
import logging
import os
import signal
import threading
import time
from pathlib import Path

import yaml
from flask import Blueprint, jsonify, render_template, request

from .config import Config, DEFAULTS, _deep_merge

logger = logging.getLogger(__name__)


def create_config_blueprint(config: Config) -> Blueprint:
    bp = Blueprint("config_editor", __name__, url_prefix="/config", template_folder="templates")

    def _config_path() -> str:
        return os.environ.get("CONFIG_PATH", "/app/data/config.yaml")

    def _load_raw() -> dict:
        path = _config_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def _save_raw(data: dict):
        path = _config_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    @bp.route("/")
    def index():
        return render_template("config.html")

    @bp.route("/api/current")
    def api_current():
        """Return the current merged config (defaults + file)."""
        raw = _load_raw()
        merged = _deep_merge(DEFAULTS, raw)
        return jsonify(merged)

    @bp.route("/api/save", methods=["POST"])
    def api_save():
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données manquantes"}), 400

        # Build a clean config dict from posted form data
        try:
            cfg = {
                "video_folder": data.get("video_folder", "/video").strip(),
                "filename_datetime_format": data.get("filename_datetime_format", "").strip(),
                "timezone": data.get("timezone", "Europe/Paris").strip(),
                "ingestion": {
                    "scan_interval_seconds": int(data.get("scan_interval_seconds", 60)),
                    "file_stable_delay_seconds": int(data.get("file_stable_delay_seconds", 120)),
                    "max_recent_files": int(data.get("max_recent_files", 5)),
                },
                "motion_filter": {
                    "sample_fps": float(data.get("motion_sample_fps", 2)),
                    "motion_threshold": int(data.get("motion_threshold", 25)),
                    "min_motion_area": int(data.get("min_motion_area", 500)),
                    "segment_padding_seconds": float(data.get("segment_padding_seconds", 1.0)),
                },
                "detector": {
                    "sample_fps": float(data.get("detector_sample_fps", 2)),
                    "confidence_threshold": float(data.get("confidence_threshold", 0.35)),
                    "vehicle_classes": data.get("vehicle_classes", ["car", "motorcycle", "bus", "truck"]),
                    "count_direction": bool(data.get("count_direction", False)),
                    "model_dir": "/app/data/models",
                },
                "dashboard": {
                    "port": 8080,
                    "default_days": int(data.get("default_days", 30)),
                },
                "database": {"path": "/app/data/vehicles.db"},
                "logging": {
                    "level": data.get("log_level", "INFO"),
                    "max_size_mb": 10,
                    "backup_count": 3,
                },
            }

            # Preserve existing counting coordinates (managed by calibration tool)
            existing = _load_raw()
            if "counting" in existing:
                cfg["counting"] = existing["counting"]

            _save_raw(cfg)
            logger.info("Configuration sauvegardée via l'interface web.")
            return jsonify({"ok": True})

        except Exception as e:
            logger.error("Erreur sauvegarde config : %s", e)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/restart", methods=["POST"])
    def api_restart():
        """Send SIGTERM to PID 1 — Docker restart: always relaunches the container."""
        def _do_restart():
            time.sleep(1.5)
            logger.info("Redémarrage demandé depuis l'interface web.")
            os.kill(1, signal.SIGTERM)

        threading.Thread(target=_do_restart, daemon=True).start()
        return jsonify({"ok": True, "message": "Redémarrage en cours…"})

    return bp
