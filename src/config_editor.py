"""Blueprint de configuration — lecture et écriture de config.yaml via l'interface web."""
import logging
import os
import signal
import threading
import time
from pathlib import Path

import yaml
from flask import Blueprint, jsonify, render_template, request

from .config import Config, DEFAULTS, _deep_merge, validate_config

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
                    "max_recent_files": int(data.get("max_recent_files", 0)),
                },
                "motion_filter": {
                    "sample_fps": float(data.get("motion_sample_fps", 2)),
                    "motion_threshold": int(data.get("motion_threshold", 25)),
                    "min_motion_area": int(data.get("min_motion_area", 500)),
                    "segment_padding_seconds": float(data.get("segment_padding_seconds", 1.0)),
                },
                "detector": {
                    "sample_fps": float(data.get("detector_sample_fps", 4)),
                    "model_name": data.get("model_name", "yolo11n"),
                    "imgsz": int(data.get("imgsz", 320)),
                    "confidence_threshold": float(data.get("confidence_threshold", 0.35)),
                    "night_confidence_threshold": float(data.get("night_confidence_threshold", 0.18)),
                    "night_enhance": bool(data.get("night_enhance", True)),
                    "roi_crop": bool(data.get("roi_crop", True)),
                    "vehicle_classes": data.get("vehicle_classes", ["car", "motorcycle", "bus", "truck"]),
                    "count_direction": bool(data.get("count_direction", False)),
                    "model_dir": "/app/data/models",
                },
                "location": {
                    "latitude": float(data.get("latitude", 43.67)),
                    "longitude": float(data.get("longitude", 1.42)),
                    "timezone_offset": int(data.get("timezone_offset", 1)),
                },
                "dashboard": {
                    "port": 8080,
                    "default_days": int(data.get("default_days", 30)),
                    "username": str(data.get("dashboard_username", "")).strip(),
                    "password": str(data.get("dashboard_password", "")).strip(),
                },
                "database": {"path": "/app/data/vehicles.db"},
                "data": {
                    "retention_days": int(data.get("data_retention_days", 0)),
                },
                "logging": {
                    "level": data.get("log_level", "INFO"),
                    "max_size_mb": 10,
                    "backup_count": 3,
                },
                "performance": {
                    "prefetch_motion": bool(data.get("prefetch_motion", True)),
                    "backlog_throttle_threshold": int(data.get("backlog_throttle_threshold", 10)),
                    "backlog_fps_factor": float(data.get("backlog_fps_factor", 0.5)),
                },
            }

            # Preserve existing counting coordinates (managed by calibration tool)
            existing = _load_raw()
            if "counting" in existing:
                cfg["counting"] = existing["counting"]

            merged_check = _deep_merge(DEFAULTS, cfg)
            warnings = validate_config(merged_check)
            for w in warnings:
                logger.warning("Config : %s", w)

            _save_raw(cfg)
            logger.info("Configuration sauvegardée via l'interface web.")
            return jsonify({"ok": True, "warnings": warnings})

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

    @bp.route("/api/estimate")
    def api_estimate():
        """Estimates processing time per 30-min video based on current config."""
        raw = _load_raw()
        merged = _deep_merge(DEFAULTS, raw)

        motion_fps = float(merged.get("motion_filter", {}).get("sample_fps", 2))
        detector_fps = float(merged.get("detector", {}).get("sample_fps", 4))
        imgsz = int(merged.get("detector", {}).get("imgsz", 320))
        roi_crop = bool(merged.get("detector", {}).get("roi_crop", True))

        # Empirical constants for NAS DS218+ / Intel Celeron J3355 with OpenVINO
        MOTION_SEC_PER_FRAME = 0.20   # seconds per sampled frame (motion OpenCV)
        DETECT_SEC_PER_FRAME_640 = 0.50   # seconds per YOLO frame at imgsz=640
        ACTIVE_FRACTION = 0.30        # typical fraction of video with motion
        VIDEO_MINUTES = 30.0

        # imgsz=320 is ~4x faster; roi_crop reduces pixels by ~60% → ~2x faster
        imgsz_factor = 4.0 if imgsz <= 320 else 1.0
        roi_factor = 2.0 if roi_crop else 1.0
        detect_sec_per_frame = DETECT_SEC_PER_FRAME_640 / (imgsz_factor * roi_factor)

        motion_frames = VIDEO_MINUTES * 60 * motion_fps
        motion_time_sec = motion_frames * MOTION_SEC_PER_FRAME

        detect_frames = VIDEO_MINUTES * 60 * ACTIVE_FRACTION * detector_fps
        detect_time_sec = detect_frames * detect_sec_per_frame

        total_min = round((motion_time_sec + detect_time_sec) / 60, 1)

        return jsonify({
            "video_minutes": VIDEO_MINUTES,
            "estimated_minutes": total_min,
            "motion_contribution_min": round(motion_time_sec / 60, 1),
            "detect_contribution_min": round(detect_time_sec / 60, 1),
            "motion_fps": motion_fps,
            "detector_fps": detector_fps,
        })

    return bp
