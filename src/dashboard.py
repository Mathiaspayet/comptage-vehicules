"""Module Tableau de bord web — Flask app avec calibration intégrée."""
import json
import logging
from datetime import date, datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .calibration import create_calibration_blueprint
from .config import Config
from .config_editor import create_config_blueprint
from .database import Database

logger = logging.getLogger(__name__)

VERSION_FILE = Path("/app/version.json")


def _read_version() -> dict:
    try:
        with open(VERSION_FILE) as f:
            return json.load(f)
    except Exception:
        return {"sha": "dev", "built_at": "—"}


def create_app(config: Config, db: Database) -> Flask:
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["JSON_SORT_KEYS"] = False

    # ── Calibration sous /calibration ──────────────────────────────
    calib_bp = create_calibration_blueprint(config)
    app.register_blueprint(calib_bp)

    # ── Configuration sous /config ─────────────────────────────────
    config_bp = create_config_blueprint(config)
    app.register_blueprint(config_bp)

    # ── Dashboard ──────────────────────────────────────────────────
    @app.route("/")
    def index():
        today = date.today().isoformat()
        return render_template("dashboard.html", today=today)

    @app.route("/api/version")
    def api_version():
        return jsonify(_read_version())

    @app.route("/api/stats/hourly")
    def api_hourly():
        day = request.args.get("date", date.today().isoformat())
        vehicle_type = request.args.get("vehicle_type", "all")
        try:
            hourly = db.get_hourly_stats(day, vehicle_type)
            summary = db.get_summary(day, vehicle_type)
            breakdown = db.get_vehicle_type_breakdown(day)
        except Exception as e:
            logger.error("Erreur stats horaires : %s", e)
            return jsonify({"error": str(e)}), 500
        return jsonify({"date": day, "hourly": hourly, "summary": summary, "breakdown": breakdown})

    @app.route("/api/stats/daily")
    def api_daily():
        days = int(request.args.get("days", config.default_days))
        vehicle_type = request.args.get("vehicle_type", "all")
        try:
            daily = db.get_daily_stats(days, vehicle_type)
        except Exception as e:
            logger.error("Erreur stats quotidiennes : %s", e)
            return jsonify({"error": str(e)}), 500
        return jsonify({"days": days, "daily": daily})

    @app.route("/api/dates")
    def api_dates():
        try:
            dates = db.get_available_dates()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"dates": dates})

    @app.route("/api/status")
    def api_status():
        try:
            status = db.get_processing_status()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        status["server_time"] = datetime.utcnow().isoformat() + "Z"
        return jsonify(status)

    @app.route("/api/journal")
    def api_journal():
        log_file = Path("/app/data/logs/comptage.log")
        lines = []
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                # Read last 200 lines efficiently
                all_lines = f.readlines()
                lines = [l.rstrip() for l in all_lines[-200:]]
        except FileNotFoundError:
            lines = ["(fichier journal non encore créé)"]
        except Exception as e:
            lines = [f"Erreur lecture journal : {e}"]
        return jsonify({"lines": lines})

    @app.route("/api/progress")
    def api_progress():
        try:
            status = db.get_processing_status()
            # Count pending files in the video folder
            video_folder = config.video_folder
            total_on_disk = 0
            exts = {".mp4", ".avi", ".mkv", ".mov"}
            if video_folder.exists():
                for entry in video_folder.iterdir():
                    if entry.is_file() and entry.suffix.lower() in exts:
                        total_on_disk += 1
                    elif entry.is_dir():
                        for sub in entry.iterdir():
                            if sub.is_file() and sub.suffix.lower() in exts:
                                total_on_disk += 1
            processed = status.get("done", 0) + status.get("errors", 0)
            pending = max(0, total_on_disk - processed)
            return jsonify({
                "total_on_disk": total_on_disk,
                "processed": status.get("done", 0),
                "errors": status.get("errors", 0),
                "pending": pending,
                "last_file": status.get("last_file"),
                "percent": round(100 * processed / total_on_disk) if total_on_disk else 0,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def run_dashboard(config: Config, db: Database):
    app = create_app(config, db)
    port = config.dashboard_port
    logger.info("Tableau de bord + calibration démarrés sur http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)
