"""Module Tableau de bord web — Flask read-only dashboard."""
import logging
from datetime import date, datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .config import Config
from .database import Database

logger = logging.getLogger(__name__)


def create_dashboard_app(config: Config, db: Database) -> Flask:
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["JSON_SORT_KEYS"] = False

    @app.route("/")
    def index():
        today = date.today().isoformat()
        return render_template("dashboard.html", today=today)

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

    return app


def run_dashboard(config: Config, db: Database):
    app = create_dashboard_app(config, db)
    port = config.dashboard_port
    logger.info("Tableau de bord démarré sur http://0.0.0.0:%d", port)
    # use_reloader=False is important when running in a thread
    app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)
