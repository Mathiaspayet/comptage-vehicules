"""Module Tableau de bord web — Flask app avec calibration intégrée."""
import csv
import io
import json
import logging
from datetime import date, datetime
from pathlib import Path

from flask import Flask, Response, jsonify, make_response, render_template, request, send_file

from .calibration import create_calibration_blueprint
from .config import Config
from .config_editor import create_config_blueprint
from .database import Database
from .debug import create_debug_blueprint
from .progress import tracker as progress_tracker

logger = logging.getLogger(__name__)

VERSION_FILE = Path("/app/version.json")

# Module-level size cache for the pending-files API endpoint.
# Maps filename → (size, timestamp when size was first seen at this value).
_pending_size_cache: dict[str, tuple[int, float]] = {}


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

    # ── Protection CSRF : tous les POST/DELETE doivent être AJAX ───
    @app.before_request
    def _check_csrf():
        if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
            return None
        if request.headers.get("X-Requested-With") != "XMLHttpRequest":
            return jsonify({"error": "Requête invalide (header X-Requested-With manquant)"}), 403

    # ── Auth basique optionnelle ────────────────────────────────────
    @app.before_request
    def _check_auth():
        username = config.dashboard_username
        if not username:
            return None
        # Healthcheck must bypass auth so Docker can reach /api/status
        if request.path == "/api/status":
            return None
        auth = request.authorization
        if not auth or auth.username != username or auth.password != config.dashboard_password:
            return Response(
                "Accès restreint — authentification requise.",
                401,
                {"WWW-Authenticate": 'Basic realm="Comptage véhicules"'},
            )

    # ── En-têtes de sécurité HTTP ───────────────────────────────────
    @app.after_request
    def _add_security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

    # ── Calibration sous /calibration ──────────────────────────────
    calib_bp = create_calibration_blueprint(config)
    app.register_blueprint(calib_bp)

    # ── Configuration sous /config ─────────────────────────────────
    config_bp = create_config_blueprint(config)
    app.register_blueprint(config_bp)

    # ── Débogage sous /debug ────────────────────────────────────────
    debug_bp = create_debug_blueprint(config)
    app.register_blueprint(debug_bp)

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
            hourly    = db.get_hourly_stats(day, vehicle_type)
            summary   = db.get_summary(day, vehicle_type)
            breakdown = db.get_vehicle_type_breakdown(day)
            direction = db.get_direction_stats(day)
        except Exception as e:
            logger.error("Erreur stats horaires : %s", e)
            return jsonify({"error": str(e)}), 500
        return jsonify({"date": day, "hourly": hourly, "summary": summary, "breakdown": breakdown, "direction": direction})

    @app.route("/api/stats/calendar")
    def api_calendar():
        import calendar as _cal
        year  = int(request.args.get("year",  date.today().year))
        month = int(request.args.get("month", date.today().month))
        try:
            days = db.get_calendar_stats(year, month)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        first_weekday = _cal.monthrange(year, month)[0]  # 0 = Monday
        last_day      = _cal.monthrange(year, month)[1]
        return jsonify({
            "year": year, "month": month,
            "first_weekday": first_weekday,
            "last_day": last_day,
            "days": days,
        })

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

    @app.route("/api/files")
    def api_files():
        status_filter = request.args.get("status", "all")
        limit = int(request.args.get("limit", 200))
        offset = int(request.args.get("offset", 0))
        max_age_days = int(request.args.get("max_age_days", 30))
        sort_by = request.args.get("sort_by", "processed_at")
        sort_dir = request.args.get("sort_dir", "desc")
        try:
            files = db.get_processed_files(limit, offset, status_filter, max_age_days, sort_by, sort_dir)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"files": files, "count": len(files)})

    @app.route("/api/files/pending")
    def api_files_pending():
        """Returns video files on disk that haven't been processed yet."""
        import time as _time
        folder = config.video_folder
        stable_delay = config.file_stable_delay
        now = _time.time()
        exts = {".mp4", ".avi", ".mkv", ".mov"}

        if not folder.exists():
            return jsonify({"pending": [], "writing": []})

        all_videos: list[Path] = []
        try:
            for entry in folder.iterdir():
                if entry.is_file() and entry.suffix.lower() in exts:
                    all_videos.append(entry)
                elif entry.is_dir():
                    try:
                        for sub in entry.iterdir():
                            if sub.is_file() and sub.suffix.lower() in exts:
                                all_videos.append(sub)
                    except PermissionError:
                        pass
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        processed_names = db.get_all_processed_filenames()
        pending = []
        writing = []

        for f in sorted(all_videos, key=lambda x: x.stat().st_mtime):
            if f.name in processed_names:
                continue
            try:
                current_size = f.stat().st_size
                age_sec = now - f.stat().st_mtime
            except FileNotFoundError:
                _pending_size_cache.pop(f.name, None)
                continue

            cached = _pending_size_cache.get(f.name)
            if cached is None or cached[0] != current_size:
                _pending_size_cache[f.name] = (current_size, now)
                stable_since = now
            else:
                stable_since = cached[1]

            is_stable = (now - stable_since) >= stable_delay
            entry = {
                "filename": f.name,
                "age_minutes": round(age_sec / 60, 1),
                "size_kb": round(current_size / 1024, 0),
            }
            if is_stable:
                pending.append(entry)
            else:
                writing.append(entry)

        return jsonify({
            "pending": pending,
            "writing": writing,
            "stable_delay_seconds": stable_delay,
        })

    @app.route("/api/files/reset", methods=["POST"])
    def api_files_reset():
        data = request.get_json() or {}
        filenames = data.get("filenames")  # None = reset all
        try:
            if filenames is None:
                count = db.unmark_all_files()
            else:
                count = db.unmark_files(filenames)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"ok": True, "reset_count": count})

    @app.route("/api/files/skip-all", methods=["POST"])
    def api_files_skip_all():
        """Mark all pending files on disk as skipped so the processor ignores them."""
        import time as _time
        folder = config.video_folder
        exts = {".mp4", ".avi", ".mkv", ".mov"}
        now = _time.time()

        all_videos: list[Path] = []
        try:
            if folder.exists():
                for entry in folder.iterdir():
                    if entry.is_file() and entry.suffix.lower() in exts:
                        all_videos.append(entry)
                    elif entry.is_dir():
                        try:
                            for sub in entry.iterdir():
                                if sub.is_file() and sub.suffix.lower() in exts:
                                    all_videos.append(sub)
                        except PermissionError:
                            pass
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        processed_names = db.get_all_processed_filenames()
        to_skip = [f.name for f in all_videos if f.name not in processed_names]
        try:
            count = db.skip_files(to_skip)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"ok": True, "skipped_count": count})

    @app.route("/api/maintenance")
    def api_maintenance():
        """Returns disk usage of /app/data subfolders."""
        import shutil
        data_dir = Path("/app/data")
        result = {}
        for sub in ["logs", "models", "vehicles.db"]:
            p = data_dir / sub
            if p.exists():
                if p.is_file():
                    result[sub] = p.stat().st_size
                else:
                    result[sub] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            else:
                result[sub] = 0
        total = sum(result.values())
        return jsonify({"sizes": result, "total": total})

    @app.route("/api/maintenance/clean-logs", methods=["POST"])
    def api_maintenance_clean_logs():
        """Delete rotated log files (.log.1, .log.2, .log.3)."""
        logs_dir = Path("/app/data/logs")
        deleted = 0
        freed = 0
        if logs_dir.exists():
            for f in logs_dir.iterdir():
                if f.is_file() and f.suffix in {".1", ".2", ".3"}:
                    freed += f.stat().st_size
                    f.unlink()
                    deleted += 1
        return jsonify({"ok": True, "deleted": deleted, "freed_bytes": freed})

    @app.route("/api/maintenance/clear-motion-cache", methods=["POST"])
    def api_maintenance_clear_motion_cache():
        """Delete all cached motion segments."""
        try:
            count = db.clear_motion_cache()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"ok": True, "deleted": count})

    @app.route("/api/maintenance/purge-data", methods=["POST"])
    def api_maintenance_purge_data():
        """Delete crossings older than `days` days."""
        data = request.get_json() or {}
        try:
            days = int(data.get("days", 90))
        except (ValueError, TypeError):
            return jsonify({"error": "Paramètre days invalide"}), 400
        if days <= 0:
            return jsonify({"error": "days doit être > 0"}), 400
        try:
            count = db.delete_old_crossings(days)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"ok": True, "deleted": count, "days": days})

    @app.route("/api/maintenance/backup-db")
    def api_maintenance_backup_db():
        """Download a copy of the SQLite database."""
        db_path = config.db_path
        if not db_path.exists():
            return jsonify({"error": "Base de données introuvable"}), 404
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return send_file(
            str(db_path),
            as_attachment=True,
            download_name=f"vehicles_backup_{now}.db",
            mimetype="application/octet-stream",
        )

    @app.route("/api/export/crossings.csv")
    def api_export_crossings():
        """Export crossing events as CSV with optional date/type filters."""
        date_from = request.args.get("date_from") or None
        date_to = request.args.get("date_to") or None
        vehicle_type = request.args.get("vehicle_type", "all")
        try:
            rows = db.get_crossings_export(date_from, date_to, vehicle_type)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["timestamp", "vehicle_type", "direction", "confidence", "source_file"],
        )
        writer.writeheader()
        writer.writerows(rows)
        resp = make_response(buf.getvalue())
        resp.headers["Content-Type"] = "text/csv; charset=utf-8"
        resp.headers["Content-Disposition"] = "attachment; filename=franchissements.csv"
        return resp

    @app.route("/api/export/files.csv")
    def api_export_files():
        """Export processed files list as CSV."""
        max_age_days = int(request.args.get("max_age_days", 0))
        try:
            rows = db.get_processed_files(
                limit=100_000, offset=0, status_filter="all", max_age_days=max_age_days
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        fields = [
            "filename", "processed_at", "status",
            "vehicle_count", "processing_duration_seconds", "error_message",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        resp = make_response(buf.getvalue())
        resp.headers["Content-Type"] = "text/csv; charset=utf-8"
        resp.headers["Content-Disposition"] = "attachment; filename=fichiers_traites.csv"
        return resp

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
            snap = progress_tracker.snapshot()
            status = db.get_processing_status()
            return jsonify({
                # Progression frame par frame du fichier en cours
                "current_file": snap["current_file"],
                "phase": snap["phase"],                   # idle / motion / detection
                "file_percent": snap["file_percent"],     # 0–100 dans le fichier courant
                "frames_done": snap["frames_done"],
                "frames_total": snap["frames_total"],
                # Progression globale de la file
                "queue_done": snap["queue_done"],
                "queue_total": snap["queue_total"],
                "queue_percent": snap["queue_percent"],   # 0–100 sur la file entière
                # Reset en cours
                "resetting": snap["resetting"],
                "reset_reason": snap["reset_reason"],
                # Stats BDD
                "db_done": status.get("done", 0),
                "db_errors": status.get("errors", 0),
                "last_file": status.get("last_file"),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def run_dashboard(config: Config, db: Database):
    app = create_app(config, db)
    port = config.dashboard_port
    logger.info("Tableau de bord + calibration démarrés sur http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)
