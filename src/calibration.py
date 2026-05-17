"""Outil de calibration — interface web pour définir la ligne de comptage et la ROI."""
import base64
import logging
import os
from pathlib import Path

import cv2
import yaml
from flask import Flask, jsonify, render_template, request

from .config import Config

logger = logging.getLogger(__name__)


def create_calibration_app(config: Config) -> Flask:
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))

    @app.route("/")
    def index():
        return render_template("calibration.html")

    @app.route("/api/frame")
    def api_frame():
        """Extract a frame from the video folder and return it as base64 JPEG."""
        video_folder = config.video_folder
        frame_path = request.args.get("video")

        if frame_path:
            video_file = Path(frame_path)
        else:
            video_file = _find_latest_video(video_folder)

        if video_file is None or not video_file.exists():
            return jsonify({"error": "Aucune vidéo trouvée dans le dossier configuré."}), 404

        frame_sec = float(request.args.get("second", 5.0))
        frame = _extract_frame(video_file, frame_sec)
        if frame is None:
            return jsonify({"error": f"Impossible d'extraire une image de {video_file.name}"}), 500

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        h, w = frame.shape[:2]
        return jsonify({
            "image": f"data:image/jpeg;base64,{b64}",
            "width": w,
            "height": h,
            "source": video_file.name,
        })

    @app.route("/api/videos")
    def api_videos():
        """List available video files for the user to choose from."""
        folder = config.video_folder
        if not folder.exists():
            return jsonify({"videos": [], "error": "Dossier vidéo introuvable."})
        exts = {".mp4", ".avi", ".mkv", ".mov"}
        videos = sorted(
            [f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts],
            reverse=True,
        )[:20]
        return jsonify({"videos": videos, "folder": str(folder)})

    @app.route("/api/current-config")
    def api_current_config():
        """Return current counting configuration."""
        return jsonify({
            "line_p1": list(config.line_p1),
            "line_p2": list(config.line_p2),
            "roi_polygon": config.roi_polygon,
        })

    @app.route("/api/save", methods=["POST"])
    def api_save():
        """Save the line and ROI coordinates to config.yaml."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données manquantes"}), 400

        line_p1 = data.get("line_p1")
        line_p2 = data.get("line_p2")
        roi_polygon = data.get("roi_polygon", [])

        if not line_p1 or not line_p2:
            return jsonify({"error": "line_p1 et line_p2 sont requis"}), 400

        config_path = os.environ.get("CONFIG_PATH", "/app/data/config.yaml")

        # Load existing config or start fresh from example
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}

        cfg.setdefault("counting", {})
        cfg["counting"]["line_p1"] = [int(line_p1[0]), int(line_p1[1])]
        cfg["counting"]["line_p2"] = [int(line_p2[0]), int(line_p2[1])]
        if roi_polygon:
            cfg["counting"]["roi_polygon"] = [[int(p[0]), int(p[1])] for p in roi_polygon]

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        except IOError as e:
            return jsonify({"error": f"Impossible d'écrire config.yaml : {e}"}), 500

        logger.info(
            "Calibration enregistrée → ligne: %s–%s, ROI: %d points",
            line_p1, line_p2, len(roi_polygon),
        )
        return jsonify({"ok": True, "saved_to": config_path})

    return app


def run_calibration(config: Config):
    app = create_calibration_app(config)
    port = config.calibration_port
    logger.info("Outil de calibration démarré sur http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _find_latest_video(folder: Path) -> Path | None:
    exts = {".mp4", ".avi", ".mkv", ".mov"}
    if not folder.exists():
        return None
    videos = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts]
    return max(videos, key=lambda f: f.stat().st_mtime) if videos else None


def _extract_frame(video_path: Path, second: float = 5.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ret, frame = cap.read()
        if not ret:
            # Fallback: first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()
