"""Outil de calibration — Flask Blueprint monté sous /calibration."""
import base64
import logging
import os
from pathlib import Path

import cv2
import yaml
from flask import Blueprint, jsonify, render_template, request

from .config import Config

logger = logging.getLogger(__name__)


def create_calibration_blueprint(config: Config) -> Blueprint:
    bp = Blueprint(
        "calibration",
        __name__,
        url_prefix="/calibration",
        template_folder="templates",
    )

    @bp.route("/")
    def index():
        return render_template("calibration.html")

    @bp.route("/api/frame")
    def api_frame():
        video_folder = config.video_folder
        frame_path = request.args.get("video")

        if frame_path:
            video_file = _find_video_by_name(video_folder, frame_path)
        else:
            video_file = _find_latest_video(video_folder)

        if video_file is None or not video_file.exists():
            return jsonify({"error": "Aucune vidéo trouvée."}), 404

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

    @bp.route("/api/videos")
    def api_videos():
        folder = config.video_folder
        if not folder.exists():
            return jsonify({"videos": [], "error": "Dossier vidéo introuvable."})
        exts = {".mp4", ".avi", ".mkv", ".mov"}
        videos = []
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in exts:
                videos.append(entry.name)
            elif entry.is_dir():
                for sub in entry.iterdir():
                    if sub.is_file() and sub.suffix.lower() in exts:
                        videos.append(sub.name)
        videos = sorted(videos, reverse=True)[:20]
        return jsonify({"videos": videos, "folder": str(folder)})

    @bp.route("/api/current-config")
    def api_current_config():
        return jsonify({
            "roi_polygon": config.roi_polygon,
        })

    @bp.route("/api/save", methods=["POST"])
    def api_save():
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données manquantes"}), 400

        roi_polygon = data.get("roi_polygon", [])

        if not roi_polygon or len(roi_polygon) < 3:
            return jsonify({"error": "Le polygone ROI doit avoir au moins 3 points."}), 400

        config_path = os.environ.get("CONFIG_PATH", "/app/data/config.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}

        cfg.setdefault("counting", {})
        cfg["counting"]["roi_polygon"] = [[int(p[0]), int(p[1])] for p in roi_polygon]

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        except IOError as e:
            return jsonify({"error": f"Impossible d'écrire config.yaml : {e}"}), 500

        logger.info("ROI enregistré → %d points", len(roi_polygon))
        return jsonify({"ok": True, "saved_to": config_path})

    return bp


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _find_video_by_name(folder: Path, name: str):
    """Search for a video file by name in folder and immediate subfolders."""
    if not folder.exists():
        return None
    exts = {".mp4", ".avi", ".mkv", ".mov"}
    for entry in folder.iterdir():
        if entry.is_file() and entry.name == name and entry.suffix.lower() in exts:
            return entry
        elif entry.is_dir():
            for sub in entry.iterdir():
                if sub.is_file() and sub.name == name:
                    return sub
    return None


def _find_latest_video(folder: Path):
    exts = {".mp4", ".avi", ".mkv", ".mov"}
    if not folder.exists():
        return None
    videos = []
    for entry in folder.iterdir():
        if entry.is_file() and entry.suffix.lower() in exts:
            videos.append(entry)
        elif entry.is_dir():
            for sub in entry.iterdir():
                if sub.is_file() and sub.suffix.lower() in exts:
                    videos.append(sub)
    return max(videos, key=lambda f: f.stat().st_mtime) if videos else None


def _extract_frame(video_path: Path, second: float = 5.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()
