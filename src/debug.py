"""Page de débogage — test de détection frame par frame et analyse audio/YOLO d'un fichier traité."""
import base64
import logging
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request, send_file

from .calibration import _extract_frame, _find_video_by_name, _find_latest_video
from .config import Config

logger = logging.getLogger(__name__)

COCO_NAMES = {2: "voiture", 3: "moto", 5: "bus", 7: "camion"}
COLORS = {2: (0, 200, 0), 3: (200, 100, 0), 5: (0, 100, 200), 7: (200, 0, 200)}


def _parse_video_start(filename: str, config: Config) -> "datetime | None":
    """Extrait le datetime de début de vidéo depuis le nom de fichier."""
    basename = Path(filename).name
    fmt = config.filename_datetime_format
    try:
        sample = datetime(2026, 1, 2, 14, 5, 6).strftime(fmt)
        prefix = basename[: len(sample)]
        return datetime.strptime(prefix, fmt)
    except Exception:
        pass
    m = re.search(r"(\d{8})[_\-](\d{6})", basename)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except Exception:
            pass
    return None


def create_debug_blueprint(config: Config, db=None, audio_filter=None) -> Blueprint:
    bp = Blueprint("debug", __name__, url_prefix="/debug", template_folder="templates")

    # ------------------------------------------------------------------ #
    # Page principale                                                      #
    # ------------------------------------------------------------------ #

    @bp.route("/")
    def index():
        return render_template("debug.html")

    # ------------------------------------------------------------------ #
    # API — test de détection sur une frame (onglet 1)                    #
    # ------------------------------------------------------------------ #

    @bp.route("/api/detect")
    def api_detect():
        """Extract a frame, run YOLO detection, return annotated image."""
        video_name = request.args.get("video")
        second = float(request.args.get("second", 5.0))
        conf_override = request.args.get("conf")

        folder = config.video_folder
        if video_name:
            video_file = _find_video_by_name(folder, video_name)
        else:
            video_file = _find_latest_video(folder)

        if video_file is None:
            return jsonify({"error": "Aucune vidéo trouvée."}), 404

        frame = _extract_frame(video_file, second)
        if frame is None:
            return jsonify({"error": "Impossible d'extraire la frame."}), 500

        try:
            from ultralytics import YOLO
            model_name = config.model_name
            model_dir = config.model_dir
            ov_path = model_dir / f"{model_name}_openvino_model"
            pt_path = model_dir / f"{model_name}.pt"

            if ov_path.exists():
                model = YOLO(str(ov_path))
            elif pt_path.exists():
                model = YOLO(str(pt_path))
            else:
                model = YOLO(f"{model_name}.pt")

            conf = float(conf_override) if conf_override else 0.15
            results = model(frame, conf=conf, verbose=False)

        except Exception as e:
            return jsonify({"error": f"Erreur YOLO : {e}"}), 500

        annotated = frame.copy()
        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cid  = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                name  = COCO_NAMES.get(cid, f"classe {cid}")
                color = COLORS.get(cid, (180, 180, 180))
                is_vehicle = cid in COCO_NAMES
                thickness = 3 if is_vehicle else 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color if is_vehicle else (120, 120, 120), thickness)
                label = f"{name} {conf_val:.0%}"
                cv2.rectangle(annotated, (x1, y1 - 20), (x1 + len(label) * 9, y1), color if is_vehicle else (120, 120, 120), -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                detections.append({"name": name, "cid": cid, "conf": round(conf_val, 3), "is_vehicle": is_vehicle, "box": [x1, y1, x2, y2]})

        roi = config.roi_polygon
        if roi:
            pts = np.array(roi, dtype=np.int32)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        h, w = frame.shape[:2]
        vehicles = [d for d in detections if d["is_vehicle"]]
        others   = [d for d in detections if not d["is_vehicle"]]

        return jsonify({
            "image": f"data:image/jpeg;base64,{b64}",
            "width": w, "height": h,
            "source": video_file.name,
            "vehicles": vehicles,
            "other_detections": others,
            "conf_used": conf,
        })

    # ------------------------------------------------------------------ #
    # API — analyse audio d'un fichier traité (onglet 2)                  #
    # ------------------------------------------------------------------ #

    @bp.route("/api/files")
    def api_files():
        """Liste les fichiers traités disponibles pour l'analyse de debug."""
        if db is None:
            return jsonify({"error": "DB non disponible"}), 503
        files = db.get_processed_files_done(limit=300)
        return jsonify({"files": files})

    @bp.route("/api/analyze")
    def api_analyze():
        """Re-analyse audio d'un fichier et retourne les segments avec les véhicules détectés."""
        if db is None or audio_filter is None:
            return jsonify({"error": "Module non disponible"}), 503

        filename = request.args.get("file", "").strip()
        if not filename:
            return jsonify({"error": "Paramètre file manquant"}), 400

        video_file = _find_video_by_name(config.video_folder, filename)
        if video_file is None:
            return jsonify({"error": f"Fichier vidéo introuvable : {filename}"}), 404

        # Re-analyse audio (rapide, ~8s)
        try:
            segments_raw = audio_filter.analyze_video(video_file)
        except Exception as e:
            logger.error("Erreur analyse audio debug : %s", e)
            return jsonify({"error": f"Erreur analyse audio : {e}"}), 500

        # Crossings depuis la DB
        crossings = db.get_crossings_for_file(filename)

        # Datetime de début de vidéo pour corréler les timestamps
        video_start_dt = _parse_video_start(filename, config)

        # Associe chaque crossing à son segment audio
        crossing_secs: list[float | None] = []
        for c in crossings:
            if video_start_dt:
                try:
                    c_dt = datetime.fromisoformat(c["timestamp"])
                    crossing_secs.append((c_dt - video_start_dt).total_seconds())
                except Exception:
                    crossing_secs.append(None)
            else:
                crossing_secs.append(None)

        segments_out = []
        matched_crossing_ids: set[int] = set()

        for i, seg in enumerate(segments_raw):
            seg_crossings = []
            for j, (c, c_sec) in enumerate(zip(crossings, crossing_secs)):
                if c_sec is not None and seg.start_sec <= c_sec <= seg.end_sec:
                    seg_crossings.append({
                        "vehicle_type": c["vehicle_type"],
                        "direction": c["direction"],
                        "confidence": round(c["confidence"], 2),
                        "sec_from_start": round(c_sec, 1),
                    })
                    matched_crossing_ids.add(j)

            segments_out.append({
                "idx": i,
                "start": round(seg.start_sec, 1),
                "end": round(seg.end_sec, 1),
                "duration": round(seg.end_sec - seg.start_sec, 1),
                "crossings": seg_crossings,
            })

        unmatched = [crossings[j] for j in range(len(crossings)) if j not in matched_crossing_ids]

        return jsonify({
            "filename": filename,
            "segments": segments_out,
            "total_segments": len(segments_out),
            "total_vehicles": len(crossings),
            "unmatched_crossings": len(unmatched),
            "video_start": video_start_dt.isoformat() if video_start_dt else None,
        })

    # ------------------------------------------------------------------ #
    # Streaming vidéo                                                      #
    # ------------------------------------------------------------------ #

    @bp.route("/video/<path:filename>")
    def serve_video(filename):
        """Sert le fichier vidéo pour le player HTML5 (supporte les Range requests)."""
        video_file = _find_video_by_name(config.video_folder, filename)
        if video_file is None:
            return jsonify({"error": "Vidéo introuvable"}), 404
        return send_file(str(video_file), mimetype="video/mp4", conditional=True)

    return bp
