"""Page de débogage — test de détection sur une frame."""
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request

from .calibration import _extract_frame, _find_video_by_name, _find_latest_video
from .config import Config

logger = logging.getLogger(__name__)

COCO_NAMES = {2: "voiture", 3: "moto", 5: "bus", 7: "camion"}
COLORS = {2: (0, 200, 0), 3: (200, 100, 0), 5: (0, 100, 200), 7: (200, 0, 200)}


def create_debug_blueprint(config: Config) -> Blueprint:
    bp = Blueprint("debug", __name__, url_prefix="/debug", template_folder="templates")

    @bp.route("/")
    def index():
        return render_template("debug.html")

    @bp.route("/api/detect")
    def api_detect():
        """Extract a frame, run YOLO detection, return annotated image."""
        video_name = request.args.get("video")
        second = float(request.args.get("second", 5.0))
        conf_override = request.args.get("conf")

        # Find video
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

        # Run YOLO
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

            conf = float(conf_override) if conf_override else 0.15  # very low threshold for debug
            results = model(frame, conf=conf, verbose=False)

        except Exception as e:
            return jsonify({"error": f"Erreur YOLO : {e}"}), 500

        # Annotate frame
        annotated = frame.copy()
        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                name  = COCO_NAMES.get(cid, f"classe {cid}")
                color = COLORS.get(cid, (180, 180, 180))
                is_vehicle = cid in COCO_NAMES

                # Draw box (green for vehicles, gray for others)
                thickness = 3 if is_vehicle else 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color if is_vehicle else (120, 120, 120), thickness)
                label = f"{name} {conf:.0%}"
                cv2.rectangle(annotated, (x1, y1 - 20), (x1 + len(label) * 9, y1), color if is_vehicle else (120, 120, 120), -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                detections.append({
                    "name": name,
                    "cid": cid,
                    "conf": round(conf, 3),
                    "is_vehicle": is_vehicle,
                    "box": [x1, y1, x2, y2],
                })

        # Draw ROI polygon
        roi = config.roi_polygon
        if roi:
            pts = np.array(roi, dtype=np.int32)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

        # Encode
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        h, w = frame.shape[:2]

        vehicles = [d for d in detections if d["is_vehicle"]]
        others   = [d for d in detections if not d["is_vehicle"]]

        return jsonify({
            "image": f"data:image/jpeg;base64,{b64}",
            "width": w,
            "height": h,
            "source": video_file.name,
            "vehicles": vehicles,
            "other_detections": others,
            "conf_used": conf,
        })

    return bp
