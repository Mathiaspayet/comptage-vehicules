"""Module Détection IA & Comptage — YOLO + ByteTrack + line crossing."""
import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import Config
from .motion_filter import Segment

_TRACKER_CFG = Path(__file__).parent / "bytetrack_custom.yaml"

logger = logging.getLogger(__name__)

# COCO class IDs relevant to vehicles
COCO_VEHICLE_CLASSES = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


@dataclass
class CrossingEvent:
    timestamp: datetime
    vehicle_type: str
    direction: str | None   # "left_to_right" | "right_to_left" | None
    confidence: float
    source_file: str


class VehicleDetector:
    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._active_class_ids = self._resolve_class_ids()

    def _resolve_class_ids(self) -> list[int]:
        ids = []
        for name in self.config.vehicle_classes:
            cid = COCO_VEHICLE_CLASSES.get(name.lower())
            if cid is not None:
                ids.append(cid)
        return ids or list(COCO_VEHICLE_CLASSES.values())

    def _load_model(self):
        """Loads the YOLO model, preferring OpenVINO format for Intel CPUs.
        Re-exports if the cached model was built with a different imgsz or model_name."""
        from ultralytics import YOLO

        model_dir = self.config.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.config.model_name          # e.g. "yolo11n", "yolov8n", "yolov8s"
        imgsz = self.config.imgsz

        ov_dir_name = f"{model_name}_openvino_model"
        ov_path     = model_dir / ov_dir_name
        meta_path   = ov_path / "_meta.json"
        pt_path     = model_dir / f"{model_name}.pt"

        # Use cached OpenVINO model if imgsz matches
        if ov_path.exists():
            cached_imgsz = None
            try:
                cached_imgsz = json.loads(meta_path.read_text())["imgsz"]
            except Exception:
                pass
            if cached_imgsz == imgsz:
                logger.info("Chargement du modèle OpenVINO %s (imgsz=%d) : %s", model_name, imgsz, ov_path)
                self._model = YOLO(str(ov_path))
                return
            else:
                logger.info(
                    "imgsz changé (%s→%d) — re-export du modèle OpenVINO %s…",
                    cached_imgsz, imgsz, model_name,
                )
                shutil.rmtree(ov_path)

        if not pt_path.exists():
            logger.info("Téléchargement du modèle %s.pt…", model_name)
        model_pt = YOLO(str(pt_path) if pt_path.exists() else f"{model_name}.pt")

        try:
            logger.info("Export OpenVINO %s (imgsz=%d)…", model_name, imgsz)
            model_pt.export(format="openvino", imgsz=imgsz, half=False)
            local_ov = Path(ov_dir_name)
            if local_ov.exists():
                shutil.copytree(local_ov, ov_path)
                shutil.rmtree(local_ov)
            pt_source = Path(f"{model_name}.pt")
            if pt_source.exists() and not pt_path.exists():
                shutil.copy(pt_source, pt_path)
            meta_path.write_text(json.dumps({"imgsz": imgsz}))
            logger.info("Modèle OpenVINO exporté : %s", ov_path)
            self._model = YOLO(str(ov_path))
        except Exception as e:
            logger.warning("Export OpenVINO échoué (%s), utilisation du modèle .pt", e)
            pt_source = Path(f"{model_name}.pt")
            if pt_source.exists() and not pt_path.exists():
                shutil.copy(pt_source, pt_path)
            self._model = YOLO(str(pt_path) if pt_path.exists() else f"{model_name}.pt")

    def process_video(
        self,
        video_path: Path,
        segments: list[Segment],
        video_start_dt: datetime | None = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        start_seg_idx: int = 0,
        initial_crossings: list | None = None,
        on_segment_done: Optional[Callable[[int, list], None]] = None,
        shutdown_event=None,
    ) -> list[CrossingEvent]:
        """Runs AI detection only on the active segments of a video.

        When resuming from a checkpoint, pass start_seg_idx to skip already-processed
        segments. Returns only the NEW CrossingEvent objects found from start_seg_idx
        onward (the caller is responsible for combining with previously saved crossings).
        """
        if self._model is None:
            self._load_model()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

        try:
            return self._process(
                cap, video_path, segments, video_start_dt, on_progress,
                start_seg_idx=start_seg_idx,
                on_segment_done=on_segment_done,
                shutdown_event=shutdown_event,
            )
        finally:
            cap.release()

    def _roi_crop_bbox(self) -> tuple[int, int, int, int] | None:
        """Returns (x0, y0, x1, y1) bounding box of the ROI polygon, or None if disabled."""
        if not self.config.roi_crop:
            return None
        roi = self.config.roi_polygon
        if not roi or len(roi) < 3:
            return None
        pts = np.array(roi)
        x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _process(
        self,
        cap: cv2.VideoCapture,
        video_path: Path,
        segments: list[Segment],
        video_start_dt: datetime | None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        start_seg_idx: int = 0,
        on_segment_done: Optional[Callable[[int, list], None]] = None,
        shutdown_event=None,
    ) -> list[CrossingEvent]:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_fps = self.config.effective_detector_fps
        step = max(1, int(fps / sample_fps))

        # Only count frames in segments we will actually process (>= start_seg_idx)
        total_active_frames = max(1, sum(
            int((seg.end_sec - seg.start_sec) * fps / step)
            for seg in segments[start_seg_idx:]
        ))

        line_p1 = np.array(self.config.line_p1, dtype=np.float32)
        line_p2 = np.array(self.config.line_p2, dtype=np.float32)

        # Presence-mode state: frame count per track ID, and confirmed IDs
        count_mode = self.config.count_mode
        track_frame_counts: dict[int, int] = {}
        track_counted: set[int] = set()

        if count_mode == "presence":
            logger.info("Mode comptage : présence de track (min=%d frames)", self.config.min_presence_frames)
        else:
            logger.info("Mode comptage : franchissement de ligne")

        # Precompute ROI crop and adjust line coordinates into cropped space
        crop_bbox = self._roi_crop_bbox()
        if crop_bbox:
            x0, y0, x1, y1 = crop_bbox
            offset = np.array([x0, y0], dtype=np.float32)
            line_p1_use = line_p1 - offset
            line_p2_use = line_p2 - offset
            logger.debug("ROI crop activé : (%d,%d)→(%d,%d) — %dx%d px au lieu de la frame entière",
                         x0, y0, x1, y1, x1 - x0, y1 - y0)
        else:
            line_p1_use = line_p1
            line_p2_use = line_p2

        track_sides: dict[int, float] = {}
        track_types: dict[int, str] = {}
        track_confs: dict[int, float] = {}
        # New crossings found in this run (not including initial_crossings)
        crossings: list[CrossingEvent] = []
        # Running list of new crossing dicts for the on_segment_done callback
        new_crossings_dicts: list[dict] = []

        source_name = video_path.name
        seg_idx = 0
        frame_idx = 0
        inside_segment = False
        analysed_frames = 0

        if on_progress:
            on_progress(0, total_active_frames)

        _seg_start_wall: float | None = None
        _seg_frames = 0

        while seg_idx < len(segments):
            # Skip segments that were already processed before the checkpoint
            if seg_idx < start_seg_idx:
                seg_idx += 1
                continue

            if shutdown_event is not None and shutdown_event.is_set():
                logger.info("Arrêt demandé — interruption après le segment %d/%d.", seg_idx, len(segments))
                break

            seg = segments[seg_idx]
            current_sec = frame_idx / fps

            if current_sec < seg.start_sec:
                if _seg_start_wall is None:
                    _seg_start_wall = time.monotonic()
                    _seg_frames = 0
                cap.set(cv2.CAP_PROP_POS_MSEC, seg.start_sec * 1000)
                frame_idx = int(seg.start_sec * fps)
                current_sec = seg.start_sec
                inside_segment = True

            if current_sec > seg.end_sec:
                # Segment finished — log timing and fire the checkpoint callback
                if _seg_start_wall is not None and _seg_frames > 0:
                    elapsed = time.monotonic() - _seg_start_wall
                    spf = elapsed / _seg_frames
                    logger.info(
                        "Segment %d/%d [%.0f→%.0f s] : %d frames en %.1f s (%.3f s/frame)",
                        seg_idx + 1, len(segments),
                        seg.start_sec, seg.end_sec,
                        _seg_frames, elapsed, spf,
                    )
                if on_segment_done is not None:
                    on_segment_done(seg_idx, list(new_crossings_dicts))
                seg_idx += 1
                inside_segment = False
                _seg_start_wall = None
                _seg_frames = 0
                continue

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0 and inside_segment:
                # Crop to ROI bounding box before inference
                inference_frame = frame[y0:y1, x0:x1] if crop_bbox else frame

                events = self._analyze_frame(
                    inference_frame,
                    frame_idx,
                    fps,
                    video_start_dt,
                    source_name,
                    line_p1_use,
                    line_p2_use,
                    track_sides,
                    track_types,
                    track_confs,
                    track_frame_counts=track_frame_counts if count_mode == "presence" else None,
                    track_counted=track_counted if count_mode == "presence" else None,
                )
                crossings.extend(events)
                new_crossings_dicts.extend([
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "vehicle_type": e.vehicle_type,
                        "direction": e.direction,
                        "confidence": e.confidence,
                        "source_file": e.source_file,
                    }
                    for e in events
                ])
                analysed_frames += 1
                _seg_frames += 1
                if on_progress:
                    on_progress(analysed_frames, total_active_frames)

            frame_idx += 1
            current_sec = frame_idx / fps
            if current_sec > seg.end_sec:
                # Segment finished — fire the checkpoint callback
                if on_segment_done is not None:
                    on_segment_done(seg_idx, list(new_crossings_dicts))
                seg_idx += 1
                inside_segment = False

        total_wall = time.monotonic() - (_seg_start_wall or time.monotonic())
        logger.info(
            "%s → %d franchissement(s) | %d frames analysées | %d/%d segments",
            video_path.name, len(crossings), analysed_frames,
            len(segments) - start_seg_idx, len(segments),
        )
        return crossings

    def _sunrise_sunset_local(self, dt: datetime) -> tuple[float, float]:
        """Returns (sunrise_hour, sunset_hour) in local time for the configured location."""
        lat = self.config.latitude
        lon = self.config.longitude
        tz_offset = self.config.timezone_offset
        n = dt.timetuple().tm_yday
        b = math.radians(360 / 365 * (n - 81))
        decl = math.radians(23.45 * math.sin(b))
        lat_r = math.radians(lat)
        cos_ha = -math.tan(lat_r) * math.tan(decl)
        cos_ha = max(-1.0, min(1.0, cos_ha))
        ha = math.degrees(math.acos(cos_ha))
        sunrise_utc = 12 - ha / 15 - lon / 15
        sunset_utc = 12 + ha / 15 - lon / 15
        return sunrise_utc + tz_offset, sunset_utc + tz_offset

    def _is_night(self, dt: datetime | None) -> bool:
        if dt is None:
            return False
        sunrise, sunset = self._sunrise_sunset_local(dt)
        h = dt.hour + dt.minute / 60
        return h < sunrise or h >= sunset

    def _enhance_frame(self, frame: np.ndarray, night: bool = False) -> np.ndarray:
        # CLAHE on L channel — normalise le contraste local, aide sur les scènes
        # en contre-jour (matin/soir avec soleil rasant) et la nuit.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        if not hasattr(self, "_clahe"):
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if not hasattr(self, "_clahe_night"):
            self._clahe_night = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = (self._clahe_night if night else self._clahe).apply(l)
        frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        if night:
            # Correction gamma : remonte les pixels sombres pour la nuit
            gamma = 0.5
            lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
            frame = lut[frame]
        return frame

    def _analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        video_start_dt: datetime | None,
        source_name: str,
        line_p1: np.ndarray,
        line_p2: np.ndarray,
        track_sides: dict,
        track_types: dict,
        track_confs: dict,
        track_frame_counts: "dict[int, int] | None" = None,
        track_counted: "set[int] | None" = None,
    ) -> list[CrossingEvent]:
        night = self._is_night(video_start_dt)
        conf = self.config.night_confidence_threshold if night else self.config.confidence_threshold
        # CLAHE appliqué sur toutes les frames (contre-jour, nuit, luminosité inégale).
        # La correction gamma supplémentaire est réservée à la nuit (night_enhance).
        if self.config.night_enhance:
            frame = self._enhance_frame(frame, night=night)

        results = self._model.track(
            frame,
            persist=True,
            tracker=str(_TRACKER_CFG),
            classes=self._active_class_ids,
            conf=conf,
            imgsz=self.config.imgsz,
            verbose=False,
        )

        crossings: list[CrossingEvent] = []
        if not results or results[0].boxes is None:
            return crossings

        boxes = results[0].boxes
        if boxes.id is None:
            return crossings

        track_ids = boxes.id.int().tolist()
        class_ids = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()

        current_sec = frame_idx / fps
        frame_dt = (video_start_dt + timedelta(seconds=current_sec)) if video_start_dt else datetime.utcnow()

        presence_mode = track_frame_counts is not None and track_counted is not None
        min_frames = self.config.min_presence_frames

        for tid, cid, conf_val, box in zip(track_ids, class_ids, confs, xyxy):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            centroid = np.array([cx, cy], dtype=np.float32)

            track_confs[tid] = max(track_confs.get(tid, 0.0), conf_val)
            track_types[tid] = _coco_id_to_name(cid)

            if presence_mode:
                # Count each unique track visible for ≥ min_frames as one vehicle.
                # Works regardless of direction of travel — no line orientation needed.
                frame_count = track_frame_counts.get(tid, 0) + 1  # type: ignore[union-attr]
                track_frame_counts[tid] = frame_count  # type: ignore[index]
                if frame_count >= min_frames and tid not in track_counted:  # type: ignore[operator]
                    track_counted.add(tid)  # type: ignore[union-attr]
                    direction = None
                    if self.config.count_direction:
                        side = _side_of_line(centroid, line_p1, line_p2)
                        direction = "left_to_right" if side > 0 else "right_to_left"
                    crossings.append(CrossingEvent(
                        timestamp=frame_dt,
                        vehicle_type=track_types[tid],
                        direction=direction,
                        confidence=track_confs[tid],
                        source_file=source_name,
                    ))
                    logger.debug(
                        "Présence : %s tid=%d (%d frames, confiance=%.2f)",
                        track_types[tid], tid, frame_count, track_confs[tid],
                    )
            else:
                # Line-crossing mode: count when a track crosses from one side to the other.
                side = _side_of_line(centroid, line_p1, line_p2)
                if tid in track_sides:
                    prev_side = track_sides[tid]
                    if prev_side != 0 and side != 0 and (
                        (prev_side > 0 and side < 0) or (prev_side < 0 and side > 0)
                    ):
                        direction = None
                        if self.config.count_direction:
                            direction = "left_to_right" if prev_side > 0 else "right_to_left"
                        crossings.append(CrossingEvent(
                            timestamp=frame_dt,
                            vehicle_type=track_types[tid],
                            direction=direction,
                            confidence=track_confs[tid],
                            source_file=source_name,
                        ))
                        logger.debug(
                            "Franchissement : %s %s (confiance=%.2f)",
                            track_types[tid], direction or "", conf_val,
                        )
                if side != 0:
                    track_sides[tid] = side

        return crossings


# ------------------------------------------------------------------ #
# Helper functions                                                     #
# ------------------------------------------------------------------ #

def _side_of_line(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Sign of the cross product — positive on one side, negative on the other."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    px = point[0] - p1[0]
    py = point[1] - p1[1]
    cross = dx * py - dy * px
    return cross


def _coco_id_to_name(cid: int) -> str:
    reverse = {v: k for k, v in COCO_VEHICLE_CLASSES.items()}
    return reverse.get(cid, "vehicle")
