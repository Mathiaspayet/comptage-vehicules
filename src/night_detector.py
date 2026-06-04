"""Détection nocturne par éblouissement des phares.

Principe :
  La nuit, un véhicule qui passe provoque un pic de luminosité bref et intense
  dans le champ de la caméra (phares avant blancs, feux arrière rouges). Ce signal
  est bien plus propre que la détection YOLO sur des images sombres et bruitées.

  1. measure_scene_brightness() échantillonne quelques frames → décide nuit/jour
     sur la luminosité RÉELLE (robuste aux saisons, météo, nuages).
  2. NightDetector construit un profil de luminosité (90e percentile dans le ROI)
     et détecte les pics au-dessus de baseline + sigma × std. Chaque pic = 1 véhicule.

  L'audio reste le détecteur principal de segments ; ce module compte les véhicules
  à l'intérieur de ces segments la nuit, sans YOLO.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import Config
from .detector import CrossingEvent
from .motion_filter import Segment

logger = logging.getLogger(__name__)


def _roi_mask(shape, polygon) -> "np.ndarray | None":
    """Construit un masque binaire du polygone ROI à la taille de l'image."""
    if not polygon or len(polygon) < 3:
        return None
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    # Clamp dans les bornes de l'image (ROI calibré sur une autre résolution)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 255)
    if int(mask.sum()) == 0:
        return None
    return mask


def measure_scene_brightness(video_path: Path, n_samples: int = 10) -> float:
    """Compat : luminosité médiane plein cadre (0-255)."""
    bright, _ = measure_scene_lighting(video_path, n_samples=n_samples)
    return bright


def measure_scene_lighting(
    video_path: Path,
    config: "Config | None" = None,
    n_samples: int = 10,
    glare_threshold: int = 240,
) -> "tuple[float, float]":
    """Échantillonne n frames et retourne (luminosité_médiane, fraction_surexposée).

    - luminosité_médiane : mesurée dans le ROI (la route) si `config.night_roi_metering`,
      sinon plein cadre. Le métering ROI est robuste au contre-jour : au coucher de
      soleil le ciel est brûlé mais la route reste sombre.
    - fraction_surexposée : proportion moyenne de pixels (plein cadre) au-dessus de
      `glare_threshold` — signature d'un soleil directement dans le viseur.
    """
    roi_metering = bool(config.night_roi_metering) if config is not None else False
    polygon = config.roi_polygon if (config is not None and roi_metering) else None
    if config is not None:
        glare_threshold = config.glare_threshold

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 128.0, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vals: list[float] = []
    glare: list[float] = []

    def _sample(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fraction surexposée sur le plein cadre (le soleil/ciel)
        glare.append(float(np.count_nonzero(gray >= glare_threshold)) / gray.size)
        # Luminosité : ROI si dispo, sinon plein cadre
        region = gray
        if polygon:
            mask = _roi_mask(gray.shape, polygon)
            if mask is not None:
                region = gray[mask > 0]
        vals.append(float(np.median(region)) if region.size else float(np.median(gray)))

    try:
        if total <= 0:
            for _ in range(n_samples):
                ret, frame = cap.read()
                if not ret:
                    break
                _sample(frame)
        else:
            for i in range(1, n_samples + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, total * i // (n_samples + 1))
                ret, frame = cap.read()
                if ret:
                    _sample(frame)
    finally:
        cap.release()

    brightness = float(np.median(vals)) if vals else 128.0
    glare_frac = float(np.median(glare)) if glare else 0.0
    return brightness, glare_frac


class NightDetector:
    def __init__(self, config: Config):
        self.config = config

    def _roi_crop_bbox(self) -> "tuple[int, int, int, int] | None":
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

    def process_video(
        self,
        video_path: Path,
        segments: list[Segment],
        video_start_dt: "datetime | None" = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        start_seg_idx: int = 0,
        on_segment_done: Optional[Callable[[int, list], None]] = None,
        shutdown_event=None,
    ) -> list[CrossingEvent]:
        """Compte les véhicules de nuit par détection de pics de phares.

        Construit d'abord une baseline de luminosité de fond (scan grossier de
        toute la vidéo), puis détecte les pics dans chaque segment audio.
        Compatible checkpoint : appelle on_segment_done après chaque segment.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            sample_fps = self.config.night_sample_fps
            step = max(1, int(fps / sample_fps))
            crop = self._roi_crop_bbox()

            baseline, bg_std = self._measure_baseline(cap, fps, crop)
            sigma = self.config.night_flash_sigma
            threshold = baseline + sigma * max(bg_std, 1.0)
            logger.info(
                "%s — nuit : baseline=%.1f std=%.1f seuil_phares=%.1f",
                video_path.name, baseline, bg_std, threshold,
            )

            source_name = video_path.name
            crossings: list[CrossingEvent] = []
            new_crossings_dicts: list[dict] = []
            total_segments = len(segments)

            for seg_idx, seg in enumerate(segments):
                if seg_idx < start_seg_idx:
                    continue
                if shutdown_event is not None and shutdown_event.is_set():
                    logger.info("Arrêt demandé — interruption nuit après segment %d/%d.", seg_idx, total_segments)
                    break

                times, brightness, centroids_x = self._segment_profile(cap, seg, fps, step, crop)
                flashes = self._detect_flashes(times, brightness, centroids_x, baseline, threshold)

                for peak_t, peak_b, direction in flashes:
                    conf = min(1.0, (peak_b - baseline) / max(threshold - baseline, 1.0) * 0.5 + 0.4)
                    ts = (video_start_dt + timedelta(seconds=peak_t)) if video_start_dt else datetime.utcnow()
                    ev = CrossingEvent(
                        timestamp=ts,
                        vehicle_type="car",
                        direction=direction if self.config.count_direction else None,
                        confidence=round(conf, 2),
                        source_file=source_name,
                    )
                    crossings.append(ev)
                    new_crossings_dicts.append({
                        "timestamp": ev.timestamp.isoformat(),
                        "vehicle_type": ev.vehicle_type,
                        "direction": ev.direction,
                        "confidence": ev.confidence,
                        "source_file": ev.source_file,
                    })

                if on_progress:
                    on_progress(seg_idx + 1, total_segments)
                if on_segment_done is not None:
                    on_segment_done(seg_idx, list(new_crossings_dicts))

            logger.info(
                "%s → %d véhicule(s) nuit | %d/%d segments",
                source_name, len(crossings), total_segments - start_seg_idx, total_segments,
            )
            return crossings
        finally:
            cap.release()

    def _measure_baseline(self, cap, fps: float, crop) -> "tuple[float, float]":
        """Scan grossier (~1 fps) de toute la vidéo pour estimer le fond sombre."""
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return 0.0, 1.0
        n = 60  # ~60 échantillons sur toute la vidéo
        vals: list[float] = []
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, total * i // n)
            ret, frame = cap.read()
            if not ret:
                continue
            vals.append(self._roi_brightness(frame, crop))
        if not vals:
            return 0.0, 1.0
        arr = np.array(vals)
        baseline = float(np.percentile(arr, 25))   # fond sombre
        bg_std = float(np.std(arr[arr <= np.percentile(arr, 50)]))  # variation du fond
        return baseline, bg_std

    def _segment_profile(self, cap, seg: Segment, fps: float, step: int, crop):
        """Retourne (times, brightness, centroids_x) échantillonnés dans le segment.
        centroids_x = position horizontale (px, repère plein cadre) des phares,
        ou NaN si pas de pixel brillant identifiable."""
        start_f = int(seg.start_sec * fps)
        end_f = int(seg.end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        times: list[float] = []
        brightness: list[float] = []
        centroids_x: list[float] = []
        f = start_f
        while f <= end_f:
            ret, frame = cap.read()
            if not ret:
                break
            if (f - start_f) % step == 0:
                b, cx = self._roi_brightness(frame, crop)
                times.append(f / fps)
                brightness.append(b)
                centroids_x.append(cx)
            f += 1
        return times, brightness, centroids_x

    @staticmethod
    def _roi_brightness(frame: np.ndarray, crop) -> "tuple[float, float]":
        """Retourne (90e percentile de luminosité, centre horizontal des phares).
        Le centroïde x est en repère plein cadre (offset ROI ajouté)."""
        x_off = 0
        if crop:
            x0, y0, x1, y1 = crop
            frame = frame[y0:y1, x0:x1]
            x_off = x0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p90 = float(np.percentile(gray, 90))
        # Centroïde horizontal des pixels les plus brillants (phares)
        bright_thresh = float(np.percentile(gray, 99))
        ys, xs = np.where(gray >= bright_thresh)
        cx = float(np.mean(xs)) + x_off if xs.size else float("nan")
        return p90, cx

    def _detect_flashes(self, times, brightness, centroids_x, baseline, threshold):
        """Détecte les régions au-dessus du seuil. Chaque région = 1 véhicule.
        La direction vient du déplacement horizontal des phares pendant le flash.
        Retourne [(peak_time, peak_brightness, direction), ...]."""
        if not brightness:
            return []
        min_sep = self.config.night_min_flash_sep_sec
        flashes: list[tuple[float, float, "str | None"]] = []

        def _direction(lo: int, hi: int) -> "str | None":
            """Sens depuis le déplacement net du centroïde des phares sur [lo, hi]."""
            xs = [centroids_x[k] for k in range(lo, hi) if not np.isnan(centroids_x[k])]
            if len(xs) < 2:
                return None
            delta = xs[-1] - xs[0]
            if abs(delta) < 5:   # déplacement négligeable → indéterminé
                return None
            return "left_to_right" if delta > 0 else "right_to_left"

        in_flash = False
        start_idx = 0
        for i, b in enumerate(brightness):
            if b >= threshold and not in_flash:
                in_flash = True
                start_idx = i
            elif b < threshold and in_flash:
                in_flash = False
                region = brightness[start_idx:i]
                pk = start_idx + int(np.argmax(region))
                flashes.append((times[pk], brightness[pk], _direction(start_idx, i)))
        if in_flash:
            region = brightness[start_idx:]
            pk = start_idx + int(np.argmax(region))
            flashes.append((times[pk], brightness[pk], _direction(start_idx, len(brightness))))

        # Fusionne les pics trop proches (même véhicule)
        merged: list[tuple[float, float, "str | None"]] = []
        for t, b, d in flashes:
            if merged and t - merged[-1][0] < min_sep:
                if b > merged[-1][1]:
                    merged[-1] = (t, b, d)
            else:
                merged.append((t, b, d))
        return merged
