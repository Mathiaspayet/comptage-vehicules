"""Prétraitement adaptatif des images — remplace la logique mode nuit/crépuscule/jour.

Principe :
  Un seul pipeline s'adapte à la luminosité mesurée de la scène.
  Plus de modes séparés avec leurs algorithmes différents.

  Étapes :
    1. measure_scene_quality() : échantillonne quelques frames → (luminance, glare_frac)
    2. AdaptivePreprocessor.preprocess() :
       a. Suppression glare si forte surexposition (soleil dans le viseur)
       b. CLAHE avec clip limit adaptatif (plus agressif quand c'est sombre)
       c. Correction gamma automatique (luminance < seuil → image éclaircie)
    3. adaptive_confidence() : seuil YOLO plus permissif la nuit

  Plus de NightDetector, plus de crépuscule adaptatif, plus de streaks.
  La condition visuelle est décrite par deux nombres (luminance, glare_frac)
  et toute la logique de prétraitement en découle.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

# Seuils de luminance ROI (0-255)
_DARK_LUM  = 40    # nuit franche : gamma fort + CLAHE agressif
_DIM_LUM   = 80    # crépuscule / gris : gamma modéré
# (au-dessus de _DIM_LUM = plein jour)

# Gammas (< 1.0 = éclaircit l'image)
_GAMMA_DARK = 0.50
_GAMMA_DIM  = 0.70

# Clip limit CLAHE
_CLIP_DARK  = 3.5
_CLIP_DIM   = 2.5
_CLIP_DAY   = 2.0

# Seuil de fraction surexposée pour le contre-jour
_GLARE_FRAC_THRESHOLD = 0.10


def measure_scene_quality(
    video_path: Path,
    config: Config,
    n_samples: int = 10,
) -> "tuple[float, float]":
    """Échantillonne n frames → (luminance_médiane, fraction_surexposée).

    - luminance_médiane (0-255) : mesurée dans le ROI si configuré.
    - fraction_surexposée (0-1) : pixels au-dessus du seuil glare → contre-jour.
    """
    polygon = config.roi_polygon if getattr(config, "night_roi_metering", True) else []
    glare_thr = config.glare_threshold

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 128.0, 0.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vals: list[float] = []
    glare: list[float] = []

    def _sample(frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        glare.append(float(np.count_nonzero(gray >= glare_thr)) / gray.size)
        if polygon and len(polygon) >= 3:
            h, w = gray.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(polygon, dtype=np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(mask, [pts], 255)
            region = gray[mask > 0]
            vals.append(float(np.median(region)) if region.size else float(np.median(gray)))
        else:
            vals.append(float(np.median(gray)))

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

    return (
        float(np.median(vals)) if vals else 128.0,
        float(np.median(glare)) if glare else 0.0,
    )


class AdaptivePreprocessor:
    """Prétraitement visuel adaptatif paramétré par (luminance, glare_frac).

    Remplace entièrement NightDetector + la logique mode nuit/crépuscule/jour.
    """

    def __init__(self, config: Config):
        self.config = config
        self._clahe_dark = cv2.createCLAHE(clipLimit=_CLIP_DARK, tileGridSize=(8, 8))
        self._clahe_dim  = cv2.createCLAHE(clipLimit=_CLIP_DIM,  tileGridSize=(8, 8))
        self._clahe_day  = cv2.createCLAHE(clipLimit=_CLIP_DAY,  tileGridSize=(8, 8))
        self._lut_dark   = _build_gamma_lut(_GAMMA_DARK)
        self._lut_dim    = _build_gamma_lut(_GAMMA_DIM)

    def preprocess(self, frame: np.ndarray, luminance: float, glare_frac: float) -> np.ndarray:
        """Applique glare → CLAHE → gamma selon la qualité de scène."""
        # 1. Suppression glare (soleil dans le viseur)
        if self.config.glare_suppression and glare_frac >= _GLARE_FRAC_THRESHOLD:
            frame = _suppress_glare(frame, self.config.glare_threshold)

        if not self.config.night_enhance:
            return frame

        # 2. CLAHE + 3. Gamma selon luminance
        if luminance < _DARK_LUM:
            frame = _apply_clahe(frame, self._clahe_dark)
            return self._lut_dark[frame]
        if luminance < _DIM_LUM:
            frame = _apply_clahe(frame, self._clahe_dim)
            return self._lut_dim[frame]

        # Jour standard : CLAHE léger uniquement
        if self.config.glare_suppression:
            frame = _apply_clahe(frame, self._clahe_day)
        return frame

    def adaptive_confidence(self, luminance: float) -> float:
        """Seuil de confiance YOLO adaptatif — plus permissif dans l'obscurité."""
        night_conf = self.config.night_confidence_threshold
        day_conf   = self.config.confidence_threshold
        if luminance < _DARK_LUM:
            return night_conf
        if luminance < _DIM_LUM:
            ratio = (luminance - _DARK_LUM) / (_DIM_LUM - _DARK_LUM)
            return night_conf + ratio * (day_conf - night_conf)
        return day_conf

    def visual_reliable(self, luminance: float, glare_frac: float) -> bool:
        """True si les conditions permettent à YOLO d'être fiable."""
        return luminance >= _DIM_LUM and glare_frac < _GLARE_FRAC_THRESHOLD

    def scene_label(self, luminance: float, glare_frac: float) -> str:
        """Étiquette de condition pour les logs et le stockage DB."""
        if glare_frac >= _GLARE_FRAC_THRESHOLD:
            return "twilight"   # contre-jour → condition difficile similaire au crépuscule
        if luminance < _DARK_LUM:
            return "night"
        if luminance < _DIM_LUM:
            return "twilight"
        return "day"

    def log_scene(self, filename: str, luminance: float, glare_frac: float) -> None:
        label = self.scene_label(luminance, glare_frac)
        cond = {
            "day":     f"JOUR (lum={luminance:.0f})",
            "twilight": f"CRÉPUSCULE/CONTRE-JOUR (lum={luminance:.0f}, glare={glare_frac:.0%})",
            "night":   f"NUIT (lum={luminance:.0f})",
        }.get(label, label)
        logger.info("%s — scène : %s", filename, cond)


# ── Helpers ──────────────────────────────────────────────────────────── #

def _suppress_glare(frame: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray >= threshold
    if not mask.any():
        return frame
    mask_u8 = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    out = frame.copy()
    out[mask_u8 > 0] = (127, 127, 127)
    return out


def _apply_clahe(frame: np.ndarray, clahe) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _build_gamma_lut(gamma: float) -> np.ndarray:
    return np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
