"""Filtre audio — détecte les segments avec activité sonore (véhicules).

Principe :
  1. Extraction PCM 16 kHz mono via ffmpeg (rapide, pas de décodage vidéo)
  2. Énergie RMS par fenêtre de 0.5 s → profil dB
  3. Auto-calibration sur les N derniers fichiers :
       seuil = percentile10_moyen + sigma × std_moyen
  4. Les segments audio sont fusionnés avec les segments mouvement (union)
"""

import logging
import subprocess
from pathlib import Path

import numpy as np

from .config import Config
from .motion_filter import Segment, _merge_segments

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000   # Hz — suffisant pour bruit moteur (< 8 kHz)


class AudioFilter:
    def __init__(self, config: Config, db):
        self.config = config
        self.db = db

    # ------------------------------------------------------------------ #
    # API publique                                                         #
    # ------------------------------------------------------------------ #

    def analyze_video(self, video_path: Path) -> list[Segment]:
        """Analyse la piste audio et retourne les segments actifs.

        Stocke toujours les stats pour la calibration.
        Retourne [] si désactivé, pas d'audio, ou pas encore calibré.
        """
        if not self.config.audio_enabled:
            return []

        samples = self._extract_audio(video_path)
        if samples is None or len(samples) < _SAMPLE_RATE:
            logger.debug("%s — pas de piste audio utilisable.", video_path.name)
            return []

        duration_sec = len(samples) / _SAMPLE_RATE
        energy_db = self._compute_energy_db(samples)

        stats = {
            "mean_db":   float(np.mean(energy_db)),
            "median_db": float(np.median(energy_db)),
            "std_db":    float(np.std(energy_db)),
            "p10_db":    float(np.percentile(energy_db, 10)),
            "p90_db":    float(np.percentile(energy_db, 90)),
        }
        self.db.add_audio_stats(video_path.name, **stats)

        logger.debug(
            "%s — audio : médiane=%.1f dB  p10=%.1f dB  p90=%.1f dB  std=%.1f dB",
            video_path.name,
            stats["median_db"], stats["p10_db"], stats["p90_db"], stats["std_db"],
        )

        threshold = self._get_threshold()
        if threshold is None:
            n = self.db.get_audio_stats_count()
            logger.info(
                "%s — calibration audio : %d/%d fichiers analysés.",
                video_path.name, n, self.config.audio_calibration_files,
            )
            return []

        segments = self._detect_segments(energy_db, threshold, duration_sec)
        total = sum(s.end_sec - s.start_sec for s in segments)
        logger.info(
            "%s — audio : seuil=%.1f dB → %d segment(s) / %.1f s (%.0f%%)",
            video_path.name, threshold, len(segments), total,
            100 * total / duration_sec if duration_sec else 0,
        )
        return segments

    def calibration_info(self) -> dict:
        """Informations de calibration pour le dashboard."""
        n = self.db.get_audio_stats_count()
        min_files = self.config.audio_calibration_files
        agg = self.db.get_audio_calibration_aggregate()
        threshold = self._get_threshold()
        return {
            "enabled":        self.config.audio_enabled,
            "calibrated":     threshold is not None,
            "files_analyzed": n,
            "files_needed":   min_files,
            "background_db":  round(agg["avg_p10_db"], 1) if agg else None,
            "noise_std_db":   round(agg["avg_std_db"], 1) if agg else None,
            "threshold_db":   round(threshold, 1) if threshold is not None else None,
            "sigma_factor":   self.config.audio_sigma_factor,
        }

    # ------------------------------------------------------------------ #
    # Implémentation interne                                               #
    # ------------------------------------------------------------------ #

    def _extract_audio(self, video_path: Path) -> "np.ndarray | None":
        """Extrait l'audio mono 16 kHz via ffmpeg pipe (float32 normalisé)."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",                        # pas de vidéo
            "-acodec", "pcm_s16le",       # PCM 16 bits
            "-ar", str(_SAMPLE_RATE),     # 16 kHz
            "-ac", "1",                   # mono
            "-f", "s16le",                # format brut
            "-loglevel", "error",
            "pipe:1",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            raw = proc.stdout.read()
            proc.wait()
        except FileNotFoundError:
            logger.debug("ffmpeg absent — filtre audio désactivé.")
            return None

        if not raw:
            return None
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _compute_energy_db(self, samples: np.ndarray) -> np.ndarray:
        """Énergie RMS (dBFS) par fenêtre temporelle."""
        window = max(1, int(self.config.audio_window_sec * _SAMPLE_RATE))
        n = len(samples) // window
        energy = np.empty(n)
        for i in range(n):
            chunk = samples[i * window: (i + 1) * window]
            rms = np.sqrt(np.mean(chunk ** 2))
            energy[i] = 20.0 * np.log10(max(rms, 1e-10))
        return energy

    def _get_threshold(self) -> "float | None":
        if self.db.get_audio_stats_count() < self.config.audio_calibration_files:
            return None
        agg = self.db.get_audio_calibration_aggregate()
        if not agg or agg.get("avg_p10_db") is None:
            return None
        threshold = (
            agg["avg_p10_db"]
            + self.config.audio_sigma_factor * agg["avg_std_db"]
        )
        return max(threshold, self.config.audio_min_energy_db)

    def _detect_segments(
        self, energy_db: np.ndarray, threshold: float, duration_sec: float
    ) -> list[Segment]:
        window_sec = self.config.audio_window_sec
        padding = self.config.audio_segment_padding

        segments: list[Segment] = []
        active_start: "float | None" = None

        for i, val in enumerate(energy_db):
            t = i * window_sec
            if val >= threshold and active_start is None:
                active_start = max(0.0, t - padding)
            elif val < threshold and active_start is not None:
                segments.append(Segment(
                    start_sec=active_start,
                    end_sec=min(duration_sec, t + padding),
                ))
                active_start = None

        if active_start is not None:
            segments.append(Segment(start_sec=active_start, end_sec=duration_sec))

        return _merge_segments(segments, gap=3.0)


def union_segments(a: list[Segment], b: list[Segment]) -> list[Segment]:
    """Union de deux listes de segments (fusion mouvement + audio)."""
    if not b:
        return a
    if not a:
        return b
    combined = sorted(a + b, key=lambda s: s.start_sec)
    return _merge_segments(combined, gap=2.0)
