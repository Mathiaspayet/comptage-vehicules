"""Détection audio améliorée — RMS + analyse spectrale FFT.

Améliorations par rapport au filtre RMS simple :
  1. FFT par fenêtre → score spectral dans les bandes véhicules (80–800 Hz)
  2. Détection à deux niveaux :
       - énergie haute (>= seuil) : détection certaine
       - énergie modérée + spectre véhicule (>= seuil-4dB ET score >= 0.35) :
         détecte les véhicules lents, électriques, distants
  3. Moins de faux positifs : vent, pluie, portes → large bande, score bas
  4. Interface identique à AudioFilter — remplacement direct, mêmes tables DB
"""

import logging
import re
import subprocess
from pathlib import Path

import numpy as np

from .config import Config
from .motion_filter import Segment, _merge_segments

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000  # Hz — suffisant pour bruit moteur (< 8 kHz)
_RE_DATETIME = re.compile(r"(\d{4})(\d{2})(\d{2})[^\d](\d{2})(\d{2})(\d{2})")

# Bandes véhicules : moteur, transmission, bruit de roulement
_VEHICLE_BAND_LOW_HZ  = 80
_VEHICLE_BAND_HIGH_HZ = 800

# Score spectral minimal pour le critère "faible-mais-véhicule"
_MIN_VEHICLE_SCORE = 0.35

# Marge en dB sous le seuil où le score spectral peut compenser
_SPECTRAL_BOOST_DB = 4.0


def _hour_from_filename(filename: str) -> "int | None":
    m = _RE_DATETIME.search(filename)
    if m:
        try:
            return int(m.group(4))
        except (ValueError, IndexError):
            pass
    return None


def _is_night_hour(hour: "int | None", night_start: int, night_end: int) -> bool:
    if hour is None:
        return False
    if night_start > night_end:
        return hour >= night_start or hour < night_end
    return night_start <= hour < night_end


class AudioDetector:
    """Détecteur audio spectral — remplace AudioFilter avec analyse FFT."""

    def __init__(self, config: Config, db):
        self.config = config
        self.db = db

    # ── API publique (compatible AudioFilter) ────────────────────────── #

    def analyze_video(self, video_path: Path) -> list[Segment]:
        """Analyse la piste audio et retourne les segments actifs.

        Interface identique à AudioFilter.analyze_video().
        """
        if not self.config.audio_enabled:
            return []

        video_hour = _hour_from_filename(video_path.name)
        samples = self._extract_audio(video_path)

        if samples is None or len(samples) < _SAMPLE_RATE:
            logger.debug("%s — pas de piste audio utilisable.", video_path.name)
            self.db.add_audio_stats(
                video_path.name, None, None, None, None, None,
                video_hour=video_hour,
            )
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
        self.db.add_audio_stats(video_path.name, **stats, video_hour=video_hour)

        is_night = _is_night_hour(
            video_hour,
            self.config.audio_night_start_hour,
            self.config.audio_night_end_hour,
        )
        threshold = self._get_threshold(is_night=is_night)

        if threshold is None:
            n = self.db.get_audio_stats_count()
            logger.info(
                "%s — calibration audio en cours : %d/%d fichiers analysés.",
                video_path.name, n, self.config.audio_calibration_files,
            )
            return []

        spectral = self._compute_spectral_scores(samples)
        segments = self._detect_segments(energy_db, spectral, threshold, duration_sec)

        total = sum(s.end_sec - s.start_sec for s in segments)
        avg_score = float(np.mean(spectral)) if len(spectral) else 0.0
        logger.info(
            "%s — audio spectral : seuil=%.1f dB score_moy=%.2f → %d segment(s) / %.1f s (%.0f%%)",
            video_path.name, threshold, avg_score, len(segments), total,
            100 * total / duration_sec if duration_sec else 0,
        )
        return segments

    def analyze_video_debug(
        self, video_path: Path, threshold_override: "float | None" = None
    ) -> dict:
        """Analyse audio avec données brutes pour visualisation debug."""
        if not self.config.audio_enabled:
            return {"error": "Filtre audio désactivé."}

        samples = self._extract_audio(video_path)
        if samples is None or len(samples) < _SAMPLE_RATE:
            return {"error": "Pas de piste audio utilisable."}

        duration_sec = len(samples) / _SAMPLE_RATE
        energy_db = self._compute_energy_db(samples)
        spectral = self._compute_spectral_scores(samples)

        video_hour = _hour_from_filename(video_path.name)
        is_night = _is_night_hour(
            video_hour,
            self.config.audio_night_start_hour,
            self.config.audio_night_end_hour,
        )
        threshold = (
            threshold_override
            if threshold_override is not None
            else self._get_threshold(is_night=is_night)
        )

        segments = []
        if threshold is not None:
            segments = self._detect_segments(energy_db, spectral, threshold, duration_sec)

        return {
            "segments": segments,
            "energy_db": energy_db.tolist(),
            "spectral_scores": spectral.tolist(),
            "window_sec": float(self.config.audio_window_sec),
            "threshold_db": float(threshold) if threshold is not None else None,
            "duration_sec": round(duration_sec, 1),
        }

    def is_calibrated(self) -> bool:
        """True si suffisamment de données pour calculer un seuil de détection."""
        return self._get_threshold(is_night=False) is not None

    def calibration_info(self) -> dict:
        """Informations de calibration pour le dashboard."""
        cs = self.config.audio_calibration_hour_start
        ce = self.config.audio_calibration_hour_end
        min_files = self.config.audio_calibration_files

        n_calib = self.db.get_audio_stats_count(night_only=True, night_start=cs, night_end=ce)
        n_all   = self.db.get_audio_stats_count()
        using_calib = n_calib >= min_files

        agg = self.db.get_audio_calibration_aggregate(
            night_only=using_calib, night_start=cs, night_end=ce
        )
        threshold_override_raw = self.db.get_state("audio_threshold_override")
        threshold_override = float(threshold_override_raw) if threshold_override_raw else None

        threshold_calc = self._get_threshold_calc(using_calib, cs, ce)
        threshold = threshold_override if threshold_override is not None else threshold_calc

        return {
            "enabled":            self.config.audio_enabled,
            "calibrated":         threshold is not None,
            "files_analyzed":     n_calib if using_calib else n_all,
            "files_needed":       min_files,
            "calib_hour_start":   cs,
            "calib_hour_end":     ce,
            "background_db":      round(agg["avg_p10_db"], 1) if agg else None,
            "noise_std_db":       round(agg["avg_std_db"], 1) if agg else None,
            "threshold_db":       round(threshold, 1) if threshold is not None else None,
            "threshold_db_calc":  round(threshold_calc, 1) if threshold_calc is not None else None,
            "threshold_override": threshold_override,
            "sigma_factor":       self.config.audio_sigma_factor,
        }

    # ── Implémentation ───────────────────────────────────────────────── #

    def _extract_audio(self, video_path: Path) -> "np.ndarray | None":
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(_SAMPLE_RATE), "-ac", "1",
            "-f", "s16le", "-loglevel", "error", "pipe:1",
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.debug("ffmpeg absent — filtre audio désactivé.")
            return None
        try:
            raw = proc.stdout.read()
            proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise
        finally:
            proc.stdout.close()

        if not raw:
            return None
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _compute_energy_db(self, samples: np.ndarray) -> np.ndarray:
        """Énergie RMS (dBFS) par fenêtre."""
        window = max(1, int(self.config.audio_window_sec * _SAMPLE_RATE))
        n = len(samples) // window
        energy = np.empty(n)
        for i in range(n):
            chunk = samples[i * window: (i + 1) * window]
            rms = np.sqrt(np.mean(chunk ** 2))
            energy[i] = 20.0 * np.log10(max(rms, 1e-10))
        return energy

    def _compute_spectral_scores(self, samples: np.ndarray) -> np.ndarray:
        """Score spectral véhicule par fenêtre — ratio puissance 80-800 Hz / totale.

        > 0.40 : signature moteur/roulement nette
        < 0.20 : bruit large bande (vent, pluie, voix haute)
        """
        window = max(1, int(self.config.audio_window_sec * _SAMPLE_RATE))
        n = len(samples) // window
        scores = np.empty(n)

        freq_res = _SAMPLE_RATE / window
        lo = max(1, int(_VEHICLE_BAND_LOW_HZ / freq_res))
        hi = min(window // 2 + 1, int(_VEHICLE_BAND_HIGH_HZ / freq_res))
        hann = np.hanning(window)

        for i in range(n):
            chunk = samples[i * window: (i + 1) * window]
            spectrum = np.abs(np.fft.rfft(chunk * hann)) ** 2
            total = np.sum(spectrum) + 1e-10
            scores[i] = np.sum(spectrum[lo:hi]) / total

        return scores

    def _detect_segments(
        self,
        energy_db: np.ndarray,
        spectral: np.ndarray,
        threshold: float,
        duration_sec: float,
    ) -> list[Segment]:
        """Détection à deux niveaux :
        - énergie >= seuil : toujours actif (véhicule audible)
        - énergie >= seuil-BOOST ET spectral >= MIN_SCORE : actif si signature véhicule
        """
        window_sec = self.config.audio_window_sec
        padding = self.config.audio_segment_padding
        boost_thresh = threshold - _SPECTRAL_BOOST_DB

        active = (energy_db >= threshold) | (
            (energy_db >= boost_thresh) & (spectral >= _MIN_VEHICLE_SCORE)
        )

        segments: list[Segment] = []
        active_start: "float | None" = None

        for i, is_active in enumerate(active):
            t = i * window_sec
            if is_active and active_start is None:
                active_start = max(0.0, t - padding)
            elif not is_active and active_start is not None:
                segments.append(Segment(
                    start_sec=active_start,
                    end_sec=min(duration_sec, t + padding),
                ))
                active_start = None

        if active_start is not None:
            segments.append(Segment(start_sec=active_start, end_sec=duration_sec))

        return _merge_segments(segments, gap=3.0)

    def _get_threshold(self, is_night: bool = False) -> "float | None":
        override = self.db.get_state("audio_threshold_override")
        floor = self.config.audio_night_min_energy_db if is_night else self.config.audio_min_energy_db
        if override:
            try:
                return max(float(override), floor)
            except (ValueError, TypeError):
                pass
        return self._get_threshold_calc(is_night=is_night)

    def _get_threshold_calc(
        self,
        using_calib: "bool | None" = None,
        cs: "int | None" = None,
        ce: "int | None" = None,
        is_night: bool = False,
    ) -> "float | None":
        needed = self.config.audio_calibration_files
        floor = self.config.audio_night_min_energy_db if is_night else self.config.audio_min_energy_db

        if cs is None:
            cs = self.config.audio_calibration_hour_start
        if ce is None:
            ce = self.config.audio_calibration_hour_end
        if using_calib is None:
            n_calib = self.db.get_audio_stats_count(night_only=True, night_start=cs, night_end=ce)
            using_calib = n_calib >= needed

        if using_calib:
            agg = self.db.get_audio_calibration_aggregate(
                night_only=True, night_start=cs, night_end=ce
            )
            if agg and agg.get("avg_p10_db") is not None:
                return max(
                    agg["avg_p10_db"] + self.config.audio_sigma_factor * agg["avg_std_db"],
                    floor,
                )

        if self.db.get_audio_stats_count() < needed:
            return None
        agg = self.db.get_audio_calibration_aggregate()
        if not agg or agg.get("avg_p10_db") is None:
            return None
        return max(
            agg["avg_p10_db"] + self.config.audio_sigma_factor * agg["avg_std_db"],
            floor,
        )


def bootstrap_calibration(
    audio: "AudioDetector", video_folder: "Path", shutdown_event=None
) -> None:
    """Bootstrap la calibration audio sur les fichiers existants.

    Interface identique à audio_filter.bootstrap_calibration().
    """
    needed = audio.config.audio_calibration_files
    cs = audio.config.audio_calibration_hour_start
    ce = audio.config.audio_calibration_hour_end

    logger.info("Bootstrap audio spectral : priorité fenêtre %dh-%dh", cs, ce)

    if audio.db.get_audio_stats_count(night_only=True, night_start=cs, night_end=ce) >= needed:
        return
    if audio.db.get_audio_stats_count() >= needed:
        return

    candidates = audio.db.get_files_missing_audio_stats(limit=max(needed * 20, 400))
    if not candidates:
        logger.info("Bootstrap audio : aucun fichier éligible trouvé.")
        return

    calib_files = [f for f in candidates if _is_night_hour(_hour_from_filename(f), cs, ce)]
    other_files  = [f for f in candidates if not _is_night_hour(_hour_from_filename(f), cs, ce)]

    for filename in calib_files + other_files:
        if shutdown_event and shutdown_event.is_set():
            break
        if audio.db.get_audio_stats_count(night_only=True, night_start=cs, night_end=ce) >= needed:
            break
        if audio.db.get_audio_stats_count() >= needed:
            break

        matches = list(video_folder.rglob(filename))
        if not matches:
            continue
        try:
            audio.analyze_video(matches[0])
        except Exception as e:
            logger.debug("Bootstrap audio — erreur sur %s : %s", filename, e)

    n_calib = audio.db.get_audio_stats_count(night_only=True, night_start=cs, night_end=ce)
    n_all   = audio.db.get_audio_stats_count()
    if n_calib >= needed:
        logger.info("Bootstrap terminé — %d fichiers fenêtre %dh-%dh.", n_calib, cs, ce)
    elif n_all >= needed:
        logger.info("Bootstrap terminé — %d fichiers (fenêtre calibration insuffisante).", n_all)
    else:
        logger.warning("Bootstrap terminé — seulement %d/%d fichiers avec audio.", n_all, needed)
