"""Filtre audio — détecte les segments avec activité sonore (véhicules).

Principe :
  1. Extraction PCM 16 kHz mono via ffmpeg (rapide, pas de décodage vidéo)
  2. Énergie RMS par fenêtre de 0.5 s → profil dB
  3. Auto-calibration sur les N derniers fichiers de NUIT (22h-06h) :
       seuil = percentile10_moyen + sigma × std_moyen
     Les fichiers de nuit ont moins de véhicules → meilleur baseline de silence.
     Fallback sur tous les fichiers si pas assez de fichiers de nuit.
  4. L'audio est maintenant le détecteur principal de segments (remplace le mouvement).
"""

import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Config
from .motion_filter import Segment, _merge_segments

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000   # Hz — suffisant pour bruit moteur (< 8 kHz)
_RE_DATETIME = re.compile(r"(\d{4})(\d{2})(\d{2})[^\d](\d{2})(\d{2})(\d{2})")


def _hour_from_filename(filename: str) -> "int | None":
    """Extrait l'heure (0-23) du nom de fichier, ou None si non parseable."""
    m = _RE_DATETIME.search(filename)
    if m:
        try:
            return int(m.group(4))
        except (ValueError, IndexError):
            pass
    return None


def _is_night_hour(hour: "int | None", night_start: int, night_end: int) -> bool:
    """Retourne True si l'heure correspond à la plage nuit (ex: 22h-06h)."""
    if hour is None:
        return False
    if night_start > night_end:   # traversée de minuit : 22h → 06h
        return hour >= night_start or hour < night_end
    return night_start <= hour < night_end


class AudioFilter:
    def __init__(self, config: Config, db):
        self.config = config
        self.db = db

    # ------------------------------------------------------------------ #
    # API publique                                                         #
    # ------------------------------------------------------------------ #

    def analyze_video(self, video_path: Path) -> list[Segment]:
        """Analyse la piste audio et retourne les segments actifs.

        Stocke toujours les stats (y compris pour les fichiers sans audio)
        pour que le bootstrap ne réessaie pas les mêmes fichiers.
        Retourne [] si désactivé, pas d'audio, ou pas encore calibré.
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
        self.db.add_audio_stats(
            video_path.name, **stats, video_hour=video_hour,
        )

        logger.debug(
            "%s — audio : médiane=%.1f dB  p10=%.1f dB  p90=%.1f dB  std=%.1f dB",
            video_path.name,
            stats["median_db"], stats["p10_db"], stats["p90_db"], stats["std_db"],
        )

        threshold = self._get_threshold()
        if threshold is None:
            n = self.db.get_audio_stats_count()
            logger.info(
                "%s — calibration audio en cours : %d/%d fichiers analysés.",
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
        night_only = self.config.audio_night_calibration
        ns = self.config.audio_night_start_hour
        ne = self.config.audio_night_end_hour
        min_files = self.config.audio_calibration_files

        n_night = self.db.get_audio_stats_count(night_only=True, night_start=ns, night_end=ne)
        n_all   = self.db.get_audio_stats_count()
        using_night = night_only and n_night >= min_files

        agg = self.db.get_audio_calibration_aggregate(
            night_only=using_night, night_start=ns, night_end=ne
        )
        threshold = self._get_threshold()
        return {
            "enabled":          self.config.audio_enabled,
            "calibrated":       threshold is not None,
            "files_analyzed":   n_night if using_night else n_all,
            "files_needed":     min_files,
            "night_calibration": night_only,
            "night_files":      n_night,
            "using_night":      using_night,
            "background_db":    round(agg["avg_p10_db"], 1) if agg else None,
            "noise_std_db":     round(agg["avg_std_db"], 1) if agg else None,
            "threshold_db":     round(threshold, 1) if threshold is not None else None,
            "sigma_factor":     self.config.audio_sigma_factor,
        }

    # ------------------------------------------------------------------ #
    # Implémentation interne                                               #
    # ------------------------------------------------------------------ #

    def _extract_audio(self, video_path: Path) -> "np.ndarray | None":
        """Extrait l'audio mono 16 kHz via ffmpeg pipe (float32 normalisé)."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(_SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
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
        """Calcule le seuil de détection.

        Préfère les fichiers de nuit pour la calibration (meilleur baseline).
        Fallback sur tous les fichiers si insuffisamment de fichiers de nuit.
        """
        needed = self.config.audio_calibration_files
        night_only = self.config.audio_night_calibration
        ns = self.config.audio_night_start_hour
        ne = self.config.audio_night_end_hour

        if night_only:
            n_night = self.db.get_audio_stats_count(night_only=True, night_start=ns, night_end=ne)
            if n_night >= needed:
                agg = self.db.get_audio_calibration_aggregate(
                    night_only=True, night_start=ns, night_end=ne
                )
                if agg and agg.get("avg_p10_db") is not None:
                    threshold = (
                        agg["avg_p10_db"]
                        + self.config.audio_sigma_factor * agg["avg_std_db"]
                    )
                    return max(threshold, self.config.audio_min_energy_db)

        # Fallback : tous les fichiers
        if self.db.get_audio_stats_count() < needed:
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


def bootstrap_calibration(audio: "AudioFilter", video_folder: "Path", shutdown_event=None) -> None:
    """Analyse les fichiers déjà traités pour bootstrapper la calibration audio.

    Priorise les fichiers de nuit (22h-06h) pour obtenir un meilleur baseline
    de bruit de fond (moins de véhicules, donc plus représentatif du silence).
    Tourne en thread daemon au démarrage.
    """
    needed = audio.config.audio_calibration_files
    night_only = audio.config.audio_night_calibration
    ns = audio.config.audio_night_start_hour
    ne = audio.config.audio_night_end_hour

    # Déjà calibré ?
    if night_only:
        if audio.db.get_audio_stats_count(night_only=True, night_start=ns, night_end=ne) >= needed:
            return
    else:
        if audio.db.get_audio_stats_count() >= needed:
            return

    candidates = audio.db.get_files_missing_audio_stats(limit=max(needed * 20, 400))
    if not candidates:
        logger.info("Bootstrap audio : aucun fichier éligible trouvé.")
        return

    # Trier : nuit en premier si calibration nuit activée
    if night_only:
        night_files = [f for f in candidates if _is_night_hour(_hour_from_filename(f), ns, ne)]
        day_files   = [f for f in candidates if not _is_night_hour(_hour_from_filename(f), ns, ne)]
        ordered = night_files + day_files
        logger.info(
            "Bootstrap audio : %d candidat(s) — %d de nuit (priorité) + %d de jour.",
            len(ordered), len(night_files), len(day_files),
        )
    else:
        ordered = candidates
        logger.info(
            "Bootstrap audio : %d candidat(s), objectif %d fichiers calibrés.",
            len(ordered), needed,
        )

    analyzed = 0
    for filename in ordered:
        if shutdown_event and shutdown_event.is_set():
            break

        # Arrêt dès que le seuil est atteint (nuit d'abord)
        if night_only:
            if audio.db.get_audio_stats_count(night_only=True, night_start=ns, night_end=ne) >= needed:
                break
        else:
            if audio.db.get_audio_stats_count() >= needed:
                break

        matches = list(video_folder.rglob(filename))
        if not matches:
            logger.debug("Bootstrap audio : %s introuvable sur le disque.", filename)
            continue

        try:
            audio.analyze_video(matches[0])
            analyzed += 1
        except Exception as e:
            logger.debug("Bootstrap audio — erreur sur %s : %s", filename, e)

    n_night = audio.db.get_audio_stats_count(night_only=True, night_start=ns, night_end=ne)
    n_all   = audio.db.get_audio_stats_count()

    if n_night >= needed:
        logger.info(
            "Bootstrap audio terminé — calibration nuit prête (%d fichiers de nuit).", n_night
        )
    elif n_all >= needed:
        logger.info(
            "Bootstrap audio terminé — calibration sur tous fichiers (%d), "
            "seulement %d fichiers de nuit disponibles.",
            n_all, n_night,
        )
    elif n_all > 0:
        logger.warning(
            "Bootstrap audio terminé — %d/%d fichiers avec audio (%d de nuit). "
            "Réduisez audio_filter.calibration_files à %d dans la config.",
            n_all, needed, n_night, n_all,
        )
    else:
        logger.warning(
            "Bootstrap audio : aucun fichier avec piste audio parmi %d candidats analysés.",
            analyzed,
        )


def union_segments(a: list[Segment], b: list[Segment]) -> list[Segment]:
    """Union de deux listes de segments (conservé pour compatibilité)."""
    if not b:
        return a
    if not a:
        return b
    combined = sorted(a + b, key=lambda s: s.start_sec)
    return _merge_segments(combined, gap=2.0)
