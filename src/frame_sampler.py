"""Sauvegarde de frames de debug (avant/après traitement) pour chaque fichier traité."""
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MAX_SEGMENTS = 3   # paires de frames max par fichier vidéo
_JPEG_QUALITY = 85


class FrameSampler:
    """Capture et sauvegarde une paire raw/proc par segment actif (max _MAX_SEGMENTS).

    Utilisé depuis detector._process() via un callback on_proc_frame injecté dans
    _analyze_frame(). Le appelant récupère les chemins via .saved_frames après traitement.
    """

    def __init__(self, output_dir: Path, source_filename: str, max_segments: int = _MAX_SEGMENTS):
        self.output_dir = output_dir
        self.source_filename = source_filename
        self.max_segments = max_segments
        self._sampled_segs: set[int] = set()
        self._saved: list[dict] = []

        stem = Path(source_filename).stem
        self._file_dir = output_dir / stem
        self._file_dir.mkdir(parents=True, exist_ok=True)

    def should_sample(self, seg_idx: int) -> bool:
        return len(self._sampled_segs) < self.max_segments and seg_idx not in self._sampled_segs

    def save(self, seg_idx: int, raw_frame: np.ndarray, proc_frame: np.ndarray) -> None:
        raw_path  = self._file_dir / f"seg{seg_idx:02d}_raw.jpg"
        proc_path = self._file_dir / f"seg{seg_idx:02d}_proc.jpg"

        cv2.imwrite(str(raw_path),  raw_frame,  [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        cv2.imwrite(str(proc_path), proc_frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])

        self._sampled_segs.add(seg_idx)
        self._saved.append({"seg_idx": seg_idx, "raw_path": str(raw_path), "proc_path": str(proc_path)})

        logger.info(
            "DEBUG FRAME [%s] seg%d → %s  (consulter via /api/files/%s/frames)",
            self.source_filename, seg_idx, self._file_dir, self.source_filename,
        )

    @property
    def saved_frames(self) -> list[dict]:
        return list(self._saved)


def purge_old_frames(output_dir: Path, max_age_hours: int = 48) -> int:
    """Supprime les sous-dossiers de frames de plus de max_age_hours heures.

    Chaque sous-dossier correspond à un fichier vidéo (nommé d'après son stem).
    On se base sur le mtime du dossier (mis à jour à la création des frames).
    """
    if not output_dir.exists():
        return 0
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    removed = 0
    for d in output_dir.iterdir():
        if not d.is_dir():
            continue
        mtime = datetime.utcfromtimestamp(d.stat().st_mtime)
        if mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            logger.debug("Purge debug frames : %s supprimé (age > %dh)", d.name, max_age_hours)
    if removed:
        logger.info("Purge debug frames : %d dossier(s) supprimé(s) (> %dh).", removed, max_age_hours)
    return removed
