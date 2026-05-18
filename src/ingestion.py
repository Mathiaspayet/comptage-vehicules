"""Module Ingestion — watches the video folder for new, complete files."""
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from .config import Config
from .database import Database

logger = logging.getLogger(__name__)

# Video file extensions accepted
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}

# Regex fallback: extracts the first YYYYMMDD and HHMMSS from the filename
# Handles formats like: Interphone-20260427-140210-1777291330846-1.mp4
_RE_DATE_TIME = re.compile(r"(\d{8})[^0-9](\d{6})")


class FileWatcher:
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        # name → (size, timestamp when size was first seen at this value)
        self._size_cache: dict[str, tuple[int, float]] = {}

    def scan_new_files(self) -> list[Path]:
        folder = self.config.video_folder
        if not folder.exists():
            logger.warning("Dossier vidéo inaccessible : %s", folder)
            return []

        video_files = _collect_video_files(folder)
        if not video_files:
            return []

        now = time.time()
        stable_delay = self.config.file_stable_delay

        # Only process files whose size has been stable for file_stable_delay seconds
        ready_files = [f for f in video_files if self._is_file_stable(f, now, stable_delay)]

        pending = [f for f in ready_files if not self.db.is_file_processed(f.name)]

        if not pending:
            return []

        max_n = self.config.max_recent_files
        if max_n > 0 and len(pending) > max_n:
            to_skip = pending[:-max_n]
            for path in to_skip:
                logger.info("Ignoré (backlog trop ancien) : %s", path.name)
                self.db.mark_file_done(path.name, vehicle_count=0)
            pending = pending[-max_n:]

        if pending:
            logger.info("%d nouveau(x) fichier(s) à traiter", len(pending))

        return pending

    def _is_file_stable(self, path: Path, now: float, stable_delay: int) -> bool:
        """Returns True if file size hasn't changed for stable_delay seconds."""
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            self._size_cache.pop(path.name, None)
            return False

        cached = self._size_cache.get(path.name)
        if cached is None or cached[0] != current_size:
            self._size_cache[path.name] = (current_size, now)
            return False

        _, stable_since = cached
        return (now - stable_since) >= stable_delay

    def extract_datetime(self, filename: str) -> datetime | None:
        """
        Extracts the video start datetime from the filename.

        Strategy:
        1. Try full strptime match with the configured format.
        2. Try prefix match: parse only the beginning of the filename up to
           the length of the format string (handles trailing IDs like
           Interphone-20260427-140210-1777291330846-1.mp4).
        3. Regex fallback: look for the first YYYYMMDD + HHMMSS pair anywhere
           in the filename — works for most camera naming conventions.
        """
        basename = Path(filename).name
        fmt = self.config.filename_datetime_format

        # 1. Full match
        try:
            return datetime.strptime(basename, fmt)
        except ValueError:
            pass

        # 2. Prefix match — strip the trailing part after the last format token
        # Build a "sample" string from the format to measure its expected length
        try:
            sample = datetime(2026, 1, 2, 14, 5, 6).strftime(fmt)
            prefix = basename[: len(sample)]
            return datetime.strptime(prefix, fmt)
        except (ValueError, IndexError):
            pass

        # 3. Regex fallback: find YYYYMMDD-HHMMSS anywhere in the filename
        m = _RE_DATE_TIME.search(basename)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                pass

        logger.warning(
            "Impossible d'extraire la date de '%s' avec le format '%s'",
            filename,
            fmt,
        )
        return None


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def _collect_video_files(folder: Path) -> list[Path]:
    """
    Collects all video files in `folder` and its immediate subfolders,
    sorted by modification time (oldest first).

    The Reolink/Interphone camera stores files in date-named subfolders:
        surveillance/Interphone/20260427PM/Interphone-20260427-140210-…mp4
    """
    files: list[Path] = []
    try:
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(entry)
            elif entry.is_dir():
                # One level deep (date subfolders)
                try:
                    for sub in entry.iterdir():
                        if sub.is_file() and sub.suffix.lower() in VIDEO_EXTENSIONS:
                            files.append(sub)
                except PermissionError:
                    logger.warning("Sous-dossier inaccessible : %s", entry)
    except PermissionError as e:
        logger.error("Impossible de lire le dossier vidéo : %s", e)

    return sorted(files, key=lambda f: f.stat().st_mtime)
