"""Tests for src/ingestion.py — FileWatcher.extract_datetime and _is_file_stable."""
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion import FileWatcher


def _make_watcher(fmt: str, db=None) -> FileWatcher:
    config = MagicMock()
    config.filename_datetime_format = fmt
    config.file_stable_delay = 120
    if db is None:
        db = MagicMock()
    return FileWatcher(config, db)


# ------------------------------------------------------------------ #
# extract_datetime                                                     #
# ------------------------------------------------------------------ #

def test_extract_datetime_full_match():
    watcher = _make_watcher("Doorbell_%Y%m%d_%H%M%S.mp4")
    result = watcher.extract_datetime("Doorbell_20260518_142030.mp4")
    assert result == datetime(2026, 5, 18, 14, 20, 30)


def test_extract_datetime_prefix_match():
    watcher = _make_watcher("Interphone-%Y%m%d-%H%M%S.mp4")
    result = watcher.extract_datetime("Interphone-20260518-142030-1777291330846-1.mp4")
    assert result == datetime(2026, 5, 18, 14, 20, 30)


def test_extract_datetime_regex_fallback():
    watcher = _make_watcher("NOMATCH")
    result = watcher.extract_datetime("cam_20260518_142030_extra.mp4")
    assert result == datetime(2026, 5, 18, 14, 20, 30)


def test_extract_datetime_no_match():
    watcher = _make_watcher("NOMATCH")
    result = watcher.extract_datetime("random_file.mp4")
    assert result is None


# ------------------------------------------------------------------ #
# _is_file_stable                                                     #
# ------------------------------------------------------------------ #

def test_is_file_stable_first_seen(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video data")
    watcher = _make_watcher("Doorbell_%Y%m%d_%H%M%S.mp4")
    now = time.time()
    result = watcher._is_file_stable(video, now, stable_delay=60)
    assert result is False


def test_is_file_stable_not_yet(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video data")
    watcher = _make_watcher("Doorbell_%Y%m%d_%H%M%S.mp4")
    now = time.time()
    # First call — populates cache
    watcher._is_file_stable(video, now, stable_delay=60)
    # Second call 1 second later — not yet stable
    result = watcher._is_file_stable(video, now + 1, stable_delay=60)
    assert result is False


def test_is_file_stable_after_delay(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video data")
    watcher = _make_watcher("Doorbell_%Y%m%d_%H%M%S.mp4")
    now = time.time()
    current_size = video.stat().st_size
    # Inject cache entry with stable_since = 61 seconds ago
    watcher._size_cache[video.name] = (current_size, now - 61)
    result = watcher._is_file_stable(video, now, stable_delay=60)
    assert result is True


def test_is_file_stable_size_changed(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"initial")
    watcher = _make_watcher("Doorbell_%Y%m%d_%H%M%S.mp4")
    now = time.time()
    # Inject cache entry with old (wrong) size and old stable_since
    watcher._size_cache[video.name] = (9999, now - 120)
    # File has different size → cache should be updated and return False
    result = watcher._is_file_stable(video, now, stable_delay=60)
    assert result is False
    # Cache should now reflect the actual size
    assert watcher._size_cache[video.name][0] == video.stat().st_size
