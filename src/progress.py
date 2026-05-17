"""Singleton thread-safe de progression du traitement vidéo."""
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgressState:
    # Fichier en cours
    current_file: str = ""
    # Phase : "idle", "motion", "detection", "done"
    phase: str = "idle"
    # Frames
    frames_done: int = 0
    frames_total: int = 0
    # File queue
    queue_done: int = 0
    queue_total: int = 0
    # Reset en cours
    resetting: bool = False
    reset_reason: str = ""


class ProgressTracker:
    """Singleton mis à jour par le pipeline, lu par le dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = ProgressState()

    def start_file(self, filename: str, queue_done: int, queue_total: int):
        with self._lock:
            self._state.current_file = filename
            self._state.phase = "motion"
            self._state.frames_done = 0
            self._state.frames_total = 0
            self._state.queue_done = queue_done
            self._state.queue_total = queue_total

    def set_phase(self, phase: str, frames_total: int = 0):
        with self._lock:
            self._state.phase = phase
            self._state.frames_total = frames_total
            self._state.frames_done = 0

    def update_frame(self, frames_done: int):
        with self._lock:
            self._state.frames_done = frames_done

    def finish_file(self):
        with self._lock:
            self._state.phase = "idle"
            self._state.frames_done = self._state.frames_total
            self._state.queue_done += 1

    def set_idle(self):
        with self._lock:
            self._state.phase = "idle"
            self._state.current_file = ""
            self._state.frames_done = 0
            self._state.frames_total = 0

    def set_resetting(self, reason: str = ""):
        with self._lock:
            self._state.resetting = True
            self._state.reset_reason = reason
            self._state.phase = "idle"
            self._state.current_file = ""

    def clear_resetting(self):
        with self._lock:
            self._state.resetting = False
            self._state.reset_reason = ""

    def snapshot(self) -> dict:
        with self._lock:
            s = self._state
            file_pct = 0
            if s.frames_total > 0:
                file_pct = round(100 * s.frames_done / s.frames_total)
            queue_pct = 0
            if s.queue_total > 0:
                queue_pct = round(100 * s.queue_done / s.queue_total)
            return {
                "current_file": s.current_file,
                "phase": s.phase,
                "frames_done": s.frames_done,
                "frames_total": s.frames_total,
                "file_percent": file_pct,
                "queue_done": s.queue_done,
                "queue_total": s.queue_total,
                "queue_percent": queue_pct,
                "resetting": s.resetting,
                "reset_reason": s.reset_reason,
            }


# Module-level singleton
tracker = ProgressTracker()
