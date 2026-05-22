"""Utilitaires partagés de segmentation temporelle.

Segment et _merge_segments sont utilisés par audio_filter, night_detector,
detector et main pour représenter des plages temporelles actives.
"""
from dataclasses import dataclass


@dataclass
class Segment:
    start_sec: float
    end_sec: float


def _merge_segments(segments: list[Segment], gap: float = 2.0) -> list[Segment]:
    """Merges segments that are closer than `gap` seconds."""
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.start_sec <= merged[-1].end_sec + gap:
            merged[-1] = Segment(merged[-1].start_sec, max(merged[-1].end_sec, seg.end_sec))
        else:
            merged.append(seg)
    return merged
