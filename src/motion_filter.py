"""Module Filtre de mouvement — fast motion detection to skip empty footage."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start_sec: float
    end_sec: float


class MotionFilter:
    def __init__(self, config: Config):
        self.config = config
        self._roi_mask: np.ndarray | None = None
        self._roi_shape: tuple | None = None

    def _build_roi_mask(self, frame_shape: tuple) -> np.ndarray:
        """Builds a binary mask for the region of interest polygon."""
        h, w = frame_shape[:2]
        if self._roi_mask is not None and self._roi_shape == (h, w):
            return self._roi_mask

        mask = np.zeros((h, w), dtype=np.uint8)
        polygon = self.config.roi_polygon
        if polygon:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            mask[:] = 255  # no ROI configured → full frame

        self._roi_mask = mask
        self._roi_shape = (h, w)
        return mask

    def analyze_video(
        self,
        video_path: Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Segment]:
        """
        Samples frames from the video and returns a list of active segments
        (time ranges where motion is detected in the ROI).
        on_progress(frames_done, frames_total) called periodically.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

        try:
            return self._process(cap, video_path, on_progress)
        finally:
            cap.release()

    def _process(
        self,
        cap: cv2.VideoCapture,
        video_path: Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Segment]:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        sample_fps = self.config.effective_motion_fps
        step = max(1, int(fps / sample_fps))  # read every Nth frame
        sampled_total = max(1, total_frames // step)

        threshold = self.config.motion_threshold
        min_area = self.config.min_motion_area
        padding = self.config.segment_padding

        segments: list[Segment] = []
        active_start: float | None = None
        prev_gray: np.ndarray | None = None
        frame_idx = 0
        sampled_done = 0

        logger.debug(
            "Filtre mouvement : %s | %.1f s | step=%d", video_path.name, duration_sec, step
        )

        if on_progress:
            on_progress(0, sampled_total)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                continue

            current_sec = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is None:
                prev_gray = gray
                frame_idx += 1
                continue

            roi_mask = self._build_roi_mask(frame.shape)

            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_and(thresh, roi_mask)

            # Morphological cleanup to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            motion_area = cv2.countNonZero(thresh)
            has_motion = motion_area >= min_area

            if has_motion and active_start is None:
                active_start = max(0.0, current_sec - padding)
            elif not has_motion and active_start is not None:
                end = min(duration_sec, current_sec + padding)
                segments.append(Segment(start_sec=active_start, end_sec=end))
                active_start = None

            prev_gray = gray
            frame_idx += 1
            sampled_done += 1
            if on_progress:
                on_progress(sampled_done, sampled_total)

        # Close any open segment at end of video
        if active_start is not None:
            segments.append(Segment(start_sec=active_start, end_sec=duration_sec))

        # Merge overlapping or adjacent segments
        segments = _merge_segments(segments)

        total_active = sum(s.end_sec - s.start_sec for s in segments)
        logger.info(
            "%s → %d segment(s) actif(s) / %.1f s (%.0f%% de la vidéo)",
            video_path.name,
            len(segments),
            total_active,
            100 * total_active / duration_sec if duration_sec else 0,
        )
        return segments


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
