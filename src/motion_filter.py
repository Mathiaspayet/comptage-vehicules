"""Module Filtre de mouvement — fast motion detection to skip empty footage."""
import logging
import subprocess
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
        self._ffmpeg_available: bool | None = None  # cached after first probe

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
            mask[:] = 255

        self._roi_mask = mask
        self._roi_shape = (h, w)
        return mask

    def analyze_video(
        self,
        video_path: Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Segment]:
        """
        Samples frames from the video and returns a list of active segments.
        on_progress(frames_done, frames_total) called periodically.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Prefer ffmpeg: -skip_frame nointra decodes only I-frames (~20x faster)
        if self._ffmpeg_available is not False:
            try:
                result = self._process_ffmpeg(
                    video_path, fps, width, height, total_frames, on_progress
                )
                self._ffmpeg_available = True
                return result
            except FileNotFoundError:
                self._ffmpeg_available = False
                logger.debug("ffmpeg absent — fallback OpenCV grab()")
            except Exception as e:
                logger.warning(
                    "ffmpeg échoué sur %s (%s) — fallback OpenCV grab()",
                    video_path.name, e,
                )

        cap = cv2.VideoCapture(str(video_path))
        try:
            return self._process_opencv(cap, video_path, fps, total_frames, on_progress)
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # ffmpeg pipe — I-frames only, grayscale, scaled (primary path)
    # ------------------------------------------------------------------

    def _process_ffmpeg(
        self,
        video_path: Path,
        fps: float,
        width: int,
        height: int,
        total_frames: int,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Segment]:
        """Decode only I-frames via ffmpeg pipe at reduced resolution.

        -skip_frame nointra skips all non-I-frames at the decoder level, making
        it ~20x faster than full decode on typical 30fps H.264 security footage.
        Output is grayscale 320px wide to minimise pipe bandwidth.
        """
        duration_sec = total_frames / fps
        sample_fps = self.config.effective_motion_fps

        # Output at 320px wide, height auto-scaled (must be divisible by 2)
        out_w = 320
        out_h = max(2, round(height * out_w / width / 2) * 2)

        # I-frames arrive at ~0.5-1 fps for typical security cameras; cap estimate
        sampled_total = max(1, int(duration_sec * min(sample_fps, 1.0)))

        threshold = self.config.motion_threshold
        # Scale min_motion_area to the reduced output resolution
        min_area = max(1, int(
            self.config.min_motion_area * (out_w / width) * (out_h / height)
        ))
        padding = self.config.segment_padding

        # Build ROI mask at original resolution, then resize to output scale
        full_mask = self._build_roi_mask((height, width))
        roi_mask = cv2.resize(full_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        cmd = [
            "ffmpeg",
            "-skip_frame", "nointra",   # only decode I-frames
            "-i", str(video_path),
            "-vf", f"fps={sample_fps},scale={out_w}:{out_h}",
            "-pix_fmt", "gray",
            "-f", "rawvideo",
            "-loglevel", "error",
            "pipe:1",
        ]

        frame_size = out_w * out_h  # grayscale: 1 byte/pixel

        logger.debug(
            "Filtre mouvement (ffmpeg I-frames) : %s | %.1f s | out=%dx%d",
            video_path.name, duration_sec, out_w, out_h,
        )

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        segments: list[Segment] = []
        active_start: float | None = None
        prev_gray: np.ndarray | None = None
        sampled_done = 0

        if on_progress:
            on_progress(0, sampled_total)

        try:
            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break

                current_sec = sampled_done / sample_fps
                gray = np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                    thresh = cv2.bitwise_and(thresh, roi_mask)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                    has_motion = cv2.countNonZero(thresh) >= min_area

                    if has_motion and active_start is None:
                        active_start = max(0.0, current_sec - padding)
                    elif not has_motion and active_start is not None:
                        end = min(duration_sec, current_sec + padding)
                        segments.append(Segment(start_sec=active_start, end_sec=end))
                        active_start = None

                prev_gray = gray
                sampled_done += 1
                if on_progress:
                    on_progress(sampled_done, sampled_total)
        finally:
            proc.stdout.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        if proc.returncode != 0 and sampled_done == 0:
            raise RuntimeError(f"ffmpeg code={proc.returncode} — aucun frame décodé")

        return self._finalize_segments(segments, active_start, duration_sec, video_path)

    # ------------------------------------------------------------------
    # OpenCV fallback — grab() skips BGR decode for non-sampled frames
    # ------------------------------------------------------------------

    def _process_opencv(
        self,
        cap: cv2.VideoCapture,
        video_path: Path,
        fps: float,
        total_frames: int,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Segment]:
        duration_sec = total_frames / fps
        sample_fps = self.config.effective_motion_fps
        step = max(1, int(fps / sample_fps))
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
            "Filtre mouvement (OpenCV) : %s | %.1f s | step=%d",
            video_path.name, duration_sec, step,
        )

        if on_progress:
            on_progress(0, sampled_total)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                roi_mask = self._build_roi_mask(frame.shape)

                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                thresh = cv2.bitwise_and(thresh, roi_mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                has_motion = cv2.countNonZero(thresh) >= min_area

                if has_motion and active_start is None:
                    active_start = max(0.0, current_sec - padding)
                elif not has_motion and active_start is not None:
                    end = min(duration_sec, current_sec + padding)
                    segments.append(Segment(start_sec=active_start, end_sec=end))
                    active_start = None

            prev_gray = gray
            frame_idx += step
            sampled_done += 1
            if on_progress:
                on_progress(sampled_done, sampled_total)

            for _ in range(step - 1):
                cap.grab()

        return self._finalize_segments(segments, active_start, duration_sec, video_path)

    # ------------------------------------------------------------------

    def _finalize_segments(
        self,
        segments: list[Segment],
        active_start: float | None,
        duration_sec: float,
        video_path: Path,
    ) -> list[Segment]:
        if active_start is not None:
            segments.append(Segment(start_sec=active_start, end_sec=duration_sec))

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
