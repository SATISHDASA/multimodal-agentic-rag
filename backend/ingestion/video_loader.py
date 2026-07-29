"""Video ingestion: audio-track transcription (Whisper) plus periodic
keyframe OCR (OpenCV + Tesseract) for on-screen text such as slides,
captions, or diagrams.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from backend.ingestion.audio_loader import AudioLoadError, transcribe_audio
from backend.utils.helpers import clean_text
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class VideoLoadError(RuntimeError):
    """Raised when a video cannot be processed."""


@dataclass
class VideoSegment:
    """A piece of content extracted from a video, tagged with its origin."""

    timestamp: float
    text: str
    kind: str  # "speech" or "on_screen_text"


def _extract_audio_track(video_path: Path, output_wav: Path) -> bool:
    """Extract the audio track from a video file into a WAV file via ffmpeg.

    Returns False (instead of raising) if the video has no audio track,
    since that is not a fatal error for on-screen-text-only videos.
    """

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_wav),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False
        )
    except FileNotFoundError as exc:
        raise VideoLoadError(
            "ffmpeg is not installed or not on PATH. Please install ffmpeg."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoLoadError("Audio extraction timed out.") from exc

    if proc.returncode != 0 or not output_wav.exists() or output_wav.stat().st_size == 0:
        logger.warning(
            "ffmpeg produced no audio output (video may be silent): %s", proc.stderr
        )
        return False
    return True


def _extract_keyframes_and_ocr(
    video_path: Path, frame_interval_seconds: float = 5.0
) -> List[VideoSegment]:
    """Sample frames at a fixed interval and OCR each for on-screen text."""

    try:
        import cv2
        import numpy as np
        import pytesseract

        from backend.utils.config import settings

        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    except ImportError as exc:  # pragma: no cover - defensive
        raise VideoLoadError(
            "Missing video dependencies. Run `pip install opencv-python-headless "
            "pytesseract numpy`."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoLoadError(f"Could not open video file: {video_path.name}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(int(fps * frame_interval_seconds), 1)

    segments: List[VideoSegment] = []
    frame_idx = 0
    seen_text_hashes: set[str] = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = clean_text(pytesseract.image_to_string(gray))
                if text and len(text) > 3:
                    text_hash = text[:100]
                    if text_hash not in seen_text_hashes:
                        seen_text_hashes.add(text_hash)
                        timestamp = frame_idx / fps
                        segments.append(
                            VideoSegment(
                                timestamp=timestamp, text=text, kind="on_screen_text"
                            )
                        )
            frame_idx += 1
    finally:
        cap.release()

    return segments


def load_video(video_path: Path) -> List[VideoSegment]:
    """Extract both spoken (Whisper) and on-screen (OCR) text from a video.

    Args:
        video_path: Path to the video file on disk.

    Returns:
        A combined, timestamp-sorted list of :class:`VideoSegment`.

    Raises:
        VideoLoadError: If the video cannot be opened/processed at all and
            no content (neither speech nor on-screen text) could be extracted.
    """

    if not video_path.exists():
        raise VideoLoadError(f"Video file not found: {video_path}")

    segments: List[VideoSegment] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "audio.wav"
        has_audio = False
        try:
            has_audio = _extract_audio_track(video_path, wav_path)
        except VideoLoadError as exc:
            logger.warning("Audio extraction step failed: %s", exc)

        if has_audio:
            try:
                transcript_segments = transcribe_audio(wav_path)
                for seg in transcript_segments:
                    segments.append(
                        VideoSegment(timestamp=seg.start, text=seg.text, kind="speech")
                    )
            except AudioLoadError as exc:
                logger.warning("Video audio transcription failed: %s", exc)

    try:
        ocr_segments = _extract_keyframes_and_ocr(video_path)
        segments.extend(ocr_segments)
    except VideoLoadError as exc:
        logger.warning("Video keyframe OCR failed: %s", exc)

    if not segments:
        raise VideoLoadError(
            f"No speech or on-screen text could be extracted from "
            f"'{video_path.name}'."
        )

    segments.sort(key=lambda s: s.timestamp)
    return segments
