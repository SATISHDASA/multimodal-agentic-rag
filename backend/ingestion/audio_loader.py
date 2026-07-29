"""Audio ingestion: speech-to-text transcription using open-source Whisper."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from backend.utils.config import settings
from backend.utils.helpers import clean_text
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AudioLoadError(RuntimeError):
    """Raised when audio transcription fails."""


@dataclass
class TranscriptSegment:
    """A single timestamped transcript segment."""

    start: float
    end: float
    text: str


@lru_cache(maxsize=1)
def _get_whisper_model():
    """Lazily load and cache the local Whisper model."""

    try:
        import whisper
    except ImportError as exc:  # pragma: no cover - defensive
        raise AudioLoadError(
            "openai-whisper is not installed. Run `pip install openai-whisper`."
        ) from exc

    logger.info("Loading Whisper model: %s", settings.whisper_model_size)
    try:
        model = whisper.load_model(settings.whisper_model_size)
    except Exception as exc:  # noqa: BLE001
        raise AudioLoadError(f"Could not load Whisper model: {exc}") from exc
    return model


def transcribe_audio(file_path: Path) -> List[TranscriptSegment]:
    """Transcribe an audio file into timestamped segments.

    Args:
        file_path: Path to the audio file on disk.

    Returns:
        A list of :class:`TranscriptSegment` covering the whole recording.

    Raises:
        AudioLoadError: If the file is missing, unreadable, or transcription
            fails (e.g. due to a corrupt/unsupported audio codec).
    """

    if not file_path.exists():
        raise AudioLoadError(f"Audio file not found: {file_path}")

    model = _get_whisper_model()

    try:
        result = model.transcribe(str(file_path), fp16=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Whisper transcription failed for %s", file_path)
        raise AudioLoadError(f"Transcription failed: {exc}") from exc

    segments_raw = result.get("segments") or []
    if not segments_raw:
        full_text = clean_text(result.get("text", ""))
        if not full_text:
            raise AudioLoadError("Transcription produced no text.")
        return [TranscriptSegment(start=0.0, end=0.0, text=full_text)]

    segments = [
        TranscriptSegment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            text=clean_text(seg.get("text", "")),
        )
        for seg in segments_raw
        if clean_text(seg.get("text", ""))
    ]
    if not segments:
        raise AudioLoadError("Transcription produced no usable segments.")
    return segments
