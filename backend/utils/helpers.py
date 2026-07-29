"""Small, dependency-light helper utilities shared across the codebase."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def new_id(prefix: str = "") -> str:
    """Generate a short, collision-resistant unique identifier."""

    raw = uuid.uuid4().hex
    return f"{prefix}{raw}" if prefix else raw


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents."""

    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_of_text(text: str) -> str:
    """Compute the SHA-256 hex digest of a UTF-8 encoded string."""

    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters from extracted text."""

    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_filename(filename: str) -> str:
    """Sanitize an arbitrary filename for safe use on disk."""

    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return filename[:200] if filename else new_id("file_")


def truncate(text: str, max_chars: int = 240) -> str:
    """Truncate text to ``max_chars`` characters, appending an ellipsis."""

    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "\u2026"


@dataclass
class Timer:
    """Tiny context-manager style timer used for logging step durations."""

    label: str
    _start: Optional[float] = field(default=None, init=False, repr=False)
    elapsed_seconds: float = field(default=0.0, init=False)

    def __enter__(self) -> "Timer":
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        import time

        if self._start is not None:
            self.elapsed_seconds = time.perf_counter() - self._start


def detect_source_type_from_filename(filename: str) -> str:
    """Infer the coarse-grained source type from a file extension.

    Returns one of: ``pdf``, ``image``, ``audio``, ``video``, or ``unknown``.
    """

    ext = Path(filename).suffix.lower().lstrip(".")
    pdf_ext = {"pdf"}
    image_ext = {"png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif"}
    audio_ext = {"mp3", "wav", "m4a", "flac", "ogg", "aac"}
    video_ext = {"mp4", "mov", "avi", "mkv", "webm"}

    if ext in pdf_ext:
        return "pdf"
    if ext in image_ext:
        return "image"
    if ext in audio_ext:
        return "audio"
    if ext in video_ext:
        return "video"
    return "unknown"
