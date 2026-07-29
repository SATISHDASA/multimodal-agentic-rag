"""Image ingestion: OCR text extraction with OpenCV pre-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.utils.helpers import clean_text
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be read or processed."""


@dataclass
class ImageContent:
    """Extracted OCR content for a single image file."""

    text: str
    width: int
    height: int


def _preprocess_for_ocr(image_bgr):
    """Apply grayscale + adaptive thresholding to improve OCR accuracy."""

    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return thresh


def load_image(file_path: Path) -> ImageContent:
    """Run OCR on an image file and return the extracted text.

    Args:
        file_path: Path to the image on disk.

    Returns:
        An :class:`ImageContent` with the OCR'd text and image dimensions.

    Raises:
        ImageLoadError: If the image cannot be opened or OCR fails entirely.
    """

    try:
        import cv2
        import numpy as np
        import pytesseract
        from PIL import Image, UnidentifiedImageError

        from backend.utils.config import settings

        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImageLoadError(
            "Missing image dependencies. Run `pip install opencv-python-headless "
            "pillow pytesseract`."
        ) from exc

    if not file_path.exists():
        raise ImageLoadError(f"Image file not found: {file_path}")

    try:
        with Image.open(file_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            width, height = pil_img.size
            image_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError as exc:
        raise ImageLoadError(f"Broken or unsupported image file: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"Could not read image '{file_path.name}': {exc}") from exc

    try:
        preprocessed = _preprocess_for_ocr(image_np)
        text = pytesseract.image_to_string(preprocessed)
        if not text.strip():
            # Fall back to OCR on the raw (non-thresholded) image.
            text = pytesseract.image_to_string(image_np)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OCR failed for image %s", file_path)
        raise ImageLoadError(f"OCR failed: {exc}") from exc

    return ImageContent(text=clean_text(text), width=width, height=height)
