"""PDF ingestion using PyMuPDF (fitz), with Tesseract OCR fallback for
scanned/image-only pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from backend.utils.helpers import clean_text
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoadError(RuntimeError):
    """Raised when a PDF cannot be parsed."""


@dataclass
class PageContent:
    """Extracted content for a single PDF page."""

    page_number: int
    text: str
    used_ocr: bool


def _ocr_page_image(page) -> str:
    """Render a PDF page to an image and run Tesseract OCR on it."""

    try:
        import pytesseract
        from PIL import Image
        import io

        from backend.utils.config import settings

        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR fallback failed for a PDF page: %s", exc)
        return ""


def load_pdf(file_path: Path, min_text_chars_for_native: int = 20) -> List[PageContent]:
    """Extract text from every page of a PDF file.

    Pages with little or no extractable native text (e.g. scanned pages)
    are rendered to images and OCR'd with Tesseract instead.

    Args:
        file_path: Path to the PDF file on disk.
        min_text_chars_for_native: Threshold below which OCR fallback triggers.

    Returns:
        A list of :class:`PageContent`, one per page.

    Raises:
        PDFLoadError: If the file cannot be opened or parsed at all.
    """

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - defensive
        raise PDFLoadError(
            "PyMuPDF is not installed. Run `pip install pymupdf`."
        ) from exc

    if not file_path.exists():
        raise PDFLoadError(f"PDF file not found: {file_path}")

    pages: List[PageContent] = []
    try:
        doc = fitz.open(file_path)
    except Exception as exc:  # noqa: BLE001
        raise PDFLoadError(f"Could not open PDF '{file_path.name}': {exc}") from exc

    try:
        for i, page in enumerate(doc):
            try:
                native_text = page.get_text("text") or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("Native text extraction failed on page %s: %s", i, exc)
                native_text = ""

            used_ocr = False
            text = native_text
            if len(native_text.strip()) < min_text_chars_for_native:
                ocr_text = _ocr_page_image(page)
                if len(ocr_text.strip()) > len(native_text.strip()):
                    text = ocr_text
                    used_ocr = True

            pages.append(
                PageContent(
                    page_number=i + 1, text=clean_text(text), used_ocr=used_ocr
                )
            )
    finally:
        doc.close()

    if not pages:
        raise PDFLoadError(f"No pages could be extracted from '{file_path.name}'.")

    return pages
