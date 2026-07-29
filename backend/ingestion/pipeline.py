"""High-level ingestion pipeline: extract -> chunk -> embed -> store.

This module orchestrates the individual loaders (PDF/image/audio/video/web)
and feeds their output through chunking, embedding, and dual storage
(SQLite for raw text/BM25, Qdrant for vectors).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from backend.database import sqlite as db
from backend.ingestion.audio_loader import AudioLoadError, transcribe_audio
from backend.ingestion.chunking import chunk_text
from backend.ingestion.image_loader import ImageLoadError, load_image
from backend.ingestion.pdf_loader import PDFLoadError, load_pdf
from backend.ingestion.video_loader import VideoLoadError, load_video
from backend.ingestion.web_loader import WebLoadError, load_website
from backend.models.embeddings import EmbeddingError, get_embedding_model
from backend.retrieval.bm25 import get_bm25_index
from backend.retrieval.qdrant_store import QdrantStoreError, get_qdrant_store
from backend.utils.helpers import detect_source_type_from_filename, safe_filename
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionError(RuntimeError):
    """Raised when an end-to-end ingestion run fails."""


def _store_and_index(
    doc_id: str, filename: str, source_type: str, raw_chunks: List[Dict[str, Any]]
) -> int:
    """Persist chunks to SQLite, embed them, and upsert into Qdrant."""

    if not raw_chunks:
        raise IngestionError(f"No content extracted from '{filename}'.")

    chunk_ids = db.insert_chunks(doc_id, raw_chunks)

    try:
        embedding_model = get_embedding_model()
        vectors = embedding_model.embed_documents([c["text"] for c in raw_chunks])
    except EmbeddingError as exc:
        raise IngestionError(f"Embedding generation failed: {exc}") from exc

    payloads = [
        {
            "chunk_id": cid,
            "doc_id": doc_id,
            "text": chunk["text"],
            "metadata": {**chunk.get("metadata", {}), "filename": filename,
                         "source_type": source_type},
        }
        for cid, chunk in zip(chunk_ids, raw_chunks)
    ]

    try:
        store = get_qdrant_store()
        store.upsert_chunks(chunk_ids, vectors, payloads)
    except QdrantStoreError as exc:
        raise IngestionError(f"Vector storage failed: {exc}") from exc

    # Rebuild the BM25 keyword index so the new chunks are searchable.
    try:
        get_bm25_index().build()
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 index rebuild failed after ingestion: %s", exc)

    return len(chunk_ids)


def ingest_pdf(file_path: Path, filename: str) -> Dict[str, Any]:
    """Ingest a PDF file end-to-end. Returns a status dict."""

    doc_id = db.create_document(filename, "pdf", str(file_path))
    try:
        pages = load_pdf(file_path)
        all_chunks: List[Dict[str, Any]] = []
        for page in pages:
            if not page.text:
                continue
            page_chunks = chunk_text(
                page.text,
                base_metadata={
                    "page": page.page_number,
                    "used_ocr": page.used_ocr,
                    "source_type": "pdf",
                },
            )
            all_chunks.extend(page_chunks)

        num_chunks = _store_and_index(doc_id, filename, "pdf", all_chunks)
        db.update_document_status(doc_id, "ready", num_chunks=num_chunks)
        return {"doc_id": doc_id, "status": "ready", "num_chunks": num_chunks}
    except (PDFLoadError, IngestionError) as exc:
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected PDF ingestion failure")
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise IngestionError(f"Unexpected error ingesting PDF: {exc}") from exc


def ingest_image(file_path: Path, filename: str) -> Dict[str, Any]:
    """Ingest an image file end-to-end via OCR. Returns a status dict."""

    doc_id = db.create_document(filename, "image", str(file_path))
    try:
        image_content = load_image(file_path)
        chunks = chunk_text(
            image_content.text,
            base_metadata={
                "source_type": "image",
                "width": image_content.width,
                "height": image_content.height,
            },
        )
        num_chunks = _store_and_index(doc_id, filename, "image", chunks)
        db.update_document_status(doc_id, "ready", num_chunks=num_chunks)
        return {"doc_id": doc_id, "status": "ready", "num_chunks": num_chunks}
    except (ImageLoadError, IngestionError) as exc:
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected image ingestion failure")
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise IngestionError(f"Unexpected error ingesting image: {exc}") from exc


def ingest_audio(file_path: Path, filename: str) -> Dict[str, Any]:
    """Ingest an audio file end-to-end via Whisper transcription."""

    doc_id = db.create_document(filename, "audio", str(file_path))
    try:
        segments = transcribe_audio(file_path)
        all_chunks: List[Dict[str, Any]] = []
        for seg in segments:
            seg_chunks = chunk_text(
                seg.text,
                base_metadata={
                    "source_type": "audio",
                    "start_time": seg.start,
                    "end_time": seg.end,
                },
            )
            all_chunks.extend(seg_chunks)

        num_chunks = _store_and_index(doc_id, filename, "audio", all_chunks)
        db.update_document_status(doc_id, "ready", num_chunks=num_chunks)
        return {"doc_id": doc_id, "status": "ready", "num_chunks": num_chunks}
    except (AudioLoadError, IngestionError) as exc:
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected audio ingestion failure")
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise IngestionError(f"Unexpected error ingesting audio: {exc}") from exc


def ingest_video(file_path: Path, filename: str) -> Dict[str, Any]:
    """Ingest a video file end-to-end (speech + on-screen text)."""

    doc_id = db.create_document(filename, "video", str(file_path))
    try:
        segments = load_video(file_path)
        all_chunks: List[Dict[str, Any]] = []
        for seg in segments:
            seg_chunks = chunk_text(
                seg.text,
                base_metadata={
                    "source_type": "video",
                    "timestamp": seg.timestamp,
                    "kind": seg.kind,
                },
            )
            all_chunks.extend(seg_chunks)

        num_chunks = _store_and_index(doc_id, filename, "video", all_chunks)
        db.update_document_status(doc_id, "ready", num_chunks=num_chunks)
        return {"doc_id": doc_id, "status": "ready", "num_chunks": num_chunks}
    except (VideoLoadError, IngestionError) as exc:
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected video ingestion failure")
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise IngestionError(f"Unexpected error ingesting video: {exc}") from exc


def ingest_website(url: str) -> Dict[str, Any]:
    """Ingest a website URL end-to-end."""

    filename = safe_filename(url)
    doc_id = db.create_document(filename, "website", url)
    try:
        page = load_website(url)
        chunks = chunk_text(
            page.text,
            base_metadata={"source_type": "website", "url": url, "title": page.title},
        )
        num_chunks = _store_and_index(doc_id, filename, "website", chunks)
        db.update_document_status(doc_id, "ready", num_chunks=num_chunks)
        return {"doc_id": doc_id, "status": "ready", "num_chunks": num_chunks}
    except (WebLoadError, IngestionError) as exc:
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected website ingestion failure")
        db.update_document_status(doc_id, "failed", error_message=str(exc))
        raise IngestionError(f"Unexpected error ingesting website: {exc}") from exc


def ingest_any(file_path: Path, filename: str | None = None) -> Dict[str, Any]:
    """Dispatch ingestion based on the file's detected source type."""

    filename = filename or file_path.name
    source_type = detect_source_type_from_filename(filename)

    dispatch = {
        "pdf": ingest_pdf,
        "image": ingest_image,
        "audio": ingest_audio,
        "video": ingest_video,
    }

    handler = dispatch.get(source_type)
    if handler is None:
        raise IngestionError(
            f"Unsupported file type for '{filename}'. Supported: PDF, image, "
            f"audio, video, or a website URL."
        )
    return handler(file_path, filename)
