"""LangChain tool definitions used by the agent graph for ingestion and
retrieval. Wrapping these as ``@tool`` functions keeps them independently
testable and gives the agent a clear, typed contract for each capability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool

from backend.ingestion.pipeline import (
    IngestionError,
    ingest_audio,
    ingest_image,
    ingest_pdf,
    ingest_video,
    ingest_website,
)
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.reranker import rerank
from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def pdf_ingestion_tool(file_path: str, filename: str) -> Dict[str, Any]:
    """Extract, chunk, embed, and index a PDF file (with OCR fallback for
    scanned pages)."""

    try:
        return ingest_pdf(Path(file_path), filename)
    except IngestionError as exc:
        return {"status": "failed", "error": str(exc)}


@tool
def image_ocr_tool(file_path: str, filename: str) -> Dict[str, Any]:
    """Run OCR on an image file and index the extracted text."""

    try:
        return ingest_image(Path(file_path), filename)
    except IngestionError as exc:
        return {"status": "failed", "error": str(exc)}


@tool
def whisper_audio_tool(file_path: str, filename: str) -> Dict[str, Any]:
    """Transcribe an audio file with Whisper and index the transcript."""

    try:
        return ingest_audio(Path(file_path), filename)
    except IngestionError as exc:
        return {"status": "failed", "error": str(exc)}


@tool
def video_extraction_tool(file_path: str, filename: str) -> Dict[str, Any]:
    """Extract speech and on-screen text from a video and index both."""

    try:
        return ingest_video(Path(file_path), filename)
    except IngestionError as exc:
        return {"status": "failed", "error": str(exc)}


@tool
def web_scraper_tool(url: str) -> Dict[str, Any]:
    """Scrape a website URL and index its readable text content."""

    try:
        return ingest_website(url)
    except IngestionError as exc:
        return {"status": "failed", "error": str(exc)}


@tool
def hybrid_retrieval_tool(query: str) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks for a query using hybrid
    (vector + BM25) search."""

    return hybrid_search(query)


@tool
def reranker_tool(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank retrieved candidate chunks by cross-encoder relevance."""

    return rerank(query, candidates, top_k=settings.top_k_final)


ALL_INGESTION_TOOLS = [
    pdf_ingestion_tool,
    image_ocr_tool,
    whisper_audio_tool,
    video_extraction_tool,
    web_scraper_tool,
]

ALL_RETRIEVAL_TOOLS = [hybrid_retrieval_tool, reranker_tool]
