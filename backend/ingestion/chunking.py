"""Text chunking utilities shared by all ingestion pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    base_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Split ``text`` into overlapping chunks suitable for embedding.

    Args:
        text: The full text to split.
        base_metadata: Metadata to attach to every resulting chunk (e.g.
            ``{"source_type": "pdf", "page": 3}``).
        chunk_size: Target chunk size in characters (defaults to settings).
        chunk_overlap: Overlap between consecutive chunks (defaults to settings).

    Returns:
        A list of dicts, each with ``text`` and ``metadata`` keys.
    """

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    base_metadata = base_metadata or {}
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)

    chunks = []
    for idx, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        metadata = dict(base_metadata)
        metadata["chunk_index"] = idx
        chunks.append({"text": piece, "metadata": metadata})

    return chunks
