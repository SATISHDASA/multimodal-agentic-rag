"""Keyword (lexical) search using BM25 over all ingested chunks.

The index is rebuilt from SQLite whenever new documents are ingested and
cached in-memory (and pickled to disk) for fast repeated queries within a
session.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.database.sqlite import get_all_chunks
from backend.utils.config import SQLITE_DIR
from backend.utils.logger import get_logger

logger = get_logger(__name__)

BM25_CACHE_PATH = SQLITE_DIR / "bm25_index.pkl"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric tokenization used consistently for BM25."""

    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index with disk persistence."""

    def __init__(self) -> None:
        self._bm25 = None
        self._chunk_records: List[Dict[str, Any]] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self) -> int:
        """(Re)build the BM25 index from all chunks currently in SQLite.

        Returns:
            The number of chunks indexed.
        """

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "rank-bm25 is not installed. Run `pip install rank-bm25`."
            ) from exc

        records = get_all_chunks()
        if not records:
            self._bm25 = None
            self._chunk_records = []
            logger.warning("BM25 index build skipped: no chunks found.")
            return 0

        tokenized_corpus = [_tokenize(r["text"]) for r in records]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunk_records = records

        self._persist()
        logger.info("BM25 index built with %d chunks", len(records))
        return len(records)

    def _persist(self) -> None:
        try:
            with open(BM25_CACHE_PATH, "wb") as fh:
                pickle.dump(
                    {"bm25": self._bm25, "records": self._chunk_records}, fh
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist BM25 index to disk: %s", exc)

    def load_from_disk(self) -> bool:
        """Attempt to load a previously persisted index. Returns success."""

        if not BM25_CACHE_PATH.exists():
            return False
        try:
            with open(BM25_CACHE_PATH, "rb") as fh:
                data = pickle.load(fh)
            self._bm25 = data.get("bm25")
            self._chunk_records = data.get("records", [])
            return self._bm25 is not None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load BM25 index from disk: %s", exc)
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return the ``top_k`` chunks most lexically similar to ``query``."""

        if self._bm25 is None:
            if not self.load_from_disk():
                logger.warning("BM25 search requested but index is not built.")
                return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            record = self._chunk_records[idx]
            results.append(
                {
                    "chunk_id": record["id"],
                    "doc_id": record["doc_id"],
                    "text": record["text"],
                    "metadata": record.get("metadata", {}),
                    "score": float(scores[idx]),
                }
            )
        return results


_bm25_index_singleton: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Return a process-wide singleton :class:`BM25Index`."""

    global _bm25_index_singleton
    if _bm25_index_singleton is None:
        _bm25_index_singleton = BM25Index()
    return _bm25_index_singleton
