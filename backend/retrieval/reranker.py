"""Cross-encoder reranking of retrieved chunks using BAAI/bge-reranker-base."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class RerankerError(RuntimeError):
    """Raised when reranking fails."""


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazily load and cache the cross-encoder reranker model."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - defensive
        raise RerankerError(
            "sentence-transformers is not installed. Run "
            "`pip install sentence-transformers`."
        ) from exc

    logger.info("Loading reranker model: %s", settings.reranker_model_name)
    try:
        model = CrossEncoder(settings.reranker_model_name, max_length=512)
    except Exception as exc:  # noqa: BLE001
        raise RerankerError(f"Could not load reranker model: {exc}") from exc
    return model


def rerank(
    query: str, candidates: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Rerank retrieved chunks by cross-encoder relevance to the query.

    Args:
        query: The user's (rewritten) question.
        candidates: List of chunk dicts, each containing at least a "text" key.
        top_k: Number of top results to return after reranking.

    Returns:
        The reranked list of chunk dicts (each with an added "rerank_score"),
        truncated to ``top_k`` entries, sorted by descending relevance.
    """

    if not candidates:
        return []

    try:
        model = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)
    except RerankerError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reranking failed; falling back to original order")
        # Graceful degradation: return the original top_k without reranking
        # rather than failing the whole request.
        for c in candidates:
            c.setdefault("rerank_score", c.get("score", 0.0))
        return candidates[:top_k]

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]
