"""Hybrid retrieval: fuse dense vector search (Qdrant) with sparse keyword
search (BM25) using min-max normalized weighted score fusion.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.models.embeddings import get_embedding_model
from backend.retrieval.bm25 import get_bm25_index
from backend.retrieval.qdrant_store import get_qdrant_store
from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _min_max_normalize(values: List[float]) -> List[float]:
    """Scale a list of scores to the [0, 1] range."""

    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def hybrid_search(
    query: str,
    top_k_vector: int | None = None,
    top_k_bm25: int | None = None,
    alpha: float | None = None,
) -> List[Dict[str, Any]]:
    """Retrieve and fuse results from dense and sparse search.

    Args:
        query: The (rewritten) user question.
        top_k_vector: How many candidates to pull from Qdrant.
        top_k_bm25: How many candidates to pull from BM25.
        alpha: Weight in [0, 1] given to the vector score; ``1 - alpha`` is
            given to the BM25 score.

    Returns:
        A list of chunk dicts sorted by descending fused score, each
        annotated with ``vector_score``, ``bm25_score``, and ``fused_score``.
    """

    top_k_vector = top_k_vector or settings.top_k_vector
    top_k_bm25 = top_k_bm25 or settings.top_k_bm25
    alpha = alpha if alpha is not None else settings.hybrid_alpha

    vector_results: List[Dict[str, Any]] = []
    try:
        embedding_model = get_embedding_model()
        query_vector = embedding_model.embed_query(query)
        store = get_qdrant_store()
        vector_results = store.search(query_vector, top_k=top_k_vector)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector search step failed, continuing with BM25 only: %s", exc)

    bm25_results: List[Dict[str, Any]] = []
    try:
        bm25_index = get_bm25_index()
        bm25_results = bm25_index.search(query, top_k=top_k_bm25)
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 search step failed, continuing with vector only: %s", exc)

    if not vector_results and not bm25_results:
        return []

    vector_scores = _min_max_normalize([r["score"] for r in vector_results])
    bm25_scores = _min_max_normalize([r["score"] for r in bm25_results])

    fused: Dict[str, Dict[str, Any]] = {}

    for result, norm_score in zip(vector_results, vector_scores):
        cid = result["chunk_id"]
        fused[cid] = {
            **result,
            "vector_score": norm_score,
            "bm25_score": 0.0,
        }

    for result, norm_score in zip(bm25_results, bm25_scores):
        cid = result["chunk_id"]
        if cid in fused:
            fused[cid]["bm25_score"] = norm_score
        else:
            fused[cid] = {
                **result,
                "vector_score": 0.0,
                "bm25_score": norm_score,
            }

    for record in fused.values():
        record["fused_score"] = (
            alpha * record["vector_score"] + (1 - alpha) * record["bm25_score"]
        )

    ranked = sorted(fused.values(), key=lambda r: r["fused_score"], reverse=True)
    return ranked
