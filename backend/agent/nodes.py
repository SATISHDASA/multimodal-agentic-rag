"""LangGraph node functions implementing the query-time agent workflow:

    rewrite query -> hybrid retrieve -> rerank -> generate answer (Groq,
    streaming) -> compute citations & confidence score.

Document/media *ingestion* (PDF/OCR/Whisper/video/web) is handled by
:mod:`backend.ingestion.pipeline` and triggered directly from the UI's
"Process" button -- this keeps heavy, long-running file processing decoupled
from the low-latency conversational graph, while still going through the
same planner/tool abstractions defined in :mod:`backend.agent.tools`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from backend.agent.planner import rewrite_query
from backend.models.groq_llm import GroqLLMError, get_groq_llm
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.reranker import rerank
from backend.utils.config import settings
from backend.utils.helpers import truncate
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    """Shared state threaded through every node of the query graph."""

    session_id: str
    question: str
    history: List[Dict[str, str]]
    rewritten_query: str
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    error: Optional[str]


_SYSTEM_PROMPT = (
    "You are a meticulous multimodal research assistant. Answer the user's "
    "question using ONLY the provided context excerpts, which may originate "
    "from PDFs, images (OCR), audio/video transcripts, or websites. "
    "Rules:\n"
    "1. Ground every claim in the context; do not use outside knowledge.\n"
    "2. If the context does not contain the answer, say so plainly.\n"
    "3. Cite sources inline using bracketed numbers like [1], [2] that "
    "correspond to the numbered context excerpts you were given.\n"
    "4. Be concise, accurate, and well organized."
)


def rewrite_query_node(state: AgentState) -> AgentState:
    """Rewrite the user's question into a standalone search query."""

    try:
        rewritten = rewrite_query(state["question"], state.get("history", []))
    except Exception as exc:  # noqa: BLE001
        logger.warning("rewrite_query_node failed, using raw question: %s", exc)
        rewritten = state["question"]
    return {**state, "rewritten_query": rewritten}


def hybrid_retrieve_node(state: AgentState) -> AgentState:
    """Run hybrid (vector + BM25) retrieval for the rewritten query."""

    query = state.get("rewritten_query") or state["question"]
    try:
        results = hybrid_search(query)
    except Exception as exc:  # noqa: BLE001
        logger.exception("hybrid_retrieve_node failed")
        return {**state, "retrieved_chunks": [], "error": f"Retrieval failed: {exc}"}
    return {**state, "retrieved_chunks": results}


def rerank_node(state: AgentState) -> AgentState:
    """Rerank retrieved chunks with the cross-encoder for final relevance."""

    candidates = state.get("retrieved_chunks", [])
    if not candidates:
        return {**state, "reranked_chunks": []}

    query = state.get("rewritten_query") or state["question"]
    try:
        reranked = rerank(query, candidates, top_k=settings.top_k_final)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rerank_node failed, using un-reranked order: %s", exc)
        reranked = candidates[: settings.top_k_final]
    return {**state, "reranked_chunks": reranked}


def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source_desc = meta.get("filename") or meta.get("url") or "unknown source"
        location = ""
        if "page" in meta:
            location = f", page {meta['page']}"
        elif "start_time" in meta:
            location = f", ~{meta['start_time']:.0f}s"
        elif "timestamp" in meta:
            location = f", ~{meta['timestamp']:.0f}s"
        lines.append(f"[{i}] Source: {source_desc}{location}\n{chunk['text']}")
    return "\n\n".join(lines)


def generate_answer_node(state: AgentState) -> AgentState:
    """Generate a grounded, cited answer using the Groq LLM."""

    chunks = state.get("reranked_chunks", [])
    question = state["question"]

    if not chunks:
        return {
            **state,
            "answer": (
                "I couldn't find any relevant information in the ingested "
                "documents to answer that question. Try uploading more "
                "sources or rephrasing your question."
            ),
            "citations": [],
            "confidence": 0.0,
        }

    context_block = _build_context_block(chunks)
    user_prompt = (
        f"Context excerpts:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question, citing sources with [n] markers."
    )

    try:
        llm = get_groq_llm()
        answer = llm.generate(
            system_prompt=_SYSTEM_PROMPT,
            messages=[*state.get("history", []), {"role": "user", "content": user_prompt}],
        )
    except GroqLLMError as exc:
        logger.exception("generate_answer_node: Groq call failed")
        return {
            **state,
            "answer": f"Sorry, I hit an error while generating the answer: {exc}",
            "citations": [],
            "confidence": 0.0,
            "error": str(exc),
        }

    return {**state, "answer": answer}


def build_citations_node(state: AgentState) -> AgentState:
    """Build a structured citation list and a heuristic confidence score."""

    chunks = state.get("reranked_chunks", [])
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        citations.append(
            {
                "index": i,
                "source": meta.get("filename") or meta.get("url") or "unknown",
                "source_type": meta.get("source_type", "unknown"),
                "location": meta.get("page")
                or meta.get("start_time")
                or meta.get("timestamp"),
                "excerpt": truncate(chunk["text"], 220),
                "relevance_score": round(
                    chunk.get("rerank_score", chunk.get("fused_score", 0.0)), 4
                ),
            }
        )

    confidence = 0.0
    if chunks:
        top_scores = [
            c.get("rerank_score", c.get("fused_score", 0.0)) for c in chunks[:3]
        ]
        # bge-reranker-base outputs raw logits; squash to (0, 1) with a
        # sigmoid so the confidence score is human-interpretable.
        import math

        squashed = [1 / (1 + math.exp(-s)) for s in top_scores]
        confidence = round(sum(squashed) / len(squashed), 3)

    return {**state, "citations": citations, "confidence": confidence}
