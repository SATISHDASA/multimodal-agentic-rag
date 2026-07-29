"""LangGraph agent graph definition for query-time answering.

Graph shape::

    START -> rewrite_query -> hybrid_retrieve -> rerank
          -> generate_answer -> build_citations -> END

Each stage is a plain node function from :mod:`backend.agent.nodes` so the
graph itself stays a thin, declarative wiring layer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from langgraph.graph import END, START, StateGraph

from backend.agent.memory import ConversationMemory
from backend.agent.nodes import (
    AgentState,
    build_citations_node,
    generate_answer_node,
    hybrid_retrieve_node,
    rerank_node,
    rewrite_query_node,
)
from backend.models.groq_llm import GroqLLMError, get_groq_llm
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _build_graph():
    """Construct and compile the LangGraph ``StateGraph`` for query answering."""

    graph = StateGraph(AgentState)

    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("hybrid_retrieve", hybrid_retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("build_citations", build_citations_node)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", "build_citations")
    graph.add_edge("build_citations", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    """Return a process-wide singleton compiled LangGraph instance."""

    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


def run_query(session_id: str, question: str) -> Dict[str, Any]:
    """Run the full agent graph for a single user question (non-streaming).

    Persists both the user's question and the assistant's answer to session
    memory as a side effect.

    Returns:
        A dict with keys: ``answer``, ``citations``, ``confidence``,
        ``retrieved_chunks``, ``reranked_chunks``, ``rewritten_query``.
    """

    memory = ConversationMemory(session_id)
    history = memory.get_history_for_llm()

    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "history": history,
    }

    graph = get_compiled_graph()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent graph execution failed")
        final_state = {
            "answer": f"An unexpected error occurred: {exc}",
            "citations": [],
            "confidence": 0.0,
        }

    memory.add_user_message(question)
    memory.add_assistant_message(
        final_state.get("answer", ""),
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", 0.0),
    )

    return {
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "confidence": final_state.get("confidence", 0.0),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "reranked_chunks": final_state.get("reranked_chunks", []),
        "rewritten_query": final_state.get("rewritten_query", question),
    }


def run_query_streaming(
    session_id: str, question: str
) -> Iterator[Dict[str, Any]]:
    """Run retrieval synchronously, then stream the LLM's answer token-by-token.

    Yields dicts of the shape ``{"type": "token", "content": str}`` while the
    answer is being generated, followed by a single final
    ``{"type": "final", ...}`` event carrying citations/confidence, once the
    full answer text has been persisted to memory.

    This generator-based approach lets the Streamlit UI render tokens live
    (for the "Streaming Responses" requirement) while still reusing the same
    retrieval/rerank/citation logic as the non-streaming path.
    """

    from backend.agent.nodes import _build_context_block, _SYSTEM_PROMPT
    from backend.retrieval.hybrid import hybrid_search
    from backend.retrieval.reranker import rerank
    from backend.agent.planner import rewrite_query
    from backend.agent.nodes import build_citations_node
    from backend.utils.config import settings

    memory = ConversationMemory(session_id)
    history = memory.get_history_for_llm()

    rewritten_query = rewrite_query(question, history)

    try:
        retrieved = hybrid_search(rewritten_query)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming retrieval failed")
        retrieved = []

    reranked = rerank(rewritten_query, retrieved, top_k=settings.top_k_final) if retrieved else []

    if not reranked:
        fallback_answer = (
            "I couldn't find any relevant information in the ingested "
            "documents to answer that question. Try uploading more sources "
            "or rephrasing your question."
        )
        yield {"type": "token", "content": fallback_answer}
        memory.add_user_message(question)
        memory.add_assistant_message(fallback_answer, citations=[], confidence=0.0)
        yield {
            "type": "final",
            "answer": fallback_answer,
            "citations": [],
            "confidence": 0.0,
            "reranked_chunks": [],
            "rewritten_query": rewritten_query,
        }
        return

    context_block = _build_context_block(reranked)
    user_prompt = (
        f"Context excerpts:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question, citing sources with [n] markers."
    )

    full_answer_parts: List[str] = []
    try:
        llm = get_groq_llm()
        for token in llm.stream(
            system_prompt=_SYSTEM_PROMPT,
            messages=[*history, {"role": "user", "content": user_prompt}],
        ):
            full_answer_parts.append(token)
            yield {"type": "token", "content": token}
    except GroqLLMError as exc:
        error_msg = f"Sorry, I hit an error while generating the answer: {exc}"
        yield {"type": "token", "content": error_msg}
        full_answer_parts = [error_msg]

    full_answer = "".join(full_answer_parts)

    citation_state = build_citations_node({"reranked_chunks": reranked})

    memory.add_user_message(question)
    memory.add_assistant_message(
        full_answer,
        citations=citation_state.get("citations", []),
        confidence=citation_state.get("confidence", 0.0),
    )

    yield {
        "type": "final",
        "answer": full_answer,
        "citations": citation_state.get("citations", []),
        "confidence": citation_state.get("confidence", 0.0),
        "reranked_chunks": reranked,
        "rewritten_query": rewritten_query,
    }
