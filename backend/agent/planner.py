"""Planning logic for the agent: input-type routing and query rewriting."""

from __future__ import annotations

from typing import List

from backend.models.groq_llm import GroqLLMError, get_groq_llm
from backend.utils.helpers import detect_source_type_from_filename
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_QUERY_REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into clear, standalone search queries for a "
    "retrieval system. Use the conversation history to resolve pronouns and "
    "implicit references (e.g. 'it', 'that document', 'the second one'). "
    "Respond with ONLY the rewritten query text, nothing else. Do not add "
    "quotation marks or explanations. If the original question is already "
    "clear and standalone, return it unchanged."
)


def plan_ingestion_tool(filename: str) -> str:
    """Return which ingestion tool should handle a given uploaded filename.

    Returns one of: ``pdf``, ``image``, ``audio``, ``video``, ``unknown``.
    """

    source_type = detect_source_type_from_filename(filename)
    logger.info("Planner routed '%s' -> %s tool", filename, source_type)
    return source_type


def rewrite_query(question: str, history: List[dict]) -> str:
    """Rewrite a (possibly context-dependent) user question into a
    standalone search query using recent chat history.

    Falls back to the original question if rewriting fails for any reason,
    so retrieval quality degrades gracefully rather than the whole request
    failing.
    """

    if not history:
        return question

    try:
        llm = get_groq_llm()
        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in history[-6:]
        )
        user_prompt = (
            f"Conversation history:\n{history_text}\n\n"
            f"Latest question: {question}\n\n"
            "Rewritten standalone query:"
        )
        rewritten = llm.generate(
            system_prompt=_QUERY_REWRITE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        rewritten = rewritten.strip().strip('"').strip()
        return rewritten if rewritten else question
    except GroqLLMError as exc:
        logger.warning("Query rewriting failed, using original question: %s", exc)
        return question
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error during query rewriting: %s", exc)
        return question
