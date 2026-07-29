"""Conversation / session memory backed by SQLite.

Provides the LangGraph agent with recent chat history formatted for the
Groq chat-completions API, and persists new turns as they happen.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.database import sqlite as db
from backend.utils.logger import get_logger

logger = get_logger(__name__)

MAX_HISTORY_TURNS = 6  # number of prior (user, assistant) turn-pairs to keep


class ConversationMemory:
    """Reads/writes chat turns for a given session id."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """Return recent history as a list of {"role", "content"} dicts."""

        messages = db.get_messages(self.session_id)
        trimmed = messages[-(MAX_HISTORY_TURNS * 2):]
        return [{"role": m["role"], "content": m["content"]} for m in trimmed]

    def get_full_history(self) -> List[Dict[str, Any]]:
        """Return the full raw message history (including citations/confidence)."""

        return db.get_messages(self.session_id)

    def add_user_message(self, content: str) -> str:
        return db.add_message(self.session_id, "user", content)

    def add_assistant_message(
        self,
        content: str,
        citations: Optional[List[dict]] = None,
        confidence: Optional[float] = None,
    ) -> str:
        return db.add_message(
            self.session_id, "assistant", content, citations=citations,
            confidence=confidence,
        )


def create_new_session(title: str = "New chat") -> str:
    """Create and return a new chat session id."""

    return db.create_session(title)
