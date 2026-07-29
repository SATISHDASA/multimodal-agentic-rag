"""SQLite persistence layer.

Stores document metadata, raw text chunks (used to build the BM25 keyword
index and to hydrate retrieval results), chat sessions, and chat messages
(including citations and confidence scores) for session memory.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional

from backend.utils.config import SQLITE_DIR
from backend.utils.helpers import new_id, utc_now_iso
from backend.utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = SQLITE_DIR / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    num_chunks INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON chat_messages (session_id);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row factory and FK support enabled."""

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they do not already exist."""

    with get_connection() as conn:
        conn.executescript(_SCHEMA)
    logger.info("SQLite database initialized at %s", DB_PATH)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def create_document(
    filename: str, source_type: str, source_path: Optional[str] = None
) -> str:
    """Insert a new document record and return its generated id."""

    doc_id = new_id("doc_")
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO documents
               (id, filename, source_type, source_path, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
            (doc_id, filename, source_type, source_path, utc_now_iso()),
        )
    return doc_id


def update_document_status(
    doc_id: str,
    status: str,
    error_message: Optional[str] = None,
    num_chunks: Optional[int] = None,
) -> None:
    """Update the processing status (and optional error/chunk count) of a document."""

    with get_connection() as conn:
        if num_chunks is not None:
            conn.execute(
                """UPDATE documents
                   SET status = ?, error_message = ?, num_chunks = ?
                   WHERE id = ?""",
                (status, error_message, num_chunks, doc_id),
            )
        else:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, doc_id),
            )


def list_documents() -> List[dict]:
    """Return metadata for all ingested documents, most recent first."""

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(doc_id: str) -> None:
    """Delete a document and its associated chunks."""

    with get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


def insert_chunks(doc_id: str, chunks: List[dict]) -> List[str]:
    """Bulk-insert chunks for a document. Each chunk dict has 'text' and 'metadata'."""

    ids: List[str] = []
    now = utc_now_iso()
    with get_connection() as conn:
        for chunk in chunks:
            chunk_id = new_id("chunk_")
            ids.append(chunk_id)
            conn.execute(
                """INSERT INTO chunks (id, doc_id, text, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    doc_id,
                    chunk["text"],
                    json.dumps(chunk.get("metadata", {})),
                    now,
                ),
            )
    return ids


def get_all_chunks() -> List[dict]:
    """Return every chunk in the database (used to (re)build the BM25 index)."""

    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM chunks").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"]) if d.get("metadata") else {}
        result.append(d)
    return result


def get_chunks_by_ids(chunk_ids: List[str]) -> List[dict]:
    """Fetch specific chunks by their ids, preserving no particular order."""

    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"]) if d.get("metadata") else {}
        result.append(d)
    return result


# --------------------------------------------------------------------------- #
# Chat sessions & messages
# --------------------------------------------------------------------------- #


def create_session(title: str = "New chat") -> str:
    """Create a new chat session and return its id."""

    session_id = new_id("session_")
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, title, created_at) VALUES (?, ?, ?)",
            (session_id, title, utc_now_iso()),
        )
    return session_id


def add_message(
    session_id: str,
    role: str,
    content: str,
    citations: Optional[List[dict]] = None,
    confidence: Optional[float] = None,
) -> str:
    """Persist a single chat message (user or assistant) to a session."""

    message_id = new_id("msg_")
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO chat_messages
               (id, session_id, role, content, citations, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                message_id,
                session_id,
                role,
                content,
                json.dumps(citations or []),
                confidence,
                utc_now_iso(),
            ),
        )
    return message_id


def get_messages(session_id: str) -> List[dict]:
    """Return the full message history for a session, chronological order."""

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM chat_messages
               WHERE session_id = ? ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["citations"] = json.loads(d["citations"]) if d.get("citations") else []
        result.append(d)
    return result


def list_sessions() -> List[dict]:
    """Return all chat sessions, most recent first."""

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]
