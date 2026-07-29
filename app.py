"""Streamlit frontend for the Multimodal Agentic RAG application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Ensure the project root is importable when Streamlit runs this file
# directly (e.g. on Streamlit Community Cloud).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.agent.graph import run_query_streaming  # noqa: E402
from backend.agent.memory import create_new_session  # noqa: E402
from backend.database import sqlite as db  # noqa: E402
from backend.ingestion.pipeline import IngestionError, ingest_any, ingest_website  # noqa: E402
from backend.utils.config import UPLOADS_DIR, settings  # noqa: E402
from backend.utils.helpers import safe_filename  # noqa: E402
from backend.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

st.set_page_config(
    page_title="Multimodal Agentic RAG",
    page_icon="\U0001F9E0",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

# --------------------------------------------------------------------------- #
# Styling (dark-mode compatible: relies on Streamlit theme variables only)
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
    .chat-citation-badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid rgba(128,128,128,0.4);
    }
    .confidence-high { color: #2ecc71; font-weight: 600; }
    .confidence-medium { color: #f1c40f; font-weight: 600; }
    .confidence-low { color: #e74c3c; font-weight: 600; }
    .source-pill {
        border-radius: 6px;
        padding: 6px 10px;
        margin-bottom: 6px;
        border: 1px solid rgba(128,128,128,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #

if "session_id" not in st.session_state:
    st.session_state.session_id = create_new_session("Streamlit session")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # list of {role, content, citations, confidence}

if "processing_log" not in st.session_state:
    st.session_state.processing_log = []


def _confidence_class(score: float) -> str:
    if score >= 0.66:
        return "confidence-high"
    if score >= 0.4:
        return "confidence-medium"
    return "confidence-low"


def _save_uploaded_file(uploaded_file) -> Path:
    filename = safe_filename(uploaded_file.name)
    dest_path = Path(UPLOADS_DIR) / filename
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


# --------------------------------------------------------------------------- #
# Sidebar: uploads + processing
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.title("\U0001F4C2 Ingest Sources")
    st.caption("Upload documents, media, or a website URL, then click Process.")

    pdf_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="pdf_uploader"
    )
    image_files = st.file_uploader(
        "Upload Image(s)",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        key="image_uploader",
    )
    audio_files = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav", "m4a", "flac", "ogg", "aac"],
        accept_multiple_files=True,
        key="audio_uploader",
    )
    video_files = st.file_uploader(
        "Upload Video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=True,
        key="video_uploader",
    )
    website_url = st.text_input(
        "Website URL", placeholder="https://example.com/article"
    )

    process_clicked = st.button("\U0001F680 Process", use_container_width=True, type="primary")

    st.divider()
    st.subheader("\U0001F4CB Ingested Documents")

    docs = db.list_documents()
    if not docs:
        st.caption("No documents ingested yet.")
    else:
        for d in docs:
            icon = {
                "ready": "\u2705",
                "pending": "\u23F3",
                "failed": "\u274C",
            }.get(d["status"], "\u2753")
            with st.expander(f"{icon} {d['filename']}"):
                st.write(f"**Type:** {d['source_type']}")
                st.write(f"**Status:** {d['status']}")
                st.write(f"**Chunks:** {d['num_chunks']}")
                if d.get("error_message"):
                    st.error(d["error_message"])
                if st.button("Delete", key=f"delete_{d['id']}"):
                    from backend.retrieval.qdrant_store import get_qdrant_store

                    try:
                        get_qdrant_store().delete_by_doc_id(d["id"])
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Vector delete failed: %s", exc)
                    db.delete_document(d["id"])
                    st.rerun()

    st.divider()
    with st.expander("\u2699\ufe0f Settings"):
        st.write(f"**LLM model:** {settings.groq_model}")
        st.write(f"**Embedding model:** {settings.embedding_model_name}")
        st.write(f"**Reranker:** {settings.reranker_model_name}")
        if not settings.groq_api_key:
            st.warning(
                "GROQ_API_KEY is not set. Add it to your .env file to enable "
                "chat responses."
            )


# --------------------------------------------------------------------------- #
# Processing logic
# --------------------------------------------------------------------------- #


def _process_all_uploads() -> None:
    tasks: List[Dict[str, Any]] = []

    for f in pdf_files or []:
        tasks.append({"kind": "file", "file": f})
    for f in image_files or []:
        tasks.append({"kind": "file", "file": f})
    for f in audio_files or []:
        tasks.append({"kind": "file", "file": f})
    for f in video_files or []:
        tasks.append({"kind": "file", "file": f})
    if website_url.strip():
        tasks.append({"kind": "url", "url": website_url.strip()})

    if not tasks:
        st.toast("No files or URL provided.", icon="\u26A0\ufe0f")
        return

    progress_bar = st.progress(0.0, text="Starting ingestion...")
    total = len(tasks)

    for i, task in enumerate(tasks):
        if task["kind"] == "file":
            uploaded_file = task["file"]
            progress_bar.progress(
                i / total, text=f"Processing {uploaded_file.name}..."
            )
            try:
                dest_path = _save_uploaded_file(uploaded_file)
                result = ingest_any(dest_path, uploaded_file.name)
                st.toast(
                    f"Ingested '{uploaded_file.name}' "
                    f"({result['num_chunks']} chunks).",
                    icon="\u2705",
                )
            except IngestionError as exc:
                st.toast(f"Failed on '{uploaded_file.name}': {exc}", icon="\u274C")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected ingestion error")
                st.toast(f"Unexpected error on '{uploaded_file.name}': {exc}", icon="\u274C")
        else:
            url = task["url"]
            progress_bar.progress(i / total, text=f"Scraping {url}...")
            try:
                result = ingest_website(url)
                st.toast(
                    f"Ingested website ({result['num_chunks']} chunks).",
                    icon="\u2705",
                )
            except IngestionError as exc:
                st.toast(f"Failed to ingest website: {exc}", icon="\u274C")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected website ingestion error")
                st.toast(f"Unexpected error ingesting website: {exc}", icon="\u274C")

    progress_bar.progress(1.0, text="Done!")
    time.sleep(0.4)
    progress_bar.empty()
    st.rerun()


if process_clicked:
    _process_all_uploads()


# --------------------------------------------------------------------------- #
# Main page: chat interface
# --------------------------------------------------------------------------- #

st.title("\U0001F9E0 Multimodal Agentic RAG")
st.caption(
    "Ask questions across PDFs, images, audio, video, and websites — "
    "answers are grounded with citations."
)

chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                confidence = msg.get("confidence", 0.0)
                conf_class = _confidence_class(confidence)
                st.markdown(
                    f'Confidence: <span class="{conf_class}">{confidence:.0%}</span>',
                    unsafe_allow_html=True,
                )
                citations = msg.get("citations", [])
                if citations:
                    with st.expander(f"\U0001F4DA Sources ({len(citations)})"):
                        for c in citations:
                            st.markdown(
                                f"**[{c['index']}] {c['source']}** "
                                f"_{c['source_type']}_"
                                + (f" — loc: {c['location']}" if c.get("location") else "")
                            )
                            st.caption(c["excerpt"])

user_question = st.chat_input("Ask a question about your ingested sources...")

if user_question:
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_question}
    )
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""
            final_payload: Dict[str, Any] = {}

            try:
                for event in run_query_streaming(
                    st.session_state.session_id, user_question
                ):
                    if event["type"] == "token":
                        streamed_text += event["content"]
                        placeholder.markdown(streamed_text + "\u258C")
                    elif event["type"] == "final":
                        final_payload = event
                placeholder.markdown(streamed_text)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Chat streaming failed in UI")
                streamed_text = f"An unexpected error occurred: {exc}"
                placeholder.markdown(streamed_text)

            confidence = final_payload.get("confidence", 0.0)
            citations = final_payload.get("citations", [])

            conf_class = _confidence_class(confidence)
            st.markdown(
                f'Confidence: <span class="{conf_class}">{confidence:.0%}</span>',
                unsafe_allow_html=True,
            )
            if citations:
                with st.expander(f"\U0001F4DA Sources ({len(citations)})"):
                    for c in citations:
                        st.markdown(
                            f"**[{c['index']}] {c['source']}** _{c['source_type']}_"
                            + (f" — loc: {c['location']}" if c.get("location") else "")
                        )
                        st.caption(c["excerpt"])

            reranked_chunks = final_payload.get("reranked_chunks", [])
            if reranked_chunks:
                with st.expander("\U0001F50D Retrieved context (debug view)"):
                    for chunk in reranked_chunks:
                        st.markdown(
                            f'<div class="source-pill">{chunk["text"][:500]}</div>',
                            unsafe_allow_html=True,
                        )

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": streamed_text,
            "citations": final_payload.get("citations", []),
            "confidence": final_payload.get("confidence", 0.0),
        }
    )
