"""FastAPI backend exposing ingestion and chat endpoints.

This API is optional for the Streamlit app (which calls the backend
functions in-process for simplicity/performance), but is provided so the
same RAG engine can be consumed by other clients over HTTP.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agent.graph import run_query
from backend.agent.memory import create_new_session
from backend.database import sqlite as db
from backend.ingestion.pipeline import IngestionError, ingest_any, ingest_website
from backend.utils.config import UPLOADS_DIR, settings
from backend.utils.helpers import safe_filename
from backend.utils.logger import get_logger

logger = get_logger(__name__)

db.init_db()

app = FastAPI(
    title="Multimodal Agentic RAG API",
    description=(
        "Backend API for ingesting PDFs, images, audio, video, and websites, "
        "and answering questions over them with citations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    session_id: Optional[str] = Field(
        default=None, description="Existing session id, or omit to create one."
    )
    question: str = Field(..., min_length=1, description="The user's question.")


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""

    session_id: str
    answer: str
    citations: List[dict]
    confidence: float
    rewritten_query: str


class WebsiteIngestRequest(BaseModel):
    """Request body for the /ingest/website endpoint."""

    url: str = Field(..., description="A fully qualified http(s) URL to ingest.")


class IngestResponse(BaseModel):
    """Response body shared by all ingestion endpoints."""

    doc_id: str
    status: str
    num_chunks: int


class DocumentInfo(BaseModel):
    """Metadata describing a previously ingested document."""

    id: str
    filename: str
    source_type: str
    status: str
    num_chunks: int
    created_at: str
    error_message: Optional[str] = None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health", tags=["system"])
def health_check() -> dict:
    """Simple liveness probe."""

    return {"status": "ok"}


@app.post("/session", tags=["chat"])
def new_session() -> dict:
    """Create a new chat session and return its id."""

    return {"session_id": create_new_session()}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question over all ingested documents, with citations."""

    session_id = request.session_id or create_new_session()
    try:
        result = run_query(session_id, request.question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        citations=result["citations"],
        confidence=result["confidence"],
        rewritten_query=result["rewritten_query"],
    )


@app.post("/ingest/file", response_model=IngestResponse, tags=["ingestion"])
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a PDF, image, audio, or video file."""

    filename = safe_filename(file.filename or "upload")
    dest_path = Path(UPLOADS_DIR) / filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Could not save uploaded file: {exc}"
        ) from exc
    finally:
        file.file.close()

    try:
        result = ingest_any(dest_path, filename)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during file ingestion")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(**result)


@app.post("/ingest/website", response_model=IngestResponse, tags=["ingestion"])
def ingest_website_endpoint(request: WebsiteIngestRequest) -> IngestResponse:
    """Ingest a website URL."""

    try:
        result = ingest_website(request.url)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during website ingestion")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(**result)


@app.get("/documents", response_model=List[DocumentInfo], tags=["ingestion"])
def list_documents_endpoint() -> List[DocumentInfo]:
    """List all ingested documents and their processing status."""

    docs = db.list_documents()
    return [
        DocumentInfo(
            id=d["id"],
            filename=d["filename"],
            source_type=d["source_type"],
            status=d["status"],
            num_chunks=d["num_chunks"],
            created_at=d["created_at"],
            error_message=d.get("error_message"),
        )
        for d in docs
    ]


@app.delete("/documents/{doc_id}", tags=["ingestion"])
def delete_document_endpoint(doc_id: str) -> dict:
    """Delete a document's metadata, chunks, and vectors."""

    from backend.retrieval.qdrant_store import get_qdrant_store

    try:
        get_qdrant_store().delete_by_doc_id(doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete vectors for doc %s: %s", doc_id, exc)

    db.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/sessions/{session_id}/messages", tags=["chat"])
def get_session_messages(session_id: str) -> list:
    """Return the full chat history for a session."""

    return db.get_messages(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )
