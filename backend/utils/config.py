"""Centralized application configuration.

All configuration is loaded from environment variables (or a local ``.env``
file) using ``pydantic-settings``. Nothing in this module ever hard-codes a
secret; only safe, non-sensitive defaults are provided.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root directory of the whole project (multimodal_agentic_rag/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SQLITE_DIR = DATA_DIR / "sqlite"
QDRANT_LOCAL_DIR = DATA_DIR / "qdrant_local"

for _dir in (DATA_DIR, UPLOADS_DIR, SQLITE_DIR, QDRANT_LOCAL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Strongly typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Groq (LLM) ----
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.2, alias="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=1024, alias="GROQ_MAX_TOKENS")

    # ---- Qdrant (vector database) ----
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        default="multimodal_rag_chunks", alias="QDRANT_COLLECTION"
    )
    # If no QDRANT_URL is configured, we transparently fall back to an
    # on-disk local Qdrant instance so the app still works out of the box.
    use_local_qdrant_fallback: bool = Field(
        default=True, alias="USE_LOCAL_QDRANT_FALLBACK"
    )

    # ---- Embeddings ----
    embedding_model_name: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5", alias="EMBEDDING_MODEL_NAME"
    )
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")

    # ---- Reranker ----
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base", alias="RERANKER_MODEL_NAME"
    )

    # ---- Whisper (audio/video transcription) ----
    whisper_model_size: str = Field(default="base", alias="WHISPER_MODEL_SIZE")

    # ---- Tesseract OCR ----
    tesseract_cmd: Optional[str] = Field(default=None, alias="TESSERACT_CMD")

    # ---- Chunking ----
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    # ---- Retrieval ----
    top_k_vector: int = Field(default=10, alias="TOP_K_VECTOR")
    top_k_bm25: int = Field(default=10, alias="TOP_K_BM25")
    top_k_final: int = Field(default=5, alias="TOP_K_FINAL")
    hybrid_alpha: float = Field(
        default=0.5,
        alias="HYBRID_ALPHA",
        description="Weight given to vector search vs BM25 in [0, 1].",
    )

    # ---- Web scraping ----
    web_request_timeout: int = Field(default=15, alias="WEB_REQUEST_TIMEOUT")
    web_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (compatible; MultimodalAgenticRAG/1.0; "
            "+https://github.com/)"
        ),
        alias="WEB_USER_AGENT",
    )

    # ---- FastAPI backend ----
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")

    # ---- Logging ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of :class:`Settings`."""

    return Settings()


settings = get_settings()
