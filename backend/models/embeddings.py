"""Text embedding generation using the free, local Nomic Embed model.

We use ``sentence-transformers`` to load ``nomic-ai/nomic-embed-text-v1.5``
fully locally (no paid API calls). The model is downloaded once from the
Hugging Face Hub and cached on disk.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Nomic embed models expect task-specific instruction prefixes for best
# retrieval quality: "search_document: " for indexed passages and
# "search_query: " for user queries.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


@lru_cache(maxsize=1)
def _get_model():
    """Lazily load and cache the sentence-transformers embedding model."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - defensive
        raise EmbeddingError(
            "sentence-transformers is not installed. Run "
            "`pip install sentence-transformers`."
        ) from exc

    logger.info("Loading embedding model: %s", settings.embedding_model_name)
    try:
        model = SentenceTransformer(
            settings.embedding_model_name, trust_remote_code=True
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load embedding model")
        raise EmbeddingError(f"Could not load embedding model: {exc}") from exc
    return model


class EmbeddingModel:
    """Facade providing document- and query-side embedding generation."""

    def __init__(self) -> None:
        self.dimension = settings.embedding_dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks for indexing."""

        if not texts:
            return []
        model = _get_model()
        prefixed = [f"{DOCUMENT_PREFIX}{t}" for t in texts]
        try:
            vectors = model.encode(
                prefixed,
                batch_size=16,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Embedding generation failed for documents")
            raise EmbeddingError(f"Embedding failure: {exc}") from exc
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query for similarity search."""

        model = _get_model()
        try:
            vector = model.encode(
                f"{QUERY_PREFIX}{text}",
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Embedding generation failed for query")
            raise EmbeddingError(f"Embedding failure: {exc}") from exc
        return vector.tolist()


_embedding_model_singleton: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Return a process-wide singleton :class:`EmbeddingModel`."""

    global _embedding_model_singleton
    if _embedding_model_singleton is None:
        _embedding_model_singleton = EmbeddingModel()
    return _embedding_model_singleton
