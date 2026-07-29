"""Vector storage and similarity search backed by Qdrant.

Uses Qdrant Cloud's free tier when ``QDRANT_URL``/``QDRANT_API_KEY`` are
configured, and transparently falls back to an embedded on-disk Qdrant
instance (still free, zero external dependency) otherwise -- convenient for
local development and for Streamlit Community Cloud deployments where no
cloud credentials have been set yet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.utils.config import QDRANT_LOCAL_DIR, settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantStoreError(RuntimeError):
    """Raised when a Qdrant operation fails."""


class QdrantStore:
    """Thin wrapper around the Qdrant client for chunk-level vector storage."""

    def __init__(self) -> None:
        self._client = None
        self.collection_name = settings.qdrant_collection

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - defensive
            raise QdrantStoreError(
                "qdrant-client is not installed. Run `pip install qdrant-client`."
            ) from exc

        try:
            if settings.qdrant_url:
                logger.info("Connecting to Qdrant Cloud at %s", settings.qdrant_url)
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=30,
                )
            elif settings.use_local_qdrant_fallback:
                logger.info(
                    "QDRANT_URL not set; using local on-disk Qdrant at %s",
                    QDRANT_LOCAL_DIR,
                )
                self._client = QdrantClient(path=str(QDRANT_LOCAL_DIR))
            else:
                raise QdrantStoreError(
                    "QDRANT_URL is not configured and local fallback is disabled."
                )
        except QdrantStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise QdrantStoreError(f"Could not connect to Qdrant: {exc}") from exc

        self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = self._client
        try:
            existing = [c.name for c in client.get_collections().collections]
            if self.collection_name not in existing:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension, distance=Distance.COSINE
                    ),
                )
                logger.info("Created Qdrant collection '%s'", self.collection_name)
        except Exception as exc:  # noqa: BLE001
            raise QdrantStoreError(f"Could not ensure Qdrant collection: {exc}") from exc

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Upsert a batch of chunk vectors with their payloads into Qdrant."""

        from qdrant_client.models import PointStruct

        if not chunk_ids:
            return

        client = self._get_client()
        points = [
            PointStruct(id=_string_id_to_int(cid), vector=vec, payload=payload)
            for cid, vec, payload in zip(chunk_ids, vectors, payloads)
        ]
        try:
            client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Qdrant upsert failed")
            raise QdrantStoreError(f"Qdrant upsert failed: {exc}") from exc

    def search(
        self, query_vector: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for the ``top_k`` most similar chunks to ``query_vector``."""

        client = self._get_client()
        try:
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Qdrant search failed")
            raise QdrantStoreError(f"Qdrant search failed: {exc}") from exc

        return [
            {
                "chunk_id": r.payload.get("chunk_id"),
                "doc_id": r.payload.get("doc_id"),
                "text": r.payload.get("text"),
                "metadata": r.payload.get("metadata", {}),
                "score": r.score,
            }
            for r in results
        ]

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all vectors belonging to a given document id."""

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Qdrant delete failed")
            raise QdrantStoreError(f"Qdrant delete failed: {exc}") from exc


def _string_id_to_int(string_id: str) -> int:
    """Deterministically map a string chunk id to a positive 63-bit integer.

    Qdrant point ids must be either an unsigned integer or a UUID; since our
    internal chunk ids are custom prefixed strings, we hash them down to a
    stable integer instead of a random one so re-upserts overwrite cleanly.
    """

    import hashlib

    digest = hashlib.sha256(string_id.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


_qdrant_store_singleton: Optional[QdrantStore] = None


def get_qdrant_store() -> QdrantStore:
    """Return a process-wide singleton :class:`QdrantStore`."""

    global _qdrant_store_singleton
    if _qdrant_store_singleton is None:
        _qdrant_store_singleton = QdrantStore()
    return _qdrant_store_singleton
