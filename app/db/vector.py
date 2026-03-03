"""Qdrant vector database client for NexusCortex.

Handles embedding generation, vector upserts, and semantic search
for the RAG engine's vector retrieval path.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from typing import TYPE_CHECKING, Any

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from app.exceptions import VectorStoreError

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# Deterministic namespace for content-based UUIDs (uuid5).
NEXUS_UUID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class VectorClient:
    """Async Qdrant client with integrated embedding generation."""

    def __init__(self, settings: Settings) -> None:
        self._host = settings.QDRANT_HOST
        self._port = settings.QDRANT_PORT
        self._collection = settings.QDRANT_COLLECTION
        self._embedding_dim = settings.EMBEDDING_DIM
        self._llm_base_url = settings.LLM_BASE_URL
        self._llm_api_key = settings.LLM_API_KEY
        self._embedding_model = settings.EMBEDDING_MODEL
        self._client = AsyncQdrantClient(host=self._host, port=self._port)
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def initialize(self) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        try:
            collections = await self._client.get_collections()
            existing = {c.name for c in collections.collections}
            if self._collection not in existing:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '%s' (dim=%d, cosine)",
                    self._collection,
                    self._embedding_dim,
                )
            else:
                logger.info(
                    "Qdrant collection '%s' already exists",
                    self._collection,
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize Qdrant collection: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the underlying Qdrant and HTTP clients."""
        await self._http_client.aclose()
        await self._client.close()
        logger.info("Qdrant client closed")

    async def upsert(self, text: str, metadata: dict[str, Any]) -> str:
        """Embed text and upsert into Qdrant.

        Args:
            text: The text content to embed and store.
            metadata: Must include 'source', 'tags', 'domain'.
                      Additional keys are stored under 'metadata'.

        Returns:
            The generated point ID as a string.
        """
        try:
            vector = await self._embed(text)
            point_id = str(uuid.uuid5(NEXUS_UUID_NAMESPACE, text))

            payload = {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "tags": metadata.get("tags", []),
                "domain": metadata.get("domain", "general"),
                "timestamp": metadata.get(
                    "timestamp",
                    datetime.now(timezone.utc).isoformat(),
                ),
                "metadata": {
                    k: v
                    for k, v in metadata.items()
                    if k not in {"source", "tags", "domain", "timestamp"}
                },
            }

            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            return point_id
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert vector: {exc}"
            ) from exc

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Embed query and search Qdrant for similar vectors.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            filter_tags: Optional list of tags to filter on (match any).

        Returns:
            List of dicts with id, score, text, and metadata.
        """
        try:
            vector = await self._embed(query)

            query_filter = None
            if filter_tags:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="tags",
                            match=MatchAny(any=filter_tags),
                        )
                    ]
                )

            results = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "text": point.payload.get("text", "") if point.payload else "",
                    "metadata": {
                        "source": point.payload.get("source", ""),
                        "tags": point.payload.get("tags", []),
                        "domain": point.payload.get("domain", ""),
                        "timestamp": point.payload.get("timestamp", ""),
                        **(point.payload.get("metadata", {})),
                    }
                    if point.payload
                    else {},
                }
                for point in results.points
            ]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to search vectors: {exc}"
            ) from exc

    async def _embed(self, text: str) -> list[float]:
        """Generate an embedding vector via the LLM embeddings endpoint.

        Calls POST {LLM_BASE_URL}/embeddings with the configured model.
        """
        url = f"{self._llm_base_url}/embeddings"
        headers = {}
        if self._llm_api_key:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"
        try:
            response = await self._http_client.post(
                url,
                json={
                    "model": self._embedding_model,
                    "input": text,
                },
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPStatusError as exc:
            raise VectorStoreError(
                f"Embedding endpoint returned {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc
        except (httpx.RequestError, KeyError, IndexError) as exc:
            raise VectorStoreError(
                f"Failed to generate embedding: {exc}"
            ) from exc
