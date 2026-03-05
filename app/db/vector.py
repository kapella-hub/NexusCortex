"""Qdrant vector database client for NexusCortex.

Handles embedding generation, vector upserts, and semantic search
for the RAG engine's vector retrieval path.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from typing import TYPE_CHECKING, Any

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.exceptions import VectorStoreError

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# Deterministic namespace for content-based UUIDs (uuid5).
NEXUS_UUID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# Maximum number of embedding vectors to cache in memory.
_EMBED_CACHE_MAX_SIZE = 512

# Maximum number of texts to send in a single batch embedding request.
_BATCH_CHUNK_SIZE = 32


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
        # LRU embedding cache: hash(text) -> embedding vector
        self._embed_cache: OrderedDict[str, list[float]] = OrderedDict()

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

        # Create payload indexes for faster filtered queries
        for field_name in ("tags", "namespace"):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Index may already exist

    async def close(self) -> None:
        """Close the underlying Qdrant and HTTP clients."""
        await self._http_client.aclose()
        await self._client.close()
        logger.info("Qdrant client closed")

    async def ping(self) -> None:
        """Verify connectivity to Qdrant. Raises on failure."""
        await self._client.get_collections()

    async def memory_count(self) -> int | None:
        """Return the number of points in the collection, or None on failure."""
        try:
            info = await self._client.get_collection(self._collection)
            return info.points_count
        except Exception:
            return None

    async def get_stats(self) -> dict:
        """Return vector store statistics.

        Scrolls a sample of points to find oldest/newest timestamps
        and namespace (domain) distribution.
        """
        try:
            info = await self._client.get_collection(self._collection)
            total = info.points_count or 0
        except Exception:
            return {
                "total": 0,
                "oldest_memory": None,
                "newest_memory": None,
                "namespace_counts": {},
            }

        oldest: str | None = None
        newest: str | None = None
        namespace_counts: dict[str, int] = {}

        # Scroll through all points to aggregate stats
        offset = None
        while True:
            try:
                records, next_offset = await self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=None,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                break

            if not records:
                break

            for point in records:
                payload = point.payload or {}
                ts = payload.get("timestamp")
                if ts:
                    ts_str = str(ts)
                    if oldest is None or ts_str < oldest:
                        oldest = ts_str
                    if newest is None or ts_str > newest:
                        newest = ts_str
                domain = payload.get("domain", "default")
                namespace_counts[domain] = namespace_counts.get(domain, 0) + 1

            if next_offset is None:
                break
            offset = next_offset

        return {
            "total": total,
            "oldest_memory": oldest,
            "newest_memory": newest,
            "namespace_counts": namespace_counts,
        }

    async def scroll_all(self, namespace: str | None = None, batch_size: int = 100):
        """Async generator that yields all points, optionally filtered by namespace.

        Yields dicts with id, text, metadata, namespace.
        """
        scroll_filter = None
        if namespace:
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="domain",
                        match=MatchAny(any=[namespace]),
                    )
                ]
            )

        offset = None
        while True:
            try:
                records, next_offset = await self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                break

            if not records:
                break

            for point in records:
                payload = point.payload or {}
                yield {
                    "id": str(point.id),
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                    "namespace": payload.get("domain", "default"),
                    "tags": payload.get("tags", []),
                    "source": payload.get("source", ""),
                    "created_at": payload.get("timestamp", ""),
                }

            if next_offset is None:
                break
            offset = next_offset

    async def list_memories(
        self,
        limit: int = 20,
        offset: int = 0,
        query: str | None = None,
        namespace: str | None = None,
    ) -> list[dict]:
        """List memories with optional search.

        If query is provided, do semantic search. Otherwise scroll through points.
        If namespace is provided, filter by domain.
        """
        effective_limit = min(limit, 100)

        try:
            if query:
                # Semantic search
                filter_tags = None
                query_filter = None
                if namespace:
                    query_filter = Filter(
                        must=[
                            FieldCondition(
                                key="domain",
                                match=MatchAny(any=[namespace]),
                            )
                        ]
                    )
                vector = await self._embed(query)
                results = await self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    query_filter=query_filter,
                    limit=effective_limit + offset,
                    with_payload=True,
                )
                points = results.points[offset:]
                return [
                    {
                        "id": str(p.id),
                        "score": p.score,
                        "text": p.payload.get("text", "") if p.payload else "",
                        "domain": p.payload.get("domain", "") if p.payload else "",
                        "tags": p.payload.get("tags", []) if p.payload else [],
                        "timestamp": p.payload.get("timestamp", "") if p.payload else "",
                        "source": p.payload.get("source", "") if p.payload else "",
                    }
                    for p in points
                ]
            else:
                # Scroll through points
                scroll_filter = None
                if namespace:
                    scroll_filter = Filter(
                        must=[
                            FieldCondition(
                                key="domain",
                                match=MatchAny(any=[namespace]),
                            )
                        ]
                    )
                records, _next_offset = await self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=effective_limit + offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points = records[offset:]
                return [
                    {
                        "id": str(p.id),
                        "score": None,
                        "text": p.payload.get("text", "") if p.payload else "",
                        "domain": p.payload.get("domain", "") if p.payload else "",
                        "tags": p.payload.get("tags", []) if p.payload else [],
                        "timestamp": p.payload.get("timestamp", "") if p.payload else "",
                        "source": p.payload.get("source", "") if p.payload else "",
                    }
                    for p in points
                ]
        except Exception as exc:
            logger.error("Failed to list memories: %s", exc)
            return []

    async def upsert(self, text: str, metadata: dict[str, Any], namespace: str = "default") -> str:
        """Embed text and upsert into Qdrant.

        Args:
            text: The text content to embed and store.
            metadata: Must include 'source', 'tags', 'domain'.
                      Additional keys are stored under 'metadata'.
            namespace: Tenant namespace for multi-tenant isolation.

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
                "namespace": namespace,
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
        namespace: str = "default",
    ) -> list[dict[str, Any]]:
        """Embed query and search Qdrant for similar vectors.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            filter_tags: Optional list of tags to filter on (match any).
            namespace: Tenant namespace for multi-tenant filtering.

        Returns:
            List of dicts with id, score, text, and metadata.
        """
        try:
            vector = await self._embed(query)

            filter_conditions = []
            if filter_tags:
                filter_conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchAny(any=filter_tags),
                    )
                )
            if namespace != "default":
                filter_conditions.append(
                    FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    )
                )
            query_filter = Filter(must=filter_conditions) if filter_conditions else None

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

    async def batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts, using cache and batching.

        Checks the embedding cache first, then sends uncached texts to the
        embedding API in chunks of up to 32. Falls back to individual _embed
        calls if the batch request fails.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors in the same order as the input texts.
        """
        if not texts:
            return []

        # Map each input position to its cache key and check for hits
        cache_keys = [
            hashlib.sha256(t.encode()).hexdigest() for t in texts
        ]
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, key in enumerate(cache_keys):
            if key in self._embed_cache:
                self._embed_cache.move_to_end(key)
                results[i] = self._embed_cache[key]
                logger.debug("batch_embed cache hit for index %d", i)
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return results  # type: ignore[return-value]

        # Batch embed uncached texts in chunks
        uncached_texts = [texts[i] for i in uncached_indices]
        try:
            uncached_embeddings = await self._batch_embed_api(uncached_texts)
        except Exception:
            logger.warning(
                "Batch embed API failed, falling back to individual embeds"
            )
            uncached_embeddings = []
            for text in uncached_texts:
                uncached_embeddings.append(await self._embed(text))

        # Store results and update cache
        for idx, embedding in zip(uncached_indices, uncached_embeddings):
            results[idx] = embedding
            self._cache_put(cache_keys[idx], embedding)

        return results  # type: ignore[return-value]

    async def _batch_embed_api(self, texts: list[str]) -> list[list[float]]:
        """Send texts to the embedding API in chunks, return ordered results."""
        url = f"{self._llm_base_url}/embeddings"
        headers = {}
        if self._llm_api_key:
            headers["Authorization"] = f"Bearer {self._llm_api_key}"

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_CHUNK_SIZE):
            chunk = texts[start : start + _BATCH_CHUNK_SIZE]
            try:
                response = await self._http_client.post(
                    url,
                    json={
                        "model": self._embedding_model,
                        "input": chunk,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                # Sort by index to ensure correct ordering
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                all_embeddings.extend(item["embedding"] for item in sorted_data)
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                raise VectorStoreError(
                    f"Batch embedding request failed: {exc}"
                ) from exc
            except (KeyError, IndexError, TypeError) as exc:
                raise VectorStoreError(
                    f"Failed to parse batch embedding response: {exc}"
                ) from exc

        return all_embeddings

    async def _embed(self, text: str) -> list[float]:
        """Generate an embedding vector via the LLM embeddings endpoint.

        Calls POST {LLM_BASE_URL}/embeddings with the configured model.
        Results are cached by content hash (SHA-256) with LRU eviction.
        """
        cache_key = hashlib.sha256(text.encode()).hexdigest()

        # Check cache
        if cache_key in self._embed_cache:
            self._embed_cache.move_to_end(cache_key)
            logger.debug("Embedding cache hit for text hash %s", cache_key[:12])
            return self._embed_cache[cache_key]

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
            embedding = data["data"][0]["embedding"]
        except httpx.HTTPStatusError as exc:
            raise VectorStoreError(
                f"Embedding endpoint returned {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc
        except (httpx.RequestError, KeyError, IndexError) as exc:
            raise VectorStoreError(
                f"Failed to generate embedding: {exc}"
            ) from exc

        # Store in cache
        self._cache_put(cache_key, embedding)

        return embedding

    async def set_feedback(
        self,
        memory_id: str,
        useful: bool,
        comment: str | None,
        timestamp: str,
    ) -> None:
        """Update a memory point's payload with feedback metadata.

        Args:
            memory_id: The Qdrant point ID to update.
            useful: Whether the memory was useful.
            comment: Optional feedback comment.
            timestamp: ISO-format timestamp of the feedback.
        """
        await self._client.set_payload(
            collection_name=self._collection,
            payload={
                "feedback_useful": useful,
                "feedback_comment": comment,
                "feedback_timestamp": timestamp,
            },
            points=[memory_id],
        )

    async def get_embedding_info(self) -> dict:
        """Return embedding model info: model name, dimensions, cache size, total vectors."""
        collection_info = await self._client.get_collection(self._collection)
        return {
            "model": self._embedding_model,
            "dimensions": self._embedding_dim,
            "cache_size": len(self._embed_cache),
            "cache_max_size": _EMBED_CACHE_MAX_SIZE,
            "total_vectors": collection_info.points_count,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embed_cache.clear()

    def _cache_put(self, key: str, value: list[float]) -> None:
        """Insert into the embedding cache with LRU eviction."""
        if key in self._embed_cache:
            self._embed_cache.move_to_end(key)
            return
        if len(self._embed_cache) >= _EMBED_CACHE_MAX_SIZE:
            self._embed_cache.popitem(last=False)
        self._embed_cache[key] = value
