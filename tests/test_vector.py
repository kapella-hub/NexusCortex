"""Tests for VectorClient (app.db.vector)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.db.vector import NEXUS_UUID_NAMESPACE, VectorClient
from app.exceptions import VectorStoreError


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        QDRANT_HOST="localhost",
        QDRANT_PORT=6333,
        QDRANT_COLLECTION="test_collection",
        EMBEDDING_DIM=768,
        LLM_BASE_URL="http://localhost:11434/v1",
        LLM_API_KEY="test-api-key",
        EMBEDDING_MODEL="test-embed",
    )


@pytest.fixture()
def mock_qdrant_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def mock_http_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def vector_client(settings: Settings, mock_qdrant_client: AsyncMock, mock_http_client: AsyncMock) -> VectorClient:
    client = VectorClient(settings)
    client._client = mock_qdrant_client
    client._http_client = mock_http_client
    return client


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_creates_collection_when_missing(
        self, vector_client, mock_qdrant_client
    ):
        # Simulate no existing collection
        collections_resp = MagicMock()
        collections_resp.collections = []
        mock_qdrant_client.get_collections.return_value = collections_resp

        await vector_client.initialize()

        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_skips_creation_when_exists(
        self, vector_client, mock_qdrant_client
    ):
        existing = MagicMock()
        existing.name = "test_collection"
        collections_resp = MagicMock()
        collections_resp.collections = [existing]
        mock_qdrant_client.get_collections.return_value = collections_resp

        await vector_client.initialize()

        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_error_raises_vector_store_error(
        self, vector_client, mock_qdrant_client
    ):
        mock_qdrant_client.get_collections.side_effect = RuntimeError("down")

        with pytest.raises(VectorStoreError, match="Failed to initialize"):
            await vector_client.initialize()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_closes_both_clients(self, vector_client, mock_qdrant_client, mock_http_client):
        """close() should close both the httpx client and the qdrant client."""
        await vector_client.close()

        mock_http_client.aclose.assert_called_once()
        mock_qdrant_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


class TestUpsert:
    @pytest.mark.asyncio
    async def test_upsert_embeds_and_stores(self, vector_client, mock_qdrant_client):
        embed_vector = [0.1] * 768

        with patch.object(
            vector_client, "_embed", new_callable=AsyncMock, return_value=embed_vector
        ):
            point_id = await vector_client.upsert(
                text="Fix auth bug",
                metadata={
                    "source": "action_log",
                    "tags": ["auth"],
                    "domain": "security",
                },
            )

        assert isinstance(point_id, str)
        assert len(point_id) == 36  # UUID format

        mock_qdrant_client.upsert.assert_called_once()
        call_kwargs = mock_qdrant_client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "test_collection"

        points = call_kwargs.kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["text"] == "Fix auth bug"
        assert points[0].payload["source"] == "action_log"
        assert points[0].payload["tags"] == ["auth"]
        assert points[0].payload["domain"] == "security"

    @pytest.mark.asyncio
    async def test_upsert_uses_deterministic_uuid(self, vector_client, mock_qdrant_client):
        """Same text should produce the same point ID (uuid5)."""
        embed_vector = [0.1] * 768

        with patch.object(
            vector_client, "_embed", new_callable=AsyncMock, return_value=embed_vector
        ):
            id1 = await vector_client.upsert("same text", {"source": "test"})
            id2 = await vector_client.upsert("same text", {"source": "test"})

        assert id1 == id2
        expected = str(uuid.uuid5(NEXUS_UUID_NAMESPACE, "same text"))
        assert id1 == expected

    @pytest.mark.asyncio
    async def test_upsert_embedding_failure(self, vector_client):
        with patch.object(
            vector_client,
            "_embed",
            new_callable=AsyncMock,
            side_effect=VectorStoreError("embed failed"),
        ):
            with pytest.raises(VectorStoreError, match="embed failed"):
                await vector_client.upsert("text", {"source": "test"})

    @pytest.mark.asyncio
    async def test_upsert_qdrant_failure(self, vector_client, mock_qdrant_client):
        with patch.object(
            vector_client, "_embed", new_callable=AsyncMock, return_value=[0.1] * 768
        ):
            mock_qdrant_client.upsert.side_effect = RuntimeError("qdrant down")
            with pytest.raises(VectorStoreError, match="Failed to upsert"):
                await vector_client.upsert("text", {"source": "test"})


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_without_filters(self, vector_client, mock_qdrant_client):
        point = MagicMock()
        point.id = "p1"
        point.score = 0.92
        point.payload = {
            "text": "result text",
            "source": "action_log",
            "tags": ["auth"],
            "domain": "security",
            "timestamp": "2025-01-01T00:00:00Z",
            "metadata": {"extra": "val"},
        }
        result_obj = MagicMock()
        result_obj.points = [point]
        mock_qdrant_client.query_points.return_value = result_obj

        with patch.object(
            vector_client, "_embed", new_callable=AsyncMock, return_value=[0.1] * 768
        ):
            results = await vector_client.search("auth bug", top_k=3)

        assert len(results) == 1
        assert results[0]["id"] == "p1"
        assert results[0]["score"] == 0.92
        assert results[0]["text"] == "result text"
        assert results[0]["metadata"]["source"] == "action_log"
        assert results[0]["metadata"]["extra"] == "val"

        # Verify no filter was passed
        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None
        assert call_kwargs["limit"] == 3

    @pytest.mark.asyncio
    async def test_search_with_tag_filter(self, vector_client, mock_qdrant_client):
        result_obj = MagicMock()
        result_obj.points = []
        mock_qdrant_client.query_points.return_value = result_obj

        with patch.object(
            vector_client, "_embed", new_callable=AsyncMock, return_value=[0.1] * 768
        ):
            await vector_client.search(
                "auth bug", top_k=5, filter_tags=["auth", "login"]
            )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        query_filter = call_kwargs["query_filter"]
        assert query_filter is not None
        # Filter should have a FieldCondition on "tags"
        assert len(query_filter.must) == 1
        assert query_filter.must[0].key == "tags"

    @pytest.mark.asyncio
    async def test_search_failure(self, vector_client, mock_qdrant_client):
        with patch.object(
            vector_client, "_embed", new_callable=AsyncMock, return_value=[0.1] * 768
        ):
            mock_qdrant_client.query_points.side_effect = RuntimeError("fail")
            with pytest.raises(VectorStoreError, match="Failed to search"):
                await vector_client.search("query")

    @pytest.mark.asyncio
    async def test_search_with_empty_payload(self, vector_client, mock_qdrant_client):
        """Points with None payload should produce empty metadata dicts."""
        point = MagicMock()
        point.id = "p2"
        point.score = 0.5
        point.payload = None
        result_obj = MagicMock()
        result_obj.points = [point]
        mock_qdrant_client.query_points.return_value = result_obj

        with patch.object(
            vector_client, "_embed", new_callable=AsyncMock, return_value=[0.1] * 768
        ):
            results = await vector_client.search("q")

        assert results[0]["text"] == ""
        assert results[0]["metadata"] == {}


# ---------------------------------------------------------------------------
# _embed
# ---------------------------------------------------------------------------


class TestEmbed:
    @pytest.mark.asyncio
    async def test_correct_api_call_format(self, vector_client, mock_http_client):
        """Verify _embed calls the persistent _http_client with correct params."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)

        result = await vector_client._embed("test text")

        assert result == [0.1, 0.2, 0.3]

        # Verify the API call
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        url = call_args[0][0]
        assert "embeddings" in url
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert json_body["model"] == "test-embed"
        assert json_body["input"] == "test text"

    @pytest.mark.asyncio
    async def test_embed_sends_auth_header(self, vector_client, mock_http_client):
        """Verify _embed sends Authorization header when API key is set."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1]}]
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)

        await vector_client._embed("test")

        call_args = mock_http_client.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-api-key"

    @pytest.mark.asyncio
    async def test_embed_http_error(self, vector_client, mock_http_client):
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        exc = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )
        mock_response.raise_for_status.side_effect = exc
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(VectorStoreError, match="Embedding endpoint returned"):
            await vector_client._embed("test")

    @pytest.mark.asyncio
    async def test_embed_malformed_response(self, vector_client, mock_http_client):
        """Missing 'data' key in response should raise VectorStoreError."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(VectorStoreError, match="Failed to generate embedding"):
            await vector_client._embed("test")

    @pytest.mark.asyncio
    async def test_embed_no_auth_header_when_key_empty(self, settings, mock_qdrant_client, mock_http_client):
        """When LLM_API_KEY is empty, no Authorization header should be sent."""
        settings_no_key = Settings(
            QDRANT_HOST="localhost",
            QDRANT_PORT=6333,
            QDRANT_COLLECTION="test_collection",
            EMBEDDING_DIM=768,
            LLM_BASE_URL="http://localhost:11434/v1",
            LLM_API_KEY="",
            EMBEDDING_MODEL="test-embed",
        )
        client = VectorClient(settings_no_key)
        client._client = mock_qdrant_client
        client._http_client = mock_http_client

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1]}]
        }
        mock_http_client.post = AsyncMock(return_value=mock_response)

        await client._embed("test")

        call_args = mock_http_client.post.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "Authorization" not in headers
