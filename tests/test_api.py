"""Tests for FastAPI API endpoints and exception handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.exceptions import (
    GraphConnectionError,
    LLMExtractionError,
    StreamIngestionError,
    VectorStoreError,
)
from app.models import RecallResponse, MemorySource


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, test_client, mock_graph, mock_vector, mock_redis):
        """Health endpoint with all services connected returns 'ok'."""
        # Mock the service probes used by the health endpoint
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_graph._ensure_driver = MagicMock(return_value=mock_driver)

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock()
        mock_vector._client = mock_qdrant

        mock_redis.ping = AsyncMock()

        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "services" in body
        assert body["services"]["graph"]["status"] == "connected"
        assert body["services"]["qdrant"]["status"] == "connected"
        assert body["services"]["redis"]["status"] == "connected"

    def test_health_degraded_when_graph_down(self, test_client, mock_graph, mock_vector, mock_redis):
        """Health returns 'degraded' when Neo4j is unreachable."""
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = AsyncMock(side_effect=RuntimeError("Neo4j down"))
        mock_graph._ensure_driver = MagicMock(return_value=mock_driver)

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock()
        mock_vector._client = mock_qdrant

        mock_redis.ping = AsyncMock()

        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["services"]["graph"]["status"] == "disconnected"
        assert body["services"]["qdrant"]["status"] == "connected"
        assert body["services"]["redis"]["status"] == "connected"

    def test_health_degraded_when_qdrant_down(self, test_client, mock_graph, mock_vector, mock_redis):
        """Health returns 'degraded' when Qdrant is unreachable."""
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_graph._ensure_driver = MagicMock(return_value=mock_driver)

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(side_effect=RuntimeError("Qdrant down"))
        mock_vector._client = mock_qdrant

        mock_redis.ping = AsyncMock()

        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["services"]["qdrant"]["status"] == "disconnected"

    def test_health_degraded_when_redis_down(self, test_client, mock_graph, mock_vector, mock_redis):
        """Health returns 'degraded' when Redis is unreachable."""
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_graph._ensure_driver = MagicMock(return_value=mock_driver)

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock()
        mock_vector._client = mock_qdrant

        mock_redis.ping = AsyncMock(side_effect=RuntimeError("Redis down"))

        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["services"]["redis"]["status"] == "disconnected"

    def test_health_degraded_includes_error_detail(self, test_client, mock_graph, mock_vector, mock_redis):
        """Disconnected services include generic detail (no internal info leak)."""
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_graph._ensure_driver = MagicMock(return_value=mock_driver)

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock()
        mock_vector._client = mock_qdrant

        mock_redis.ping = AsyncMock()

        resp = test_client.get("/health")
        body = resp.json()
        assert body["services"]["graph"]["detail"] is not None
        # Detail should be generic — no internal host/port/driver info leaked
        assert body["services"]["graph"]["detail"] == "Service unreachable"


# ---------------------------------------------------------------------------
# POST /memory/recall
# ---------------------------------------------------------------------------


class TestMemoryRecall:
    def test_recall_returns_recall_response(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.query_related.return_value = [
            {
                "name": "auth",
                "description": "Authentication module",
                "label": "Concept",
                "distance": 1,
            }
        ]
        mock_vector.search.return_value = [
            {
                "id": "v1",
                "score": 0.85,
                "text": "Fix login timeout",
                "metadata": {"source": "action_log", "tags": ["auth"], "domain": "general"},
            }
        ]

        resp = test_client.post(
            "/memory/recall",
            json={"task": "Fix auth bug", "tags": ["auth"], "top_k": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "context_block" in body
        assert "sources" in body
        assert "score" in body
        assert isinstance(body["sources"], list)

    def test_recall_empty_stores(self, test_client, mock_graph, mock_vector):
        """Both stores return nothing -- should still produce a valid response."""
        mock_graph.query_related.return_value = []
        mock_vector.search.return_value = []

        resp = test_client.post(
            "/memory/recall", json={"task": "unknown topic"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"] == []
        assert body["score"] == 0.0
        assert "No relevant memories found" in body["context_block"]

    def test_recall_missing_task_field(self, test_client):
        resp = test_client.post("/memory/recall", json={})
        assert resp.status_code == 422  # validation error


# ---------------------------------------------------------------------------
# POST /memory/learn
# ---------------------------------------------------------------------------


class TestMemoryLearn:
    def test_learn_stores_and_returns_ids(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.merge_action_log.return_value = "graph-id-1"
        mock_vector.upsert.return_value = "vector-id-1"

        resp = test_client.post(
            "/memory/learn",
            json={
                "action": "Increased pool size",
                "outcome": "Timeout resolved",
                "resolution": "Updated config",
                "tags": ["db"],
                "domain": "infra",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "stored"
        assert body["graph_id"] == "graph-id-1"
        assert body["vector_id"] == "vector-id-1"

    def test_learn_without_resolution(
        self, test_client, mock_graph, mock_vector
    ):
        resp = test_client.post(
            "/memory/learn",
            json={"action": "a", "outcome": "o"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "stored"

    def test_learn_constructs_correct_text_for_vector(
        self, test_client, mock_graph, mock_vector
    ):
        """Verify the text sent to vector.upsert includes action|outcome|resolution."""
        mock_graph.merge_action_log.return_value = "id"
        mock_vector.upsert.return_value = "vid"

        test_client.post(
            "/memory/learn",
            json={
                "action": "Rebuild index",
                "outcome": "Search speed improved",
                "resolution": "Ran REINDEX",
                "tags": ["search"],
                "domain": "db",
            },
        )

        call_args = mock_vector.upsert.call_args
        text_arg = call_args.kwargs.get("text") or call_args[1].get("text") or call_args[0][0]
        assert "Rebuild index" in text_arg
        assert "Search speed improved" in text_arg
        assert "Resolution: Ran REINDEX" in text_arg

    def test_learn_missing_required_fields(self, test_client):
        resp = test_client.post("/memory/learn", json={"action": "only action"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /memory/stream
# ---------------------------------------------------------------------------


class TestMemoryStream:
    def test_stream_single_event(self, test_client, mock_redis):
        resp = test_client.post(
            "/memory/stream",
            json={
                "source": "ci",
                "payload": {"build": "123", "status": "failed"},
                "tags": ["ci"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert body["queued"] == 1
        mock_redis.pipeline.assert_called_once()
        mock_redis._pipeline.lpush.assert_called_once()
        mock_redis._pipeline.execute.assert_called_once()

    def test_stream_batch_events(self, test_client, mock_redis):
        events = [
            {"source": "ci", "payload": {"id": "1"}, "tags": ["a"]},
            {"source": "ci", "payload": {"id": "2"}, "tags": ["b"]},
            {"source": "ci", "payload": {"id": "3"}, "tags": ["c"]},
        ]
        resp = test_client.post("/memory/stream", json=events)
        assert resp.status_code == 200
        body = resp.json()
        assert body["queued"] == 3
        assert mock_redis._pipeline.lpush.call_count == 3

    def test_stream_batch_exceeds_max_returns_422(self, test_client, mock_redis):
        """Sending more than MAX_BATCH_SIZE (100) events returns 422."""
        events = [
            {"source": "ci", "payload": {"id": str(i)}}
            for i in range(101)
        ]
        resp = test_client.post("/memory/stream", json=events)
        assert resp.status_code == 422
        assert "Batch size exceeds maximum" in resp.json()["detail"]

    def test_stream_redis_failure(self, test_client, mock_redis):
        """Redis pipeline failure should surface as a 502."""
        mock_redis._pipeline.execute = AsyncMock(side_effect=Exception("connection refused"))

        resp = test_client.post(
            "/memory/stream",
            json={"source": "ci", "payload": {"x": 1}},
        )
        assert resp.status_code == 502

    def test_stream_missing_payload(self, test_client):
        resp = test_client.post(
            "/memory/stream", json={"source": "ci"}
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Exception handler mapping
# ---------------------------------------------------------------------------


class TestExceptionHandlers:
    def test_graph_error_on_recall_graceful_degradation(
        self, test_client, mock_graph, mock_vector
    ):
        """When graph fails during recall, RAG engine catches it and returns
        vector-only results (200, not 503)."""
        mock_graph.query_related.side_effect = GraphConnectionError("down")
        mock_vector.search.return_value = [
            {
                "id": "v1",
                "score": 0.7,
                "text": "Vector result survives graph failure",
                "metadata": {"source": "action_log", "tags": [], "domain": "general"},
            }
        ]

        resp = test_client.post(
            "/memory/recall", json={"task": "test graceful degradation"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["sources"]) == 1
        assert body["sources"][0]["store"] == "vector"

    def test_graph_error_on_learn_returns_503(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.merge_action_log.side_effect = GraphConnectionError(
            "Neo4j unreachable"
        )
        resp = test_client.post(
            "/memory/learn",
            json={"action": "a", "outcome": "o"},
        )
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Knowledge graph service unavailable"

    def test_vector_error_on_learn_returns_502(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.merge_action_log.return_value = "id"
        mock_vector.upsert.side_effect = VectorStoreError("Qdrant down")

        resp = test_client.post(
            "/memory/learn",
            json={"action": "a", "outcome": "o"},
        )
        assert resp.status_code == 502
        assert resp.json()["detail"] == "Vector store service error"

    def test_stream_ingestion_error_returns_502(
        self, test_client, mock_redis
    ):
        mock_redis._pipeline.execute = AsyncMock(side_effect=Exception("Redis down"))
        resp = test_client.post(
            "/memory/stream",
            json={"source": "s", "payload": {"k": "v"}},
        )
        assert resp.status_code == 502
