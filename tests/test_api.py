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
        mock_graph.ping = AsyncMock()
        mock_vector.ping = AsyncMock()
        mock_vector.memory_count = AsyncMock(return_value=42)
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
        mock_graph.ping = AsyncMock(side_effect=RuntimeError("Neo4j down"))
        mock_vector.ping = AsyncMock()
        mock_vector.memory_count = AsyncMock(return_value=None)
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
        mock_graph.ping = AsyncMock()
        mock_vector.ping = AsyncMock(side_effect=RuntimeError("Qdrant down"))
        mock_vector.memory_count = AsyncMock(return_value=None)
        mock_redis.ping = AsyncMock()

        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["services"]["qdrant"]["status"] == "disconnected"

    def test_health_degraded_when_redis_down(self, test_client, mock_graph, mock_vector, mock_redis):
        """Health returns 'degraded' when Redis is unreachable."""
        mock_graph.ping = AsyncMock()
        mock_vector.ping = AsyncMock()
        mock_vector.memory_count = AsyncMock(return_value=None)
        mock_redis.ping = AsyncMock(side_effect=RuntimeError("Redis down"))

        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["services"]["redis"]["status"] == "disconnected"

    def test_health_caches_response(self, test_client, mock_graph, mock_vector, mock_redis):
        """Second health call within TTL should not re-probe backends."""
        mock_graph.ping = AsyncMock()
        mock_vector.ping = AsyncMock()
        mock_vector.memory_count = AsyncMock(return_value=42)
        mock_redis.ping = AsyncMock()

        # First call — probes everything
        resp1 = test_client.get("/health")
        assert resp1.status_code == 200

        # Reset mocks to track second call
        mock_graph.ping.reset_mock()
        mock_vector.ping.reset_mock()
        mock_vector.memory_count.reset_mock()
        mock_redis.ping.reset_mock()

        # Second call — should use cache, no backend calls
        resp2 = test_client.get("/health")
        assert resp2.status_code == 200
        assert resp2.json() == resp1.json()

        mock_graph.ping.assert_not_called()
        mock_vector.ping.assert_not_called()
        mock_redis.ping.assert_not_called()

    def test_health_cache_expires(self, test_client, mock_graph, mock_vector, mock_redis):
        """Health cache should expire after TTL."""
        from datetime import datetime, timezone, timedelta

        mock_graph.ping = AsyncMock()
        mock_vector.ping = AsyncMock()
        mock_vector.memory_count = AsyncMock(return_value=42)
        mock_redis.ping = AsyncMock()

        # First call
        test_client.get("/health")

        # Simulate time passing beyond TTL
        from app import main as main_module
        main_module._health_cache_time = datetime.now(timezone.utc) - timedelta(seconds=30)

        mock_graph.ping.reset_mock()

        # Call after cache expired — should re-probe
        resp = test_client.get("/health")
        assert resp.status_code == 200
        mock_graph.ping.assert_called_once()

    def test_health_degraded_includes_error_detail(self, test_client, mock_graph, mock_vector, mock_redis):
        """Disconnected services include generic detail (no internal info leak)."""
        mock_graph.ping = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_vector.ping = AsyncMock()
        mock_vector.memory_count = AsyncMock(return_value=None)
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

    def test_learn_graph_failure_returns_partial(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.merge_action_log.side_effect = RuntimeError("neo4j down")
        mock_vector.upsert.return_value = "vector-id-1"

        resp = test_client.post(
            "/memory/learn",
            json={"action": "test", "outcome": "test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "partial"
        assert body["graph_id"] is None
        assert body["vector_id"] == "vector-id-1"

    def test_learn_vector_failure_returns_partial(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.merge_action_log.return_value = "graph-id-1"
        mock_vector.upsert.side_effect = RuntimeError("qdrant down")

        resp = test_client.post(
            "/memory/learn",
            json={"action": "test", "outcome": "test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "partial"
        assert body["graph_id"] == "graph-id-1"
        assert body["vector_id"] is None

    def test_learn_both_fail_returns_503(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.merge_action_log.side_effect = RuntimeError("neo4j down")
        mock_vector.upsert.side_effect = RuntimeError("qdrant down")

        resp = test_client.post(
            "/memory/learn",
            json={"action": "test", "outcome": "test"},
        )
        assert resp.status_code == 503


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

    def test_graph_error_on_learn_returns_partial(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.merge_action_log.side_effect = GraphConnectionError(
            "Neo4j unreachable"
        )
        resp = test_client.post(
            "/memory/learn",
            json={"action": "a", "outcome": "o"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "partial"
        assert body["graph_id"] is None

    def test_vector_error_on_learn_returns_partial(
        self, test_client, mock_graph, mock_vector
    ):
        mock_graph.merge_action_log.return_value = "id"
        mock_vector.upsert.side_effect = VectorStoreError("Qdrant down")

        resp = test_client.post(
            "/memory/learn",
            json={"action": "a", "outcome": "o"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "partial"
        assert body["vector_id"] is None

    def test_stream_ingestion_error_returns_502(
        self, test_client, mock_redis
    ):
        mock_redis._pipeline.execute = AsyncMock(side_effect=Exception("Redis down"))
        resp = test_client.post(
            "/memory/stream",
            json={"source": "s", "payload": {"k": "v"}},
        )
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /memory/feedback
# ---------------------------------------------------------------------------


class TestMemoryFeedback:
    def test_feedback_success(self, test_client, mock_vector):
        mock_vector.set_feedback = AsyncMock()

        resp = test_client.post(
            "/memory/feedback",
            json={"memory_ids": ["id-1", "id-2"], "useful": True, "comment": "helpful"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["updated"] == 2
        assert mock_vector.set_feedback.call_count == 2

    def test_feedback_partial_failure(self, test_client, mock_vector):
        call_count = 0

        async def set_feedback_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Qdrant error")

        mock_vector.set_feedback = AsyncMock(side_effect=set_feedback_side_effect)

        resp = test_client.post(
            "/memory/feedback",
            json={"memory_ids": ["id-1", "id-2", "id-3"], "useful": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["updated"] == 2  # 1 of 3 failed

    def test_feedback_empty_ids_rejected(self, test_client):
        resp = test_client.post(
            "/memory/feedback",
            json={"memory_ids": [], "useful": True},
        )
        assert resp.status_code == 422

    def test_feedback_missing_useful_rejected(self, test_client):
        resp = test_client.post(
            "/memory/feedback",
            json={"memory_ids": ["id-1"]},
        )
        assert resp.status_code == 422
