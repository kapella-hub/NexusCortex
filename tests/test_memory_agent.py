"""Tests for Memory Agent worker (app.workers.memory_agent)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from app.workers.memory_agent import (
    AGENT_LOCK_KEY,
    _cosine_similarity,
    _fire_webhook_sync,
    _has_supersedes_link,
    _merge_cluster,
    backlink_reinforcement_pass,
    cluster_coherence_pass,
    confidence_decay_pass,
    deep_contradiction_pass,
    duplicate_detection_pass,
    orphan_cleanup_pass,
    run_memory_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    """Create a mock settings object with agent defaults."""
    defaults = {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": 6333,
        "QDRANT_COLLECTION": "nexus_memories",
        "REDIS_URL": "redis://localhost:6379",
        "REDIS_STREAM_KEY": "nexus:event_stream",
        "LLM_BASE_URL": "http://localhost:11434/v1",
        "LLM_MODEL": "test-model",
        "LLM_API_KEY": "test-key",
        "AGENT_ENABLED": True,
        "AGENT_SCHEDULE_HOURS": 6,
        "AGENT_DUPLICATE_THRESHOLD": 0.9,
        "AGENT_CONFIDENCE_DECAY_DAYS": 180,
        "AGENT_BATCH_LIMIT": 100,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_point(pid, payload=None, vector=None, score=None):
    """Create a mock Qdrant point."""
    p = MagicMock()
    p.id = pid
    p.payload = payload or {}
    p.vector = vector or [0.1, 0.2, 0.3]
    if score is not None:
        p.score = score
    return p


def _make_query_result(points):
    """Create a mock QueryResponse with .points attribute."""
    result = MagicMock()
    result.points = points
    return result


# ---------------------------------------------------------------------------
# Test 1: Duplicate Detection & Merge
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_neo4j_driver")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_duplicate_detection_llm_merge(
        self, mock_settings, mock_qdrant, mock_neo4j, mock_webhook
    ):
        """LLM path: similar memories merged via LLM synthesis."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        # Two similar memories
        p1 = _make_point("id-1", {"text": "Use Postgres for storage", "status": "active",
                                    "domain": "db", "tags": [], "confirmed_count": 1,
                                    "contradicted_count": 0}, [0.9, 0.1, 0.0])
        p2 = _make_point("id-2", {"text": "Postgres is the storage backend", "status": "active",
                                    "domain": "db", "tags": [], "confirmed_count": 0,
                                    "contradicted_count": 0}, [0.89, 0.11, 0.01])

        # scroll returns both
        client.scroll.return_value = ([p1, p2], None)

        # query_points for p1 finds p2 as duplicate (score > 0.9)
        sp2 = _make_point("id-2", score=0.95)
        client.query_points.return_value = _make_query_result([sp2])

        # LLM merge succeeds
        with patch("app.workers.memory_agent.httpx.post") as mock_httpx:
            llm_resp = MagicMock()
            llm_resp.raise_for_status = MagicMock()
            llm_resp.json.return_value = {
                "choices": [{"message": {"content": json.dumps({
                    "text": "Use Postgres as the primary storage backend",
                    "domain": "db", "tags": ["postgres"]
                })}}]
            }
            mock_httpx.return_value = llm_resp

            # Neo4j session mock
            mock_session = MagicMock()
            mock_neo4j.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_neo4j.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

            result = duplicate_detection_pass()

        assert result["status"] == "ok"
        assert result["merged"] == 1
        assert result["details"][0]["method"] == "llm"
        assert result["details"][0]["merged_into"] == "id-1"
        assert result["details"][0]["superseded"] == ["id-2"]

        # Verify superseded memory status updated in Qdrant
        client.set_payload.assert_any_call(
            collection_name="nexus_memories",
            payload={"status": "superseded", "superseded_by": "id-1"},
            points=["id-2"],
        )

        # Verify keeper text updated with LLM output
        client.set_payload.assert_any_call(
            collection_name="nexus_memories",
            payload={"text": "Use Postgres as the primary storage backend"},
            points=["id-1"],
        )

        # Verify SUPERSEDES edge created
        mock_session.run.assert_called()

        # Verify webhook fired
        mock_webhook.assert_called_with(
            settings.REDIS_URL,
            "agent.merged",
            {"merged_into": "id-1", "superseded": ["id-2"], "method": "llm"},
        )

    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_neo4j_driver")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_duplicate_detection_fallback(
        self, mock_settings, mock_qdrant, mock_neo4j, mock_webhook
    ):
        """Fallback path: LLM fails, highest confidence memory kept."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        p1 = _make_point("id-1", {"text": "Short", "status": "active",
                                    "domain": "db", "tags": [],
                                    "confirmed_count": 0, "contradicted_count": 0},
                         [0.9, 0.1, 0.0])
        p2 = _make_point("id-2", {"text": "Better memory with more detail",
                                    "status": "active", "domain": "db", "tags": [],
                                    "confirmed_count": 3, "contradicted_count": 0},
                         [0.89, 0.11, 0.01])

        client.scroll.return_value = ([p1, p2], None)

        sp2 = _make_point("id-2", score=0.95)
        client.query_points.return_value = _make_query_result([sp2])

        with patch("app.workers.memory_agent.httpx.post") as mock_httpx:
            mock_httpx.side_effect = Exception("LLM down")

            mock_session = MagicMock()
            mock_neo4j.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_neo4j.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

            result = duplicate_detection_pass()

        assert result["merged"] == 1
        assert result["details"][0]["method"] == "fallback"
        # Fallback keeps highest confidence memory (id-2, confirmed_count=3)
        assert result["details"][0]["merged_into"] == "id-2"
        assert result["details"][0]["superseded"] == ["id-1"]

        # Verify the lower-confidence memory was superseded
        client.set_payload.assert_any_call(
            collection_name="nexus_memories",
            payload={"status": "superseded", "superseded_by": "id-2"},
            points=["id-1"],
        )

        # Verify webhook
        mock_webhook.assert_called_once()
        webhook_payload = mock_webhook.call_args[0][2]
        assert webhook_payload["method"] == "fallback"
        assert webhook_payload["merged_into"] == "id-2"


# ---------------------------------------------------------------------------
# Test 2: Orphan Cleanup
# ---------------------------------------------------------------------------


class TestOrphanCleanup:
    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_neo4j_driver")
    @patch("app.workers.memory_agent.get_settings")
    def test_orphan_cleanup(self, mock_settings, mock_neo4j, mock_webhook):
        """Orphan nodes (degree-0) are deleted and webhook fired."""
        settings = _make_settings()
        mock_settings.return_value = settings

        # Neo4j returns orphan nodes
        mock_record1 = {"label": "Concept", "name": "stale_concept"}
        mock_record2 = {"label": "Action", "name": "orphan_action"}
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([mock_record1, mock_record2]))

        mock_tx = MagicMock()
        mock_tx.run.return_value = mock_result
        mock_tx.commit = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value.__enter__ = MagicMock(return_value=mock_tx)
        mock_session.begin_transaction.return_value.__exit__ = MagicMock(return_value=False)

        mock_neo4j.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        result = orphan_cleanup_pass()

        assert result["status"] == "ok"
        assert len(result["nodes_removed"]) == 2
        assert result["nodes_removed"][0]["label"] == "Concept"
        assert result["nodes_removed"][1]["name"] == "orphan_action"

        # Verify webhook
        mock_webhook.assert_called_once_with(
            settings.REDIS_URL,
            "agent.orphan_cleaned",
            {"nodes_removed": result["nodes_removed"]},
        )


# ---------------------------------------------------------------------------
# Test 3: Deep Contradiction Scan
# ---------------------------------------------------------------------------


class TestDeepContradiction:
    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._has_supersedes_link", return_value=False)
    @patch("app.workers.memory_agent._get_neo4j_driver")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_deep_contradiction_found(
        self, mock_settings, mock_qdrant, mock_neo4j, mock_link, mock_webhook
    ):
        """Contradictory memories in 0.85-0.95 range auto-supersede stale side."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        p1 = _make_point("mem-1", {
            "text": "Use MySQL", "status": "active", "domain": "db",
            "timestamp": "2025-01-01T00:00:00Z",
            "confirmed_count": 0, "contradicted_count": 0,
        }, [0.5, 0.5, 0.0])
        p2 = _make_point("mem-2", {
            "text": "Use Postgres instead of MySQL", "status": "active", "domain": "db",
            "timestamp": "2026-01-01T00:00:00Z",
            "confirmed_count": 2, "contradicted_count": 0,
        }, [0.5, 0.5, 0.01])

        client.scroll.return_value = ([p1, p2], None)

        # p1 finds p2 as similar with 0.90 score
        sp2 = _make_point("mem-2", score=0.90)
        # p2 finds p1 but pair already processed
        sp1 = _make_point("mem-1", score=0.90)
        client.query_points.side_effect = [
            _make_query_result([sp2]),
            _make_query_result([sp1]),
        ]

        mock_session = MagicMock()
        mock_neo4j.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        result = deep_contradiction_pass()

        assert result["status"] == "ok"
        assert result["contradictions_found"] == 1
        detail = result["details"][0]
        # mem-1 is stale (lower confidence, older), mem-2 is kept
        assert detail["kept"] == "mem-2"
        assert detail["superseded"] == "mem-1"
        assert detail["similarity"] == 0.9

        # Verify stale memory superseded in Qdrant
        client.set_payload.assert_called_with(
            collection_name="nexus_memories",
            payload={"status": "superseded", "superseded_by": "mem-2"},
            points=["mem-1"],
        )

        # Verify webhook
        mock_webhook.assert_called_once()
        wh_payload = mock_webhook.call_args[0][2]
        assert wh_payload["kept"] == "mem-2"
        assert wh_payload["superseded"] == "mem-1"


# ---------------------------------------------------------------------------
# Test 4: Backlink Reinforcement
# ---------------------------------------------------------------------------


class TestBacklinkReinforcement:
    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_neo4j_driver")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_backlink_reinforcement(
        self, mock_settings, mock_qdrant, mock_neo4j, mock_webhook
    ):
        """Memories with no BACKLINK edges get backlinks created."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        # Neo4j returns orphan vector IDs (no BACKLINK edges)
        mock_result = MagicMock()
        mock_record = {"vector_id": "vid-1"}
        mock_result.__iter__ = MagicMock(return_value=iter([mock_record]))

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result

        mock_neo4j.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        # Qdrant retrieve returns the memory with its vector
        retrieved = _make_point("vid-1", {"text": "Memory about databases"}, [0.5, 0.3, 0.2])
        client.retrieve.return_value = [retrieved]

        # Qdrant scroll returns empty (no additional orphans needed)
        client.scroll.return_value = ([], None)

        # query_points finds similar memories in backlink range
        sim1 = _make_point("vid-2", score=0.65)
        sim2 = _make_point("vid-3", score=0.50)
        sim3 = _make_point("vid-4", score=0.30)  # below 0.4, should be skipped
        client.query_points.return_value = _make_query_result([sim1, sim2, sim3])

        result = backlink_reinforcement_pass()

        assert result["status"] == "ok"
        assert result["memories_linked"] == 1
        assert result["details"][0]["memory_id"] == "vid-1"
        assert result["details"][0]["new_backlinks"] == 2  # vid-2 and vid-3, not vid-4

        # Verify BACKLINK edges created in Neo4j (2 calls for the 2 valid matches)
        # Each call creates bidirectional edges
        assert mock_session.run.call_count >= 3  # 1 for initial query + 2 for MERGE calls

        # Verify webhook
        mock_webhook.assert_called_once()
        wh_payload = mock_webhook.call_args[0][2]
        assert wh_payload["memory_id"] == "vid-1"
        assert wh_payload["new_backlinks"] == 2


# ---------------------------------------------------------------------------
# Test 5: Confidence Decay
# ---------------------------------------------------------------------------


class TestConfidenceDecay:
    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_auto_deprecate_low_confidence(self, mock_settings, mock_qdrant, mock_webhook):
        """Memories with confidence < 0.5 are auto-deprecated."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        # Old memory with low confidence: (1+0)/(1+2) = 0.33 < 0.5
        p1 = _make_point("old-1", {
            "text": "Stale fact", "status": "active",
            "timestamp": "2024-01-01T00:00:00Z",
            "confirmed_count": 0, "contradicted_count": 2,
        })

        client.scroll.return_value = ([p1], None)

        result = confidence_decay_pass()

        assert result["status"] == "ok"
        assert result["decayed"] == 1
        assert result["details"][0]["action"] == "deprecated"
        assert result["details"][0]["memory_id"] == "old-1"

        # Verify status set to deprecated
        client.set_payload.assert_called_with(
            collection_name="nexus_memories",
            payload={"status": "deprecated"},
            points=["old-1"],
        )

        # Verify webhook
        mock_webhook.assert_called_once()
        assert mock_webhook.call_args[0][1] == "agent.confidence_decayed"

    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_decrement_confirmed_count(self, mock_settings, mock_qdrant, mock_webhook):
        """Memories with confidence >= 0.5 get confirmed_count decremented."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        # Old memory with decent confidence: (1+3)/(1+0) = 4.0 >= 0.5
        p1 = _make_point("old-2", {
            "text": "Valid but old", "status": "active",
            "timestamp": "2024-01-01T00:00:00Z",
            "confirmed_count": 3, "contradicted_count": 0,
        })

        client.scroll.return_value = ([p1], None)

        result = confidence_decay_pass()

        assert result["decayed"] == 1
        assert result["details"][0]["action"] == "reduced"
        assert result["details"][0]["new_count"] == 2  # 3 - 1

        # Verify confirmed_count updated
        client.set_payload.assert_called_with(
            collection_name="nexus_memories",
            payload={"confirmed_count": 2},
            points=["old-2"],
        )

    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_decrement_floors_at_zero(self, mock_settings, mock_qdrant, mock_webhook):
        """confirmed_count=1 decrements to 0 (floor)."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        # confidence: (1+1)/(1+0) = 2.0 >= 0.5, so reduces (not deprecates)
        p1 = _make_point("old-3", {
            "text": "Old one-confirmed", "status": "active",
            "timestamp": "2024-01-01T00:00:00Z",
            "confirmed_count": 1, "contradicted_count": 0,
        })

        client.scroll.return_value = ([p1], None)

        result = confidence_decay_pass()

        assert result["details"][0]["new_count"] == 0  # max(0, 1-1)
        client.set_payload.assert_called_with(
            collection_name="nexus_memories",
            payload={"confirmed_count": 0},
            points=["old-3"],
        )

    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_skips_zero_confirmed_high_confidence(self, mock_settings, mock_qdrant, mock_webhook):
        """Memories with confirmed_count=0 and confidence >= 0.5 are skipped (no-op)."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        p1 = _make_point("old-4", {
            "text": "Zero confirmed, no contradiction", "status": "active",
            "timestamp": "2024-01-01T00:00:00Z",
            "confirmed_count": 0, "contradicted_count": 0,
        })

        client.scroll.return_value = ([p1], None)

        result = confidence_decay_pass()

        assert result["decayed"] == 0
        client.set_payload.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: Cluster Coherence
# ---------------------------------------------------------------------------


class TestClusterCoherence:
    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_neo4j_driver")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_reclassify_outlier(self, mock_settings, mock_qdrant, mock_neo4j, mock_webhook):
        """An outlier memory gets reclassified to a better-fit domain."""
        settings = _make_settings()
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        # Domain "db" has 3 memories, one is an outlier closer to "infra"
        db_vec = [1.0, 0.0, 0.0]
        db_vec2 = [0.99, 0.01, 0.0]
        db_vec3 = [0.98, 0.02, 0.0]
        outlier_vec = [0.1, 0.9, 0.0]  # very different from db cluster

        # Domain "infra" has 3 memories closer to the outlier
        infra_vec = [0.1, 0.95, 0.0]
        infra_vec2 = [0.12, 0.93, 0.0]
        infra_vec3 = [0.08, 0.97, 0.0]

        points = [
            _make_point("db-1", {"text": "DB1", "status": "active", "domain": "db"}, db_vec),
            _make_point("db-2", {"text": "DB2", "status": "active", "domain": "db"}, db_vec2),
            _make_point("db-3", {"text": "DB3", "status": "active", "domain": "db"}, db_vec3),
            _make_point("outlier", {"text": "Infra stuff", "status": "active", "domain": "db"}, outlier_vec),
            _make_point("infra-1", {"text": "I1", "status": "active", "domain": "infra"}, infra_vec),
            _make_point("infra-2", {"text": "I2", "status": "active", "domain": "infra"}, infra_vec2),
            _make_point("infra-3", {"text": "I3", "status": "active", "domain": "infra"}, infra_vec3),
        ]

        client.scroll.return_value = (points, None)

        # Neo4j mock for domain update
        mock_tx = MagicMock()
        mock_tx.run = MagicMock()
        mock_tx.commit = MagicMock()

        mock_session = MagicMock()
        mock_session.begin_transaction.return_value.__enter__ = MagicMock(return_value=mock_tx)
        mock_session.begin_transaction.return_value.__exit__ = MagicMock(return_value=False)

        mock_neo4j.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

        result = cluster_coherence_pass()

        assert result["status"] == "ok"
        assert result["reclassified"] >= 1

        # Find the outlier reclassification
        reclassified = [r for r in result["details"] if r["memory_id"] == "outlier"]
        assert len(reclassified) == 1
        assert reclassified[0]["from_domain"] == "db"
        assert reclassified[0]["to_domain"] == "infra"

        # Verify Qdrant domain updated
        client.set_payload.assert_any_call(
            collection_name="nexus_memories",
            payload={"domain": "infra"},
            points=["outlier"],
        )

        # Verify webhook fired
        mock_webhook.assert_called()
        wh_calls = [c for c in mock_webhook.call_args_list
                     if c[0][1] == "agent.reclassified"]
        assert len(wh_calls) >= 1


# ---------------------------------------------------------------------------
# Test 7: Lock Guard
# ---------------------------------------------------------------------------


class TestLockGuard:
    @patch("app.workers.memory_agent.get_settings")
    @patch("app.workers.memory_agent._get_redis_client")
    def test_lock_prevents_concurrent_runs(self, mock_get_redis, mock_settings):
        """Redis SETNX lock prevents a second concurrent run."""
        settings = _make_settings()
        mock_settings.return_value = settings

        mock_redis = MagicMock()
        # Lock already held — SETNX returns False
        mock_redis.set.return_value = False
        mock_get_redis.return_value = mock_redis

        result = run_memory_agent()

        assert result["status"] == "locked"
        # Verify SETNX called with correct key and TTL
        mock_redis.set.assert_called_once_with(
            AGENT_LOCK_KEY, "1", nx=True, ex=6 * 3600,
        )


# ---------------------------------------------------------------------------
# Test 8: Agent Disabled
# ---------------------------------------------------------------------------


class TestAgentDisabled:
    @patch("app.workers.memory_agent.get_settings")
    def test_agent_disabled_skips_all(self, mock_settings):
        """AGENT_ENABLED=false skips all passes entirely."""
        settings = _make_settings(AGENT_ENABLED=False)
        mock_settings.return_value = settings

        result = run_memory_agent()

        assert result["status"] == "disabled"


# ---------------------------------------------------------------------------
# Test 9: Batch Limit
# ---------------------------------------------------------------------------


class TestBatchLimit:
    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_confidence_decay_respects_batch_limit(
        self, mock_settings, mock_qdrant, mock_webhook
    ):
        """Each pass processes at most AGENT_BATCH_LIMIT items."""
        batch_limit = 3
        settings = _make_settings(AGENT_BATCH_LIMIT=batch_limit)
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        # Create more memories than batch_limit, all old enough to decay
        points = [
            _make_point(f"mem-{i}", {
                "text": f"Memory {i}", "status": "active",
                "timestamp": "2024-01-01T00:00:00Z",
                "confirmed_count": 5, "contradicted_count": 0,
            })
            for i in range(10)
        ]

        client.scroll.return_value = (points, None)

        result = confidence_decay_pass()

        # Should not process more than batch_limit
        assert result["decayed"] <= batch_limit

    @patch("app.workers.memory_agent._fire_webhook_sync")
    @patch("app.workers.memory_agent._get_qdrant_client")
    @patch("app.workers.memory_agent.get_settings")
    def test_scroll_respects_batch_limit(self, mock_settings, mock_qdrant, mock_webhook):
        """Scroll pagination stops at AGENT_BATCH_LIMIT."""
        batch_limit = 5
        settings = _make_settings(AGENT_BATCH_LIMIT=batch_limit)
        mock_settings.return_value = settings

        client = MagicMock()
        mock_qdrant.return_value = client

        # First scroll returns batch_limit points, second should not happen
        points = [
            _make_point(f"mem-{i}", {
                "text": f"Memory {i}", "status": "active",
                "domain": "test", "tags": [],
                "confirmed_count": 0, "contradicted_count": 0,
            }, [float(i) / 10, 0.5, 0.5])
            for i in range(batch_limit)
        ]
        client.scroll.return_value = (points, None)
        client.query_points.return_value = _make_query_result([])

        result = duplicate_detection_pass()

        # scroll limit should be capped by batch_limit
        scroll_call = client.scroll.call_args
        assert scroll_call.kwargs.get("limit", scroll_call[1].get("limit", 999)) <= batch_limit


# ---------------------------------------------------------------------------
# Test 10: Pass Isolation
# ---------------------------------------------------------------------------


class TestPassIsolation:
    @patch("app.workers.memory_agent.get_settings")
    @patch("app.workers.memory_agent._get_redis_client")
    def test_one_pass_failure_does_not_block_others(self, mock_get_redis, mock_settings):
        """A failing pass should not prevent subsequent passes from running."""
        settings = _make_settings()
        mock_settings.return_value = settings

        mock_redis = MagicMock()
        mock_redis.set.return_value = True  # Lock acquired
        mock_get_redis.return_value = mock_redis

        # Patch all six passes: first raises, rest succeed
        with patch("app.workers.memory_agent.duplicate_detection_pass") as p1, \
             patch("app.workers.memory_agent.orphan_cleanup_pass") as p2, \
             patch("app.workers.memory_agent.deep_contradiction_pass") as p3, \
             patch("app.workers.memory_agent.backlink_reinforcement_pass") as p4, \
             patch("app.workers.memory_agent.confidence_decay_pass") as p5, \
             patch("app.workers.memory_agent.cluster_coherence_pass") as p6:

            p1.side_effect = RuntimeError("Pass 1 exploded")
            p2.return_value = {"status": "ok", "nodes_removed": []}
            p3.side_effect = RuntimeError("Pass 3 exploded")
            p4.return_value = {"status": "ok", "memories_linked": 0, "details": []}
            p5.return_value = {"status": "ok", "decayed": 0, "details": []}
            p6.return_value = {"status": "ok", "reclassified": 0, "details": []}

            result = run_memory_agent()

        assert result["status"] == "ok"
        # Failed passes should be recorded as errors
        assert result["passes"]["duplicate_detection"]["status"] == "error"
        assert result["passes"]["deep_contradiction"]["status"] == "error"
        # Successful passes should have their results
        assert result["passes"]["orphan_cleanup"]["status"] == "ok"
        assert result["passes"]["backlink_reinforcement"]["status"] == "ok"
        assert result["passes"]["confidence_decay"]["status"] == "ok"
        assert result["passes"]["cluster_coherence"]["status"] == "ok"

        # All 6 passes should have been attempted
        p1.assert_called_once()
        p2.assert_called_once()
        p3.assert_called_once()
        p4.assert_called_once()
        p5.assert_called_once()
        p6.assert_called_once()


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestCoseSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0
