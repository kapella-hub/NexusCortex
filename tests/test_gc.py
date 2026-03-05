"""Tests for Garbage Collection worker (app.workers.gc)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.workers.gc import prune_memories, _prune, _prune_qdrant, _prune_neo4j_orphans


# ---------------------------------------------------------------------------
# prune_memories — top-level Celery task
# ---------------------------------------------------------------------------


class TestPruneMemories:
    @patch("app.workers.gc._prune")
    def test_successful_prune(self, mock_prune):
        mock_prune.return_value = {
            "status": "completed",
            "pruned_vector": 5,
            "pruned_graph": 3,
        }
        result = prune_memories()
        assert result["status"] == "completed"
        assert result["pruned_vector"] == 5
        assert result["pruned_graph"] == 3

    @patch("app.workers.gc._prune")
    def test_unhandled_error_returns_error_status(self, mock_prune):
        mock_prune.side_effect = RuntimeError("unexpected")
        result = prune_memories()
        assert result["status"] == "error"
        assert result["pruned_vector"] == 0
        assert result["pruned_graph"] == 0


# ---------------------------------------------------------------------------
# _prune_qdrant
# ---------------------------------------------------------------------------


class TestPruneQdrant:
    @patch("app.workers.gc._get_qdrant_client")
    def test_prunes_old_memories_without_feedback(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create mock points: 2 old without positive feedback, 1 old with positive feedback
        old_no_feedback = MagicMock()
        old_no_feedback.id = "point-1"
        old_no_feedback.payload = {"timestamp": "2024-01-01T00:00:00", "feedback_useful": False}

        old_no_feedback_2 = MagicMock()
        old_no_feedback_2.id = "point-2"
        old_no_feedback_2.payload = {"timestamp": "2024-01-01T00:00:00"}  # no feedback at all

        old_with_feedback = MagicMock()
        old_with_feedback.id = "point-3"
        old_with_feedback.payload = {"timestamp": "2024-01-01T00:00:00", "feedback_useful": True}

        # scroll returns (points, next_offset)
        mock_client.scroll.return_value = (
            [old_no_feedback, old_no_feedback_2, old_with_feedback],
            None,  # no more pages
        )

        mock_settings = MagicMock()
        mock_settings.QDRANT_COLLECTION = "test_collection"

        result = _prune_qdrant(mock_settings, "2025-01-01T00:00:00")

        # Should prune 2 (the ones without positive feedback)
        assert result == 2
        # delete should have been called
        mock_client.delete.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("app.workers.gc._get_qdrant_client")
    def test_preserves_memories_with_positive_feedback(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Only points with positive feedback
        point_with_feedback = MagicMock()
        point_with_feedback.id = "point-1"
        point_with_feedback.payload = {"timestamp": "2024-01-01T00:00:00", "feedback_useful": True}

        mock_client.scroll.return_value = ([point_with_feedback], None)

        mock_settings = MagicMock()
        mock_settings.QDRANT_COLLECTION = "test_collection"

        result = _prune_qdrant(mock_settings, "2025-01-01T00:00:00")

        # Nothing should be pruned
        assert result == 0
        mock_client.delete.assert_not_called()

    @patch("app.workers.gc._get_qdrant_client")
    def test_no_old_memories(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.scroll.return_value = ([], None)

        mock_settings = MagicMock()
        mock_settings.QDRANT_COLLECTION = "test_collection"

        result = _prune_qdrant(mock_settings, "2025-01-01T00:00:00")
        assert result == 0
        mock_client.delete.assert_not_called()

    @patch("app.workers.gc._get_qdrant_client")
    def test_qdrant_error_returns_zero(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.scroll.side_effect = RuntimeError("Qdrant down")

        mock_settings = MagicMock()
        mock_settings.QDRANT_COLLECTION = "test_collection"

        result = _prune_qdrant(mock_settings, "2025-01-01T00:00:00")
        assert result == 0


# ---------------------------------------------------------------------------
# _prune_neo4j_orphans
# ---------------------------------------------------------------------------


class TestPruneNeo4jOrphans:
    @patch("app.workers.gc._get_neo4j_driver")
    def test_deletes_orphaned_nodes(self, mock_get_driver):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=7)

        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_get_driver.return_value = mock_driver

        result = _prune_neo4j_orphans()
        assert result == 7

    @patch("app.workers.gc._get_neo4j_driver")
    def test_neo4j_error_returns_zero(self, mock_get_driver):
        mock_get_driver.side_effect = RuntimeError("Neo4j down")
        result = _prune_neo4j_orphans()
        assert result == 0


# ---------------------------------------------------------------------------
# _prune — integration of both stores
# ---------------------------------------------------------------------------


class TestPrune:
    @patch("app.workers.gc._prune_neo4j_orphans")
    @patch("app.workers.gc._prune_qdrant")
    @patch("app.workers.gc.get_settings")
    def test_combines_both_stores(self, mock_settings, mock_prune_qdrant, mock_prune_neo4j):
        mock_settings.return_value = MagicMock(MAX_MEMORY_AGE_DAYS=180)
        mock_prune_qdrant.return_value = 10
        mock_prune_neo4j.return_value = 5

        result = _prune()
        assert result["status"] == "completed"
        assert result["pruned_vector"] == 10
        assert result["pruned_graph"] == 5
