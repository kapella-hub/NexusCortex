"""Tests for automatic backlink discovery."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.backlinks import BACKLINK_MAX_THRESHOLD, BACKLINK_MIN_THRESHOLD, discover_backlinks


@pytest.fixture
def mock_vector():
    v = AsyncMock()
    return v


@pytest.fixture
def mock_graph():
    g = AsyncMock()
    return g


class TestDiscoverBacklinks:
    """Tests for the discover_backlinks function."""

    @pytest.mark.asyncio
    async def test_finds_related_memories_in_range(self, mock_vector, mock_graph):
        """Should return memories with similarity between MIN and MAX thresholds."""
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": "related-1", "score": 0.65, "text": "Related memory about database"},
            {"id": "related-2", "score": 0.55, "text": "Another related memory"},
        ])

        result = await discover_backlinks(
            vector=mock_vector,
            graph=mock_graph,
            new_text="Database migration to Postgres",
            new_vector_id="new-id",
            new_graph_id="graph-id",
            domain="infrastructure",
        )

        assert len(result) == 2
        assert result[0]["id"] == "related-1"
        assert result[0]["score"] == 0.65
        assert result[1]["id"] == "related-2"

    @pytest.mark.asyncio
    async def test_skips_self(self, mock_vector, mock_graph):
        """Should not include the new memory itself in backlinks."""
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": "new-id", "score": 1.0, "text": "Self"},
            {"id": "related-1", "score": 0.6, "text": "Related"},
        ])

        result = await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id="graph-id", domain="test",
        )

        assert len(result) == 1
        assert result[0]["id"] == "related-1"

    @pytest.mark.asyncio
    async def test_skips_above_contradiction_threshold(self, mock_vector, mock_graph):
        """Should not include memories above the contradiction threshold."""
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": "contradiction", "score": 0.90, "text": "Too similar"},
            {"id": "related", "score": 0.70, "text": "Just right"},
        ])

        result = await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id="graph-id", domain="test",
        )

        assert len(result) == 1
        assert result[0]["id"] == "related"

    @pytest.mark.asyncio
    async def test_skips_below_minimum_threshold(self, mock_vector, mock_graph):
        """Should not include memories below the minimum threshold."""
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": "weak", "score": 0.2, "text": "Too dissimilar"},
        ])

        result = await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id="graph-id", domain="test",
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_creates_graph_edges(self, mock_vector, mock_graph):
        """Should create BACKLINK edges in the graph for each discovered link."""
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": "related-1", "score": 0.65, "text": "Related memory"},
        ])

        await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id="graph-id", domain="test",
        )

        mock_graph.create_backlink.assert_called_once_with(
            source_id="graph-id",
            target_vector_id="related-1",
            score=0.65,
        )

    @pytest.mark.asyncio
    async def test_no_graph_edges_without_graph_id(self, mock_vector, mock_graph):
        """Should not create graph edges when graph_id is None."""
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": "related-1", "score": 0.65, "text": "Related"},
        ])

        result = await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id=None, domain="test",
        )

        assert len(result) == 1
        mock_graph.create_backlink.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_top_k_limit(self, mock_vector, mock_graph):
        """Should respect the top_k limit."""
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": f"related-{i}", "score": 0.7 - i * 0.05, "text": f"Memory {i}"}
            for i in range(10)
        ])

        result = await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id="graph-id", domain="test",
            top_k=3,
        )

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_handles_vector_failure_gracefully(self, mock_vector, mock_graph):
        """Should return empty list if vector search fails."""
        mock_vector.find_similar = AsyncMock(side_effect=Exception("Connection failed"))

        result = await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id="graph-id", domain="test",
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_graph_edge_failure_gracefully(self, mock_vector, mock_graph):
        """Should still return backlinks even if graph edge creation fails."""
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": "related-1", "score": 0.65, "text": "Related"},
        ])
        mock_graph.create_backlink = AsyncMock(side_effect=Exception("Graph error"))

        result = await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id="graph-id", domain="test",
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_truncates_text_in_result(self, mock_vector, mock_graph):
        """Should truncate text to 200 chars in result."""
        long_text = "A" * 500
        mock_vector.find_similar = AsyncMock(return_value=[
            {"id": "related-1", "score": 0.65, "text": long_text},
        ])

        result = await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id=None, domain="test",
        )

        assert len(result[0]["text"]) == 200

    @pytest.mark.asyncio
    async def test_cross_domain_discovery(self, mock_vector, mock_graph):
        """Should search across all domains (domain=None)."""
        mock_vector.find_similar = AsyncMock(return_value=[])

        await discover_backlinks(
            vector=mock_vector, graph=mock_graph,
            new_text="test", new_vector_id="new-id",
            new_graph_id=None, domain="infrastructure",
        )

        # Verify domain=None is passed to enable cross-domain linking
        mock_vector.find_similar.assert_called_once()
        call_kwargs = mock_vector.find_similar.call_args
        assert call_kwargs.kwargs.get("domain") is None or call_kwargs[1].get("domain") is None


class TestBacklinkThresholds:
    """Test threshold constants."""

    def test_min_threshold_reasonable(self):
        assert 0.3 <= BACKLINK_MIN_THRESHOLD <= 0.5

    def test_max_threshold_below_contradiction(self):
        assert BACKLINK_MAX_THRESHOLD < 0.85

    def test_max_above_min(self):
        assert BACKLINK_MAX_THRESHOLD > BACKLINK_MIN_THRESHOLD
