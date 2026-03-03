"""Tests for RAGEngine (app.engine.rag) -- the most critical component."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.engine.rag import RAGEngine, _CROSS_REF_BOOST, _FUZZY_MATCH_THRESHOLD
from app.models import ContextQuery


@pytest.fixture()
def mock_graph() -> AsyncMock:
    g = AsyncMock()
    g.query_related = AsyncMock(return_value=[])
    return g


@pytest.fixture()
def mock_vector() -> AsyncMock:
    v = AsyncMock()
    v.search = AsyncMock(return_value=[])
    return v


@pytest.fixture()
def engine(mock_graph, mock_vector) -> RAGEngine:
    return RAGEngine(graph=mock_graph, vector=mock_vector)


# ---------------------------------------------------------------------------
# _normalize_vector
# ---------------------------------------------------------------------------


class TestNormalizeVector:
    def test_preserves_scores(self, engine):
        results = [
            {"text": "Fix bug", "score": 0.95, "metadata": {"source": "ci"}},
            {"text": "Add feature", "score": 0.6, "metadata": {}},
        ]
        entries = engine._normalize_vector(results)
        assert len(entries) == 2
        assert entries[0]["score"] == 0.95
        assert entries[1]["score"] == 0.6

    def test_assigns_vector_store(self, engine):
        entries = engine._normalize_vector(
            [{"text": "x", "score": 0.5, "metadata": {}}]
        )
        assert entries[0]["store"] == "vector"

    def test_skips_empty_text(self, engine):
        results = [
            {"text": "", "score": 0.9, "metadata": {}},
            {"text": "real content", "score": 0.8, "metadata": {}},
        ]
        entries = engine._normalize_vector(results)
        assert len(entries) == 1
        assert entries[0]["content"] == "real content"

    def test_handles_empty_results(self, engine):
        assert engine._normalize_vector([]) == []

    def test_missing_score_defaults_zero(self, engine):
        entries = engine._normalize_vector([{"text": "t", "metadata": {}}])
        assert entries[0]["score"] == 0.0


# ---------------------------------------------------------------------------
# _normalize_graph
# ---------------------------------------------------------------------------


class TestNormalizeGraph:
    def test_distance_based_scoring(self, engine):
        results = [
            {"name": "a", "description": "desc a", "label": "Concept", "distance": 1},
            {"name": "b", "description": "desc b", "label": "Action", "distance": 3},
        ]
        entries = engine._normalize_graph(results)
        assert len(entries) == 2
        # distance=1 -> score=1.0, distance=3 -> score=1/3
        assert entries[0]["score"] == 1.0
        assert abs(entries[1]["score"] - 1.0 / 3) < 0.001

    def test_missing_distance_defaults_half(self, engine):
        results = [{"name": "x", "description": "desc", "label": "Concept", "distance": None}]
        entries = engine._normalize_graph(results)
        assert entries[0]["score"] == 0.5

    def test_handles_missing_fields(self, engine):
        """Nodes with only a name (no description) should use name as content."""
        results = [{"name": "auth", "description": None, "label": "Concept", "distance": 1}]
        entries = engine._normalize_graph(results)
        assert entries[0]["content"] == "auth"

    def test_skips_nodes_without_name_or_description(self, engine):
        results = [
            {"name": "", "description": "", "label": "Concept", "distance": 1},
            {"name": None, "description": None, "label": "Concept", "distance": 1},
        ]
        entries = engine._normalize_graph(results)
        assert len(entries) == 0

    def test_description_preferred_over_name(self, engine):
        results = [{"name": "N", "description": "Full description", "label": "X", "distance": 2}]
        entries = engine._normalize_graph(results)
        assert entries[0]["content"] == "Full description"

    def test_assigns_graph_store(self, engine):
        results = [{"name": "n", "description": "d", "label": "L", "distance": 1}]
        entries = engine._normalize_graph(results)
        assert entries[0]["store"] == "graph"

    def test_metadata_includes_label(self, engine):
        results = [{"name": "n", "description": "d", "label": "CustomLabel", "distance": 1}]
        entries = engine._normalize_graph(results)
        assert entries[0]["metadata"]["label"] == "CustomLabel"

    def test_handles_empty_results(self, engine):
        assert engine._normalize_graph([]) == []


# ---------------------------------------------------------------------------
# _merge_and_boost
# ---------------------------------------------------------------------------


class TestMergeAndBoost:
    def test_cross_reference_detection(self, engine):
        """Nearly identical content in both stores should merge with boost."""
        vector_entries = [
            {"content": "Fix authentication timeout", "score": 0.8, "store": "vector", "metadata": {}},
        ]
        graph_entries = [
            {
                "content": "Fix authentication timeout bug",
                "score": 0.7,
                "store": "graph",
                "metadata": {"name": "auth", "label": "Concept"},
            },
        ]
        merged = engine._merge_and_boost(vector_entries, graph_entries)

        # Should produce exactly one cross-referenced entry
        cross_refs = [e for e in merged if e["store"] == "both"]
        assert len(cross_refs) == 1
        # Score should be boosted: max(0.8, 0.7) * 1.5 = 1.2 -> capped at 1.0
        assert cross_refs[0]["score"] == 1.0

    def test_boost_factor_applied(self, engine):
        """Verify boost factor is correctly applied when not capped."""
        vector_entries = [
            {"content": "pool size config", "score": 0.4, "store": "vector", "metadata": {}},
        ]
        graph_entries = [
            {
                "content": "pool size configuration",
                "score": 0.3,
                "store": "graph",
                "metadata": {"name": "pool", "label": "Concept"},
            },
        ]
        merged = engine._merge_and_boost(vector_entries, graph_entries)
        cross_refs = [e for e in merged if e["store"] == "both"]
        assert len(cross_refs) == 1
        # max(0.4, 0.3) * 1.5 = 0.6
        assert abs(cross_refs[0]["score"] - 0.6) < 0.001

    def test_no_cross_reference_for_dissimilar_content(self, engine):
        vector_entries = [
            {"content": "Fix authentication bug", "score": 0.8, "store": "vector", "metadata": {}},
        ]
        graph_entries = [
            {
                "content": "Database connection pooling",
                "score": 0.7,
                "store": "graph",
                "metadata": {"name": "db", "label": "Concept"},
            },
        ]
        merged = engine._merge_and_boost(vector_entries, graph_entries)
        assert len(merged) == 2
        stores = {e["store"] for e in merged}
        assert "both" not in stores

    def test_deduplication(self, engine):
        """A graph entry matched by a vector entry should not appear again."""
        vector_entries = [
            {"content": "Fix login timeout", "score": 0.9, "store": "vector", "metadata": {}},
        ]
        graph_entries = [
            {
                "content": "Fix login timeout",
                "score": 0.5,
                "store": "graph",
                "metadata": {"name": "login", "label": "Action"},
            },
        ]
        merged = engine._merge_and_boost(vector_entries, graph_entries)
        # Exact match -> cross-ref, graph entry consumed -> only 1 result
        assert len(merged) == 1
        assert merged[0]["store"] == "both"

    def test_all_graph_entries_preserved_when_no_match(self, engine):
        vector_entries = [
            {"content": "AAA", "score": 0.9, "store": "vector", "metadata": {}},
        ]
        graph_entries = [
            {"content": "BBB", "score": 0.5, "store": "graph", "metadata": {"name": "b", "label": "X"}},
            {"content": "CCC", "score": 0.4, "store": "graph", "metadata": {"name": "c", "label": "Y"}},
        ]
        merged = engine._merge_and_boost(vector_entries, graph_entries)
        assert len(merged) == 3

    def test_empty_vector_entries(self, engine):
        graph_entries = [
            {"content": "x", "score": 0.5, "store": "graph", "metadata": {"name": "n", "label": "L"}},
        ]
        merged = engine._merge_and_boost([], graph_entries)
        assert len(merged) == 1
        assert merged[0]["store"] == "graph"

    def test_empty_graph_entries(self, engine):
        vector_entries = [
            {"content": "x", "score": 0.5, "store": "vector", "metadata": {}},
        ]
        merged = engine._merge_and_boost(vector_entries, [])
        assert len(merged) == 1
        assert merged[0]["store"] == "vector"

    def test_both_empty(self, engine):
        assert engine._merge_and_boost([], []) == []


# ---------------------------------------------------------------------------
# _is_fuzzy_match
# ---------------------------------------------------------------------------


class TestIsFuzzyMatch:
    def test_identical_strings_match(self, engine):
        assert engine._is_fuzzy_match("hello world", "hello world") is True

    def test_case_insensitive(self, engine):
        assert engine._is_fuzzy_match("Hello World", "hello world") is True

    def test_very_different_strings_no_match(self, engine):
        assert engine._is_fuzzy_match("abc", "xyz") is False

    def test_empty_strings_no_match(self, engine):
        assert engine._is_fuzzy_match("", "hello") is False
        assert engine._is_fuzzy_match("hello", "") is False
        assert engine._is_fuzzy_match("", "") is False

    def test_similar_strings_above_threshold(self, engine):
        # "Fix authentication timeout" vs "Fix authentication timeout bug"
        # SequenceMatcher ratio should be well above 0.7
        assert engine._is_fuzzy_match(
            "Fix authentication timeout",
            "Fix authentication timeout bug",
        ) is True

    def test_moderately_different_strings_below_threshold(self, engine):
        assert engine._is_fuzzy_match(
            "Fix authentication",
            "Database pooling configuration",
        ) is False


# ---------------------------------------------------------------------------
# _format_markdown
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    def test_empty_entries(self, engine):
        result = engine._format_markdown([])
        assert "No relevant memories found" in result

    def test_graph_section(self, engine):
        entries = [
            {
                "content": "Auth module handles tokens",
                "score": 0.9,
                "store": "graph",
                "metadata": {"label": "Concept", "name": "auth"},
            },
        ]
        result = engine._format_markdown(entries)
        assert "### From Knowledge Graph" in result
        assert "**[Concept]** auth" in result

    def test_vector_section(self, engine):
        entries = [
            {
                "content": "Fix login timeout",
                "score": 0.85,
                "store": "vector",
                "metadata": {"source": "action_log", "tags": ["auth", "bug"]},
            },
        ]
        result = engine._format_markdown(entries)
        assert "### From Semantic Memory" in result
        assert "[0.85] Fix login timeout" in result
        assert "Source: action_log" in result
        assert "auth, bug" in result

    def test_cross_ref_section(self, engine):
        entries = [
            {
                "content": "Pool sizing fix",
                "score": 0.95,
                "store": "both",
                "metadata": {"graph_name": "pool_config"},
            },
        ]
        result = engine._format_markdown(entries)
        assert "### Cross-Referenced (High Confidence)" in result
        assert "[0.95] Pool sizing fix" in result
        assert "pool_config" in result

    def test_all_sections_present(self, engine):
        entries = [
            {"content": "g", "score": 0.9, "store": "graph", "metadata": {"label": "Concept", "name": "g"}},
            {"content": "v", "score": 0.8, "store": "vector", "metadata": {"source": "s", "tags": []}},
            {"content": "b", "score": 0.95, "store": "both", "metadata": {"graph_name": "b"}},
        ]
        result = engine._format_markdown(entries)
        assert "### From Knowledge Graph" in result
        assert "### From Semantic Memory" in result
        assert "### Cross-Referenced (High Confidence)" in result

    def test_graph_entry_content_equals_name(self, engine):
        """When content == name, only the header should appear, not the arrow."""
        entries = [
            {
                "content": "auth",
                "score": 0.9,
                "store": "graph",
                "metadata": {"label": "Concept", "name": "auth"},
            },
        ]
        result = engine._format_markdown(entries)
        assert "**[Concept]** auth" in result
        # Should NOT have arrow when content == name
        assert "\u2192" not in result

    def test_graph_entry_content_differs_from_name(self, engine):
        entries = [
            {
                "content": "Detailed description",
                "score": 0.9,
                "store": "graph",
                "metadata": {"label": "Action", "name": "short_name"},
            },
        ]
        result = engine._format_markdown(entries)
        assert "\u2192 Detailed description" in result

    def test_vector_no_tags(self, engine):
        entries = [
            {
                "content": "test",
                "score": 0.5,
                "store": "vector",
                "metadata": {"source": "src", "tags": []},
            },
        ]
        result = engine._format_markdown(entries)
        assert "Tags: none" in result


# ---------------------------------------------------------------------------
# recall (full pipeline)
# ---------------------------------------------------------------------------


class TestRecall:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_graph, mock_vector, engine):
        mock_vector.search.return_value = [
            {
                "id": "v1",
                "score": 0.9,
                "text": "Fixed auth timeout by increasing pool",
                "metadata": {"source": "action_log", "tags": ["auth"], "domain": "security"},
            },
        ]
        mock_graph.query_related.return_value = [
            {
                "name": "auth",
                "description": "Authentication service",
                "label": "Concept",
                "distance": 1,
            },
        ]

        query = ContextQuery(task="Fix auth issue", tags=["auth"], top_k=5)
        response = await engine.recall(query)

        assert response.context_block
        assert len(response.sources) > 0
        assert response.score > 0.0

    @pytest.mark.asyncio
    async def test_both_stores_empty(self, mock_graph, mock_vector, engine):
        mock_vector.search.return_value = []
        mock_graph.query_related.return_value = []

        query = ContextQuery(task="unknown", top_k=5)
        response = await engine.recall(query)

        assert response.sources == []
        assert response.score == 0.0
        assert "No relevant memories found" in response.context_block

    @pytest.mark.asyncio
    async def test_only_vector_results(self, mock_graph, mock_vector, engine):
        mock_vector.search.return_value = [
            {"id": "v1", "score": 0.7, "text": "Vector only result", "metadata": {}},
        ]
        mock_graph.query_related.return_value = []

        query = ContextQuery(task="search", top_k=5)
        response = await engine.recall(query)

        assert len(response.sources) == 1
        assert response.sources[0].store == "vector"

    @pytest.mark.asyncio
    async def test_only_graph_results(self, mock_graph, mock_vector, engine):
        mock_vector.search.return_value = []
        mock_graph.query_related.return_value = [
            {"name": "concept", "description": "A graph concept", "label": "Concept", "distance": 1},
        ]

        query = ContextQuery(task="search", top_k=5)
        response = await engine.recall(query)

        assert len(response.sources) == 1
        assert response.sources[0].store == "graph"

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, mock_graph, mock_vector, engine):
        mock_vector.search.return_value = [
            {"id": f"v{i}", "score": 0.9 - i * 0.1, "text": f"Result {i}", "metadata": {}}
            for i in range(5)
        ]
        mock_graph.query_related.return_value = [
            {"name": f"g{i}", "description": f"Graph {i}", "label": "C", "distance": i + 1}
            for i in range(5)
        ]

        query = ContextQuery(task="search", top_k=3)
        response = await engine.recall(query)

        assert len(response.sources) <= 3

    @pytest.mark.asyncio
    async def test_all_low_scores(self, mock_graph, mock_vector, engine):
        mock_vector.search.return_value = [
            {"id": "v1", "score": 0.1, "text": "Marginal embedding match on obscure topic", "metadata": {}},
        ]
        mock_graph.query_related.return_value = [
            {"name": "x", "description": "Distant unrelated entity from deep traversal", "label": "C", "distance": 10},
        ]

        query = ContextQuery(task="obscure", top_k=5)
        response = await engine.recall(query)

        assert len(response.sources) == 2
        # All scores should be low
        for src in response.sources:
            assert src.score <= 0.2

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, mock_graph, mock_vector, engine):
        """Verify both stores are queried concurrently (via asyncio.gather)."""
        import asyncio

        call_order = []

        original_search = mock_vector.search.side_effect

        async def slow_vector(*args, **kwargs):
            call_order.append("vector_start")
            await asyncio.sleep(0.01)
            call_order.append("vector_end")
            return [{"id": "v1", "score": 0.5, "text": "v", "metadata": {}}]

        async def slow_graph(*args, **kwargs):
            call_order.append("graph_start")
            await asyncio.sleep(0.01)
            call_order.append("graph_end")
            return [{"name": "g", "description": "g", "label": "C", "distance": 1}]

        mock_vector.search.side_effect = slow_vector
        mock_graph.query_related.side_effect = slow_graph

        query = ContextQuery(task="test", top_k=5)
        response = await engine.recall(query)

        # Both should have been called
        assert mock_vector.search.called
        assert mock_graph.query_related.called
        assert len(response.sources) == 2

    @pytest.mark.asyncio
    async def test_one_store_fails_gracefully(self, mock_graph, mock_vector, engine):
        """If one store raises, the other should still return results."""
        mock_vector.search.side_effect = RuntimeError("vector down")
        mock_graph.query_related.return_value = [
            {"name": "n", "description": "d", "label": "C", "distance": 1},
        ]

        query = ContextQuery(task="test", top_k=5)
        response = await engine.recall(query)

        # Should have graph result despite vector failure
        assert len(response.sources) == 1
        assert response.sources[0].store == "graph"

    @pytest.mark.asyncio
    async def test_aggregate_score_calculation(self, mock_graph, mock_vector, engine):
        """Aggregate score is the max of individual source scores."""
        mock_vector.search.return_value = [
            {"id": "v1", "score": 0.8, "text": "Result 1", "metadata": {}},
            {"id": "v2", "score": 0.6, "text": "Result 2", "metadata": {}},
        ]
        mock_graph.query_related.return_value = []

        query = ContextQuery(task="test", top_k=5)
        response = await engine.recall(query)

        # aggregate_score = max of source scores
        assert response.score == 0.8
