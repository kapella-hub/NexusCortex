"""Tests for Neo4jClient (app.db.graph)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.db.graph import Neo4jClient
from app.exceptions import GraphConnectionError
from app.models import ActionLog


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="test",
    )


@pytest.fixture()
def client_with_driver(settings: Settings) -> Neo4jClient:
    """Neo4jClient with a mocked async driver already assigned.

    The mock supports two patterns used in the source code:

    1. ``session.run(...)`` — used by query_related, query_resolutions,
       ensure_indexes (simple reads / DDL).
    2. ``session.begin_transaction() -> tx -> tx.run(...)`` — used by
       merge_action_log and merge_knowledge_nodes (explicit transactions).

    Both ``session`` and ``tx`` are stored on the client instance for
    convenient assertion access in tests.
    """
    client = Neo4jClient(settings)

    # --- Transaction mock (for merge_action_log / merge_knowledge_nodes) ---
    mock_tx = AsyncMock()
    mock_tx.run = AsyncMock()
    mock_tx.commit = AsyncMock()

    # begin_transaction() returns an async CM whose __aenter__ yields the tx
    mock_tx_ctx = AsyncMock()
    mock_tx_ctx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx_ctx.__aexit__ = AsyncMock(return_value=False)

    # --- Session mock ---
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()
    mock_session.begin_transaction = MagicMock(return_value=mock_tx_ctx)

    # Build async context manager for driver.session()
    mock_sess_ctx = AsyncMock()
    mock_sess_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_sess_ctx.__aexit__ = AsyncMock(return_value=False)

    # Driver — session() is a regular (sync) call returning the async CM
    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_sess_ctx

    client._driver = mock_driver
    client._mock_session = mock_session
    client._mock_tx = mock_tx
    return client


# ---------------------------------------------------------------------------
# _ensure_driver / connection guard
# ---------------------------------------------------------------------------


class TestEnsureDriver:
    def test_raises_when_driver_not_connected(self, settings):
        client = Neo4jClient(settings)
        assert client._driver is None
        with pytest.raises(GraphConnectionError, match="not connected"):
            client._ensure_driver()

    def test_returns_driver_when_connected(self, client_with_driver):
        driver = client_with_driver._ensure_driver()
        assert driver is not None


# ---------------------------------------------------------------------------
# _content_hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self, settings):
        client = Neo4jClient(settings)
        h1 = client._content_hash("hello")
        h2 = client._content_hash("hello")
        assert h1 == h2

    def test_different_inputs_different_hashes(self, settings):
        client = Neo4jClient(settings)
        h1 = client._content_hash("hello")
        h2 = client._content_hash("world")
        assert h1 != h2

    def test_returns_configurable_hex_chars(self, settings):
        client = Neo4jClient(settings)
        h = client._content_hash("test")
        assert len(h) == settings.CONTENT_HASH_LENGTH
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_extracts_meaningful_words(self):
        kws = Neo4jClient._extract_keywords("Fix the authentication timeout bug")
        assert "authentication" in kws
        assert "timeout" in kws
        assert "the" not in kws  # stopword

    def test_respects_max_keywords(self):
        kws = Neo4jClient._extract_keywords("a very long sentence with many meaningful words here", max_keywords=2)
        assert len(kws) <= 2

    def test_removes_short_tokens(self):
        kws = Neo4jClient._extract_keywords("go do it or be an ox")
        # All 2-char or shorter tokens + stopwords should be filtered
        for kw in kws:
            assert len(kw) >= 3

    def test_empty_input(self):
        assert Neo4jClient._extract_keywords("") == []

    def test_only_stopwords(self):
        assert Neo4jClient._extract_keywords("the is are was") == []


# ---------------------------------------------------------------------------
# merge_action_log
# ---------------------------------------------------------------------------


class TestMergeActionLog:
    @pytest.mark.asyncio
    async def test_merge_action_log_with_resolution(self, client_with_driver):
        tx = client_with_driver._mock_tx
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "elem-id-123"})
        tx.run = AsyncMock(return_value=mock_result)

        log = ActionLog(
            action="Increased pool",
            outcome="Timeout fixed",
            resolution="Changed config",
            tags=["db", "perf"],
            domain="infra",
        )
        result_id = await client_with_driver.merge_action_log(log)

        assert result_id == "elem-id-123"
        tx.run.assert_called_once()
        tx.commit.assert_called_once()

        # Verify Cypher parameters include content hashes
        call_kwargs = tx.run.call_args.kwargs
        assert call_kwargs["domain"] == "infra"
        assert call_kwargs["action"] == "Increased pool"
        assert call_kwargs["outcome"] == "Timeout fixed"
        assert call_kwargs["resolution"] == "Changed config"
        assert call_kwargs["tags"] == ["db", "perf"]
        assert call_kwargs["action_id"] == client_with_driver._content_hash("Increased pool")
        assert call_kwargs["outcome_id"] == client_with_driver._content_hash("Timeout fixed")
        assert call_kwargs["resolution_id"] == client_with_driver._content_hash("Changed config")

    @pytest.mark.asyncio
    async def test_merge_action_log_without_resolution(self, client_with_driver):
        tx = client_with_driver._mock_tx
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "elem-id-456"})
        tx.run = AsyncMock(return_value=mock_result)

        log = ActionLog(action="a", outcome="o")
        result_id = await client_with_driver.merge_action_log(log)
        assert result_id == "elem-id-456"

        call_kwargs = tx.run.call_args.kwargs
        assert call_kwargs["resolution"] is None
        assert call_kwargs["resolution_id"] is None

    @pytest.mark.asyncio
    async def test_merge_action_log_no_result_raises(self, client_with_driver):
        tx = client_with_driver._mock_tx
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        tx.run = AsyncMock(return_value=mock_result)

        log = ActionLog(action="a", outcome="o")
        with pytest.raises(GraphConnectionError, match="returned no result"):
            await client_with_driver.merge_action_log(log)

    @pytest.mark.asyncio
    async def test_merge_action_log_driver_error(self, client_with_driver):
        tx = client_with_driver._mock_tx
        tx.run = AsyncMock(side_effect=RuntimeError("connection lost"))

        log = ActionLog(action="a", outcome="o")
        with pytest.raises(GraphConnectionError, match="Failed to merge action log"):
            await client_with_driver.merge_action_log(log)


# ---------------------------------------------------------------------------
# merge_knowledge_nodes
# ---------------------------------------------------------------------------


class TestMergeKnowledgeNodes:
    @pytest.mark.asyncio
    async def test_label_sanitization(self, client_with_driver):
        tx = client_with_driver._mock_tx
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"cnt": 1})
        tx.run = AsyncMock(return_value=mock_result)

        nodes = [
            {
                "id": "n1",
                "label": "Drop;Table--",
                "properties": {"desc": "malicious label"},
            }
        ]
        await client_with_driver.merge_knowledge_nodes(nodes, [])

        call_args = tx.run.call_args
        query = call_args[0][0]
        # The sanitized label should only keep alphanumerics/underscores
        assert "DropTable" in query
        assert ";" not in query
        assert "--" not in query

    @pytest.mark.asyncio
    async def test_empty_label_defaults_to_entity(self, client_with_driver):
        tx = client_with_driver._mock_tx
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"cnt": 1})
        tx.run = AsyncMock(return_value=mock_result)

        nodes = [{"id": "n1", "label": "!@#$", "properties": {}}]
        await client_with_driver.merge_knowledge_nodes(nodes, [])

        query = tx.run.call_args[0][0]
        assert "Entity" in query

    @pytest.mark.asyncio
    async def test_node_and_edge_grouping(self, client_with_driver):
        tx = client_with_driver._mock_tx
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"cnt": 2})
        tx.run = AsyncMock(return_value=mock_result)

        nodes = [
            {"id": "n1", "label": "Concept", "properties": {"name": "a"}},
            {"id": "n2", "label": "Concept", "properties": {"name": "b"}},
            {"id": "n3", "label": "Action", "properties": {"name": "c"}},
        ]
        edges = [
            {"source": "n1", "target": "n2", "type": "RELATES_TO"},
            {"source": "n2", "target": "n3", "type": "CAUSED"},
        ]

        count = await client_with_driver.merge_knowledge_nodes(nodes, edges)

        # 2 node label groups (Concept, Action) + 2 edge type groups (RELATES_TO, CAUSED) = 4 calls
        assert tx.run.call_count == 4
        # count = 2 per call * 4 = 8
        assert count == 8
        tx.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_nodes_and_edges(self, client_with_driver):
        count = await client_with_driver.merge_knowledge_nodes([], [])
        assert count == 0


# ---------------------------------------------------------------------------
# query_related
# ---------------------------------------------------------------------------


class TestQueryRelated:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self, client_with_driver):
        session = client_with_driver._mock_session

        record1 = {"name": "auth", "description": "Auth module", "label": "Concept", "distance": 1}
        record2 = {"name": "login", "description": "Login flow", "label": "Action", "distance": 2}
        records = [record1, record2]

        mock_result = MagicMock()

        async def _aiter():
            for r in records:
                yield r

        mock_result.__aiter__ = lambda self: _aiter()
        session.run = AsyncMock(return_value=mock_result)

        results = await client_with_driver.query_related("authentication module", limit=10)
        assert len(results) == 2
        assert results[0]["name"] == "auth"
        assert results[1]["label"] == "Action"

    @pytest.mark.asyncio
    async def test_query_related_driver_error(self, client_with_driver):
        session = client_with_driver._mock_session
        session.run = AsyncMock(side_effect=RuntimeError("timeout"))

        with pytest.raises(GraphConnectionError, match="Failed to query related"):
            await client_with_driver.query_related("anything meaningful")

    @pytest.mark.asyncio
    async def test_query_related_empty_keywords(self, client_with_driver):
        """If only stopwords are provided, should return empty list without querying."""
        results = await client_with_driver.query_related("the is a")
        assert results == []

    @pytest.mark.asyncio
    async def test_query_related_builds_fulltext_query(self, client_with_driver):
        """Verify the Cypher query uses fulltext index search."""
        session = client_with_driver._mock_session

        mock_result = MagicMock()

        async def _aiter():
            return
            yield  # empty async generator

        mock_result.__aiter__ = lambda self: _aiter()
        session.run = AsyncMock(return_value=mock_result)

        await client_with_driver.query_related("authentication timeout", limit=5)

        # Verify the query was called with fulltext search
        call_args = session.run.call_args
        query = call_args[0][0]
        assert "fulltext.queryNodes" in query
        # Should have search_terms parameter
        call_kwargs = call_args.kwargs
        assert "search_terms" in call_kwargs


# ---------------------------------------------------------------------------
# query_resolutions
# ---------------------------------------------------------------------------


class TestQueryResolutions:
    @pytest.mark.asyncio
    async def test_returns_matching_resolutions(self, client_with_driver):
        session = client_with_driver._mock_session
        record = {
            "resolution": "Restart service",
            "error": "OOM killed",
            "id": "r1",
        }

        mock_result = MagicMock()

        async def _aiter():
            yield record

        mock_result.__aiter__ = lambda self: _aiter()
        session.run = AsyncMock(return_value=mock_result)

        results = await client_with_driver.query_resolutions("OOM")
        assert len(results) == 1
        assert results[0]["resolution"] == "Restart service"

    @pytest.mark.asyncio
    async def test_error_propagation(self, client_with_driver):
        session = client_with_driver._mock_session
        session.run = AsyncMock(side_effect=RuntimeError("network error"))

        with pytest.raises(GraphConnectionError, match="Failed to query resolutions"):
            await client_with_driver.query_resolutions("error")
