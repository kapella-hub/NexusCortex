"""Shared pytest fixtures for NexusCortex test suite."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_graph, get_redis, get_vector


@pytest.fixture()
def test_settings() -> Settings:
    """Settings with test-friendly defaults (nothing connects to real services)."""
    return Settings(
        APP_NAME="NexusCortex-Test",
        DEBUG=True,
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="test",
        QDRANT_HOST="localhost",
        QDRANT_PORT=6333,
        QDRANT_COLLECTION="test_memory",
        REDIS_URL="redis://localhost:6379/15",
        REDIS_STREAM_KEY="nexus:test_stream",
        REDIS_BATCH_SIZE=10,
        LLM_BASE_URL="http://localhost:11434/v1",
        LLM_MODEL="test-model",
        LLM_API_KEY="test-key",
        EMBEDDING_MODEL="test-embed",
        EMBEDDING_DIM=768,
    )


@pytest.fixture()
def mock_graph() -> AsyncMock:
    """AsyncMock standing in for Neo4jClient."""
    graph = AsyncMock()
    graph.connect = AsyncMock()
    graph.close = AsyncMock()
    graph.merge_action_log = AsyncMock(return_value="neo4j-element-id-1")
    graph.merge_knowledge_nodes = AsyncMock(return_value=3)
    graph.query_related = AsyncMock(return_value=[])
    graph.query_resolutions = AsyncMock(return_value=[])
    return graph


@pytest.fixture()
def mock_vector() -> AsyncMock:
    """AsyncMock standing in for VectorClient."""
    vector = AsyncMock()
    vector.initialize = AsyncMock()
    vector.close = AsyncMock()
    vector.upsert = AsyncMock(return_value="vec-uuid-1")
    vector.search = AsyncMock(return_value=[])
    vector._embed = AsyncMock(return_value=[0.1] * 768)
    return vector


@pytest.fixture()
def mock_redis() -> AsyncMock:
    """AsyncMock standing in for redis.asyncio.Redis."""
    r = AsyncMock()
    r.lpush = AsyncMock(return_value=1)
    r.rpop = AsyncMock(return_value=None)
    r.aclose = AsyncMock()
    return r


@pytest.fixture()
def test_client(mock_graph: AsyncMock, mock_vector: AsyncMock, mock_redis: AsyncMock) -> TestClient:
    """FastAPI TestClient with all external dependencies mocked via dependency overrides.

    We replace the real lifespan with a no-op so the TestClient does not
    attempt to connect to Neo4j, Qdrant, or Redis during startup.
    """

    async def _override_graph():
        return mock_graph

    async def _override_vector():
        return mock_vector

    async def _override_redis():
        return mock_redis

    # Swap out the lifespan to avoid real connections
    original_router_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _noop_lifespan(a: FastAPI):
        yield

    app.router.lifespan_context = _noop_lifespan

    app.dependency_overrides[get_graph] = _override_graph
    app.dependency_overrides[get_vector] = _override_vector
    app.dependency_overrides[get_redis] = _override_redis

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()
    app.router.lifespan_context = original_router_lifespan
