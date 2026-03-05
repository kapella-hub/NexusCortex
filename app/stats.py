"""Stats router for NexusCortex — memory statistics endpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter

from app.models import MemoryStats

if TYPE_CHECKING:
    from app.db.graph import Neo4jClient
    from app.db.vector import VectorClient

logger = logging.getLogger(__name__)


def create_stats_router(graph: Neo4jClient, vector: VectorClient, redis_client) -> APIRouter:
    """Create the stats router with injected dependencies."""
    router = APIRouter(tags=["stats"])

    @router.get("/memory/stats", response_model=MemoryStats)
    async def memory_stats() -> MemoryStats:
        """Return aggregated memory statistics from all stores."""
        # Query all three stores concurrently
        import asyncio

        graph_stats_task = asyncio.create_task(graph.get_stats())
        vector_stats_task = asyncio.create_task(vector.get_stats())

        # DLQ depth from Redis
        try:
            dlq_depth = await redis_client.llen("nexus:event_stream:dlq")
        except Exception:
            dlq_depth = 0

        graph_stats = await graph_stats_task
        vector_stats = await vector_stats_task

        return MemoryStats(
            total_memories=vector_stats.get("total", 0),
            graph_nodes=graph_stats.get("node_count", 0),
            graph_edges=graph_stats.get("edge_count", 0),
            domains=graph_stats.get("domains", []),
            top_tags=graph_stats.get("top_tags", []),
            dlq_depth=dlq_depth,
            oldest_memory=vector_stats.get("oldest_memory"),
            newest_memory=vector_stats.get("newest_memory"),
            namespace_counts=vector_stats.get("namespace_counts", {}),
        )

    return router
