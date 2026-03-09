"""Garbage Collection worker — periodic pruning of old, low-value memories.

Removes expired memories from Qdrant and orphaned nodes from Neo4j
based on configurable age and feedback thresholds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from qdrant_client import QdrantClient

from app.config import get_settings
from app.workers.sleep_cycle import celery_app, _get_neo4j_driver

logger = logging.getLogger(__name__)


def _get_qdrant_client() -> QdrantClient:
    """Create a synchronous Qdrant client from settings."""
    s = get_settings()
    return QdrantClient(host=s.QDRANT_HOST, port=s.QDRANT_PORT)


@celery_app.task(name="app.workers.gc.prune_memories")
def prune_memories() -> dict[str, Any]:
    """Remove memories older than MAX_MEMORY_AGE_DAYS with score below PRUNE_SCORE_THRESHOLD.

    1. Query Qdrant for points where created_at is older than MAX_MEMORY_AGE_DAYS
    2. Among those, delete points with no positive feedback (feedback_useful != True)
    3. For Neo4j, find orphaned nodes (nodes with no relationships) and delete them
    4. Log how many memories were pruned
    5. Return {"status": "completed", "pruned_vector": N, "pruned_graph": M}
    """
    try:
        return _prune()
    except Exception:
        logger.exception("Unhandled error in prune_memories")
        return {"status": "error", "pruned_vector": 0, "pruned_graph": 0}


def _prune() -> dict[str, Any]:
    """Core pruning logic."""
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.MAX_MEMORY_AGE_DAYS)
    cutoff_iso = cutoff.isoformat()

    pruned_vector = _prune_qdrant(settings, cutoff_iso)
    pruned_graph = _prune_neo4j_orphans()

    logger.info(
        "GC completed: pruned %d vector points, %d orphaned graph nodes",
        pruned_vector,
        pruned_graph,
    )

    return {
        "status": "completed",
        "pruned_vector": pruned_vector,
        "pruned_graph": pruned_graph,
    }


def _prune_qdrant(settings: Any, cutoff_iso: str) -> int:
    """Delete old Qdrant points that have no positive feedback.

    Scrolls through all points and filters client-side for:
      - timestamp < cutoff (old memories)
      - feedback_useful != True (no positive feedback)
    Then deletes matching points by ID.
    """
    client = _get_qdrant_client()
    collection = settings.QDRANT_COLLECTION
    pruned = 0

    try:
        offset = None
        batch_size = 100
        ids_to_delete: list[str] = []

        while True:
            results = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_offset = results

            for point in points:
                payload = point.payload or {}
                ts = payload.get("timestamp", "")

                # Compare ISO timestamps as strings (lexicographic comparison
                # works correctly for ISO 8601 format)
                if not ts or ts >= cutoff_iso:
                    continue

                # Keep memories that have received positive feedback
                if payload.get("feedback_useful") is True:
                    continue

                ids_to_delete.append(str(point.id))

            if next_offset is None or not points:
                break
            offset = next_offset

        # Delete collected point IDs
        if ids_to_delete:
            from qdrant_client.models import PointIdsList
            client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=ids_to_delete),
            )
            pruned = len(ids_to_delete)

    except Exception:
        logger.exception("Error pruning Qdrant points")

    finally:
        client.close()

    return pruned


def _prune_neo4j_orphans() -> int:
    """Delete orphaned Neo4j nodes (nodes with no relationships)."""
    try:
        driver = _get_neo4j_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (n) "
                "WHERE (n:Domain OR n:Concept OR n:Action "
                "       OR n:Outcome OR n:Resolution OR n:EventStream) "
                "  AND NOT (n)--() "
                "WITH n LIMIT 1000 "
                "DETACH DELETE n "
                "RETURN count(n) AS cnt"
            )
            record = result.single()
            return record["cnt"] if record else 0
    except Exception:
        logger.exception("Error pruning orphaned Neo4j nodes")
        return 0
