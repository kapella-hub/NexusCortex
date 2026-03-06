"""Automatic backlink discovery for NexusCortex.

Inspired by Obsidian's bidirectional linking. When a new memory is stored,
this module finds semantically related memories (moderate similarity, below
the contradiction threshold) and creates RELATES_TO edges in the knowledge
graph, building a web of connections automatically.
"""

from __future__ import annotations

import logging

from app.db.graph import Neo4jClient
from app.db.vector import VectorClient

logger = logging.getLogger(__name__)

# Similarity range for backlinks: above this is "related", above 0.85 is contradiction
BACKLINK_MIN_THRESHOLD = 0.4
BACKLINK_MAX_THRESHOLD = 0.84  # Just below contradiction threshold


async def discover_backlinks(
    vector: VectorClient,
    graph: Neo4jClient,
    new_text: str,
    new_vector_id: str,
    new_graph_id: str | None,
    domain: str,
    namespace: str = "default",
    top_k: int = 5,
) -> list[dict]:
    """Find semantically related memories and create bidirectional graph links.

    Searches for memories with moderate similarity (0.4–0.84) to the new
    memory. For each match, creates a RELATES_TO edge in Neo4j (if graph
    IDs are available). Skips memories that are already superseded by this
    memory (handled by contradiction detection).

    Returns list of discovered backlinks: [{"id": ..., "score": ..., "text": ...}]
    """
    backlinks: list[dict] = []

    try:
        similar = await vector.find_similar(
            text=new_text,
            namespace=namespace,
            domain=None,  # Cross-domain backlinks — broader discovery
            threshold=BACKLINK_MIN_THRESHOLD,
            top_k=top_k + 2,  # Fetch extra to account for filtering
        )

        for match in similar:
            # Skip self
            if match["id"] == new_vector_id:
                continue

            # Skip if above contradiction threshold (handled elsewhere)
            if match["score"] >= BACKLINK_MAX_THRESHOLD + 0.01:
                continue

            # Skip if below minimum
            if match["score"] < BACKLINK_MIN_THRESHOLD:
                continue

            backlinks.append({
                "id": match["id"],
                "score": round(match["score"], 4),
                "text": match.get("text", "")[:200],
            })

            # Create bidirectional graph edge if we have graph IDs
            if new_graph_id:
                try:
                    await graph.create_backlink(
                        source_id=new_graph_id,
                        target_vector_id=match["id"],
                        score=match["score"],
                    )
                except Exception:
                    logger.debug("Failed to create backlink edge for %s", match["id"])

            if len(backlinks) >= top_k:
                break

    except Exception:
        logger.warning("Backlink discovery failed, proceeding without backlinks")

    return backlinks
