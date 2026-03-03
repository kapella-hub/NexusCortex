"""Sleep Cycle worker — background Celery task for cognitive consolidation.

Pops raw event batches from Redis, sends them through an LLM for knowledge
extraction, and writes the resulting nodes/edges to Neo4j.
"""

from __future__ import annotations

import atexit
import json
import logging
from typing import Any

import httpx
import redis
from celery import Celery
from neo4j import GraphDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app — settings loaded lazily via get_settings()
# ---------------------------------------------------------------------------


def _create_celery_app() -> Celery:
    """Create and configure the Celery app with settings loaded at call time."""
    s = get_settings()
    _app = Celery(
        "nexus_cortex",
        broker=s.CELERY_BROKER_URL,
        backend=s.CELERY_RESULT_BACKEND,
    )
    _app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        beat_schedule={
            "sleep-cycle-consolidation": {
                "task": "app.workers.sleep_cycle.process_event_batch",
                "schedule": 60.0,
            },
        },
    )
    return _app


celery_app = _create_celery_app()

# ---------------------------------------------------------------------------
# System prompt — used verbatim for LLM knowledge extraction
# ---------------------------------------------------------------------------

CONSOLIDATION_SYSTEM_PROMPT = (
    "You are the cognitive consolidation engine for Nexus Cortex. "
    "Your job is to analyze the following batch of raw logs, errors, or events, "
    "deduplicate the noise, and extract a structured Knowledge Graph.\n\n"
    "Identify the core concepts, the actions taken, any errors triggered, and how they relate.\n"
    "You must return ONLY a valid JSON object with the following structure. "
    "Do not include markdown formatting or explanations.\n\n"
    "{\n"
    '  "nodes": [\n'
    '    {"id": "unique_string", "label": "Concept|Action|Outcome|Resolution", '
    '"properties": {"description": "summary of the node"}}\n'
    "  ],\n"
    '  "edges": [\n'
    '    {"source": "node_id", "target": "node_id", '
    '"type": "RELATES_TO|CAUSED|RESOLVED_BY|UTILIZES"}\n'
    "  ]\n"
    "}"
)

# ---------------------------------------------------------------------------
# Cached Neo4j driver — reused across batch calls
# ---------------------------------------------------------------------------

_neo4j_driver = None


def _get_neo4j_driver():
    """Return a module-level Neo4j driver, creating it on first call."""
    global _neo4j_driver
    if _neo4j_driver is None:
        s = get_settings()
        _neo4j_driver = GraphDatabase.driver(
            s.NEO4J_URI,
            auth=(s.NEO4J_USER, s.NEO4J_PASSWORD),
        )
        atexit.register(_close_neo4j_driver)
    return _neo4j_driver


def _close_neo4j_driver():
    """Close the cached Neo4j driver on process exit."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only nodes that have required fields."""
    valid = []
    for node in nodes:
        if (
            isinstance(node, dict)
            and isinstance(node.get("id"), str)
            and isinstance(node.get("label"), str)
            and isinstance(node.get("properties"), dict)
        ):
            valid.append(node)
        else:
            logger.warning("Dropping invalid node: %s", node)
    return valid


def _validate_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only edges that have required fields."""
    valid = []
    for edge in edges:
        if (
            isinstance(edge, dict)
            and isinstance(edge.get("source"), str)
            and isinstance(edge.get("target"), str)
            and isinstance(edge.get("type"), str)
        ):
            valid.append(edge)
        else:
            logger.warning("Dropping invalid edge: %s", edge)
    return valid


# ---------------------------------------------------------------------------
# Neo4j sync write — mirrors Neo4jClient.merge_knowledge_nodes but sync
# ---------------------------------------------------------------------------

def _write_to_neo4j(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> int:
    """Write knowledge nodes and edges using a synchronous Neo4j session."""
    count = 0
    driver = _get_neo4j_driver()
    with driver.session() as session:
        # Merge nodes grouped by label
        if nodes:
            label_groups: dict[str, list[dict[str, Any]]] = {}
            for node in nodes:
                label = node.get("label", "Entity")
                label_groups.setdefault(label, []).append(node)

            for label, group in label_groups.items():
                safe_label = "".join(
                    c for c in label if c.isalnum() or c == "_"
                )
                if not safe_label:
                    safe_label = "Entity"

                query = (
                    "UNWIND $nodes AS node "
                    f"MERGE (n:{safe_label} {{id: node.id}}) "
                    "SET n += node.properties "
                    "RETURN count(n) AS cnt"
                )
                result = session.run(query, nodes=group)
                record = result.single()
                if record:
                    count += record["cnt"]

        # Merge edges grouped by relationship type
        if edges:
            rel_groups: dict[str, list[dict[str, Any]]] = {}
            for edge in edges:
                rel_type = edge.get("type", "RELATED_TO")
                rel_groups.setdefault(rel_type, []).append(edge)

            for rel_type, group in rel_groups.items():
                safe_type = "".join(
                    c for c in rel_type if c.isalnum() or c == "_"
                )
                if not safe_type:
                    safe_type = "RELATED_TO"

                query = (
                    "UNWIND $edges AS edge "
                    "MATCH (src {id: edge.source}) "
                    "MATCH (tgt {id: edge.target}) "
                    f"MERGE (src)-[r:{safe_type}]->(tgt) "
                    "RETURN count(r) AS cnt"
                )
                result = session.run(query, edges=group)
                record = result.single()
                if record:
                    count += record["cnt"]

    return count


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(name="app.workers.sleep_cycle.process_event_batch")
def process_event_batch() -> dict[str, Any]:
    """Pop a batch of events from Redis, extract knowledge via LLM, write to Neo4j."""
    try:
        return _process_batch()
    except Exception:
        logger.exception("Unhandled error in process_event_batch")
        return {"status": "error", "nodes": 0, "edges": 0}


def _process_batch() -> dict[str, Any]:
    """Core batch processing logic, separated for clarity."""
    settings = get_settings()
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

    # Pop up to REDIS_BATCH_SIZE items using a pipeline for efficiency
    pipe = redis_client.pipeline()
    for _ in range(settings.REDIS_BATCH_SIZE):
        pipe.rpop(settings.REDIS_STREAM_KEY)
    results = pipe.execute()
    raw_items: list[str] = [r for r in results if r is not None]

    if not raw_items:
        return {"status": "empty", "nodes": 0, "edges": 0}

    # Parse events and build batch text
    events: list[dict[str, Any]] = []
    for raw in raw_items:
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed event JSON: %.200s", raw)
            events.append({"raw": raw})

    # Truncate individual event payloads to limit LLM prompt injection surface
    _MAX_EVENT_CHARS = 2000
    truncated_events = []
    for evt in events:
        serialized = json.dumps(evt, default=str)
        if len(serialized) > _MAX_EVENT_CHARS:
            serialized = serialized[:_MAX_EVENT_CHARS] + "...(truncated)"
        truncated_events.append(serialized)

    batch_text = "\n---\n".join(truncated_events)

    # Call LLM
    try:
        llm_response = httpx.post(
            f"{settings.LLM_BASE_URL}/chat/completions",
            json={
                "model": settings.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": batch_text},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            timeout=120.0,
        )
        llm_response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("LLM API call failed: %s", exc)
        _send_to_dlq(redis_client, raw_items, settings)
        return {"status": "llm_error", "nodes": 0, "edges": 0}

    # Parse LLM JSON
    try:
        content = llm_response.json()["choices"][0]["message"]["content"]
        knowledge = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse LLM response: %s", exc)
        _send_to_dlq(redis_client, raw_items, settings)
        return {"status": "parse_error", "nodes": 0, "edges": 0}

    # Validate structure
    raw_nodes = knowledge.get("nodes", [])
    raw_edges = knowledge.get("edges", [])

    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        logger.error("LLM returned invalid structure: nodes/edges not lists")
        _send_to_dlq(redis_client, raw_items, settings)
        return {"status": "validation_error", "nodes": 0, "edges": 0}

    nodes = _validate_nodes(raw_nodes)
    edges = _validate_edges(raw_edges)

    if not nodes and not edges:
        logger.warning("No valid nodes or edges extracted from batch")
        return {"status": "empty_extraction", "nodes": 0, "edges": 0}

    # Write to Neo4j
    try:
        written = _write_to_neo4j(nodes, edges)
    except Exception as exc:
        logger.error("Neo4j write failed: %s", exc)
        _send_to_dlq(redis_client, raw_items, settings)
        return {"status": "neo4j_error", "nodes": 0, "edges": 0}

    logger.info(
        "Sleep Cycle processed batch: %d events -> %d nodes, %d edges (%d written)",
        len(events),
        len(nodes),
        len(edges),
        written,
    )
    return {
        "status": "ok",
        "nodes": len(nodes),
        "edges": len(edges),
        "written": written,
    }


def _send_to_dlq(
    redis_client: redis.Redis,
    items: list[str],
    settings: Any | None = None,
) -> None:
    """Push failed batch items to the dead-letter queue."""
    if settings is None:
        settings = get_settings()
    dlq_key = f"{settings.REDIS_STREAM_KEY}:dlq"
    try:
        for item in items:
            redis_client.lpush(dlq_key, item)
        logger.info("Sent %d items to DLQ (%s)", len(items), dlq_key)
    except Exception:
        logger.exception("Failed to send items to DLQ")
