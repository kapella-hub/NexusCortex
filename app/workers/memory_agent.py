"""Memory Agent worker — autonomous knowledge custodian.

Runs as a periodic Celery task performing six analysis and repair passes
against the knowledge corpus: duplicate detection, orphan cleanup,
deep contradiction scan, backlink reinforcement, confidence decay,
and cluster coherence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import redis
import redis.asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from app.config import get_settings
from app.workers.sleep_cycle import _get_neo4j_driver, _get_redis_client, celery_app

logger = logging.getLogger(__name__)

# Redis lock key for preventing overlapping agent runs
AGENT_LOCK_KEY = "nexus:memory_agent:lock"

# LLM prompt for merge synthesis
MERGE_SYSTEM_PROMPT = (
    "You are a knowledge consolidation engine. You are given a set of duplicate "
    "memory entries that describe the same concept or action. Synthesize them into "
    "a single improved memory that combines the best information from all entries.\n\n"
    "Return ONLY a JSON object with these fields:\n"
    '  {"text": "the synthesized memory text", "domain": "the domain", "tags": ["tag1", "tag2"]}\n\n'
    "Do not include markdown fencing or explanations. Output valid JSON only."
)


def _get_qdrant_client() -> QdrantClient:
    """Create a synchronous Qdrant client from settings."""
    s = get_settings()
    return QdrantClient(host=s.QDRANT_HOST, port=s.QDRANT_PORT)


def _fire_webhook_sync(redis_url: str, event_type: str, payload: dict[str, Any]) -> None:
    """Fire webhooks synchronously from the Celery worker context.

    Creates a temporary async event loop to call fire_webhooks.
    """
    try:
        from app.webhooks import fire_webhooks

        async def _fire():
            r = redis.asyncio.from_url(redis_url, decode_responses=True)
            try:
                await fire_webhooks(r, event_type, payload)
            finally:
                await r.aclose()

        asyncio.run(_fire())
    except Exception:
        logger.warning("Failed to fire webhook %s", event_type)


# ---------------------------------------------------------------------------
# Pass 1: Duplicate Detection & Merge
# ---------------------------------------------------------------------------


def duplicate_detection_pass() -> dict[str, Any]:
    """Find near-duplicate active memories and merge them.

    Scrolls Qdrant for active memories, groups by >threshold similarity,
    then LLM-synthesizes a merged memory (fallback: keep highest confidence).
    """
    settings = get_settings()
    client = _get_qdrant_client()
    collection = settings.QDRANT_COLLECTION
    threshold = settings.AGENT_DUPLICATE_THRESHOLD
    batch_limit = settings.AGENT_BATCH_LIMIT
    merged_count = 0
    results: list[dict] = []

    try:
        # Scroll all active memories
        memories: list[dict] = []
        offset = None
        while len(memories) < batch_limit:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value="active"))]
                ),
                limit=min(100, batch_limit - len(memories)),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                memories.append({
                    "id": str(p.id),
                    "text": p.payload.get("text", "") if p.payload else "",
                    "domain": p.payload.get("domain", "") if p.payload else "",
                    "tags": p.payload.get("tags", []) if p.payload else [],
                    "confirmed_count": p.payload.get("confirmed_count", 0) if p.payload else 0,
                    "contradicted_count": p.payload.get("contradicted_count", 0) if p.payload else 0,
                    "vector": p.vector,
                    "payload": p.payload or {},
                })
            if next_offset is None or not points:
                break
            offset = next_offset

        if len(memories) < 2:
            return {"status": "ok", "merged": 0, "details": []}

        # Find duplicate clusters using pairwise similarity
        # Use Qdrant search for each memory to find near-duplicates
        processed_ids: set[str] = set()
        clusters: list[list[dict]] = []

        for mem in memories:
            if mem["id"] in processed_ids:
                continue

            # Search for similar memories
            similar_results = client.query_points(
                collection_name=collection,
                query=mem["vector"],
                query_filter=Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value="active"))]
                ),
                limit=10,
                with_payload=True,
            )

            cluster = [mem]
            for point in similar_results.points:
                pid = str(point.id)
                if pid == mem["id"] or pid in processed_ids:
                    continue
                if point.score >= threshold:
                    # Find the full memory dict for this point
                    match = next((m for m in memories if m["id"] == pid), None)
                    if match:
                        cluster.append(match)

            if len(cluster) >= 2:
                clusters.append(cluster)
                for c in cluster:
                    processed_ids.add(c["id"])

        # Process each cluster
        for cluster in clusters:
            if merged_count >= batch_limit:
                break

            merge_result = _merge_cluster(client, cluster, settings)
            if merge_result:
                results.append(merge_result)
                merged_count += 1

    except Exception:
        logger.exception("Error in duplicate_detection_pass")
    finally:
        client.close()

    # Fire webhooks for each merge
    for r in results:
        _fire_webhook_sync(
            settings.REDIS_URL,
            "agent.merged",
            r,
        )

    return {"status": "ok", "merged": merged_count, "details": results}


def _merge_cluster(
    client: QdrantClient,
    cluster: list[dict],
    settings: Any,
) -> dict | None:
    """Merge a cluster of duplicate memories. Returns merge info or None."""
    # Try LLM merge first
    merged_text = None
    method = "fallback"

    try:
        texts = [m["text"] for m in cluster]
        prompt = "Merge these duplicate memories into one:\n\n" + "\n---\n".join(texts)

        response = httpx.post(
            f"{settings.LLM_BASE_URL}/chat/completions",
            json={
                "model": settings.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            timeout=60.0,
        )
        response.raise_for_status()
        msg = response.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        # Fallback: some models (e.g. qwen3) put output in a reasoning field
        if not content.strip():
            content = msg.get("reasoning") or ""
        parsed = json.loads(content)
        merged_text = parsed.get("text", "")

        # Validate: merged text should be reasonable length
        min_len = min(len(t) for t in texts) // 2
        if len(merged_text) >= min_len:
            method = "llm"
        else:
            merged_text = None
    except Exception:
        logger.warning("LLM merge failed, using fallback")

    # Fallback: keep highest confidence memory
    if merged_text is None:
        best = max(
            cluster,
            key=lambda m: (1 + m["confirmed_count"]) / (1 + m["contradicted_count"]),
        )
        merged_text = best["text"]

    # The "winner" is the best memory (highest confidence); supersede the rest
    if method == "llm":
        keeper = cluster[0]
    else:
        keeper = best
    superseded_ids = [m["id"] for m in cluster if m["id"] != keeper["id"]]

    # Update superseded memories' status
    for sid in superseded_ids:
        try:
            client.set_payload(
                collection_name=settings.QDRANT_COLLECTION,
                payload={
                    "status": "superseded",
                    "superseded_by": keeper["id"],
                },
                points=[sid],
            )
        except Exception:
            logger.warning("Failed to supersede memory %s", sid)

    # Update keeper with merged text if LLM produced one
    if method == "llm":
        try:
            client.set_payload(
                collection_name=settings.QDRANT_COLLECTION,
                payload={"text": merged_text},
                points=[keeper["id"]],
            )
        except Exception:
            logger.warning("Failed to update merged text for %s", keeper["id"])

    # Create SUPERSEDES edges in Neo4j
    try:
        driver = _get_neo4j_driver()
        with driver.session() as session:
            for sid in superseded_ids:
                session.run(
                    "MERGE (ref_new:MemoryRef {vector_id: $keeper_id}) "
                    "MERGE (ref_old:MemoryRef {vector_id: $old_id}) "
                    "MERGE (ref_new)-[:SUPERSEDES {reason: $reason, detected: 'agent', timestamp: datetime()}]->(ref_old)",
                    keeper_id=keeper["id"],
                    old_id=sid,
                    reason=f"Agent duplicate merge ({method})",
                )
    except Exception:
        logger.warning("Failed to create SUPERSEDES edges for merge")

    return {
        "merged_into": keeper["id"],
        "superseded": superseded_ids,
        "method": method,
    }


# ---------------------------------------------------------------------------
# Pass 2: Orphan Cleanup
# ---------------------------------------------------------------------------


def orphan_cleanup_pass() -> dict[str, Any]:
    """Delete orphaned nodes (degree 0) from Neo4j."""
    settings = get_settings()
    batch_limit = settings.AGENT_BATCH_LIMIT
    nodes_removed: list[dict] = []

    try:
        driver = _get_neo4j_driver()
        with driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(
                    "MATCH (n) "
                    "WHERE (n:Domain OR n:Concept OR n:Action "
                    "       OR n:Outcome OR n:Resolution OR n:MemoryRef) "
                    "  AND NOT (n)--() "
                    "WITH n LIMIT $limit "
                    "WITH n, labels(n)[0] AS label, "
                    "     COALESCE(n.name, n.description, n.id, n.vector_id, 'unnamed') AS name "
                    "DETACH DELETE n "
                    "RETURN label, name",
                    limit=batch_limit,
                )
                for record in result:
                    nodes_removed.append({
                        "label": record["label"],
                        "name": record["name"],
                    })
                tx.commit()
    except Exception:
        logger.exception("Error in orphan_cleanup_pass")

    if nodes_removed:
        _fire_webhook_sync(
            settings.REDIS_URL,
            "agent.orphan_cleaned",
            {"nodes_removed": nodes_removed},
        )

    return {"status": "ok", "nodes_removed": nodes_removed}


# ---------------------------------------------------------------------------
# Pass 3: Deep Contradiction Scan
# ---------------------------------------------------------------------------


def deep_contradiction_pass() -> dict[str, Any]:
    """Find contradictions missed at learn-time across different domains."""
    settings = get_settings()
    client = _get_qdrant_client()
    collection = settings.QDRANT_COLLECTION
    batch_limit = settings.AGENT_BATCH_LIMIT
    contradictions: list[dict] = []

    try:
        # Scroll active memories
        memories: list[dict] = []
        offset = None
        while len(memories) < batch_limit:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value="active"))]
                ),
                limit=min(100, batch_limit - len(memories)),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                memories.append({
                    "id": str(p.id),
                    "text": p.payload.get("text", "") if p.payload else "",
                    "domain": p.payload.get("domain", "") if p.payload else "",
                    "timestamp": p.payload.get("timestamp", "") if p.payload else "",
                    "confirmed_count": p.payload.get("confirmed_count", 0) if p.payload else 0,
                    "contradicted_count": p.payload.get("contradicted_count", 0) if p.payload else 0,
                    "vector": p.vector,
                })
            if next_offset is None or not points:
                break
            offset = next_offset

        processed_pairs: set[tuple[str, str]] = set()

        for mem in memories:
            if len(contradictions) >= batch_limit:
                break

            # Search for similar memories in 0.85-0.95 range (not already SUPERSEDES-linked)
            similar_results = client.query_points(
                collection_name=collection,
                query=mem["vector"],
                query_filter=Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value="active"))]
                ),
                limit=5,
                with_payload=True,
            )

            for point in similar_results.points:
                pid = str(point.id)
                if pid == mem["id"]:
                    continue

                pair_key = tuple(sorted([mem["id"], pid]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                if not (0.85 <= point.score <= 0.95):
                    continue

                # Check if already linked by SUPERSEDES in Neo4j
                if _has_supersedes_link(mem["id"], pid):
                    continue

                # Determine which is stale: older + lower confidence loses
                other = next((m for m in memories if m["id"] == pid), None)
                if other is None:
                    continue

                mem_confidence = (1 + mem["confirmed_count"]) / (1 + mem["contradicted_count"])
                other_confidence = (1 + other["confirmed_count"]) / (1 + other["contradicted_count"])

                # Stale side: lower confidence, or if equal, older
                if mem_confidence < other_confidence or (
                    mem_confidence == other_confidence and mem["timestamp"] < other["timestamp"]
                ):
                    stale, keeper = mem, other
                else:
                    stale, keeper = other, mem

                # Supersede the stale memory
                try:
                    client.set_payload(
                        collection_name=collection,
                        payload={
                            "status": "superseded",
                            "superseded_by": keeper["id"],
                        },
                        points=[stale["id"]],
                    )
                except Exception:
                    logger.warning("Failed to supersede memory %s", stale["id"])
                    continue

                # Create SUPERSEDES edge
                try:
                    driver = _get_neo4j_driver()
                    with driver.session() as session:
                        session.run(
                            "MERGE (ref_new:MemoryRef {vector_id: $keeper_id}) "
                            "MERGE (ref_old:MemoryRef {vector_id: $old_id}) "
                            "MERGE (ref_new)-[:SUPERSEDES {reason: $reason, detected: 'agent', timestamp: datetime()}]->(ref_old)",
                            keeper_id=keeper["id"],
                            old_id=stale["id"],
                            reason=f"Deep contradiction scan (similarity={point.score:.3f})",
                        )
                except Exception:
                    logger.warning("Failed to create SUPERSEDES edge")

                contradiction_info = {
                    "kept": keeper["id"],
                    "superseded": stale["id"],
                    "similarity": round(point.score, 4),
                }
                contradictions.append(contradiction_info)

                _fire_webhook_sync(
                    settings.REDIS_URL,
                    "agent.contradiction_found",
                    contradiction_info,
                )

    except Exception:
        logger.exception("Error in deep_contradiction_pass")
    finally:
        client.close()

    return {"status": "ok", "contradictions_found": len(contradictions), "details": contradictions}


def _has_supersedes_link(id_a: str, id_b: str) -> bool:
    """Check if two memories are already linked by SUPERSEDES in Neo4j."""
    try:
        driver = _get_neo4j_driver()
        with driver.session() as session:
            result = session.run(
                "OPTIONAL MATCH (a:MemoryRef {vector_id: $id_a})-[:SUPERSEDES]-(b:MemoryRef {vector_id: $id_b}) "
                "RETURN a IS NOT NULL AND b IS NOT NULL AS linked",
                id_a=id_a, id_b=id_b,
            )
            record = result.single()
            return bool(record and record["linked"])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pass 4: Backlink Reinforcement
# ---------------------------------------------------------------------------


def backlink_reinforcement_pass() -> dict[str, Any]:
    """Find memories with zero backlinks and create missing connections."""
    settings = get_settings()
    client = _get_qdrant_client()
    collection = settings.QDRANT_COLLECTION
    batch_limit = settings.AGENT_BATCH_LIMIT
    results: list[dict] = []

    try:
        # Find MemoryRef nodes with no BACKLINK edges
        driver = _get_neo4j_driver()
        orphan_vector_ids: list[str] = []

        with driver.session() as session:
            result = session.run(
                "MATCH (ref:MemoryRef) "
                "WHERE NOT (ref)-[:BACKLINK]-() "
                "RETURN ref.vector_id AS vector_id "
                "LIMIT $limit",
                limit=batch_limit,
            )
            orphan_vector_ids = [record["vector_id"] for record in result if record["vector_id"]]

        # Also find active memories not in MemoryRef at all
        if len(orphan_vector_ids) < batch_limit:
            remaining = batch_limit - len(orphan_vector_ids)
            offset = None
            checked = 0
            while checked < remaining:
                points, next_offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="status", match=MatchValue(value="active"))]
                    ),
                    limit=min(100, remaining - checked),
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                for p in points:
                    pid = str(p.id)
                    if pid not in orphan_vector_ids:
                        # Check if this memory has a MemoryRef with backlinks
                        with driver.session() as session:
                            check = session.run(
                                "OPTIONAL MATCH (ref:MemoryRef {vector_id: $vid})-[:BACKLINK]-() "
                                "RETURN ref IS NOT NULL AS has_backlinks",
                                vid=pid,
                            )
                            rec = check.single()
                            if not (rec and rec["has_backlinks"]):
                                orphan_vector_ids.append(pid)
                    checked += 1
                    if checked >= remaining:
                        break
                if next_offset is None or not points:
                    break
                offset = next_offset

        # For each orphan, discover backlinks
        for vid in orphan_vector_ids:
            if len(results) >= batch_limit:
                break

            # Get the memory's vector
            try:
                points_data = client.retrieve(
                    collection_name=collection,
                    ids=[vid],
                    with_payload=True,
                    with_vectors=True,
                )
                if not points_data:
                    continue
                point = points_data[0]
                vector = point.vector
                text = point.payload.get("text", "") if point.payload else ""
            except Exception:
                continue

            if not vector or not text:
                continue

            # Find similar memories in backlink range (0.4-0.84)
            similar_results = client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value="active"))]
                ),
                limit=7,
                with_payload=True,
            )

            new_backlinks = 0
            for sp in similar_results.points:
                sp_id = str(sp.id)
                if sp_id == vid:
                    continue
                if sp.score < 0.4 or sp.score > 0.84:
                    continue

                # Create bidirectional BACKLINK edge
                try:
                    with driver.session() as session:
                        session.run(
                            "MERGE (ref_a:MemoryRef {vector_id: $vid_a}) "
                            "MERGE (ref_b:MemoryRef {vector_id: $vid_b}) "
                            "MERGE (ref_a)-[:BACKLINK {score: $score, created: datetime()}]->(ref_b) "
                            "MERGE (ref_b)-[:BACKLINK {score: $score, created: datetime()}]->(ref_a)",
                            vid_a=vid, vid_b=sp_id, score=sp.score,
                        )
                    new_backlinks += 1
                except Exception:
                    logger.warning("Failed to create backlink %s <-> %s", vid, sp_id)

            if new_backlinks > 0:
                backlink_info = {"memory_id": vid, "new_backlinks": new_backlinks}
                results.append(backlink_info)
                _fire_webhook_sync(
                    settings.REDIS_URL,
                    "agent.backlinks_added",
                    backlink_info,
                )

    except Exception:
        logger.exception("Error in backlink_reinforcement_pass")
    finally:
        client.close()

    return {"status": "ok", "memories_linked": len(results), "details": results}


# ---------------------------------------------------------------------------
# Pass 5: Confidence Decay
# ---------------------------------------------------------------------------


def confidence_decay_pass() -> dict[str, Any]:
    """Decay confidence on old, unconfirmed memories."""
    settings = get_settings()
    client = _get_qdrant_client()
    collection = settings.QDRANT_COLLECTION
    batch_limit = settings.AGENT_BATCH_LIMIT
    decay_days = settings.AGENT_CONFIDENCE_DECAY_DAYS
    cutoff = (datetime.now(timezone.utc) - timedelta(days=decay_days)).isoformat()
    results: list[dict] = []

    try:
        offset = None
        processed = 0

        while processed < batch_limit:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value="active"))]
                ),
                limit=min(100, batch_limit - processed),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for p in points:
                if processed >= batch_limit:
                    break

                payload = p.payload or {}
                pid = str(p.id)

                # Check age — use last_confirmed_at if available, else timestamp
                age_ts = payload.get("last_confirmed_at") or payload.get("timestamp", "")
                if not age_ts or age_ts > cutoff:
                    continue

                confirmed = payload.get("confirmed_count", 0)
                contradicted = payload.get("contradicted_count", 0)
                confidence = (1 + confirmed) / (1 + contradicted)

                # Skip memories that can't be further decayed
                if confirmed == 0 and confidence >= 0.5:
                    continue

                if confidence < 0.5:
                    # Auto-deprecate
                    try:
                        client.set_payload(
                            collection_name=collection,
                            payload={"status": "deprecated"},
                            points=[pid],
                        )
                        decay_info = {
                            "memory_id": pid,
                            "action": "deprecated",
                            "new_count": confirmed,
                        }
                        results.append(decay_info)
                        _fire_webhook_sync(
                            settings.REDIS_URL,
                            "agent.confidence_decayed",
                            decay_info,
                        )
                    except Exception:
                        logger.warning("Failed to deprecate memory %s", pid)
                else:
                    # Decrement confirmed_count (floor at 0)
                    new_count = max(0, confirmed - 1)
                    try:
                        client.set_payload(
                            collection_name=collection,
                            payload={"confirmed_count": new_count},
                            points=[pid],
                        )
                        decay_info = {
                            "memory_id": pid,
                            "action": "reduced",
                            "new_count": new_count,
                        }
                        results.append(decay_info)
                        _fire_webhook_sync(
                            settings.REDIS_URL,
                            "agent.confidence_decayed",
                            decay_info,
                        )
                    except Exception:
                        logger.warning("Failed to decay confidence for %s", pid)

                processed += 1

            if next_offset is None or not points:
                break
            offset = next_offset

    except Exception:
        logger.exception("Error in confidence_decay_pass")
    finally:
        client.close()

    return {"status": "ok", "decayed": len(results), "details": results}


# ---------------------------------------------------------------------------
# Pass 6: Cluster Coherence
# ---------------------------------------------------------------------------


def cluster_coherence_pass() -> dict[str, Any]:
    """Check domain coherence and reclassify outlier memories."""
    settings = get_settings()
    client = _get_qdrant_client()
    collection = settings.QDRANT_COLLECTION
    batch_limit = settings.AGENT_BATCH_LIMIT
    results: list[dict] = []

    try:
        # Scroll all active memories, group by domain (bounded to avoid OOM)
        domains: dict[str, list[dict]] = {}
        offset = None
        max_scroll = batch_limit * 10  # upper bound on total memories scanned
        total_scanned = 0

        while total_scanned < max_scroll:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="status", match=MatchValue(value="active"))]
                ),
                limit=min(100, max_scroll - total_scanned),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            for p in points:
                total_scanned += 1
                payload = p.payload or {}
                domain = payload.get("domain", "general")
                domains.setdefault(domain, []).append({
                    "id": str(p.id),
                    "text": payload.get("text", ""),
                    "domain": domain,
                    "vector": p.vector,
                })

            if next_offset is None or not points:
                break
            offset = next_offset

        # Compute domain centroids (average vector)
        domain_centroids: dict[str, list[float]] = {}
        for domain, mems in domains.items():
            if len(mems) < 3:
                continue
            vectors = [m["vector"] for m in mems if m["vector"]]
            if not vectors:
                continue
            dim = len(vectors[0])
            centroid = [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]
            domain_centroids[domain] = centroid

        # For each domain with 3+ memories, find outliers
        for domain, mems in domains.items():
            if len(mems) < 3 or domain not in domain_centroids:
                continue
            if len(results) >= batch_limit:
                break

            centroid = domain_centroids[domain]

            # Compute similarity of each memory to its domain centroid
            similarities: list[tuple[dict, float]] = []
            for mem in mems:
                if not mem["vector"]:
                    continue
                sim = _cosine_similarity(mem["vector"], centroid)
                similarities.append((mem, sim))

            if len(similarities) < 3:
                continue

            scores = [s for _, s in similarities]
            mean_sim = statistics.mean(scores)
            std_sim = statistics.stdev(scores) if len(scores) > 1 else 0.0

            if std_sim == 0:
                continue

            # Flag outliers: more than 1 stddev below mean
            threshold = mean_sim - std_sim

            for mem, sim in similarities:
                if sim >= threshold:
                    continue
                if len(results) >= batch_limit:
                    break

                # Check if a better domain exists
                best_domain = domain
                best_sim = sim
                for other_domain, other_centroid in domain_centroids.items():
                    if other_domain == domain:
                        continue
                    other_sim = _cosine_similarity(mem["vector"], other_centroid)
                    if other_sim > best_sim + 0.1:  # Require 0.1 margin
                        best_domain = other_domain
                        best_sim = other_sim

                if best_domain != domain:
                    # Reclassify
                    try:
                        client.set_payload(
                            collection_name=collection,
                            payload={"domain": best_domain},
                            points=[mem["id"]],
                        )
                    except Exception:
                        logger.warning("Failed to reclassify memory %s", mem["id"])
                        continue

                    # Update domain relationship in Neo4j
                    try:
                        driver = _get_neo4j_driver()
                        with driver.session() as session:
                            with session.begin_transaction() as tx:
                                # Remove old domain relationship
                                tx.run(
                                    "MATCH (ref:MemoryRef {vector_id: $vid})-[r:RELATES_TO]->(d:Domain {name: $old_domain}) "
                                    "DELETE r",
                                    vid=mem["id"], old_domain=domain,
                                )
                                # Create new domain relationship
                                tx.run(
                                    "MERGE (ref:MemoryRef {vector_id: $vid}) "
                                    "MERGE (d:Domain {name: $new_domain}) "
                                    "MERGE (ref)-[:RELATES_TO]->(d)",
                                    vid=mem["id"], new_domain=best_domain,
                                )
                                tx.commit()
                    except Exception:
                        logger.warning("Failed to update Neo4j domain for %s", mem["id"])

                    reclassify_info = {
                        "memory_id": mem["id"],
                        "from_domain": domain,
                        "to_domain": best_domain,
                        "similarity_improvement": round(best_sim - sim, 4),
                    }
                    results.append(reclassify_info)
                    _fire_webhook_sync(
                        settings.REDIS_URL,
                        "agent.reclassified",
                        reclassify_info,
                    )

    except Exception:
        logger.exception("Error in cluster_coherence_pass")
    finally:
        client.close()

    return {"status": "ok", "reclassified": len(results), "details": results}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------


@celery_app.task(name="app.workers.memory_agent.run_memory_agent")
def run_memory_agent() -> dict[str, Any]:
    """Run all six memory agent passes sequentially.

    Checks AGENT_ENABLED kill switch, acquires Redis SETNX lock,
    runs each pass with isolated error handling, and releases the lock.
    """
    settings = get_settings()

    # Kill switch
    if not settings.AGENT_ENABLED:
        logger.info("Memory agent disabled via AGENT_ENABLED=false")
        return {"status": "disabled"}

    # Acquire lock
    redis_client = _get_redis_client()
    lock_ttl = settings.AGENT_SCHEDULE_HOURS * 3600
    acquired = redis_client.set(AGENT_LOCK_KEY, "1", nx=True, ex=lock_ttl)
    if not acquired:
        logger.info("Memory agent lock already held, skipping run")
        return {"status": "locked"}

    logger.info("Memory agent starting — 6 passes")
    results: dict[str, Any] = {"status": "ok", "passes": {}}

    passes = [
        ("duplicate_detection", duplicate_detection_pass),
        ("orphan_cleanup", orphan_cleanup_pass),
        ("deep_contradiction", deep_contradiction_pass),
        ("backlink_reinforcement", backlink_reinforcement_pass),
        ("confidence_decay", confidence_decay_pass),
        ("cluster_coherence", cluster_coherence_pass),
    ]

    for name, func in passes:
        try:
            logger.info("Memory agent: starting %s pass", name)
            result = func()
            results["passes"][name] = result
            logger.info("Memory agent: %s pass completed — %s", name, result.get("status", "unknown"))
        except Exception:
            logger.exception("Memory agent: %s pass failed", name)
            results["passes"][name] = {"status": "error"}

    # Release lock
    try:
        redis_client.delete(AGENT_LOCK_KEY)
    except Exception:
        logger.warning("Failed to release memory agent lock")

    logger.info("Memory agent completed all passes")
    return results
