"""Neo4j graph database client for NexusCortex.

Handles action log persistence, knowledge graph writes from the Sleep Cycle,
and graph traversal queries for the RAG engine.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

import neo4j
from neo4j import AsyncGraphDatabase

from app.exceptions import GraphConnectionError

if TYPE_CHECKING:
    from app.config import Settings
    from app.models import ActionLog

logger = logging.getLogger(__name__)

# Stopwords filtered out during keyword extraction for graph queries.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "of", "in",
    "to", "for", "with", "on", "at", "from", "by", "as", "into", "about",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "same", "than", "too", "very",
    "just", "because", "if", "when", "while", "where", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it", "its",
})


class Neo4jClient:
    """Async Neo4j client for the NexusCortex knowledge graph."""

    def __init__(self, settings: Settings) -> None:
        self._uri = settings.NEO4J_URI
        self._user = settings.NEO4J_USER
        self._password = settings.NEO4J_PASSWORD
        self._driver: neo4j.AsyncDriver | None = None

    async def connect(self) -> None:
        """Create the async Neo4j driver."""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            await self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self._uri)
        except Exception as exc:
            raise GraphConnectionError(
                f"Failed to connect to Neo4j at {self._uri}: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    def _ensure_driver(self) -> neo4j.AsyncDriver:
        if self._driver is None:
            raise GraphConnectionError("Neo4j driver is not connected")
        return self._driver

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        """Return a stable short hash (first 16 hex chars of SHA-256)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
        """Extract meaningful keywords from text for graph CONTAINS queries.

        Splits on non-alphanumeric boundaries, removes stopwords and short
        tokens, then returns the longest unique words (up to *max_keywords*).
        """
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
        seen: set[str] = set()
        unique: list[str] = []
        for tok in tokens:
            if len(tok) < 3 or tok in _STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            unique.append(tok)
        # Prefer longer (more specific) words first.
        unique.sort(key=len, reverse=True)
        return unique[:max_keywords]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create Neo4j indexes if they do not already exist.

        Should be called once during application startup.
        """
        index_statements = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Action) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Outcome) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Resolution) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Domain) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.name)",
        ]
        driver = self._ensure_driver()
        try:
            async with driver.session() as session:
                for stmt in index_statements:
                    await session.run(stmt)
            logger.info("Neo4j indexes ensured")
        except Exception as exc:
            raise GraphConnectionError(
                f"Failed to ensure Neo4j indexes: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def merge_action_log(self, log: ActionLog) -> str:
        """Persist an action log as a Domain->Action->Outcome->Resolution chain.

        Creates Concept nodes for each tag and links them to the Action.
        Returns the element ID of the Action node.

        All writes are wrapped in a single explicit transaction so that
        a failure partway through does not leave partial data.
        """
        action_id = self._content_hash(log.action)
        outcome_id = self._content_hash(log.outcome)

        query = """
        MERGE (d:Domain {name: $domain})
        MERGE (a:Action {id: $action_id})
        SET a.description = $action
        MERGE (o:Outcome {id: $outcome_id})
        SET o.description = $outcome
        MERGE (a)-[:RELATES_TO]->(d)
        MERGE (a)-[:CAUSED]->(o)
        WITH a, o, d
        FOREACH (_ IN CASE WHEN $resolution_id IS NOT NULL THEN [1] ELSE [] END |
            MERGE (r:Resolution {id: $resolution_id})
            SET r.description = $resolution
            MERGE (o)-[:RESOLVED_BY]->(r)
            MERGE (r)-[:UTILIZES]->(d)
        )
        FOREACH (tag IN $tags |
            MERGE (c:Concept {name: tag})
            MERGE (a)-[:RELATES_TO]->(c)
        )
        RETURN elementId(a) AS id
        """

        resolution_id = (
            self._content_hash(log.resolution) if log.resolution else None
        )

        driver = self._ensure_driver()
        try:
            async with driver.session() as session:
                async with session.begin_transaction() as tx:
                    result = await tx.run(
                        query,
                        domain=log.domain,
                        action=log.action,
                        action_id=action_id,
                        outcome=log.outcome,
                        outcome_id=outcome_id,
                        resolution=log.resolution,
                        resolution_id=resolution_id,
                        tags=log.tags,
                    )
                    record = await result.single()
                    if record is None:
                        raise GraphConnectionError(
                            "merge_action_log returned no result"
                        )
                    action_element_id = record["id"]
                    await tx.commit()
                return action_element_id
        except GraphConnectionError:
            raise
        except Exception as exc:
            raise GraphConnectionError(
                f"Failed to merge action log: {exc}"
            ) from exc

    async def merge_knowledge_nodes(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> int:
        """Bulk-write knowledge nodes and edges from Sleep Cycle LLM output.

        Uses UNWIND with dynamic labels and relationship types.
        Does NOT rely on APOC procedures.

        Each node dict must have: id, label, properties.
        Each edge dict must have: source, target, type.

        All writes are wrapped in a single explicit transaction.
        Returns total count of created/merged items.
        """
        driver = self._ensure_driver()
        count = 0

        try:
            async with driver.session() as session:
                async with session.begin_transaction() as tx:
                    # --- Merge nodes with dynamic labels ---
                    # Neo4j does not support fully dynamic labels in
                    # standard Cypher without APOC.  We batch by label
                    # so each query uses a static label in the MERGE.
                    if nodes:
                        label_groups: dict[str, list[dict[str, Any]]] = {}
                        for node in nodes:
                            label = node.get("label", "Entity")
                            label_groups.setdefault(label, []).append(node)

                        for label, group in label_groups.items():
                            # Validate label is a safe identifier (letters,
                            # digits, underscores only) to prevent injection.
                            safe_label = "".join(
                                c for c in label if c.isalnum() or c == "_"
                            )
                            if not safe_label:
                                safe_label = "Entity"

                            merge_query = (
                                "UNWIND $nodes AS node "
                                f"MERGE (n:{safe_label} {{id: node.id}}) "
                                "SET n += node.properties "
                                "RETURN count(n) AS cnt"
                            )
                            result = await tx.run(merge_query, nodes=group)
                            record = await result.single()
                            if record:
                                count += record["cnt"]

                    # --- Merge edges with dynamic relationship types ---
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

                            edge_query = (
                                "UNWIND $edges AS edge "
                                "MATCH (src {id: edge.source}) "
                                "MATCH (tgt {id: edge.target}) "
                                f"MERGE (src)-[r:{safe_type}]->(tgt) "
                                "RETURN count(r) AS cnt"
                            )
                            result = await tx.run(edge_query, edges=group)
                            record = await result.single()
                            if record:
                                count += record["cnt"]

                    await tx.commit()

        except GraphConnectionError:
            raise
        except Exception as exc:
            raise GraphConnectionError(
                f"Failed to merge knowledge nodes: {exc}"
            ) from exc

        return count

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def query_related(
        self, concept: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find nodes related to a concept via graph traversal (up to 3 hops).

        Extracts keywords from the input text and builds case-insensitive
        CONTAINS conditions so that long free-text queries still match.
        """
        keywords = self._extract_keywords(concept)
        if not keywords:
            return []

        # Build OR-chained CONTAINS conditions for each keyword.
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        for i, kw in enumerate(keywords):
            pname = f"kw{i}"
            params[pname] = kw
            conditions.append(
                f"(toLower(start.name) CONTAINS ${pname} "
                f"OR toLower(start.description) CONTAINS ${pname})"
            )
        where_clause = " OR ".join(conditions)

        query = f"""
        MATCH (start)-[r*1..3]-(related)
        WHERE {where_clause}
        RETURN DISTINCT related.name AS name,
               related.description AS description,
               labels(related)[0] AS label,
               length(r) AS distance
        ORDER BY distance ASC
        LIMIT $limit
        """
        driver = self._ensure_driver()
        try:
            async with driver.session() as session:
                result = await session.run(query, **params)
                return [dict(record) async for record in result]
        except GraphConnectionError:
            raise
        except Exception as exc:
            raise GraphConnectionError(
                f"Failed to query related concepts: {exc}"
            ) from exc

    async def query_resolutions(
        self, error_pattern: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Find resolutions for outcomes matching the given error pattern."""
        query = """
        MATCH (o:Outcome)-[:RESOLVED_BY]->(r:Resolution)
        WHERE o.description CONTAINS $error_pattern
        RETURN r.description AS resolution,
               o.description AS error,
               elementId(r) AS id
        LIMIT $limit
        """
        driver = self._ensure_driver()
        try:
            async with driver.session() as session:
                result = await session.run(
                    query, error_pattern=error_pattern, limit=limit
                )
                return [dict(record) async for record in result]
        except GraphConnectionError:
            raise
        except Exception as exc:
            raise GraphConnectionError(
                f"Failed to query resolutions: {exc}"
            ) from exc
