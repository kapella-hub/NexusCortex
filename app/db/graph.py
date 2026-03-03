"""Neo4j graph database client for NexusCortex.

Handles action log persistence, knowledge graph writes from the Sleep Cycle,
and graph traversal queries for the RAG engine.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
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

# Domain-specific stopwords: common in engineering contexts but low
# discriminative power for graph lookups.
_DOMAIN_STOPWORDS = frozenset({
    "error", "bug", "fix", "issue", "problem", "code", "file",
    "change", "make", "need", "want", "like", "just", "also",
})

# Synonym map for query expansion.
_SYNONYMS: dict[str, str] = {
    "db": "database",
    "database": "db",
    "auth": "authentication",
    "authentication": "auth",
    "config": "configuration",
    "configuration": "config",
    "env": "environment",
    "environment": "env",
    "deps": "dependencies",
    "dependencies": "deps",
    "perf": "performance",
    "performance": "perf",
    "mem": "memory",
    "memory": "mem",
    "conn": "connection",
    "connection": "conn",
    "err": "error",
    "req": "request",
    "request": "req",
    "res": "response",
    "response": "res",
}

# Valid node labels used in fulltext index and label filtering.
_GRAPH_LABELS = frozenset({"Domain", "Concept", "Action", "Outcome", "Resolution"})


class Neo4jClient:
    """Async Neo4j client for the NexusCortex knowledge graph."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
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
                max_connection_pool_size=self._settings.NEO4J_POOL_SIZE,
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

    def _content_hash(self, text: str) -> str:
        """Return a stable short hash of SHA-256, length from settings."""
        length = self._settings.CONTENT_HASH_LENGTH
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Sanitize a label to alphanumeric + underscore only."""
        safe = "".join(c for c in label if c.isalnum() or c == "_")
        return safe if safe else "Entity"

    @staticmethod
    def _canonicalize(name: str) -> str:
        """Canonicalize an entity name for consistent MERGE matching.

        Lowercase, strip whitespace, collapse multiple spaces, replace
        hyphens with underscores.
        """
        name = name.lower().strip()
        name = re.sub(r"\s+", " ", name)
        name = name.replace("-", "_")
        return name

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 7) -> list[str]:
        """Extract meaningful keywords from text for graph queries.

        Uses inverse-frequency weighting (word length + rarity heuristic)
        and bigram extraction for compound terms. Filters general and
        domain-specific stopwords.
        """
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
        # Filter tokens
        filtered: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if len(tok) < 3 or tok in _STOPWORDS or tok in _DOMAIN_STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            filtered.append(tok)

        # Count occurrences for inverse-frequency weighting
        counts = Counter(tokens)

        # Score by specificity: longer words and rarer words score higher.
        # inverse frequency proxy: 1 / count. length bonus: len(word).
        def _specificity(word: str) -> float:
            freq = counts.get(word, 1)
            return len(word) + (1.0 / freq)

        # Build bigrams from consecutive non-stopword tokens
        bigrams: list[str] = []
        non_stop = [
            t for t in tokens
            if len(t) >= 3 and t not in _STOPWORDS and t not in _DOMAIN_STOPWORDS
        ]
        for i in range(len(non_stop) - 1):
            bigram = f"{non_stop[i]} {non_stop[i + 1]}"
            if bigram not in seen:
                seen.add(bigram)
                bigrams.append(bigram)

        # Combine unigrams + bigrams, score all
        candidates = filtered + bigrams

        # Bigrams get a bonus (sum of component lengths + bonus)
        def _score(candidate: str) -> float:
            if " " in candidate:
                parts = candidate.split()
                return sum(_specificity(p) for p in parts) + 2.0  # bigram bonus
            return _specificity(candidate)

        candidates.sort(key=_score, reverse=True)
        return candidates[:max_keywords]

    @staticmethod
    def _expand_keywords(keywords: list[str]) -> list[str]:
        """Expand keywords with synonyms from the synonym map.

        For each keyword, if a synonym exists in _SYNONYMS, include both
        the original and the synonym.
        """
        expanded: list[str] = []
        seen: set[str] = set()
        for kw in keywords:
            # Handle bigrams: expand each word in the bigram
            if " " in kw:
                if kw not in seen:
                    seen.add(kw)
                    expanded.append(kw)
                continue
            if kw not in seen:
                seen.add(kw)
                expanded.append(kw)
            synonym = _SYNONYMS.get(kw)
            if synonym and synonym not in seen:
                seen.add(synonym)
                expanded.append(synonym)
        return expanded

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
            # Fulltext index across all major node types for relevance-scored search.
            (
                "CREATE FULLTEXT INDEX node_fulltext IF NOT EXISTS "
                "FOR (n:Domain|Concept|Action|Outcome|Resolution) "
                "ON EACH [n.name, n.description]"
            ),
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
        Edges may optionally include source_label and target_label for
        label-aware matching (improves performance). Falls back to
        label-less match if labels are not provided.

        Entity names are canonicalized before MERGE to reduce duplicates.

        All writes are wrapped in a single explicit transaction.
        Returns total count of created/merged items.
        """
        driver = self._ensure_driver()
        count = 0

        try:
            async with driver.session() as session:
                async with session.begin_transaction() as tx:
                    # --- Merge nodes with dynamic labels ---
                    if nodes:
                        label_groups: dict[str, list[dict[str, Any]]] = {}
                        for node in nodes:
                            label = node.get("label", "Entity")
                            # Canonicalize name in properties if present
                            props = dict(node.get("properties", {}))
                            if "name" in props:
                                props["name"] = self._canonicalize(props["name"])
                            canonicalized_node = {
                                "id": node["id"],
                                "label": label,
                                "properties": props,
                            }
                            label_groups.setdefault(label, []).append(canonicalized_node)

                        for label, group in label_groups.items():
                            safe_label = self._sanitize_label(label)

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
                            safe_type = self._sanitize_label(rel_type)
                            if safe_type == "Entity":
                                safe_type = "RELATED_TO"

                            # Check if edges in this group have label info
                            sample = group[0]
                            has_labels = "source_label" in sample and "target_label" in sample

                            if has_labels:
                                # Group edges by (source_label, target_label) for
                                # label-aware matching
                                label_pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
                                labelless: list[dict[str, Any]] = []
                                for edge in group:
                                    sl = edge.get("source_label")
                                    tl = edge.get("target_label")
                                    if sl and tl:
                                        safe_sl = self._sanitize_label(sl)
                                        safe_tl = self._sanitize_label(tl)
                                        label_pair_groups.setdefault(
                                            (safe_sl, safe_tl), []
                                        ).append(edge)
                                    else:
                                        labelless.append(edge)

                                for (safe_sl, safe_tl), lp_group in label_pair_groups.items():
                                    edge_query = (
                                        "UNWIND $edges AS edge "
                                        f"MATCH (src:{safe_sl} {{id: edge.source}}) "
                                        f"MATCH (tgt:{safe_tl} {{id: edge.target}}) "
                                        f"MERGE (src)-[r:{safe_type}]->(tgt) "
                                        "RETURN count(r) AS cnt"
                                    )
                                    result = await tx.run(edge_query, edges=lp_group)
                                    record = await result.single()
                                    if record:
                                        count += record["cnt"]

                                # Fall back to label-less for edges without labels
                                if labelless:
                                    edge_query = (
                                        "UNWIND $edges AS edge "
                                        "MATCH (src {id: edge.source}) "
                                        "MATCH (tgt {id: edge.target}) "
                                        f"MERGE (src)-[r:{safe_type}]->(tgt) "
                                        "RETURN count(r) AS cnt"
                                    )
                                    result = await tx.run(edge_query, edges=labelless)
                                    record = await result.single()
                                    if record:
                                        count += record["cnt"]
                            else:
                                # No label info — use original label-less match
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

        Tries fulltext index search first for relevance-scored results.
        Falls back to CONTAINS-based keyword matching if the fulltext
        index is not available.
        """
        keywords = self._extract_keywords(concept)
        if not keywords:
            return []

        # Expand keywords with synonyms
        expanded = self._expand_keywords(keywords)

        # Cap results to prevent explosion
        effective_limit = min(limit, 50)

        # Try fulltext search first
        result = await self._query_related_fulltext(expanded, effective_limit)
        if result is not None:
            return result

        # Fallback to CONTAINS-based search
        return await self._query_related_contains(expanded, effective_limit)

    async def _query_related_fulltext(
        self, keywords: list[str], limit: int
    ) -> list[dict[str, Any]] | None:
        """Try fulltext index search. Returns None if index doesn't exist."""
        # Build Lucene query string: OR-join all keywords
        # Escape special Lucene characters in keywords
        def _escape_lucene(term: str) -> str:
            special = r'+-&|!(){}[]^"~*?:\/'
            return "".join(f"\\{c}" if c in special else c for c in term)

        search_terms = " OR ".join(_escape_lucene(kw) for kw in keywords)

        query = """
        CALL db.index.fulltext.queryNodes('node_fulltext', $search_terms)
        YIELD node, score
        WITH node AS start, score
        WHERE start:Domain OR start:Concept OR start:Action
              OR start:Outcome OR start:Resolution
        OPTIONAL MATCH (start)-[r*1..3]-(related)
        WHERE related:Domain OR related:Concept OR related:Action
              OR related:Outcome OR related:Resolution
        WITH DISTINCT
            COALESCE(related, start) AS result,
            score,
            CASE WHEN related IS NULL THEN 0 ELSE length(r) END AS distance
        RETURN result.name AS name,
               result.description AS description,
               labels(result)[0] AS label,
               distance,
               score
        ORDER BY score DESC, distance ASC
        LIMIT $limit
        """

        driver = self._ensure_driver()
        try:
            async with driver.session() as session:
                result = await session.run(
                    query, search_terms=search_terms, limit=limit
                )
                return [dict(record) async for record in result]
        except Exception:
            # Fulltext index may not exist yet — fall back gracefully
            logger.debug(
                "Fulltext index query failed, falling back to CONTAINS search"
            )
            return None

    async def _query_related_contains(
        self, keywords: list[str], limit: int
    ) -> list[dict[str, Any]]:
        """Fallback CONTAINS-based keyword search with label filtering."""
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

        # Label-bounded traversal
        query = f"""
        MATCH (start)-[r*1..3]-(related)
        WHERE (start:Domain OR start:Concept OR start:Action
               OR start:Outcome OR start:Resolution)
          AND ({where_clause})
          AND (related:Domain OR related:Concept OR related:Action
               OR related:Outcome OR related:Resolution)
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
