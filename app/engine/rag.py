"""RAG (Retrieval-Augmented Generation) engine for NexusCortex.

Performs concurrent dual-retrieval from the knowledge graph (Neo4j) and
semantic memory (Qdrant), merges and scores results, and produces a
Markdown context block suitable for LLM system prompt injection.
"""

from __future__ import annotations

import asyncio
import logging
from difflib import SequenceMatcher
from typing import Any

from app.db.graph import Neo4jClient
from app.db.vector import VectorClient
from app.models import ContextQuery, MemorySource, RecallResponse

logger = logging.getLogger(__name__)

# Boost factor applied when a concept is found in both stores.
_CROSS_REF_BOOST = 1.5

# Minimum SequenceMatcher ratio to consider two text entries as matching.
_FUZZY_MATCH_THRESHOLD = 0.7


class RAGEngine:
    """Dual-retrieval cognitive engine combining graph and vector search."""

    def __init__(self, graph: Neo4jClient, vector: VectorClient) -> None:
        self._graph = graph
        self._vector = vector

    async def recall(self, query: ContextQuery) -> RecallResponse:
        """Retrieve, merge, score, and format memory context for an LLM.

        Steps:
            1. Concurrent dual-retrieval from Qdrant and Neo4j.
            2. Normalize scores to [0, 1].
            3. Fuzzy-match entries across stores; boost cross-referenced items.
            4. Deduplicate, sort by score descending, take top_k.
            5. Format as structured Markdown.
        """
        vector_results, graph_results = await self._dual_retrieve(query)

        vector_entries = self._normalize_vector(vector_results)
        graph_entries = self._normalize_graph(graph_results)

        merged = self._merge_and_boost(vector_entries, graph_entries)

        # Sort descending by score, take top_k.
        merged.sort(key=lambda e: e["score"], reverse=True)
        merged = merged[: query.top_k]

        context_block = self._format_markdown(merged)
        sources = [
            MemorySource(
                store=entry["store"],
                content=entry["content"],
                score=round(entry["score"], 4),
                metadata=entry.get("metadata", {}),
            )
            for entry in merged
        ]
        aggregate_score = (
            round(max(s.score for s in sources), 4)
            if sources
            else 0.0
        )

        return RecallResponse(
            context_block=context_block,
            sources=sources,
            score=aggregate_score,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def _dual_retrieve(
        self, query: ContextQuery
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run vector and graph queries concurrently.

        If one store fails, log the error and return empty results for that
        store so the other can still contribute.
        """

        async def _safe_vector() -> list[dict[str, Any]]:
            try:
                return await self._vector.search(
                    query.task,
                    top_k=query.top_k,
                    filter_tags=query.tags or None,
                )
            except Exception:
                logger.exception("Vector search failed")
                return []

        async def _safe_graph() -> list[dict[str, Any]]:
            try:
                return await self._graph.query_related(
                    query.task, limit=query.top_k
                )
            except Exception:
                logger.exception("Graph query failed")
                return []

        vector_results, graph_results = await asyncio.gather(
            _safe_vector(), _safe_graph()
        )
        return vector_results, graph_results

    # ------------------------------------------------------------------
    # Score normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_vector(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert raw vector search results into scored entries.

        Vector scores are already cosine similarity in [0, 1].
        """
        entries: list[dict[str, Any]] = []
        for r in results:
            text = r.get("text", "")
            if not text:
                continue
            entries.append(
                {
                    "content": text,
                    "score": float(r.get("score", 0.0)),
                    "store": "vector",
                    "metadata": r.get("metadata", {}),
                }
            )
        return entries

    @staticmethod
    def _normalize_graph(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert raw graph traversal results into scored entries.

        Graph scores use ``1.0 / distance`` where *distance* is the path
        length returned by Neo4j (number of hops, minimum 1).
        """
        entries: list[dict[str, Any]] = []
        for r in results:
            name = r.get("name") or ""
            description = r.get("description") or ""
            label = r.get("label") or "Entity"
            distance = r.get("distance")

            # Build readable content from whatever fields are present.
            content = description if description else name
            if not content:
                continue

            score = 1.0 / max(int(distance), 1) if distance else 0.5

            entries.append(
                {
                    "content": content,
                    "score": score,
                    "store": "graph",
                    "metadata": {
                        "name": name,
                        "label": label,
                        "distance": distance,
                    },
                }
            )
        return entries

    # ------------------------------------------------------------------
    # Merge, boost, and deduplicate
    # ------------------------------------------------------------------

    def _merge_and_boost(
        self,
        vector_entries: list[dict[str, Any]],
        graph_entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge entries from both stores, boosting cross-referenced items.

        For each pair of (vector_entry, graph_entry) whose text content has
        a fuzzy overlap ratio > 0.7, the higher-scoring entry is kept with
        its score multiplied by the boost factor and its store set to
        ``"both"``.  The other entry is consumed (removed).
        """
        # Track which graph entries have been consumed by a cross-ref match.
        graph_consumed: set[int] = set()
        merged: list[dict[str, Any]] = []

        for v_entry in vector_entries:
            matched = False
            for g_idx, g_entry in enumerate(graph_entries):
                if g_idx in graph_consumed:
                    continue
                if self._is_fuzzy_match(v_entry["content"], g_entry["content"]):
                    # Cross-referenced: combine the best score with boost.
                    best_score = max(v_entry["score"], g_entry["score"])
                    boosted_score = min(best_score * _CROSS_REF_BOOST, 1.0)

                    merged.append(
                        {
                            "content": v_entry["content"],
                            "score": boosted_score,
                            "store": "both",
                            "metadata": {
                                **v_entry.get("metadata", {}),
                                "graph_name": g_entry.get("metadata", {}).get(
                                    "name", ""
                                ),
                                "graph_label": g_entry.get("metadata", {}).get(
                                    "label", ""
                                ),
                            },
                        }
                    )
                    graph_consumed.add(g_idx)
                    matched = True
                    break

            if not matched:
                merged.append(v_entry)

        # Add remaining (un-consumed) graph entries.
        for g_idx, g_entry in enumerate(graph_entries):
            if g_idx not in graph_consumed:
                merged.append(g_entry)

        return merged

    @staticmethod
    def _is_fuzzy_match(a: str, b: str) -> bool:
        """Return True if the two strings are sufficiently similar.

        Compares only the first 200 characters for performance, since
        SequenceMatcher is O(N*M) on the full content strings.
        """
        if not a or not b:
            return False
        a_trimmed = a[:200].lower()
        b_trimmed = b[:200].lower()
        ratio = SequenceMatcher(None, a_trimmed, b_trimmed).ratio()
        return ratio > _FUZZY_MATCH_THRESHOLD

    # ------------------------------------------------------------------
    # Markdown formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_markdown(entries: list[dict[str, Any]]) -> str:
        """Build a structured Markdown block for LLM system prompt injection."""
        if not entries:
            return "## Relevant Memory Context\n\nNo relevant memories found."

        graph_lines: list[str] = []
        vector_lines: list[str] = []
        cross_lines: list[str] = []

        for entry in entries:
            store = entry["store"]
            content = entry["content"]
            score = entry["score"]
            meta = entry.get("metadata", {})

            if store == "both":
                graph_name = meta.get("graph_name", "")
                ref = f" \u27f7 Graph: {graph_name}" if graph_name else ""
                cross_lines.append(f"- [{score:.2f}] {content}{ref}")
            elif store == "graph":
                label = meta.get("label", "Entity")
                name = meta.get("name", "")
                header = f"**[{label}]** {name}" if name else f"**[{label}]**"
                if content != name:
                    graph_lines.append(f"- {header} \u2192 {content}")
                else:
                    graph_lines.append(f"- {header}")
            else:  # vector
                source = meta.get("source", "")
                tags = meta.get("tags", [])
                tag_str = ", ".join(tags) if tags else "none"
                vector_lines.append(f"- [{score:.2f}] {content}")
                vector_lines.append(
                    f"  - Source: {source}, Tags: {tag_str}"
                )

        sections: list[str] = ["## Relevant Memory Context"]

        if graph_lines:
            sections.append("")
            sections.append("### From Knowledge Graph")
            sections.extend(graph_lines)

        if vector_lines:
            sections.append("")
            sections.append("### From Semantic Memory")
            sections.extend(vector_lines)

        if cross_lines:
            sections.append("")
            sections.append("### Cross-Referenced (High Confidence)")
            sections.extend(cross_lines)

        return "\n".join(sections)
