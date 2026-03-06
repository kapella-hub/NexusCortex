"""RAG (Retrieval-Augmented Generation) engine for NexusCortex.

Performs concurrent dual-retrieval from the knowledge graph (Neo4j) and
semantic memory (Qdrant), merges and scores results, and produces a
Markdown context block suitable for LLM system prompt injection.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.db.graph import Neo4jClient
from app.db.vector import VectorClient
from app.models import ContextQuery, MemorySource, RecallResponse

logger = logging.getLogger(__name__)

# Regex pattern for detecting error-related queries.
_ERROR_PATTERN = re.compile(
    r"\b(error|fail|bug|crash|exception|broken|wrong|issue)\b",
    re.IGNORECASE,
)

# Minimum Jaccard similarity to consider two text entries as matching.
_JACCARD_MATCH_THRESHOLD = 0.3


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _min_max_normalize(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize entry scores to [0, 1] via min-max normalization.

    If the set has a single item, its score is capped at 1.0.
    If all scores are identical, they are all set to 1.0.
    """
    if not entries:
        return entries

    scores = [e["score"] for e in entries]
    min_s = min(scores)
    max_s = max(scores)

    if len(entries) == 1:
        entries[0]["score"] = min(entries[0]["score"], 1.0)
        return entries

    if max_s == min_s:
        for e in entries:
            e["score"] = 1.0
        return entries

    for e in entries:
        e["score"] = (e["score"] - min_s) / (max_s - min_s)

    return entries


class RAGEngine:
    """Dual-retrieval cognitive engine combining graph and vector search."""

    STATUS_MULTIPLIERS = {
        "active": 1.0,
        "superseded": 0.5,
        "deprecated": 0.1,
        "archived": 0.0,
    }

    def __init__(
        self,
        graph: Neo4jClient,
        vector: VectorClient,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._graph = graph
        self._vector = vector
        self._settings = settings or get_settings()
        self._http_client = http_client

    async def recall(self, query: ContextQuery) -> RecallResponse:
        """Retrieve, merge, score, and format memory context for an LLM.

        Steps:
            1. Concurrent dual-retrieval from Qdrant and Neo4j.
            2. Optionally query resolutions for error-related tasks.
            3. Normalize scores to [0, 1] per source via min-max.
            4. Fuzzy-match entries across stores; boost cross-referenced items.
            5. Apply memory decay based on age.
            6. Deduplicate, sort by score descending, take top_k.
            7. Optionally re-rank via LLM.
            8. Format as structured Markdown.
        """
        vector_results, graph_results = await self._dual_retrieve(query)

        vector_entries = self._normalize_vector(vector_results)
        graph_entries = self._format_graph_entries(graph_results, query.task)

        # Wire query_resolutions for error-related queries.
        if _ERROR_PATTERN.search(query.task):
            resolution_entries = await self._fetch_resolutions(query.task)
            graph_entries.extend(resolution_entries)

        # Min-max normalize each source independently.
        vector_entries = _min_max_normalize(vector_entries)
        graph_entries = _min_max_normalize(graph_entries)

        merged = self._merge_and_boost(vector_entries, graph_entries)

        # Apply memory decay.
        self._apply_decay(merged)

        # Apply lifecycle scoring (status + confidence multipliers).
        merged = self._apply_lifecycle_scoring(merged)

        # Sort descending by score, take top_k (or more for re-ranking).
        merged.sort(key=lambda e: e["score"], reverse=True)

        # Re-ranking pass (gated behind config).
        if self._settings.RERANK_ENABLED:
            candidate_count = query.top_k * self._settings.RERANK_CANDIDATES_MULTIPLIER
            candidates = merged[:candidate_count]
            merged = await self._rerank(query.task, candidates)
            merged.sort(key=lambda e: e["score"], reverse=True)

        merged = merged[: query.top_k]

        context_block = self._format_markdown(merged, query.task, query.top_k)
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
    # Streaming recall
    # ------------------------------------------------------------------

    async def recall_streaming(self, query: ContextQuery) -> AsyncGenerator[dict, None]:
        """Yield recall results progressively.

        Yields:
            {"type": "source", "data": MemorySource-like dict}
            {"type": "context", "data": {"context_block": str, "score": float}}
            {"type": "done", "data": {"request_id": str, "total_sources": int}}
        """
        sources: list[dict[str, Any]] = []

        # Fire both searches concurrently, yield results as each completes
        async def _vector_search() -> tuple[str, list[dict[str, Any]]]:
            try:
                results = await self._vector.search(
                    query.task,
                    top_k=query.top_k,
                    filter_tags=query.tags or None,
                    namespace=getattr(query, "namespace", "default"),
                    include_archived=getattr(query, "include_archived", False),
                )
                return "vector", results
            except Exception:
                logger.exception("Vector search failed in streaming recall")
                return "vector", []

        async def _graph_search() -> tuple[str, list[dict[str, Any]]]:
            try:
                results = await self._graph.query_related(
                    query.task, limit=query.top_k,
                    namespace=getattr(query, "namespace", "default"),
                )
                return "graph", results
            except Exception:
                logger.exception("Graph query failed in streaming recall")
                return "graph", []

        vector_task = asyncio.create_task(_vector_search())
        graph_task = asyncio.create_task(_graph_search())

        for coro in asyncio.as_completed([vector_task, graph_task]):
            store_name, results = await coro

            if store_name == "vector":
                for r in results:
                    text = r.get("text", "")
                    if not text:
                        continue
                    source = {
                        "store": "vector",
                        "content": text,
                        "score": round(float(r.get("score", 0.0)), 4),
                        "metadata": r.get("metadata", {}),
                    }
                    sources.append(source)
                    yield {"type": "source", "data": source}
            else:
                for r in results:
                    name = r.get("name") or ""
                    description = r.get("description") or ""
                    content = description if description else name
                    if not content:
                        continue
                    source = {
                        "store": "graph",
                        "content": content,
                        "score": round(1.0 / max(float(r.get("distance", 1) or 1), 1), 4),
                        "metadata": {
                            "name": name,
                            "label": r.get("label", "Entity"),
                        },
                    }
                    sources.append(source)
                    yield {"type": "source", "data": source}

        # Build context block from all sources
        context_block = self._format_markdown(sources, query.task, query.top_k)
        score = max((s["score"] for s in sources), default=0.0)

        yield {
            "type": "context",
            "data": {"context_block": context_block, "score": round(score, 4)},
        }
        yield {
            "type": "done",
            "data": {
                "request_id": str(uuid.uuid4()),
                "total_sources": len(sources),
            },
        }

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
                    namespace=getattr(query, "namespace", "default"),
                    include_archived=getattr(query, "include_archived", False),
                )
            except Exception:
                logger.exception("Vector search failed")
                return []

        async def _safe_graph() -> list[dict[str, Any]]:
            try:
                return await self._graph.query_related(
                    query.task, limit=query.top_k,
                    namespace=getattr(query, "namespace", "default"),
                )
            except Exception:
                logger.exception("Graph query failed")
                return []

        vector_results, graph_results = await asyncio.gather(
            _safe_vector(), _safe_graph()
        )
        return vector_results, graph_results

    # ------------------------------------------------------------------
    # Resolution retrieval
    # ------------------------------------------------------------------

    async def _fetch_resolutions(self, task: str) -> list[dict[str, Any]]:
        """Query resolution nodes for error-related tasks.

        Extracts keywords from the task and queries each against the graph's
        query_resolutions method. Results are formatted as graph entries with
        a 1.2x bonus score.
        """
        from app.db.graph import Neo4jClient as _NC

        keywords = _NC._extract_keywords(task)
        if not keywords:
            return []

        entries: list[dict[str, Any]] = []
        for kw in keywords:
            try:
                results = await self._graph.query_resolutions(kw, limit=3)
                for r in results:
                    resolution = r.get("resolution") or ""
                    error = r.get("error") or ""
                    if not resolution:
                        continue
                    content = f"Resolution: {resolution}"
                    if error:
                        content = f"[Error: {error}] {content}"
                    entries.append(
                        {
                            "content": content,
                            "score": 1.2,  # bonus score (will be normalized)
                            "store": "graph",
                            "metadata": {
                                "name": "resolution",
                                "label": "Resolution",
                                "source_type": "resolution",
                            },
                        }
                    )
            except Exception:
                logger.exception("query_resolutions failed for keyword '%s'", kw)

        # Deduplicate by content.
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for e in entries:
            if e["content"] not in seen:
                seen.add(e["content"])
                unique.append(e)
        return unique

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

    def _format_graph_entries(
        self,
        results: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """Convert raw graph traversal results into scored entries.

        Uses a composite score: weight * text_sim + (1 - weight) * (1/distance)
        where text_sim is Jaccard token overlap between query and node text.
        """
        weight = self._settings.GRAPH_RELEVANCE_WEIGHT
        query_tokens = _tokenize(query)

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

            # Distance component: 1/distance, default 0.5 if no distance.
            dist_score = 1.0 / max(int(distance), 1) if distance else 0.5

            # Text similarity component: Jaccard on query vs name+description.
            node_text = f"{name} {description}".strip()
            node_tokens = _tokenize(node_text)
            text_sim = _jaccard_similarity(query_tokens, node_tokens)

            score = weight * text_sim + (1 - weight) * dist_score

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

        For each pair of (vector_entry, graph_entry) whose content matches
        (via substring check or Jaccard similarity), the higher-scoring entry
        is kept with its score multiplied by the boost factor and its store
        set to ``"both"``.  The other entry is consumed (removed).
        """
        boost_factor = self._settings.BOOST_FACTOR

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
                    boosted_score = min(best_score * boost_factor, 1.0)

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

        Primary: case-insensitive substring check (either direction).
        Fallback: Jaccard similarity on word token sets, threshold 0.3.
        """
        if not a or not b:
            return False
        a_lower = a.lower()
        b_lower = b.lower()

        # Primary: substring check (graph content in vector content or vice versa).
        if b_lower in a_lower or a_lower in b_lower:
            return True

        # Fallback: Jaccard token similarity.
        a_tokens = _tokenize(a)
        b_tokens = _tokenize(b)
        return _jaccard_similarity(a_tokens, b_tokens) >= _JACCARD_MATCH_THRESHOLD

    # ------------------------------------------------------------------
    # Memory decay
    # ------------------------------------------------------------------

    def _apply_decay(self, entries: list[dict[str, Any]]) -> None:
        """Apply exponential memory decay based on entry age.

        Formula: decayed_score = score * 2^(-age_days / half_life)
        If half_life is 0 or entry has no timestamp, score is unchanged.
        """
        half_life = self._settings.MEMORY_DECAY_HALF_LIFE_DAYS
        if half_life <= 0:
            return

        now = datetime.now(timezone.utc)
        for entry in entries:
            timestamp_str = entry.get("metadata", {}).get("timestamp")
            if not timestamp_str:
                continue
            try:
                ts = datetime.fromisoformat(str(timestamp_str))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = (now - ts).total_seconds() / 86400.0
                if age_days > 0:
                    decay = 2.0 ** (-age_days / half_life)
                    entry["score"] *= decay
            except (ValueError, TypeError):
                # Unparseable timestamp — skip decay for this entry.
                continue

    # ------------------------------------------------------------------
    # Lifecycle scoring
    # ------------------------------------------------------------------

    def _apply_lifecycle_scoring(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply lifecycle status and confidence multipliers to scored items.

        Each item's score is adjusted by:
          score * status_multiplier * confidence_factor

        Where confidence_factor = (1 + confirmed_count) / (1 + contradicted_count)

        Items with status="archived" are removed entirely (score=0).
        Items are annotated with lifecycle metadata for the context block.
        """
        result = []
        for item in items:
            metadata = item.get("metadata", {})
            status = metadata.get("status", "active")

            # Status multiplier
            multiplier = self.STATUS_MULTIPLIERS.get(status, 1.0)
            if multiplier == 0.0:
                continue  # Skip archived

            # Confidence factor
            confirmed = metadata.get("confirmed_count", 0)
            contradicted = metadata.get("contradicted_count", 0)
            confidence = (1 + confirmed) / (1 + contradicted)

            item["score"] = item["score"] * multiplier * confidence

            # Annotate for context block
            if status != "active":
                item["_lifecycle_status"] = status
            if metadata.get("superseded_by"):
                item["_superseded_by"] = metadata["superseded_by"]

            result.append(item)

        return result

    # ------------------------------------------------------------------
    # Re-ranking via LLM
    # ------------------------------------------------------------------

    async def _rerank(
        self,
        task: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-rank candidates by asking the LLM to score relevance.

        Calls the configured LLM endpoint with a relevance scoring prompt.
        If the call fails for any candidate, the original score is kept.
        """
        if not candidates:
            return candidates

        base_url = self._settings.LLM_BASE_URL
        model = self._settings.LLM_MODEL
        api_key = self._settings.LLM_API_KEY

        url = f"{base_url}/chat/completions"
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        client = self._http_client or httpx.AsyncClient(timeout=15.0)
        try:
            tasks = [
                self._rerank_single(client, url, headers, model, task, c)
                for c in candidates
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if not self._http_client:
                await client.aclose()

        reranked: list[dict[str, Any]] = []
        for candidate, result in zip(candidates, results):
            if isinstance(result, Exception):
                logger.warning("Re-rank failed for entry: %s", result)
            elif result is not None:
                candidate["score"] = result
            # else: unparseable response — keep original score
            reranked.append(candidate)

        return reranked

    @staticmethod
    def _parse_rerank_score(text: str) -> float | None:
        """Extract a [0, 1] score from LLM rerank response text.

        Returns None if no valid score can be parsed.
        """
        match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
        if match:
            return float(match.group(1))
        try:
            value = float(text)
            if 0.0 <= value <= 1.0:
                return value
        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    async def _rerank_single(
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        model: str,
        task: str,
        candidate: dict[str, Any],
    ) -> float | None:
        """Ask the LLM to score a single candidate's relevance to the task.

        Returns None if the response cannot be parsed as a valid score.
        """
        prompt = (
            "Rate the relevance of this memory to the task on a scale of 0-1. "
            "Respond with ONLY a decimal number between 0 and 1.\n\n"
            f"Task: {task}\n"
            f"Memory: {candidate['content']}"
        )
        response = await client.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return RAGEngine._parse_rerank_score(text)

    # ------------------------------------------------------------------
    # Markdown formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_markdown(
        entries: list[dict[str, Any]],
        task: str = "",
        top_k: int = 5,
    ) -> str:
        """Build a structured Markdown block for LLM system prompt injection.

        Output format:
            ## Memory Recall ({n} results, confidence: {high/medium/low})
            1. [{score}] ({source}) {content}
            ...
            > Query: "{task}" | Top {top_k} results
        """
        if not entries:
            return "## Memory Recall (0 results, confidence: low)\n\nNo relevant memories found."

        # Determine confidence band from aggregate (max) score.
        max_score = max(e["score"] for e in entries)
        if max_score > 0.7:
            confidence = "high"
        elif max_score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        n = len(entries)
        lines: list[str] = [
            f"## Memory Recall ({n} result{'s' if n != 1 else ''}, confidence: {confidence})",
            "",
        ]

        # Entries are already sorted by score descending.
        for i, entry in enumerate(entries, 1):
            store = entry["store"]
            content = entry["content"]
            score = entry["score"]

            # Map internal store names to display labels.
            source_label = store
            if store == "both":
                source_label = "graph+vector"

            # Lifecycle status label.
            status_label = ""
            lifecycle_status = entry.get("_lifecycle_status")
            if lifecycle_status:
                status_label = f" [{lifecycle_status.upper()}]"
            superseded_by = entry.get("_superseded_by")
            if superseded_by:
                status_label += f" (superseded by {superseded_by})"

            lines.append(f"{i}. [{score:.0%}] ({source_label}) {content}{status_label}")

        lines.append("")
        lines.append(f'> Query: "{task}" | Top {top_k}')

        return "\n".join(lines)
