"""NexusCortex MCP Server.

Exposes NexusCortex memory operations as MCP tools via Streamable HTTP transport.
Calls the NexusCortex REST API internally using httpx.
"""

from __future__ import annotations

import atexit
import asyncio
import os

import httpx
from fastmcp import FastMCP

NEXUS_API_URL = os.getenv("NEXUS_API_URL", "http://api:8000")
NEXUS_API_KEY = os.getenv("NEXUS_API_KEY", "")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))

mcp = FastMCP("NexusCortex")

_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return a shared httpx client, lazily initialized."""
    global _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            headers: dict[str, str] = {}
            if NEXUS_API_KEY:
                headers["X-API-Key"] = NEXUS_API_KEY
            _client = httpx.AsyncClient(
                base_url=NEXUS_API_URL, timeout=30.0, headers=headers
            )
    return _client


def _format_error(exc: httpx.HTTPStatusError) -> str:
    """Format an HTTP error into a human-readable message with suggestions."""
    status = exc.response.status_code
    if status == 503:
        return (
            "Error: NexusCortex API is unavailable. "
            "Suggestion: Check service health with memory_health tool."
        )
    if status == 401:
        return (
            "Error: Authentication failed. "
            "Suggestion: Check API key configuration."
        )
    if status == 422:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = exc.response.text
        return (
            f"Error: Invalid input \u2014 {detail}. "
            "Suggestion: Check parameter values."
        )
    return (
        f"Error: API returned {status}. "
        "Suggestion: Check NexusCortex logs."
    )


def _connection_error(exc: httpx.RequestError) -> str:
    """Format a connection/transport error."""
    return (
        f"Error: Cannot connect to NexusCortex API at {NEXUS_API_URL}. "
        "Suggestion: Ensure NexusCortex is running."
    )


@mcp.tool()
async def memory_recall(task: str, tags: list[str] | None = None, top_k: int = 5) -> str:
    """Recall relevant memories for a task.

    Searches both the knowledge graph and vector store, returning merged
    context as Markdown suitable for LLM injection.

    When to use: Call BEFORE starting any task to check for relevant past
    experience. This gives you solutions to similar problems, known pitfalls,
    and established patterns from previous work.

    When NOT to use: Don't call for simple or trivial tasks that don't benefit
    from historical context (e.g. "print hello world").

    Args:
        task: Description of what the agent is trying to do.
        tags: Optional filter tags (e.g. ["auth", "bugfix"]).
        top_k: Maximum number of results to return (1-100, default 5).
    """
    try:
        client = await _get_client()
        body: dict = {"task": task, "top_k": top_k}
        if tags:
            body["tags"] = tags
        resp = await client.post("/memory/recall", json=body)
        resp.raise_for_status()
        return resp.json()["context_block"]
    except httpx.HTTPStatusError as exc:
        return _format_error(exc)
    except httpx.RequestError as exc:
        return _connection_error(exc)


@mcp.tool()
async def memory_learn(
    action: str,
    outcome: str,
    resolution: str | None = None,
    tags: list[str] | None = None,
    domain: str = "general",
) -> str:
    """Store an action/outcome pair in long-term memory.

    Records what was done, what happened, and optionally how it was resolved.

    When to use: Call AFTER completing a task to record the action and its
    outcome. This builds institutional memory so future tasks can benefit from
    your experience. Use for structured action/outcome pairs where you know
    what happened and why.

    When NOT to use: Don't use for raw event streams (CI logs, IDE events,
    monitoring data). Use memory_stream instead, which buffers events for
    background processing by the Sleep Cycle worker.

    Args:
        action: What the agent did.
        outcome: What resulted from the action.
        resolution: How the outcome was resolved (if applicable).
        tags: Categorization tags (e.g. ["database", "performance"]).
        domain: Knowledge domain (default "general").
    """
    try:
        client = await _get_client()
        body: dict = {"action": action, "outcome": outcome, "domain": domain}
        if resolution is not None:
            body["resolution"] = resolution
        if tags:
            body["tags"] = tags
        resp = await client.post("/memory/learn", json=body)
        resp.raise_for_status()
        truncated = action[:80]
        suffix = "..." if len(action) > 80 else ""
        return (
            f"Stored memory in domain '{domain}': '{truncated}{suffix}'. "
            "Outcome and resolution are now available for future recall."
        )
    except httpx.HTTPStatusError as exc:
        return _format_error(exc)
    except httpx.RequestError as exc:
        return _connection_error(exc)


@mcp.tool()
async def memory_stream(
    source: str,
    payload: dict,
    tags: list[str] | None = None,
) -> str:
    """Ingest a raw event into the processing queue.

    Events are buffered in Redis and consolidated by the Sleep Cycle worker,
    which periodically extracts structured knowledge using an LLM.

    When to use: For high-volume raw events (CI logs, IDE events, monitoring
    data, build outputs) that need background processing. The Sleep Cycle
    worker will automatically extract structured knowledge from these events.

    When NOT to use: For structured action/outcome pairs where you already
    know what happened and what the result was, use memory_learn instead.
    memory_learn writes directly to both stores; memory_stream queues for
    later processing.

    Args:
        source: Origin of the event (e.g. "ci-pipeline", "ide-plugin").
        payload: Arbitrary event data as key-value pairs.
        tags: Optional categorization tags.
    """
    try:
        client = await _get_client()
        body: dict = {"source": source, "payload": payload}
        if tags:
            body["tags"] = tags
        resp = await client.post("/memory/stream", json=body)
        resp.raise_for_status()
        data = resp.json()
        return f"Queued {data['queued']} event(s)"
    except httpx.HTTPStatusError as exc:
        return _format_error(exc)
    except httpx.RequestError as exc:
        return _connection_error(exc)


@mcp.tool()
async def memory_health() -> str:
    """Check NexusCortex service health.

    Returns connectivity status of all backend services (Neo4j, Qdrant, Redis).

    When to use: When other memory tools return errors, or to verify the
    service is operational before starting a workflow. Good first step when
    diagnosing connectivity issues.

    When NOT to use: No reason to avoid this tool. It is lightweight and
    read-only.
    """
    try:
        client = await _get_client()
        resp = await client.get("/health")
        resp.raise_for_status()
        data = resp.json()
        lines = [f"Status: {data['status']}"]
        for name, svc in data.get("services", {}).items():
            detail = f" ({svc['detail']})" if svc.get("detail") else ""
            lines.append(f"  {name}: {svc['status']}{detail}")
        return "\n".join(lines)
    except httpx.HTTPStatusError as exc:
        return _format_error(exc)
    except httpx.RequestError as exc:
        return _connection_error(exc)


def _shutdown_client() -> None:
    """Close the httpx client on process exit."""
    global _client
    if _client is not None and not _client.is_closed:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(_client.aclose())
            # If loop is running during atexit, we can't safely close.
            # The process is exiting — let the OS reclaim the socket.
        except RuntimeError:
            pass
        _client = None


atexit.register(_shutdown_client)


if __name__ == "__main__":
    mcp.run(transport="http", host=MCP_HOST, port=MCP_PORT, stateless_http=True)
