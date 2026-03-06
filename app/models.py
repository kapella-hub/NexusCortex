"""Pydantic models for NexusCortex API requests and responses."""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class ContextQuery(BaseModel):
    """Query for memory recall — describes what the agent is trying to do."""

    task: str = Field(..., min_length=1, max_length=2000)
    tags: list[Annotated[str, Field(max_length=100)]] = Field(default=[], max_length=20)
    top_k: int = Field(default=5, ge=1, le=100)
    namespace: str = Field(default="default", min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_-]+$")
    include_archived: bool = Field(default=False)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task": "Fix the authentication timeout bug in the login flow",
                    "tags": ["auth", "bugfix"],
                    "top_k": 5,
                }
            ]
        }
    }


class ActionLog(BaseModel):
    """Record of an agent action and its outcome, optionally with a resolution."""

    action: str = Field(..., min_length=1, max_length=5000)
    outcome: str = Field(..., min_length=1, max_length=5000)
    resolution: str | None = Field(default=None, max_length=5000)
    tags: list[Annotated[str, Field(max_length=100)]] = Field(default=[], max_length=20)
    domain: str = Field(default="general", min_length=1, max_length=200)
    namespace: str = Field(default="default", min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_-]+$")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "Increased connection pool size from 5 to 20",
                    "outcome": "Database timeout errors resolved",
                    "resolution": "Updated DB_POOL_SIZE in config and restarted service",
                    "tags": ["database", "performance"],
                    "domain": "infrastructure",
                }
            ]
        }
    }


_MAX_PAYLOAD_SERIALIZED_BYTES = 50_000  # 50 KB limit per event payload


class GenericEventIngest(BaseModel):
    """Arbitrary event data for stream ingestion into the Redis queue."""

    source: str = Field(..., min_length=1, max_length=500)
    payload: dict[str, Any]
    timestamp: datetime | None = None
    tags: list[Annotated[str, Field(max_length=100)]] = Field(default=[], max_length=20)
    namespace: str = Field(default="default", min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_-]+$")

    @model_validator(mode="after")
    def _validate_payload_size(self) -> "GenericEventIngest":
        """Reject payloads that exceed the serialized byte limit."""
        import json as _json

        serialized = _json.dumps(self.payload, default=str)
        if len(serialized) > _MAX_PAYLOAD_SERIALIZED_BYTES:
            msg = f"Payload exceeds maximum size of {_MAX_PAYLOAD_SERIALIZED_BYTES} bytes"
            raise ValueError(msg)
        return self

    def model_post_init(self, __context: Any) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "ci-pipeline",
                    "payload": {
                        "build_id": "abc-123",
                        "status": "failed",
                        "error": "OOM in test suite",
                    },
                    "tags": ["ci", "failure"],
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class MemorySource(BaseModel):
    """Provenance record for a single memory retrieval result."""

    store: Literal["graph", "vector", "both"]
    content: str
    score: float
    metadata: dict[str, Any] = {}


class RecallResponse(BaseModel):
    """Response from /memory/recall — merged, scored memory context."""

    context_block: str
    sources: list[MemorySource]
    score: float
    request_id: str | None = None
    namespace: str = "default"


class LearnResponse(BaseModel):
    """Response from /memory/learn — confirmation of stored action log."""

    status: str
    graph_id: str | None = None
    vector_id: str | None = None
    namespace: str = "default"
    superseded: list[str] = []
    backlinks: list[dict] = []


class StreamResponse(BaseModel):
    """Response from /memory/stream — confirmation of queued events."""

    status: str
    queued: int


class ServiceStatus(BaseModel):
    """Health status of an individual service dependency."""

    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """Response from /health — service connectivity status."""

    status: str
    services: dict[str, ServiceStatus]
    version: str = "0.6.0"
    uptime_seconds: float | None = None
    memory_count: int | None = None


# ---------------------------------------------------------------------------
# Error & Feedback Models
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error response with machine-readable code."""

    error_code: str
    detail: str
    request_id: str | None = None
    suggestion: str | None = None


class FeedbackRequest(BaseModel):
    """Relevance feedback for recalled memories."""

    memory_ids: list[str] = Field(..., min_length=1, max_length=50)
    useful: bool
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Response from /memory/feedback."""

    status: str
    updated: int


# ---------------------------------------------------------------------------
# Stats & Transfer Models
# ---------------------------------------------------------------------------


class MemoryStats(BaseModel):
    """Response from /memory/stats — aggregated memory statistics."""

    total_memories: int
    graph_nodes: int
    graph_edges: int
    domains: list[str]
    top_tags: list[dict]  # [{"tag": "auth", "count": 15}, ...]
    dlq_depth: int
    oldest_memory: str | None  # ISO timestamp
    newest_memory: str | None  # ISO timestamp
    namespace_counts: dict[str, int]  # {"default": 42, "agent-1": 15}


class ImportResponse(BaseModel):
    """Response from /memory/import."""

    status: str
    imported_memories: int = 0
    imported_nodes: int = 0
    imported_edges: int = 0
    errors: list[str] = []


# ---------------------------------------------------------------------------
# Knowledge Lifecycle Models
# ---------------------------------------------------------------------------


class DeprecateRequest(BaseModel):
    """Request to change memory status."""

    memory_ids: list[str] = Field(..., min_length=1, max_length=50)
    status: Literal["deprecated", "superseded", "archived"] = Field(...)
    reason: str = Field(..., min_length=1, max_length=2000)
    superseded_by: str | None = Field(default=None)


class DeprecateResponse(BaseModel):
    """Response from /memory/deprecate."""

    status: str
    updated: int


class ConfirmRequest(BaseModel):
    """Request to confirm memories are still valid."""

    memory_ids: list[str] = Field(..., min_length=1, max_length=50)


class ConfirmResponse(BaseModel):
    """Response from /memory/confirm."""

    status: str
    confirmed: int


class BacklinksResponse(BaseModel):
    """Response from /memory/{id}/backlinks."""

    memory_id: str
    backlinks: list[dict] = []
    total: int = 0


class MemoryHistoryResponse(BaseModel):
    """Response from /memory/{id}/history."""

    memory_id: str
    status: str
    superseded_by: dict | None = None
    supersedes: list[dict] = []
    confirmed_count: int = 0
    contradicted_count: int = 0
    last_confirmed_at: str | None = None
