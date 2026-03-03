"""Pydantic models for NexusCortex API requests and responses."""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class ContextQuery(BaseModel):
    """Query for memory recall — describes what the agent is trying to do."""

    task: str = Field(..., min_length=1, max_length=2000)
    tags: list[Annotated[str, Field(max_length=100)]] = Field(default=[], max_length=20)
    top_k: int = Field(default=5, ge=1, le=100)

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


class GenericEventIngest(BaseModel):
    """Arbitrary event data for stream ingestion into the Redis queue."""

    source: str = Field(..., min_length=1, max_length=500)
    payload: dict[str, Any]
    timestamp: datetime | None = None
    tags: list[Annotated[str, Field(max_length=100)]] = Field(default=[], max_length=20)

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


class LearnResponse(BaseModel):
    """Response from /memory/learn — confirmation of stored action log."""

    status: str
    graph_id: str | None = None
    vector_id: str | None = None


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
