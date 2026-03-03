"""NexusCortex FastAPI application.

Memory-as-a-Service layer providing persistent cognitive memory for LLM agents.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import redis.asyncio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import get_settings
from app.db.graph import Neo4jClient
from app.db.vector import VectorClient
from app.engine.rag import RAGEngine
from app.exceptions import (
    GraphConnectionError,
    LLMExtractionError,
    NexusCortexError,
    StreamIngestionError,
    VectorStoreError,
)
from app.models import (
    ActionLog,
    ContextQuery,
    GenericEventIngest,
    HealthResponse,
    LearnResponse,
    RecallResponse,
    ServiceStatus,
    StreamResponse,
)

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 100
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.state.graph_client = Neo4jClient(settings)
    app.state.vector_client = VectorClient(settings)
    app.state.redis_client = redis.asyncio.from_url(settings.REDIS_URL)

    await app.state.graph_client.connect()
    await app.state.vector_client.initialize()

    yield

    await app.state.vector_client.close()
    await app.state.graph_client.close()
    await app.state.redis_client.aclose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NexusCortex",
    description="Memory-as-a-Service for LLM agents",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RequestBodySizeLimitMiddleware:
    """Reject requests whose Content-Length exceeds MAX_REQUEST_BODY_BYTES."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            content_length = headers.get(b"content-length")
            if content_length is not None and int(content_length) > self.max_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(RequestBodySizeLimitMiddleware)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def get_graph(request: Request) -> Neo4jClient:
    return request.app.state.graph_client


async def get_vector(request: Request) -> VectorClient:
    return request.app.state.vector_client


async def get_redis(request: Request) -> redis.asyncio.Redis:
    return request.app.state.redis_client


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(GraphConnectionError)
async def handle_graph_error(request: Request, exc: GraphConnectionError) -> JSONResponse:
    logger.error("Graph connection error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Knowledge graph service unavailable"})


@app.exception_handler(VectorStoreError)
async def handle_vector_error(request: Request, exc: VectorStoreError) -> JSONResponse:
    logger.error("Vector store error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "Vector store service error"})


@app.exception_handler(LLMExtractionError)
async def handle_llm_error(request: Request, exc: LLMExtractionError) -> JSONResponse:
    logger.error("LLM extraction error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "LLM extraction service error"})


@app.exception_handler(StreamIngestionError)
async def handle_stream_error(request: Request, exc: StreamIngestionError) -> JSONResponse:
    logger.error("Stream ingestion error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "Event stream ingestion error"})


@app.exception_handler(NexusCortexError)
async def handle_nexus_error(request: Request, exc: NexusCortexError) -> JSONResponse:
    logger.error("NexusCortex error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal service error"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(
    graph: Annotated[Neo4jClient, Depends(get_graph)],
    vector: Annotated[VectorClient, Depends(get_vector)],
    redis_client: Annotated[redis.asyncio.Redis, Depends(get_redis)],
) -> HealthResponse:
    """Return service health status by probing each backend."""
    services: dict[str, ServiceStatus] = {}

    # Redis
    try:
        await redis_client.ping()
        services["redis"] = ServiceStatus(status="connected")
    except Exception as exc:
        services["redis"] = ServiceStatus(status="disconnected", detail=str(exc))

    # Neo4j
    try:
        driver = graph._ensure_driver()
        await driver.verify_connectivity()
        services["graph"] = ServiceStatus(status="connected")
    except Exception as exc:
        services["graph"] = ServiceStatus(status="disconnected", detail=str(exc))

    # Qdrant
    try:
        await vector._client.get_collections()
        services["qdrant"] = ServiceStatus(status="connected")
    except Exception as exc:
        services["qdrant"] = ServiceStatus(status="disconnected", detail=str(exc))

    all_connected = all(s.status == "connected" for s in services.values())
    return HealthResponse(
        status="ok" if all_connected else "degraded",
        services=services,
    )


@app.post("/memory/recall", response_model=RecallResponse)
async def memory_recall(
    query: ContextQuery,
    graph: Annotated[Neo4jClient, Depends(get_graph)],
    vector: Annotated[VectorClient, Depends(get_vector)],
) -> RecallResponse:
    """Dual-retrieval memory recall: graph + vector search, merged and scored."""
    engine = RAGEngine(graph=graph, vector=vector)
    return await engine.recall(query)


@app.post("/memory/learn", response_model=LearnResponse)
async def memory_learn(
    log: ActionLog,
    graph: Annotated[Neo4jClient, Depends(get_graph)],
    vector: Annotated[VectorClient, Depends(get_vector)],
) -> LearnResponse:
    """Store an action log in both the knowledge graph and vector store."""
    graph_id = await graph.merge_action_log(log)

    text = f"{log.action} | {log.outcome}"
    if log.resolution:
        text += f" | Resolution: {log.resolution}"

    vector_id = await vector.upsert(
        text=text,
        metadata={
            "source": "action_log",
            "tags": log.tags,
            "domain": log.domain,
        },
    )

    return LearnResponse(status="stored", graph_id=graph_id, vector_id=vector_id)


@app.post("/memory/stream", response_model=StreamResponse)
async def memory_stream(
    events: GenericEventIngest | list[GenericEventIngest],
    redis_client: Annotated[redis.asyncio.Redis, Depends(get_redis)],
) -> StreamResponse:
    """Push event(s) onto the Redis ingestion queue for background processing."""
    settings = get_settings()

    if isinstance(events, GenericEventIngest):
        events = [events]

    if len(events) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}",
        )

    try:
        for event in events:
            await redis_client.lpush(
                settings.REDIS_STREAM_KEY,
                event.model_dump_json(),
            )
    except Exception as exc:
        raise StreamIngestionError(f"Failed to push events to Redis: {exc}") from exc

    return StreamResponse(status="queued", queued=len(events))
