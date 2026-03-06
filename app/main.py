"""NexusCortex FastAPI application.

Memory-as-a-Service layer providing persistent cognitive memory for LLM agents.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

import httpx
import redis.asyncio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import get_settings
from app.backlinks import discover_backlinks
from app.contradiction import detect_and_supersede
from app.dashboard import create_dashboard_router
from app.db.graph import Neo4jClient
from app.db.vector import VectorClient
from app.embedding_admin import create_embedding_router
from app.engine.rag import RAGEngine
from app.lifecycle import create_lifecycle_router
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
    ErrorDetail,
    FeedbackRequest,
    FeedbackResponse,
    GenericEventIngest,
    HealthResponse,
    LearnResponse,
    RecallResponse,
    ServiceStatus,
    StreamResponse,
)
from app.stats import create_stats_router
from app.streaming import create_streaming_router
from app.transfer import create_transfer_router
from app.webhooks import create_webhook_router, fire_webhooks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 100
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.state.graph_client = Neo4jClient(settings)
    app.state.vector_client = VectorClient(settings)
    app.state.redis_client = redis.asyncio.from_url(settings.REDIS_URL)
    app.state.http_client = httpx.AsyncClient(timeout=15.0)
    app.state.start_time = datetime.now(timezone.utc)

    await app.state.graph_client.connect()
    await app.state.graph_client.ensure_indexes()
    await app.state.vector_client.initialize()

    app.state.rag_engine = RAGEngine(
        graph=app.state.graph_client,
        vector=app.state.vector_client,
        settings=settings,
        http_client=app.state.http_client,
    )

    # Register feature routers
    app.include_router(create_dashboard_router(
        graph=app.state.graph_client,
        vector=app.state.vector_client,
        redis_client=app.state.redis_client,
    ))
    app.include_router(create_webhook_router(app.state.redis_client))
    app.include_router(create_stats_router(
        graph=app.state.graph_client,
        vector=app.state.vector_client,
        redis_client=app.state.redis_client,
    ))
    app.include_router(create_transfer_router(
        graph=app.state.graph_client,
        vector=app.state.vector_client,
    ))
    app.include_router(create_streaming_router(
        rag_engine=app.state.rag_engine,
        graph=app.state.graph_client,
        vector=app.state.vector_client,
    ))
    app.include_router(create_embedding_router(app.state.vector_client))
    app.include_router(create_lifecycle_router(
        graph=app.state.graph_client,
        vector=app.state.vector_client,
    ))

    yield

    await app.state.http_client.aclose()
    await app.state.vector_client.close()
    await app.state.graph_client.close()
    await app.state.redis_client.aclose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NexusCortex",
    description="Memory-as-a-Service for LLM agents",
    version="0.6.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class _BodyTooLargeError(Exception):
    """Internal sentinel raised when chunked request body exceeds the limit."""


class RequestBodySizeLimitMiddleware:
    """Reject requests exceeding MAX_REQUEST_BODY_BYTES.

    Checks Content-Length header upfront and also tracks actual bytes
    received to guard against chunked-encoding bypass.
    """

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

            # Track actual bytes for chunked/streaming requests
            total_bytes = 0
            max_bytes = self.max_bytes

            async def receive_with_limit() -> dict:
                nonlocal total_bytes
                message = await receive()
                if message.get("type") == "http.request":
                    body = message.get("body", b"")
                    total_bytes += len(body)
                    if total_bytes > max_bytes:
                        raise _BodyTooLargeError()
                return message

            try:
                await self.app(scope, receive_with_limit, send)
            except _BodyTooLargeError:
                try:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large"},
                    )
                    await response(scope, receive, send)
                except Exception:
                    pass
            return
        await self.app(scope, receive, send)


class APIKeyMiddleware:
    """Validate X-API-Key header when API_KEY is configured.

    Skips validation for /health, /docs, /redoc, and /openapi.json.
    If API_KEY is None (not configured), all requests pass through.
    """

    SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}
    SKIP_PREFIXES = ("/dashboard",)

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._api_key = get_settings().API_KEY

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            if self._api_key is not None:
                path = scope.get("path", "")
                if path not in self.SKIP_PATHS and not path.startswith(self.SKIP_PREFIXES):
                    headers = dict(scope.get("headers", []))
                    api_key = headers.get(b"x-api-key", b"").decode("utf-8", errors="replace")
                    if not hmac.compare_digest(api_key, self._api_key):
                        request_id = headers.get(b"x-request-id", b"").decode("utf-8", errors="replace") or None
                        error = ErrorDetail(
                            error_code="UNAUTHORIZED",
                            detail="Invalid or missing API key",
                            request_id=request_id,
                        )
                        response = JSONResponse(
                            status_code=401,
                            content=error.model_dump(exclude_none=True),
                        )
                        await response(scope, receive, send)
                        return
        await self.app(scope, receive, send)


class RequestIDMiddleware:
    """Inject a unique X-Request-Id into each request and response.

    If the client provides an X-Request-Id header, it is preserved.
    Otherwise, a new UUID is generated.
    The ID is stored in request scope for use by exception handlers.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        existing_id = headers.get(b"x-request-id", b"").decode("utf-8", errors="replace")
        # Sanitize client-supplied request IDs: allow only safe chars, max 128 chars
        if existing_id and len(existing_id) <= 128 and all(
            c.isalnum() or c in "-_." for c in existing_id
        ):
            request_id = existing_id
        else:
            request_id = str(uuid.uuid4())

        # Store in scope for downstream access
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        async def send_with_request_id(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("utf-8")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


def _get_request_id(request: Request) -> str | None:
    """Extract request ID from request state, with fallback."""
    try:
        return request.state.request_id
    except AttributeError:
        return None


# Middleware order matters: outermost (first added) wraps innermost (last added).
# Execution order: RequestBodySize -> CORS -> APIKey -> RequestID -> App
app.add_middleware(RequestBodySizeLimitMiddleware)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-Id"],
)

app.add_middleware(APIKeyMiddleware)
app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def get_graph(request: Request) -> Neo4jClient:
    return request.app.state.graph_client


async def get_vector(request: Request) -> VectorClient:
    return request.app.state.vector_client


async def get_redis(request: Request) -> redis.asyncio.Redis:
    return request.app.state.redis_client


async def get_rag_engine(request: Request) -> RAGEngine:
    return request.app.state.rag_engine


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RateLimitExceeded)
async def handle_rate_limit(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    error = ErrorDetail(
        error_code="RATE_LIMITED",
        detail="Rate limit exceeded. Please slow down.",
        request_id=_get_request_id(request),
        suggestion="Reduce request frequency or contact the administrator.",
    )
    return JSONResponse(status_code=429, content=error.model_dump(exclude_none=True))


@app.exception_handler(GraphConnectionError)
async def handle_graph_error(request: Request, exc: GraphConnectionError) -> JSONResponse:
    logger.error("Graph connection error: %s", exc)
    error = ErrorDetail(
        error_code="GRAPH_UNAVAILABLE",
        detail="Knowledge graph service unavailable",
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=503, content=error.model_dump(exclude_none=True))


@app.exception_handler(VectorStoreError)
async def handle_vector_error(request: Request, exc: VectorStoreError) -> JSONResponse:
    logger.error("Vector store error: %s", exc)
    error = ErrorDetail(
        error_code="VECTOR_ERROR",
        detail="Vector store service error",
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=502, content=error.model_dump(exclude_none=True))


@app.exception_handler(LLMExtractionError)
async def handle_llm_error(request: Request, exc: LLMExtractionError) -> JSONResponse:
    logger.error("LLM extraction error: %s", exc)
    error = ErrorDetail(
        error_code="LLM_ERROR",
        detail="LLM extraction service error",
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=502, content=error.model_dump(exclude_none=True))


@app.exception_handler(StreamIngestionError)
async def handle_stream_error(request: Request, exc: StreamIngestionError) -> JSONResponse:
    logger.error("Stream ingestion error: %s", exc)
    error = ErrorDetail(
        error_code="STREAM_ERROR",
        detail="Event stream ingestion error",
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=502, content=error.model_dump(exclude_none=True))


@app.exception_handler(NexusCortexError)
async def handle_nexus_error(request: Request, exc: NexusCortexError) -> JSONResponse:
    logger.error("NexusCortex error: %s", exc)
    error = ErrorDetail(
        error_code="INTERNAL_ERROR",
        detail="Internal service error",
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=500, content=error.model_dump(exclude_none=True))


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
        logger.warning("Redis health check failed: %s", exc)
        services["redis"] = ServiceStatus(status="disconnected", detail="Service unreachable")

    # Neo4j
    try:
        await graph.ping()
        services["graph"] = ServiceStatus(status="connected")
    except Exception as exc:
        logger.warning("Neo4j health check failed: %s", exc)
        services["graph"] = ServiceStatus(status="disconnected", detail="Service unreachable")

    # Qdrant
    try:
        await vector.ping()
        services["qdrant"] = ServiceStatus(status="connected")
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        services["qdrant"] = ServiceStatus(status="disconnected", detail="Service unreachable")

    # Uptime
    uptime_seconds: float | None = None
    try:
        start_time = app.state.start_time
        uptime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
    except AttributeError:
        pass

    # Memory count from Qdrant
    memory_count = await vector.memory_count()

    all_connected = all(s.status == "connected" for s in services.values())
    return HealthResponse(
        status="ok" if all_connected else "degraded",
        services=services,
        version="0.6.0",
        uptime_seconds=uptime_seconds,
        memory_count=memory_count,
    )


@app.post("/memory/recall", response_model=RecallResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT)
async def memory_recall(
    request: Request,
    query: ContextQuery,
    engine: Annotated[RAGEngine, Depends(get_rag_engine)],
) -> RecallResponse:
    """Dual-retrieval memory recall: graph + vector search, merged and scored."""
    result = await engine.recall(query)
    result.request_id = _get_request_id(request)
    result.namespace = query.namespace
    return result


@app.post("/memory/learn", response_model=LearnResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT)
async def memory_learn(
    request: Request,
    log: ActionLog,
    graph: Annotated[Neo4jClient, Depends(get_graph)],
    vector: Annotated[VectorClient, Depends(get_vector)],
    redis_client: Annotated[redis.asyncio.Redis, Depends(get_redis)],
) -> LearnResponse:
    """Store an action log in both the knowledge graph and vector store."""
    text = f"{log.action}. The outcome was: {log.outcome}."
    if log.resolution:
        text += f" Resolution: {log.resolution}"

    graph_result, vector_result = await asyncio.gather(
        graph.merge_action_log(log, namespace=log.namespace),
        vector.upsert(
            text=text,
            metadata={
                "source": "action_log",
                "tags": log.tags,
                "domain": log.domain,
            },
            namespace=log.namespace,
        ),
        return_exceptions=True,
    )

    graph_failed = isinstance(graph_result, BaseException)
    vector_failed = isinstance(vector_result, BaseException)

    if graph_failed and vector_failed:
        logger.error("Both stores failed: graph=%s, vector=%s", graph_result, vector_result)
        raise GraphConnectionError(f"Both stores failed during learn: {graph_result}")

    if graph_failed:
        logger.error("Graph write failed (vector succeeded): %s", graph_result)
        return LearnResponse(
            status="partial",
            graph_id=None,
            vector_id=str(vector_result),
            namespace=log.namespace,
        )

    if vector_failed:
        logger.error("Vector write failed (graph succeeded): %s", vector_result)
        return LearnResponse(
            status="partial",
            graph_id=str(graph_result),
            vector_id=None,
            namespace=log.namespace,
        )

    # Contradiction detection — auto-supersede similar old memories
    superseded: list[str] = []
    try:
        superseded = await detect_and_supersede(
            vector=vector,
            graph=graph,
            new_text=text,
            new_vector_id=str(vector_result),
            new_graph_id=str(graph_result),
            domain=log.domain,
            namespace=log.namespace,
        )
    except Exception:
        logger.warning("Contradiction detection failed, continuing")

    # Auto-discover backlinks — find related memories and create graph edges
    backlinks: list[dict] = []
    try:
        backlinks = await discover_backlinks(
            vector=vector,
            graph=graph,
            new_text=text,
            new_vector_id=str(vector_result),
            new_graph_id=str(graph_result),
            domain=log.domain,
            namespace=log.namespace,
        )
    except Exception:
        logger.warning("Backlink discovery failed, continuing")

    # Fire webhooks in background (best-effort)
    try:
        asyncio.create_task(fire_webhooks(
            redis_client, "memory.learned",
            {"graph_id": graph_result, "vector_id": vector_result, "action": log.action,
             "superseded": superseded, "backlinks_found": len(backlinks)},
            namespace=log.namespace,
        ))
    except Exception:
        pass

    return LearnResponse(
        status="stored",
        graph_id=graph_result,
        vector_id=vector_result,
        namespace=log.namespace,
        superseded=superseded,
        backlinks=backlinks,
    )


@app.post("/memory/stream", response_model=StreamResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT)
async def memory_stream(
    request: Request,
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
        pipe = redis_client.pipeline()
        for event in events:
            pipe.lpush(settings.REDIS_STREAM_KEY, event.model_dump_json())
        await pipe.execute()
    except Exception as exc:
        raise StreamIngestionError(f"Failed to push events to Redis: {exc}") from exc

    # Fire webhooks in background (best-effort)
    try:
        asyncio.create_task(fire_webhooks(
            redis_client, "stream.ingested",
            {"count": len(events), "source": events[0].source},
            namespace=events[0].namespace if events else "default",
        ))
    except Exception:
        pass

    return StreamResponse(status="queued", queued=len(events))


@app.post("/memory/feedback", response_model=FeedbackResponse)
@limiter.limit(lambda: get_settings().RATE_LIMIT)
async def memory_feedback(
    request: Request,
    feedback: FeedbackRequest,
    vector: Annotated[VectorClient, Depends(get_vector)],
) -> FeedbackResponse:
    """Record relevance feedback for recalled memories.

    Updates the Qdrant payload metadata for each referenced memory point
    with the feedback signal (useful/not useful) and optional comment.
    """
    updated = 0
    timestamp = datetime.now(timezone.utc).isoformat()
    for memory_id in feedback.memory_ids:
        try:
            await vector.set_feedback(
                memory_id=memory_id,
                useful=feedback.useful,
                comment=feedback.comment,
                timestamp=timestamp,
            )
            updated += 1
        except Exception:
            logger.warning("Failed to update feedback for memory %s", memory_id)

    return FeedbackResponse(status="recorded", updated=updated)
