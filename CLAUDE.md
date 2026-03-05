# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NexusCortex is a domain-agnostic Memory-as-a-Service (MaaS) layer that provides persistent cognitive memory for LLM agents. It tracks procedural memory, declarative memory, and mistake resolution across arbitrary software projects and automated workflows.

## Tech Stack

- **API**: FastAPI (Python 3.11+)
- **Graph DB**: Neo4j 5.x (knowledge graph)
- **Vector DB**: Qdrant (semantic embeddings)
- **Queue/Cache**: Redis 7+ (event stream)
- **Background Worker**: Celery (Sleep Cycle consolidation)
- **LLM**: OpenAI-compatible API (Ollama/vLLM/OpenAI)
- **Embeddings**: nomic-embed-text (768-dim, configurable)

## Architecture

```
app/
├── config.py          # Pydantic BaseSettings, env var loading
├── models.py          # Request/response Pydantic models
├── exceptions.py      # NexusCortexError hierarchy
├── main.py            # FastAPI app, lifespan, routes, DI, router integration
├── mcp_server.py      # MCP server (Streamable HTTP, 4 tools)
├── dashboard.py       # Web dashboard router (memory browser, graph viz, DLQ)
├── webhooks.py        # Webhook registration and event firing
├── stats.py           # Memory statistics endpoint
├── transfer.py        # Export/import API (JSONL streaming)
├── streaming.py       # SSE streaming recall endpoint
├── embedding_admin.py # Embedding model admin (status, re-embed)
├── static/
│   └── dashboard.html # Self-contained dashboard SPA (zero dependencies)
├── db/
│   ├── graph.py       # Neo4j async client, Cypher queries
│   └── vector.py      # Qdrant async client, embedding + search
├── engine/
│   └── rag.py         # Dual-retrieval RAG engine (vector + graph merge)
└── workers/
    ├── sleep_cycle.py # Celery worker: Redis → LLM extraction → Neo4j
    ├── gc.py          # Celery task: memory expiry & garbage collection
    └── reembed.py     # Celery task: re-embed all vectors with new model
```

### API Endpoints
- `POST /memory/recall` — Dual-retrieval RAG query, returns Markdown for LLM injection
- `POST /memory/recall/stream` — SSE streaming recall (progressive results)
- `POST /memory/learn` — Logs actions/outcomes/resolutions to both graph + vector (parallel writes)
- `POST /memory/stream` — High-volume event ingest → Redis queue (pipeline batching)
- `POST /memory/feedback` — Submit feedback on memory usefulness
- `GET /memory/stats` — Memory statistics (counts, domains, tags, DLQ depth)
- `GET /memory/export` — Export all memories as JSONL stream
- `POST /memory/import` — Bulk import memories from JSONL
- `GET /health` — Service connectivity status with version, uptime, memory count
- `GET /dashboard/` — Web dashboard (memory browser, graph visualization, DLQ manager)
- `POST /webhooks/` — Register webhook callbacks for memory events
- `GET /admin/embeddings/status` — Embedding model info and cache stats
- `POST /admin/embeddings/reembed` — Trigger re-embedding with progress tracking

### MCP Server (port 8080)
Exposes 4 MCP tools via Streamable HTTP (`/mcp` endpoint):
- `memory_recall` — Recall relevant memories for a task (returns Markdown)
- `memory_learn` — Store action/outcome pairs in long-term memory
- `memory_stream` — Ingest raw events into the processing queue
- `memory_health` — Check service health status

### Data Flow
- `/recall` → RAG engine queries Qdrant (semantic) + Neo4j (graph) concurrently → merges/boosts scores → Markdown
- `/learn` → writes action chain to Neo4j (Domain→Action→Outcome→Resolution) + embeds to Qdrant
- `/stream` → LPUSH to Redis → Celery Beat (60s) → LLM extraction → Neo4j MERGE

### Graph Schema
- **Nodes**: Namespace, Domain, Concept, Action, Outcome, Resolution, EventStream
- **Edges**: CONTAINS, RELATES_TO, CAUSED, RESOLVED_BY, UTILIZES

## Commands

### Run the API server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run all services via Docker
```bash
docker-compose up --build
```

### Run tests
```bash
pytest tests/ -v
pytest tests/test_rag.py -v          # single test file
pytest tests/test_models.py -k "test_context_query" -v  # single test
```

### Run Celery worker + beat (separate processes)
```bash
celery -A app.workers.sleep_cycle worker --loglevel=info
celery -A app.workers.sleep_cycle beat --loglevel=info
```

### Run MCP server standalone
```bash
python -m app.mcp_server
```

## Configuration

All configuration via environment variables or `.env` file. See `.env.example` for all options. Key settings:
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_POOL_SIZE`
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- `REDIS_URL`, `REDIS_STREAM_KEY`
- `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`
- `API_KEY`, `CORS_ORIGINS`, `RATE_LIMIT` (security)
- `BOOST_FACTOR`, `GRAPH_RELEVANCE_WEIGHT`, `CONTENT_HASH_LENGTH` (RAG tuning)
- `RERANK_ENABLED`, `RERANK_CANDIDATES_MULTIPLIER`, `MEMORY_DECAY_HALF_LIFE_DAYS` (advanced)
- `NEXUS_API_URL`, `MCP_HOST`, `MCP_PORT`, `NEXUS_API_KEY` (MCP server)
- `DEFAULT_NAMESPACE`, `MAX_MEMORY_AGE_DAYS`, `GC_SCHEDULE_HOURS` (multi-tenant & GC)
- `DLQ_MAX_SIZE`, `REEMBED_BATCH_SIZE`, `PRUNE_SCORE_THRESHOLD` (operations)

## Key Design Decisions

- **No APOC dependency**: Neo4j Cypher uses standard MERGE with dynamic labels grouped by type and sanitized against injection
- **Dual-store scoring**: Items found in both Qdrant and Neo4j get a configurable boost (default 1.5x) via substring + Jaccard similarity matching (threshold 0.3)
- **Score normalization**: Min-max normalization per source to [0,1] before merging, with composite graph scoring (text relevance + distance)
- **Graceful degradation**: If one store fails during recall, the other still contributes results; `/learn` returns partial success when one store fails
- **Sleep Cycle DLQ**: Failed batches go to `nexus:event_stream:dlq` Redis key for later reprocessing
- **Sleep Cycle transactions**: All Neo4j writes in the Sleep Cycle worker are wrapped in explicit transactions
- **Cypher injection prevention**: Dynamic labels/relationship types are sanitized to alphanumeric + underscore only
- **Fulltext index search**: Neo4j Lucene-backed fulltext index for relevance-scored graph queries (falls back to CONTAINS on `ClientError` only)
- **Case-insensitive resolution queries**: `query_resolutions` uses `toLower()` on both sides for reliable matching
- **API key authentication**: Optional X-API-Key middleware (skips /health, /docs), settings cached at init
- **Rate limiting**: slowapi-based rate limiting (configurable, default 60/min)
- **Memory decay**: Exponential decay based on entry age (configurable half-life, default 90 days)
- **LLM re-ranking**: Optional re-ranking pass via LLM (gated behind RERANK_ENABLED config), shared httpx client, robust score parsing with word-boundary regex
- **Embedding cache**: OrderedDict-based LRU cache (512 entries) with O(1) eviction for vector embeddings
- **Entity canonicalization**: Lowercase + normalize names before MERGE to reduce duplicates
- **Redis pipeline batching**: Pipeline pattern for both event ingest and Sleep Cycle rpop
- **Singleton RAGEngine**: Created once in lifespan with shared httpx client, injected via FastAPI DI
- **Request body limit**: Content-Length pre-check + chunked-encoding guard with proper 413 responses
- **Config validation**: CONTENT_HASH_LENGTH >= 16 enforced, startup warnings for empty secrets
- **Port security**: Backend ports (Neo4j, Qdrant, Redis) bound to 127.0.0.1; API/MCP on 0.0.0.0
- **Celery beat separation**: Worker and beat run as independent Docker services to prevent task storms
- **Public health methods**: `ping()` and `memory_count()` on clients — health endpoint uses public API only
- **Multi-tenant namespaces**: Optional `namespace` field on all requests (default "default"), filters in both Neo4j and Qdrant
- **Namespace graph**: `Namespace` nodes linked to `Domain` via `CONTAINS` edges for scoped queries
- **Web dashboard**: Self-contained SPA with force-directed graph visualization, zero external dependencies
- **Webhook system**: HMAC-SHA256 signed callbacks, event-type + namespace filtering, stored in Redis
- **Memory GC**: Scheduled Celery task prunes old memories below score threshold, preserves positive feedback
- **DLQ cap**: `LTRIM` after every DLQ push prevents unbounded Redis growth (configurable max 10K)
- **Export/Import**: JSONL streaming for backup/migration, includes both vector and graph data
- **SSE streaming recall**: Progressive results via Server-Sent Events, no extra dependencies
- **Embedding hot-swap**: Re-embed all vectors with progress tracking via Celery task state
- **MCP client lock**: `asyncio.Lock` guards lazy client initialization against TOCTOU races
- **Lucene phrase search**: Bigrams wrapped in double quotes for correct Lucene phrase matching

## Development Practices

- Claude maintains detailed documentation of all changes and decisions
- CLAUDE.md is kept up-to-date as the project evolves
- All significant architectural decisions and progress are logged
- Teams workflow: architect → implement → test → security review

## Documentation

- `docs/ARCHITECTURE.md` — Full architecture specification with module contracts
- `docs/SECURITY_REVIEW.md` — Security review findings and fixes
- `docs/plans/2026-03-05-major-improvements.md` — v0.5.0 implementation plan (15 fixes)
