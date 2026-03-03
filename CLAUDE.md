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
├── main.py            # FastAPI app, lifespan, routes, DI
├── db/
│   ├── graph.py       # Neo4j async client, Cypher queries
│   └── vector.py      # Qdrant async client, embedding + search
├── engine/
│   └── rag.py         # Dual-retrieval RAG engine (vector + graph merge)
└── workers/
    └── sleep_cycle.py # Celery worker: Redis → LLM extraction → Neo4j
```

### API Endpoints
- `POST /memory/recall` — Dual-retrieval RAG query, returns Markdown for LLM injection
- `POST /memory/learn` — Logs actions/outcomes/resolutions to both graph + vector
- `POST /memory/stream` — High-volume event ingest → Redis queue
- `GET /health` — Service connectivity status

### Data Flow
- `/recall` → RAG engine queries Qdrant (semantic) + Neo4j (graph) concurrently → merges/boosts scores → Markdown
- `/learn` → writes action chain to Neo4j (Domain→Action→Outcome→Resolution) + embeds to Qdrant
- `/stream` → LPUSH to Redis → Celery Beat (60s) → LLM extraction → Neo4j MERGE

### Graph Schema
- **Nodes**: Domain, Concept, Action, Outcome, Resolution, EventStream
- **Edges**: RELATES_TO, CAUSED, RESOLVED_BY, UTILIZES

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

### Run Celery worker
```bash
celery -A app.workers.sleep_cycle worker --beat --loglevel=info
```

## Configuration

All configuration via environment variables or `.env` file. See `.env.example` for all options. Key settings:
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- `REDIS_URL`, `REDIS_STREAM_KEY`
- `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`

## Key Design Decisions

- **No APOC dependency**: Neo4j Cypher uses standard MERGE with dynamic labels grouped by type and sanitized against injection
- **Dual-store scoring**: Items found in both Qdrant and Neo4j get a 1.5x score boost via fuzzy text matching (SequenceMatcher > 0.7)
- **Graceful degradation**: If one store fails during recall, the other still contributes results
- **Sleep Cycle DLQ**: Failed batches go to `nexus:event_stream:dlq` Redis key for later reprocessing
- **Cypher injection prevention**: Dynamic labels/relationship types are sanitized to alphanumeric + underscore only

## Development Practices

- Claude maintains detailed documentation of all changes and decisions
- CLAUDE.md is kept up-to-date as the project evolves
- All significant architectural decisions and progress are logged
- Teams workflow: architect → implement → test → security review

## Documentation

- `docs/ARCHITECTURE.md` — Full architecture specification with module contracts
- `docs/SECURITY_REVIEW.md` — Security review findings and fixes
