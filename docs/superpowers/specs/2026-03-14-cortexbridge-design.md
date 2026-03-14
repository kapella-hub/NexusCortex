# CortexBridge — Shadow Context Manager

## Overview

CortexBridge is a lightweight MCP server that maintains a live "shadow" of an AI agent's working state. When an LLM's context window gets compressed or a session restarts, the agent pulls its shadow context — a concise document containing its plan, progress, decisions, and file knowledge — and picks up exactly where it left off.

**Primary problem**: Mid-task amnesia. AI agents lose track of their plan, re-read files, contradict earlier decisions, and waste context re-establishing state after compression events.

**Primary consumer**: Claude Code (generalize to other agents later).

**Deployment**: VPS at `/opt/CortexBridge`, same server as NexusCortex. Reuses existing Redis and Ollama infrastructure.

## Core Concepts

### Session

A bounded unit of work (e.g., "implement namespace normalization"). Has a lifecycle: `active` → `paused` → `completed`. Can span multiple Claude Code conversations. Each agent has at most one active session at a time.

### Shadow Context

A structured, always-current document assembled on-demand from a session's components. This is the single artifact an agent reads to restore its full working state after context loss. It answers three questions: *What am I doing? What do I know? What's left?*

### Shadow Context Components

| Component | Purpose | Write pattern |
|-----------|---------|---------------|
| **Plan** | Ordered steps with completion status | Agent writes/updates as it plans and completes steps |
| **Decisions** | Key choices made and rationale | Agent logs when making non-obvious choices |
| **File Map** | Files read/modified with compact summaries | Agent logs after reading or editing files |
| **Progress** | What's done, what's in flight, what's next | Agent updates at natural milestones |
| **Scratchpad** | Arbitrary key-value notes | Agent writes anything it wants to remember |

## MCP Tools

Six tools exposed via Streamable HTTP MCP:

### `ctx_start_session`

Create a new session with a goal description.

**Input**: `{ goal: string, tags?: string[], parent_session_id?: string }`
**Output**: `{ session_id: string, created_at: string }`
**Behavior**: If an active session already exists, it is automatically paused.

### `ctx_update`

Write a structured update to the active session's shadow context.

**Input**: `{ category: "plan" | "decision" | "file" | "progress" | "scratch", content: string, key?: string }`
**Output**: `{ status: "ok", component_count: int }`
**Behavior**:
- `plan`: Replaces the full plan text (Markdown checklist format). Agent sends the complete current plan each time, not diffs.
- `decision`: Appends a new decision entry.
- `file`: Upserts a file summary keyed by path. `key` = file path, `content` = summary.
- `progress`: Appends a progress log entry with auto-timestamp.
- `scratch`: Upserts a scratchpad entry. `key` = name, `content` = value.

### `ctx_get_shadow`

Returns the assembled shadow context document for a session.

**Input**: `{ session_id?: string }` (defaults to active session)
**Output**: `{ session_id: string, goal: string, status: string, shadow: string }`
**Behavior**: Assembles all components into a Markdown document. This is the "restore my brain" call. Designed to be injected into an LLM prompt as-is.

### `ctx_complete_session`

Mark a session as completed. Triggers distillation to NexusCortex.

**Input**: `{ session_id?: string, outcome?: string }` (defaults to active session)
**Output**: `{ status: "completed", nexus_memory_id?: string }`
**Behavior**: Sets session status to `completed`. Distills the shadow context into an action/outcome pair and calls NexusCortex `/memory/learn`. Returns the NexusCortex memory ID if successful.

### `ctx_list_sessions`

List recent sessions.

**Input**: `{ status?: "active" | "paused" | "completed", limit?: int }`
**Output**: `{ sessions: [{ session_id, goal, status, created_at, updated_at }] }`

### `ctx_resume_session`

Resume a paused session.

**Input**: `{ session_id: string }`
**Output**: Same as `ctx_get_shadow` — returns the shadow context so the agent can immediately continue.
**Behavior**: Sets session status back to `active`. Pauses any currently active session.

## Shadow Context Document Format

```markdown
## Session: implement namespace normalization
**Status**: active | **Started**: 2026-03-14T02:00:00Z | **Updated**: 2026-03-14T02:15:00Z

### Plan
- [x] Step 1: Add normalize_namespace function to models.py
- [x] Step 2: Add field validators to 3 request models
- [ ] Step 3: Fix broken tests
- [ ] Step 4: Run full test suite

### Decisions
- [02:05] Using field_validator not model_validator — simpler for single-field validation
- [02:08] Response models intentionally NOT normalized — they reflect stored values
- [02:10] strip() kept in normalize function for migration task (API regex blocks whitespace anyway)

### Files Known
- **app/models.py** — Added normalize_namespace() function. Added field_validator("namespace") to ContextQuery, ActionLog, GenericEventIngest. 274 lines.
- **tests/test_models.py** — 7 new normalization tests added to TestNamespaceValidation class. 2 existing tests updated for normalized assertions. 632 lines.

### Progress
- [02:06] Steps 1-2 complete, committed as 12b7200
- [02:12] Currently on step 3: fixing broken test assertions

### Scratchpad
- namespace_pattern: ^[a-zA-Z0-9_-]+$
- broken_test_line: 397
```

## Architecture

```
┌─────────────────────┐
│   Claude Code       │
│   (MCP Client)      │
│                     │
│  ctx_update()       │──────────┐
│  ctx_get_shadow()   │          │
│  ctx_start_session()│          │
└─────────────────────┘          │
                                 │ MCP (Streamable HTTP)
                                 │ Port 8070
                                 ▼
┌─────────────────────────────────────────┐
│         CortexBridge                     │
│         (FastAPI + MCP Server)           │
│                                          │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Session Mgr  │  │ Shadow Assembler│  │
│  │ (CRUD, state)│  │ (Markdown gen)  │  │
│  └──────┬───────┘  └────────┬────────┘  │
│         │                    │           │
│  ┌──────▼────────────────────▼────────┐  │
│  │          Redis (sessions DB)       │  │
│  │  Hash per session, sorted sets     │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Distiller    │  │ File Summarizer │  │
│  │ (→ NexusCortex)│ │ (Ollama, optional)│
│  └──────────────┘  └─────────────────┘  │
└──────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌──────────────┐    ┌──────────────┐
│ NexusCortex  │    │   Ollama     │
│ /memory/learn│    │ (summaries)  │
└──────────────┘    └──────────────┘
```

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| API/MCP | FastAPI + fastmcp | Same stack as NexusCortex, proven pattern |
| Storage | Redis (existing instance) | Sessions are ephemeral, Redis is fast, already running |
| MCP Protocol | Streamable HTTP on port 8070 | Same pattern as NexusCortex MCP on 8080 |
| File Summarizer | Ollama (existing instance) | Only for large files (>200 lines). Optional — degrades gracefully |
| Long-term persistence | NexusCortex API | Called on session completion to persist learnings |
| Containerization | Docker Compose | Extends existing NexusCortex compose or standalone |

## Redis Data Model

```
# Session metadata (Hash)
cortexbridge:session:{session_id}
  → goal: string
  → status: "active" | "paused" | "completed"
  → created_at: ISO timestamp
  → updated_at: ISO timestamp
  → tags: JSON array
  → parent_session_id: string | null
  → outcome: string | null

# Active session pointer (String)
cortexbridge:active_session
  → session_id

# Plan (String — full replacement on each update)
cortexbridge:session:{session_id}:plan
  → Markdown checklist text

# Decisions (List — append-only)
cortexbridge:session:{session_id}:decisions
  → JSON: { timestamp, content }

# File map (Hash — keyed by file path)
cortexbridge:session:{session_id}:files
  → field = file_path, value = JSON: { summary, lines, last_action }

# Progress log (List — append-only)
cortexbridge:session:{session_id}:progress
  → JSON: { timestamp, content }

# Scratchpad (Hash — keyed by name)
cortexbridge:session:{session_id}:scratch
  → field = key, value = content string

# Session index (Sorted Set — scored by updated_at for listing)
cortexbridge:sessions
  → member = session_id, score = epoch timestamp
```

## Session Lifecycle

1. **Start**: Agent calls `ctx_start_session(goal)`. Creates Redis keys, sets active pointer.
2. **Work**: Agent calls `ctx_update` as it works. Minimal overhead — just Redis writes.
3. **Context loss**: Agent calls `ctx_get_shadow()`. Gets full state back in one call.
4. **Pause**: Implicit when a new session starts, or explicit via starting another session.
5. **Resume**: Agent calls `ctx_resume_session(id)`. Gets shadow context, resumes work.
6. **Complete**: Agent calls `ctx_complete_session()`. Distills to NexusCortex, keeps in Redis for history.
7. **Expiry**: Completed sessions expire from Redis after 7 days (configurable). The long-term memory lives in NexusCortex.

## Distillation to NexusCortex

When a session completes, CortexBridge calls NexusCortex `POST /memory/learn`:

```json
{
  "action": "{goal} — {plan summary}",
  "outcome": "{outcome or auto-generated from progress}",
  "resolution": "{key decisions joined}",
  "tags": ["{session tags}", "cortexbridge"],
  "domain": "development",
  "namespace": "{configurable, default: cortexbridge}"
}
```

The distillation is structured extraction, not LLM-dependent. It concatenates plan steps marked done, joins decisions, and uses the outcome if provided. No Ollama needed.

## File Summarizer (Optional)

When `ctx_update(category="file")` is called without a summary (just a file path), CortexBridge can optionally auto-summarize:

- Files <= 200 lines: Store first line (shebang/docstring) + line count. No LLM.
- Files > 200 lines: Call Ollama for a 2-3 sentence summary. Cache result in Redis.
- If Ollama is unavailable: Fall back to first-line + line-count.

This is the only LLM-dependent feature and it degrades gracefully.

## Configuration

All via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | `redis://localhost:6379/2` | Redis connection (different DB from NexusCortex) |
| `MCP_PORT` | `8070` | MCP server port |
| `NEXUS_API_URL` | `http://localhost:8100` | NexusCortex API for distillation |
| `NEXUS_API_KEY` | `""` | NexusCortex API key (if required) |
| `NEXUS_NAMESPACE` | `cortexbridge` | Namespace for distilled memories |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama for file summarization |
| `LLM_MODEL` | `qwen3` | Model for summarization |
| `SESSION_TTL_DAYS` | `7` | How long completed sessions persist in Redis |
| `MAX_SESSIONS` | `100` | Max sessions kept in index |
| `AUTO_SUMMARIZE` | `true` | Enable auto file summarization |

## What This Does NOT Do

- Does NOT manage the context window directly (that's the LLM client's job)
- Does NOT intercept or modify agent messages
- Does NOT store full file contents (only summaries)
- Does NOT replace NexusCortex — it feeds into it
- Does NOT require the agent to change its workflow radically — just add a few tool calls at natural points
