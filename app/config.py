from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API
    APP_NAME: str = "NexusCortex"
    DEBUG: bool = False
    API_KEY: str | None = None
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    RATE_LIMIT: str = "60/minute"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""  # Must be set via environment / .env
    NEO4J_POOL_SIZE: int = 50

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "nexus_memory"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_STREAM_KEY: str = "nexus:event_stream"
    REDIS_BATCH_SIZE: int = 50
    DLQ_MAX_SIZE: int = 10_000

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # LLM
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "llama3"
    LLM_API_KEY: str = ""  # Must be set via environment / .env

    # Embedding
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIM: int = 768

    # RAG Engine
    BOOST_FACTOR: float = 1.5
    GRAPH_RELEVANCE_WEIGHT: float = 0.4
    CONTENT_HASH_LENGTH: int = 32

    # Re-ranking
    RERANK_ENABLED: bool = False
    RERANK_CANDIDATES_MULTIPLIER: int = 2

    # Re-embed
    REEMBED_BATCH_SIZE: int = 50

    # Memory Decay
    MEMORY_DECAY_HALF_LIFE_DAYS: int = 90

    # Garbage Collection
    MAX_MEMORY_AGE_DAYS: int = 180
    PRUNE_SCORE_THRESHOLD: float = 0.3
    GC_SCHEDULE_HOURS: int = 24

    # Namespace
    DEFAULT_NAMESPACE: str = "default"

    # Memory Agent
    AGENT_ENABLED: bool = True
    AGENT_SCHEDULE_HOURS: int = 6
    AGENT_DUPLICATE_THRESHOLD: float = 0.9
    AGENT_CONFIDENCE_DECAY_DAYS: int = 180
    AGENT_BATCH_LIMIT: int = 100

    # MCP Server
    NEXUS_API_URL: str = "http://localhost:8000"
    NEXUS_API_KEY: str | None = None
    MCP_HOST: str = "0.0.0.0"
    MCP_PORT: int = 8080

    @field_validator("CONTENT_HASH_LENGTH")
    @classmethod
    def validate_hash_length(cls, v: int) -> int:
        if v < 16:
            raise ValueError("CONTENT_HASH_LENGTH must be >= 16 to avoid hash collisions")
        return v

    def model_post_init(self, __context):
        import logging as _logging
        _log = _logging.getLogger("app.config")
        if not self.NEO4J_PASSWORD:
            _log.warning("NEO4J_PASSWORD is empty — set via environment or .env")
        if not self.LLM_API_KEY:
            _log.warning("LLM_API_KEY is empty — set via environment or .env")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
