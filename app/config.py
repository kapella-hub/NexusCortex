from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API
    APP_NAME: str = "NexusCortex"
    DEBUG: bool = False
    API_KEY: str | None = None
    CORS_ORIGINS: list[str] = ["http://localhost:*"]
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

    # Memory Decay
    MEMORY_DECAY_HALF_LIFE_DAYS: int = 90

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
