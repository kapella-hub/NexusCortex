"""NexusCortex exception hierarchy.

Maps to HTTP error responses via FastAPI exception handlers.
Never leak stack traces to clients.
"""


class NexusCortexError(Exception):
    """Base exception for all NexusCortex errors."""


class GraphConnectionError(NexusCortexError):
    """Neo4j connection or query failure."""


class VectorStoreError(NexusCortexError):
    """Qdrant operation failure."""


class LLMExtractionError(NexusCortexError):
    """LLM call or JSON parse failure."""


class StreamIngestionError(NexusCortexError):
    """Redis push failure."""


class ConfigurationError(NexusCortexError):
    """Raised when configuration is invalid or missing."""
