class IngestionError(Exception):
    """Base class."""


class ValidationFailed(IngestionError):
    """Malformed or unsupported record/request."""


class NotFound(IngestionError):
    """Unknown source/version, or not visible to the given brand."""


class IndexingError(IngestionError):
    """Embedding/indexing failed. The version stays non-searchable."""


class SearchServiceError(IngestionError):
    """Retrieval/index infrastructure failure. NEVER the same as 'no results'."""
