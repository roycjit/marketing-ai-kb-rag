"""Domain-level exceptions."""


class DomainError(Exception):
    """Base for all domain errors."""

    pass


class RepositoryError(DomainError):
    """Raised when a repository operation fails."""

    pass


class ValidationError(DomainError):
    """Raised when domain validation rules are violated."""

    pass


class ConflictResolutionError(DomainError):
    """Raised when document conflict resolution cannot produce a deterministic winner."""

    pass
