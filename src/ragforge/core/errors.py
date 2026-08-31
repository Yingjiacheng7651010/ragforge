"""Shared error types for ragforge."""


class RAGForgeError(Exception):
    """Base error for all ragforge failures, carrying a stable error code."""

    def __init__(self, message: str, *, code: str = "E_UNKNOWN") -> None:
        super().__init__(message)
        self.message = message
        self.code = code
