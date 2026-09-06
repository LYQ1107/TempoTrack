"""Structured exceptions used by CLI exit-code mapping."""

class DataUnavailable(RuntimeError):
    """Required local data is missing or does not satisfy its contract."""

class WeightUnavailable(RuntimeError):
    """A requested model/checkpoint is missing or unverified."""

class DependencyUnavailable(RuntimeError):
    """An optional but required runtime dependency is unavailable."""

class ImplementationIncomplete(RuntimeError):
    """A requested research path is not implemented end-to-end."""

class GateNotPassed(RuntimeError):
    """A required repair or data gate has not passed."""

class ArtifactMismatch(RuntimeError):
    """An artifact does not match its declared provenance/hash."""
