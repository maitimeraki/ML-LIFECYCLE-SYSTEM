"""
Custom exception hierarchy for precise error handling across the platform.
"""


class MLPlatformError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# --- Data Errors ---
class DataIngestionError(MLPlatformError):
    pass


class DataValidationError(MLPlatformError):
    pass


class SchemaViolationError(DataValidationError):
    pass


class DataQualityError(DataValidationError):
    pass


class InsufficientDataError(MLPlatformError):
    pass


# --- Drift Errors ---
class DriftDetectionError(MLPlatformError):
    pass


# --- Training Errors ---
class TrainingError(MLPlatformError):
    pass


class HyperparameterTuningError(TrainingError):
    pass


# --- Evaluation Errors ---
class ModelValidationError(MLPlatformError):
    pass


class ApprovalGateError(MLPlatformError):
    pass


# --- Registry Errors ---
class ModelRegistryError(MLPlatformError):
    pass


class ModelNotFoundError(ModelRegistryError):
    pass


class ArtifactStorageError(ModelRegistryError):
    pass


# --- Deployment Errors ---
class DeploymentError(MLPlatformError):
    pass


class RollbackError(DeploymentError):
    pass


class HealthCheckError(DeploymentError):
    pass


# --- Serving Errors ---
class PredictionError(MLPlatformError):
    pass


class ModelLoadError(PredictionError):
    pass


class InputValidationError(PredictionError):
    pass