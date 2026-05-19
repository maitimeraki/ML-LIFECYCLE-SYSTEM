"""
Central enumeration definitions used across the platform.
"""

from enum import Enum


class ModelStatus(str, Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    VALIDATED = "validated"
    VALIDATION_FAILED = "validation_failed"
    APPROVED = "approved"
    DEPLOYING = "deploying"
    SERVING = "serving"            # Currently serving traffic (champion)
    CANARY = "canary"              # In canary evaluation
    SHADOW = "shadow"              # Shadow mode (no live traffic impact)
    RETIRED = "retired"
    ROLLED_BACK = "rolled_back"


class DatasetStatus(str, Enum):
    INGESTED = "ingested"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    APPROVED_FOR_TRAINING = "approved_for_training"
    USED = "used"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RetrainDecision(str, Enum):
    NOT_NEEDED = "not_needed"
    RECOMMENDED = "recommended"
    REQUIRED = "required"
    URGENT = "urgent"


class DeploymentStrategy(str, Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    SHADOW = "shadow"
    DIRECT = "direct"  # Only for development


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"