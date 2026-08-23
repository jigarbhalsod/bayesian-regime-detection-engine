from enum import Enum


class DeploymentStatus(str, Enum):
    """
    Represents the current operational status of a deployment.
    """

    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    STOPPED = "stopped"