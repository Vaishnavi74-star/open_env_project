"""
EV Charging Scheduler - Production OpenEnv Environment

A sophisticated multi-objective optimization environment for electric vehicle
charging scheduling with dynamic pricing, grid constraints, and priority management.
"""

__version__ = "1.0.0"
__author__ = "AI Engineering Team"

try:
    from .env import EVChargingEnvironment, EVChargingEnvironmentWrapper
except Exception as e:
    import warnings
    warnings.warn(f"Could not import env module: {e}")
    EVChargingEnvironment = None
    EVChargingEnvironmentWrapper = None

try:
    from .models import (
        EnvironmentConfig,
        Observation,
        ChargingAction,
        StepResult,
        VehicleStatus,
        ChargingStationStatus,
        GridStatus,
    )
except Exception as e:
    import warnings
    warnings.warn(f"Could not import models module: {e}")

try:
    from .tasks import (
        Task,
        TaskGrader,
        TaskResult,
        create_easy_task,
        create_medium_task,
        create_hard_task,
    )
    # Provide SimpleTask alias for compatibility
    SimpleTask = Task
except Exception as e:
    import warnings
    warnings.warn(f"Could not import tasks module: {e}")

try:
    from .simple_tasks import (
        SimpleTask,
        create_easy_task,
        create_medium_task,
        create_hard_task,
    )
except Exception as e:
    import warnings
    warnings.warn(f"Could not import simple_tasks module: {e}")

try:
    from .baselines import (
        BaseAgent,
        RandomAgent,
        GreedyAgent,
        PriorityAwareAgent,
        OptimalSearchAgent,
    )
except Exception as e:
    import warnings
    warnings.warn(f"Could not import baselines module: {e}")

__all__ = [
    # Environment
    "EVChargingEnvironment",
    "EVChargingEnvironmentWrapper",
    # Models
    "EnvironmentConfig",
    "Observation",
    "ChargingAction",
    "StepResult",
    "VehicleStatus",
    "ChargingStationStatus",
    "GridStatus",
    # Tasks
    "Task",
    "SimpleTask",
    "TaskGrader",
    "TaskResult",
    "create_easy_task",
    "create_medium_task",
    "create_hard_task",
    # Baselines
    "BaseAgent",
    "RandomAgent",
    "GreedyAgent",
    "PriorityAwareAgent",
    "OptimalSearchAgent",
]
