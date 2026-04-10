"""
Simple task definitions for EV Charging Scheduler Environment.
This module provides compatibility exports for SimpleTask, create_easy_task,
create_medium_task, and create_hard_task used by the OpenEnv evaluation framework.
"""

# Re-export everything needed from the tasks package
from .tasks import (
    Task as SimpleTask,
    TaskResult,
    TaskGrader,
    create_easy_task,
    create_medium_task,
    create_hard_task,
)

__all__ = [
    "SimpleTask",
    "TaskResult",
    "TaskGrader",
    "create_easy_task",
    "create_medium_task",
    "create_hard_task",
]
