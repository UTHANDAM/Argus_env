"""
ARGUS — Typed Models
====================
All Action, Observation, and State types inherit from OpenEnv base classes.
Pydantic models ensure type safety and automatic JSON serialization.

Design decisions:
- All Action fields are Optional because each step only uses the fields
  relevant to the current task. This avoids requiring agents to guess
  which fields exist before seeing the task prompt.
- evidence defaults to [] not None to avoid null-checking in graders.
"""

from typing import List, Optional
from openenv.core.env_server import Action, Observation, State


class ArgusAction(Action):
    """Action schema for ARGUS ML Research Integrity Environment.

    Fields are populated depending on the current task:
    - missing_baseline: For Task 1 (Missing Baseline Detection)
    - cherry_picked_variant, estimated_std_low/high: For Task 2 (Cherry-Pick Detection)
    - contamination_risk, evidence: For Task 3 (Benchmark Contamination Assessment)
    """
    # Task 1: Missing Baseline Detection
    missing_baseline: Optional[str] = None

    # Task 2: Cherry-Pick Detection
    cherry_picked_variant: Optional[str] = None
    estimated_std_low: Optional[float] = None
    estimated_std_high: Optional[float] = None

    # Task 3: Benchmark Contamination Assessment
    contamination_risk: Optional[float] = None
    evidence: List[str] = []  # Defaults to empty list, not None — simplifies grader logic


class ArgusObservation(Observation):
    """Observation returned by the environment after each step.

    - prompt contains the full research scenario text the agent must analyze
    - feedback contains grading explanation from the previous step (null on first)
    - done=True signals episode completion (all 3 tasks finished)
    """
    task_name: str = ""
    prompt: str = ""
    feedback: Optional[str] = None
    done: bool = False
    reward: float = 0.0


class ArgusState(State):
    """Internal episode state exposed via the state() API.

    Tracks which task is active, cumulative reward, and step count.
    """
    current_task: str = "missing_baseline"
    step_count: int = 0
    episode_id: str = ""
    total_reward: float = 0.0