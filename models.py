from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ArgusAction(Action):
    """Typed action for the ARGUS environment."""

    missing_baseline: Optional[str] = Field(
        default=None,
        description="Exact name of the omitted baseline method for the easy task.",
    )
    cherry_picked_variant: Optional[str] = Field(
        default=None,
        description="Name of the suspiciously cherry-picked variant for the medium task.",
    )
    estimated_std_range: Optional[List[float]] = Field(
        default=None,
        description="Two-element list [low, high] describing the estimated standard deviation range.",
    )
    contamination_risk: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Estimated contamination risk score between 0.0 and 1.0 for the hard task.",
    )
    evidence: Optional[List[str]] = Field(
        default=None,
        description="Evidence signals supporting the contamination estimate.",
    )


class ArgusObservation(Observation):
    """Observation returned by the ARGUS environment."""

    task_name: str = Field(
        default="",
        description="Task identifier such as easy, medium, or hard.",
    )
    task_instruction: str = Field(
        default="",
        description="Instruction describing what the agent must determine.",
    )
    context: str = Field(
        default="",
        description="Research-paper-like context used for grading the answer.",
    )
    task_difficulty: str = Field(
        default="",
        description="Difficulty label for the current task.",
    )
    case_id: str = Field(
        default="",
        description="Deterministic scenario identifier for the current case.",
    )


class ArgusState(State):
    """Extended state object for ARGUS episodes."""

    task_name: str = Field(default="", description="Current task key.")
    task_difficulty: str = Field(default="", description="Current task difficulty.")
    case_id: str = Field(default="", description="Scenario identifier for the active case.")
    episode_index: int = Field(default=0, ge=0, description="Zero-based episode counter.")
    task_cursor: int = Field(default=0, ge=0, description="Round-robin task cursor used when no task is supplied.")
    last_reward: float = Field(default=0.0, description="Reward produced by the most recent step.")
