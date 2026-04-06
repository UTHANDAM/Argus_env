from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

class ArgusAction(Action):
    severity: Optional[str] = None
    affected_service: Optional[str] = None
    failure_type: Optional[str] = None
    missing_baseline: Optional[str] = None
    cherry_picked_variant: Optional[str] = None
    estimated_std_low: Optional[float] = None
    estimated_std_high: Optional[float] = None
    contamination_risk: Optional[float] = None
    evidence: List[str] = []
    remediation_steps: List[str] = []

class ArgusObservation(Observation):
    task_name: str = ""
    prompt: str = ""
    feedback: Optional[str] = None
    done: bool = False
    reward: float = 0.0

class ArgusState(State):
    current_task: str = "missing_baseline"
    step_count: int = 0
    episode_id: str = ""
    total_reward: float = 0.0
    