from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import ArgusAction, ArgusObservation


class ArgusEnv(EnvClient[ArgusAction, ArgusObservation, State]):

    def _step_payload(self, action: ArgusAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[ArgusObservation]:
        obs_data = payload.get("observation", payload)
        observation = ArgusObservation(
            task_name=obs_data.get("task_name", ""),
            prompt=obs_data.get("prompt", ""),
            feedback=obs_data.get("feedback"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
