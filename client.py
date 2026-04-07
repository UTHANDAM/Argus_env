"""
ARGUS — Client
===============
EnvClient subclass for connecting to ARGUS environments via HTTP.
Supports both remote HF Spaces and local Docker containers.

Design decision: _parse_result extracts observation fields from the
nested "observation" key, while reward/done come from the top-level
response. This matches the OpenEnv create_app() response format.
"""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import ArgusAction, ArgusObservation


class ArgusEnv(EnvClient[ArgusAction, ArgusObservation, State]):
    """Client for interacting with an ARGUS environment.

    Usage:
        # Remote HF Space
        env = ArgusEnv(base_url="https://uthandam-argus-env.hf.space")

        # Local Docker
        env = await ArgusEnv.from_docker_image("argus-env:latest")
    """

    def _step_payload(self, action: ArgusAction) -> Dict:
        """Serialize action to JSON payload for the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[ArgusObservation]:
        """Parse server response into typed StepResult.

        Server response format:
        {
            "observation": {"task_name": ..., "prompt": ..., "feedback": ...},
            "reward": 0.0,
            "done": false
        }
        """
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
        """Parse state response from server."""
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
