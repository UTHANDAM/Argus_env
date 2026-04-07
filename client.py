"""OpenEnv client for ARGUS."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
	from .models import ArgusAction, ArgusObservation, ArgusState
except ImportError:  # pragma: no cover - direct source-tree execution
	from models import ArgusAction, ArgusObservation, ArgusState


class ArgusEnv(EnvClient[ArgusAction, ArgusObservation, ArgusState]):
	"""Typed client for the ARGUS environment."""

	def _step_payload(self, action: ArgusAction) -> Dict[str, Any]:
		return action.model_dump(exclude_none=True)

	def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ArgusObservation]:
		observation_payload = payload.get("observation", {})

		observation = ArgusObservation(
			task_name=observation_payload.get("task_name", ""),
			task_instruction=observation_payload.get("task_instruction", ""),
			context=observation_payload.get("context", ""),
			task_difficulty=observation_payload.get("task_difficulty", ""),
			case_id=observation_payload.get("case_id", ""),
			stage_index=observation_payload.get("stage_index", 1),
			stage_count=observation_payload.get("stage_count", 1),
			stage_name=observation_payload.get("stage_name", ""),
			stage_kind=observation_payload.get("stage_kind", ""),
			stage_weight=observation_payload.get("stage_weight", 0.0),
			next_focus=observation_payload.get("next_focus", ""),
			episode_reward=observation_payload.get("episode_reward", 0.0),
			feedback=observation_payload.get("feedback", ""),
			done=payload.get("done", observation_payload.get("done", False)),
			reward=payload.get("reward", observation_payload.get("reward")),
			metadata=observation_payload.get("metadata", {}),
		)

		return StepResult(
			observation=observation,
			reward=payload.get("reward"),
			done=payload.get("done", False),
		)

	def _parse_state(self, payload: Dict[str, Any]) -> ArgusState:
		return ArgusState(
			episode_id=payload.get("episode_id"),
			step_count=payload.get("step_count", 0),
			task_name=payload.get("task_name", ""),
			task_difficulty=payload.get("task_difficulty", ""),
			case_id=payload.get("case_id", ""),
			episode_index=payload.get("episode_index", 0),
			task_cursor=payload.get("task_cursor", 0),
			last_reward=payload.get("last_reward", 0.0),
			episode_reward=payload.get("episode_reward", 0.0),
			stage_index=payload.get("stage_index", 0),
			stage_count=payload.get("stage_count", 0),
			stage_name=payload.get("stage_name", ""),
			stage_kind=payload.get("stage_kind", ""),
			last_feedback=payload.get("last_feedback", ""),
		)
