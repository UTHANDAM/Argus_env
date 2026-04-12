from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from client import ArgusEnv
from models import ArgusAction, ArgusObservation


# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Check for required environment variables
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("ARGUS_BENCHMARK", "argus_env")
SCORE_CAP = 1.0

TASK_RUNS: Sequence[Tuple[str, int]] = (
    ("easy", 11),
    ("medium", 22),
    ("hard", 33),
)

TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_STEPS = 6
SUCCESS_SCORE_THRESHOLD = 0.7

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a meticulous ML research integrity analyst.
    You are solving one stage of a multi-step investigation at a time.
    Return exactly one JSON object and nothing else.
    Do not provide explanations, markdown, or code fences.
    Only include the fields relevant to the current stage schema.
    """
).strip()


def _stage_info(observation: ArgusObservation) -> Tuple[str, int, int]:
    metadata = observation.metadata or {}
    stage_kind = (observation.stage_kind or metadata.get("stage_kind") or "").lower()
    stage_index = int(observation.stage_index or metadata.get("stage_index", 1) or 1)
    stage_count = int(observation.stage_count or metadata.get("stage_count", 1) or 1)
    return stage_kind, stage_index, stage_count


def _schema_hint(task_name: str, stage_kind: str) -> str:
    task_name = (task_name or "").lower()
    stage_kind = (stage_kind or "").lower()

    if task_name == "easy":
        return '{"missing_baseline":"string"}'

    if task_name == "medium":
        if stage_kind == "variant_probe":
            return '{"cherry_picked_variant":"string"}'
        if stage_kind == "range_probe":
            return '{"estimated_std_range":[low,high]}'
        return '{"evidence":["signal"]}'

    if stage_kind == "risk_probe":
        return '{"contamination_risk":0.0}'

    return '{"contamination_risk":0.0,"evidence":["signal"]}'


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def _compact_action_log(action: Dict[str, Any]) -> str:
    if not action:
        return "{}"

    parts: List[str] = []
    for key in sorted(action.keys()):
        if key == "metadata" and action[key] in ({}, None):
            continue
        value = action[key]
        if isinstance(value, list):
            formatted = "[" + "|".join(str(item).replace(" ", "_") for item in value) + "]"
        else:
            formatted = str(value).replace(" ", "_")
        parts.append(f"{key}={formatted}")
    return ";".join(parts)


def _parse_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    return {}


def _build_user_prompt(observation: ArgusObservation, history: List[str]) -> str:
    stage_kind, stage_index, stage_count = _stage_info(observation)
    schema_hint = _schema_hint(observation.task_name or observation.task_difficulty or "", stage_kind)
    history_block = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(
        f"""
        Task: {observation.task_name}
        Difficulty: {observation.task_difficulty}
        Stage: {stage_index}/{stage_count}
        Stage kind: {stage_kind}
        Stage name: {observation.stage_name}

        Instruction:
        {observation.task_instruction}

        Feedback from the previous stage:
        {observation.feedback or 'None'}

        Context:
        {observation.context}

        Recent trajectory:
        {history_block}

        Return exactly one JSON object matching this stage schema:
        {schema_hint}
        """
    ).strip()


def _generate_action_dict(openai_client: Optional[OpenAI], observation: ArgusObservation, history: List[str]) -> Dict[str, Any]:
    if openai_client is None:
        raise RuntimeError("OpenAI client is not available.")

    try:
        completion = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(observation, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = _parse_json_object(content)
        if parsed:
            return parsed
        raise ValueError("Model response did not contain a valid JSON action for the current ARGUS stage.")
    except Exception as exc:
        raise RuntimeError(f"HF-router inference failed: {exc}") from exc


async def _open_env_client() -> ArgusEnv:
    if LOCAL_IMAGE_NAME:
        return await ArgusEnv.from_docker_image(LOCAL_IMAGE_NAME)

    client = ArgusEnv(base_url="http://127.0.0.1:7860")
    await client.connect()
    return client


async def _run_episode(task_name: str, seed: int, openai_client: Optional[OpenAI]) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []
    env: Optional[ArgusEnv] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await _open_env_client()
        result = await env.reset(task=task_name, seed=seed)
        observation = result.observation
        done = bool(result.done)

        while steps_taken < MAX_STEPS and not done:
            action_dict = _generate_action_dict(openai_client, observation, history)

            try:
                action = ArgusAction(**action_dict)
                action_log = _compact_action_log(action.model_dump(exclude_none=True))
                error = None
            except Exception as exc:
                raise RuntimeError(f"Generated action did not validate against ArgusAction: {exc}") from exc

            step_result = await env.step(action)
            reward = float(step_result.reward or 0.0)
            done = bool(step_result.done)

            rewards.append(reward)
            steps_taken += 1
            log_step(step=steps_taken, action=action_log, reward=reward, done=done, error=error)

            history.append(
                f"Stage {steps_taken}: kind={observation.stage_kind or 'unknown'} reward={reward:.2f} action={action_log} feedback={observation.feedback or 'None'}"
            )

            observation = step_result.observation

        score = float(observation.episode_reward or 0.0)
        score = min(SCORE_CAP, score)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        score = 0.0
        success = False
        if steps_taken == 0:
            log_step(step=1, action="{}", reward=0.0, done=True, error=str(exc))
        else:
            log_step(step=steps_taken + 1, action="{}", reward=0.0, done=True, error=str(exc))
    finally:
        try:
            if env is not None:
                await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


async def main() -> None:
    openai_client: Optional[OpenAI] = client

    for task_name, seed in TASK_RUNS:
        await _run_episode(task_name, seed, openai_client)


if __name__ == "__main__":
    asyncio.run(main())
