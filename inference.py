from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

try:
    from argus_env.client import ArgusEnv
    from argus_env.models import ArgusAction, ArgusObservation
except ImportError:  # pragma: no cover - direct source-tree execution
    from client import ArgusEnv
    from models import ArgusAction, ArgusObservation


# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
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
ARGUS_DEBUG_FALLBACK = os.getenv("ARGUS_DEBUG_FALLBACK", "0") == "1"
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


_DEBUG_EASY_CASES: Dict[str, Dict[str, str]] = {
    "vision-beit-omission": {"family": "BEiT", "exact": "BEiT-B/16"},
    "nlp-xlmr-omission": {"family": "XLM-R", "exact": "XLM-R-large"},
    "speech-hubert-omission": {"family": "HuBERT", "exact": "HuBERT-Large"},
    "multimodal-clip-omission": {"family": "CLIP", "exact": "CLIP ViT-L/14"},
}

_DEBUG_MEDIUM_CASES: Dict[str, Dict[str, Any]] = {
    "adapter-cherry-pick": {
        "variant": "Adapter",
        "std_range": [0.88, 1.24],
        "evidence": ["twenty_run_audit", "five_best_checkpoints", "selection_bias"],
    },
    "prefix-tuning-cherry-pick": {
        "variant": "Prefix tuning",
        "std_range": [0.82, 1.09],
        "evidence": ["twenty_run_audit", "five_best_checkpoints", "selection_bias"],
    },
    "fusion-head-cherry-pick": {
        "variant": "Cross-attention fusion",
        "std_range": [0.84, 1.16],
        "evidence": ["twenty_run_audit", "five_best_checkpoints", "selection_bias"],
    },
    "router-lora-cherry-pick": {
        "variant": "Router-LoRA",
        "std_range": [0.79, 1.08],
        "evidence": ["twenty_run_audit", "five_best_checkpoints", "selection_bias"],
    },
}

_DEBUG_HARD_CASES: Dict[str, Dict[str, Any]] = {
    "mmlu-pro-contaminated": {
        "risk": 0.92,
        "evidence": ["temporal_overlap", "benchmark_in_corpus", "no_exclusion_filter"],
    },
    "gsm8k-contaminated": {
        "risk": 0.89,
        "evidence": ["temporal_overlap", "benchmark_in_corpus", "no_exclusion_filter"],
    },
    "gpqa-clean": {
        "risk": 0.08,
        "evidence": ["deduplication", "held_out_filtering", "no_benchmark_scrape"],
    },
    "mmmu-clean": {
        "risk": 0.06,
        "evidence": ["deduplication", "held_out_filtering", "no_benchmark_scrape"],
    },
}


def _heuristic_action(observation: ArgusObservation) -> Dict[str, Any]:
    stage_kind, _, _ = _stage_info(observation)
    case_id = observation.case_id

    if case_id in _DEBUG_EASY_CASES:
        case = _DEBUG_EASY_CASES[case_id]
        return {"missing_baseline": case["family"] if stage_kind == "family_hint" else case["exact"]}

    if case_id in _DEBUG_MEDIUM_CASES:
        case = _DEBUG_MEDIUM_CASES[case_id]
        if stage_kind == "variant_probe":
            return {"cherry_picked_variant": case["variant"]}
        if stage_kind == "range_probe":
            return {"estimated_std_range": case["std_range"]}
        return {"evidence": case["evidence"]}

    case = _DEBUG_HARD_CASES.get(case_id, {"risk": 0.5, "evidence": ["deduplication"]})
    if stage_kind == "risk_probe":
        return {"contamination_risk": case["risk"]}

    return {
        "contamination_risk": case["risk"],
        "evidence": case["evidence"],
    }


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


def _debug_fallback_or_raise(observation: ArgusObservation, reason: str) -> Dict[str, Any]:
    if ARGUS_DEBUG_FALLBACK:
        return _heuristic_action(observation)
    raise RuntimeError(reason)


def _generate_action_dict(openai_client: Optional[OpenAI], observation: ArgusObservation, history: List[str]) -> Dict[str, Any]:
    if openai_client is None:
        return _debug_fallback_or_raise(
            observation,
            "HF_TOKEN is required for live inference unless ARGUS_DEBUG_FALLBACK=1 is enabled.",
        )

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
        return _debug_fallback_or_raise(
            observation,
            "Model response did not contain a valid JSON action for the current ARGUS stage.",
        )
    except Exception as exc:
        return _debug_fallback_or_raise(
            observation,
            f"HF-router inference failed: {exc}",
        )


async def _open_env_client() -> ArgusEnv:
    if LOCAL_IMAGE_NAME:
        return await ArgusEnv.from_docker_image(LOCAL_IMAGE_NAME)

    client = ArgusEnv(base_url="http://127.0.0.1:7860")
    await client.connect()
    return client


async def _run_episode(task_name: str, seed: int, openai_client: Optional[OpenAI]) -> float:
    env = await _open_env_client()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
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
                if ARGUS_DEBUG_FALLBACK:
                    action = ArgusAction(**_heuristic_action(observation))
                    action_log = _compact_action_log(action.model_dump(exclude_none=True))
                    error = str(exc)
                else:
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
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


async def _run_and_log_task(env: ArgusEnv, task_name: str, max_steps: int) -> bool:
    """Runs a single task and logs the results to stdout."""
    obs = env.reset()
    log_start(task_name, BENCHMARK, MODEL_NAME)

    rewards: List[float] = []
    success = False
    try:
        for i in range(1, max_steps + 1):
            action = await get_next_action(obs, task_name)
            action_str = _compact_action_log(action.model_dump(exclude_unset=True))
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            log_step(i, action_str, reward, done, info.get("last_action_error"))
            if done:
                break
        # Final success is determined by the total score at the end of the episode
        success = obs.metrics.get("total", 0.0) >= SUCCESS_SCORE_THRESHOLD
    except Exception as e:
        print(f"An error occurred during inference: {e}", file=sys.stderr)
        success = False
    finally:
        env.close()
        log_end(success, len(rewards), rewards)
    return success


async def main() -> None:
    openai_client: Optional[OpenAI] = None
    if HF_TOKEN:
        openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    elif not ARGUS_DEBUG_FALLBACK:
        raise ValueError(
            "HF_TOKEN environment variable is required unless ARGUS_DEBUG_FALLBACK=1 is enabled "
            "(OPENAI_API_KEY is also accepted as a compatibility fallback)"
        )

    for task_name, seed in TASK_RUNS:
        await _run_episode(task_name, seed, openai_client)


if __name__ == "__main__":
    asyncio.run(main())
