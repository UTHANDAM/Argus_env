from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

try:
    from argus_env.client import ArgusEnv
    from argus_env.models import ArgusAction, ArgusObservation
except ImportError:  # pragma: no cover - direct source-tree execution
    from client import ArgusEnv
    from models import ArgusAction, ArgusObservation


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL") or os.getenv("SPACE_URL") or os.getenv("OPENENV_URL")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
BENCHMARK = os.getenv("ARGUS_BENCHMARK", "argus_env")

TASK_RUNS: Sequence[Tuple[str, int]] = (
    ("easy", 11),
    ("medium", 22),
    ("hard", 33),
)

TEMPERATURE = 0.0
MAX_TOKENS = 256

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a meticulous ML research integrity analyst.
    Return exactly one JSON object and nothing else.
    Do not provide explanations, markdown, or code fences.
    The JSON keys must match the requested task schema exactly.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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


def _easy_guess(context: str) -> str:
    patterns = (
        r"discusses\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)\s+as\s+a\s+standard\s+baseline",
        r"cites\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)\s+as\s+the\s+stronger\s+published\s+baseline",
        r"cites\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(".,;:")

    tokens = re.findall(r"[A-Z][A-Za-z0-9\-\/+.]+", context)
    return tokens[0] if tokens else "UnknownBaseline"


def _medium_guess(context: str) -> Dict[str, Any]:
    entries: List[Tuple[str, float]] = []
    for raw_line in context.splitlines():
        line = raw_line.strip()
        match = re.search(
            r"^(?P<variant>[^:]+):.*?(?:±|StdDev=|Std=)\s*(?P<std>[0-9.]+)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            entries.append((match.group("variant").strip(), float(match.group("std"))))

    if not entries:
        return {
            "cherry_picked_variant": "Unknown",
            "estimated_std_range": [0.8, 1.2],
        }

    suspicious_variant, suspicious_std = min(entries, key=lambda item: item[1])
    other_stds = [std for variant, std in entries if variant != suspicious_variant]

    if other_stds:
        median_std = statistics.median(other_stds)
        low = round(max(0.05, median_std * 0.8), 2)
        high = round(median_std * 1.2, 2)
    else:
        low = round(max(0.05, suspicious_std * 8.0), 2)
        high = round(max(low + 0.1, suspicious_std * 12.0), 2)

    return {
        "cherry_picked_variant": suspicious_variant,
        "estimated_std_range": [low, high],
    }


def _hard_guess(context: str) -> Dict[str, Any]:
    lower_context = context.lower()
    contaminated_signals = [
        "solution threads",
        "not explicitly excluded",
        "public educational material",
        "benchmark questions appear",
        "benchmark in the corpus",
        "model released after the benchmark",
        "training cutoff",
        "benchmark released before training ended",
    ]

    clean_signals = [
        "deduplicated",
        "held-out",
        "no benchmark-specific scrape",
        "exact and fuzzy matching",
        "removed with exact and fuzzy matching",
    ]

    evidence: List[str] = []
    if any(signal in lower_context for signal in contaminated_signals):
        risk = 0.9
        if "training cutoff" in lower_context or "benchmark released" in lower_context:
            evidence.append("temporal_overlap")
        if "solution threads" in lower_context or "benchmark questions appear" in lower_context or "public educational material" in lower_context:
            evidence.append("benchmark_in_corpus")
        if "not explicitly excluded" in lower_context or "public educational material" in lower_context:
            evidence.append("no_exclusion_filter")
    elif any(signal in lower_context for signal in clean_signals):
        risk = 0.1
        if "deduplicated" in lower_context or "exact and fuzzy matching" in lower_context:
            evidence.append("deduplication")
        if "held-out" in lower_context or "removed with exact and fuzzy matching" in lower_context:
            evidence.append("held_out_filtering")
        if "no benchmark-specific scrape" in lower_context:
            evidence.append("no_benchmark_scrape")
    else:
        risk = 0.5
        evidence = ["temporal_overlap"] if "benchmark" in lower_context else []

    if not evidence:
        evidence = ["temporal_overlap"] if risk > 0.5 else ["deduplication"]

    return {
        "contamination_risk": round(risk, 2),
        "evidence": evidence[:3],
    }


def _heuristic_action(observation: ArgusObservation) -> Dict[str, Any]:
    task_name = (observation.task_name or observation.task_difficulty or "").lower()
    if task_name == "easy":
        return {"missing_baseline": _easy_guess(observation.context)}
    if task_name == "medium":
        return _medium_guess(observation.context)
    return _hard_guess(observation.context)


def _build_user_prompt(observation: ArgusObservation) -> str:
    schema_hint = {
        "easy": '{"missing_baseline":"string"}',
        "medium": '{"cherry_picked_variant":"string","estimated_std_range":[low,high]}',
        "hard": '{"contamination_risk":0.0,"evidence":["signal"]}',
    }.get((observation.task_name or observation.task_difficulty or "").lower(), "{}")

    return textwrap.dedent(
        f"""
        Task: {observation.task_name}
        Difficulty: {observation.task_difficulty}
        Instruction:
        {observation.task_instruction}

        Context:
        {observation.context}

        Return exactly one JSON object matching this schema:
        {schema_hint}
        """
    ).strip()


def _generate_action_dict(openai_client: Optional[OpenAI], observation: ArgusObservation) -> Dict[str, Any]:
    if openai_client is None:
        return _heuristic_action(observation)

    try:
        completion = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(observation)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = _parse_json_object(content)
        if parsed:
            return parsed
    except Exception:
        pass

    return _heuristic_action(observation)


async def _open_env_client() -> ArgusEnv:
    if ENV_URL:
        client = ArgusEnv(base_url=ENV_URL)
        await client.connect()
        return client

    if LOCAL_IMAGE_NAME:
        return await ArgusEnv.from_docker_image(LOCAL_IMAGE_NAME)

    client = ArgusEnv(base_url="http://127.0.0.1:8000")
    await client.connect()
    return client


async def _run_episode(task_name: str, seed: int, openai_client: Optional[OpenAI]) -> float:
    env = await _open_env_client()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = await env.reset(task=task_name, seed=seed)
        observation = reset_result.observation
        action_dict = _generate_action_dict(openai_client, observation)

        try:
            action = ArgusAction(**action_dict)
            action_log = _compact_action_log(action.model_dump(exclude_none=True))
            error = None
        except Exception as exc:
            action = ArgusAction(**_heuristic_action(observation))
            action_log = _compact_action_log(action.model_dump(exclude_none=True))
            error = str(exc)

        step_result = await env.step(action)
        reward = float(step_result.reward or 0.0)
        done = bool(step_result.done)

        rewards.append(reward)
        steps_taken = 1
        score = max(0.0, min(1.0, reward))
        success = score >= 0.5

        log_step(step=1, action=action_log, reward=reward, done=done, error=error)

    except Exception as exc:
        score = 0.0
        success = False
        log_step(step=1, action="{}", reward=0.0, done=True, error=str(exc))
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    openai_client: Optional[OpenAI] = None
    if HF_TOKEN:
        openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_name, seed in TASK_RUNS:
        await _run_episode(task_name, seed, openai_client)


if __name__ == "__main__":
    asyncio.run(main())
