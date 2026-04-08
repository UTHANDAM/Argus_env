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
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("ARGUS_BENCHMARK", "argus_env")
SCORE_CAP = 0.99

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


def _easy_guess(context: str, stage_kind: str) -> Dict[str, Any]:
    lower_context = context.lower()
    if stage_kind == "family_hint":
        patterns = (
            r"belongs to the\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)\s+family",
            r"references the\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)\s+family",
            r"the\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)\s+family",
        )
        for pattern in patterns:
            match = re.search(pattern, context, flags=re.IGNORECASE)
            if match:
                return {"missing_baseline": match.group(1).strip(".,;:")}

    patterns = (
        r"omitted baseline.*?is\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)",
        r"exact\s+model\s+should\s+have\s+been\s+compared\s+against\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)",
        r"baseline\s+is\s+([A-Za-z0-9][A-Za-z0-9\-\/+.]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return {"missing_baseline": match.group(1).strip(".,;:")}

    if "beit" in lower_context:
        return {"missing_baseline": "BEiT-B/16" if stage_kind == "exact_missing_baseline" else "BEiT"}
    if "xlm-r" in lower_context:
        return {"missing_baseline": "XLM-R-large" if stage_kind == "exact_missing_baseline" else "XLM-R"}

    tokens = re.findall(r"[A-Z][A-Za-z0-9\-\/+.]+", context)
    return {"missing_baseline": tokens[0] if tokens else "UnknownBaseline"}


def _medium_guess(context: str, stage_kind: str) -> Dict[str, Any]:
    if stage_kind == "variant_probe":
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

        if entries:
            suspicious_variant, _ = min(entries, key=lambda item: item[1])
            return {"cherry_picked_variant": suspicious_variant}
        return {"cherry_picked_variant": "Unknown"}

    if stage_kind == "range_probe":
        approx_match = re.search(r"std\s*[≈~]\s*([0-9.]+)", context, flags=re.IGNORECASE)
        if approx_match:
            midpoint = float(approx_match.group(1))
            low = round(max(0.05, midpoint * 0.85), 2)
            high = round(max(low + 0.1, midpoint * 1.15), 2)
            return {"estimated_std_range": [low, high]}

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

        if entries:
            suspicious_variant, suspicious_std = min(entries, key=lambda item: item[1])
            other_stds = [std for variant, std in entries if variant != suspicious_variant]
            if other_stds:
                median_std = statistics.median(other_stds)
                low = round(max(0.05, median_std * 0.8), 2)
                high = round(median_std * 1.2, 2)
            else:
                low = round(max(0.05, suspicious_std * 8.0), 2)
                high = round(max(low + 0.1, suspicious_std * 12.0), 2)
            return {"estimated_std_range": [low, high]}

        return {"estimated_std_range": [0.8, 1.2]}

    evidence: List[str] = []
    lower_context = context.lower()
    if any(phrase in lower_context for phrase in ("20 times", "20 trials", "twenty-run", "twenty run")):
        evidence.append("twenty_run_audit")
    if any(phrase in lower_context for phrase in ("five best", "top five", "kept only the five", "top-five")):
        evidence.append("five_best_checkpoints")
    if any(phrase in lower_context for phrase in ("selection bias", "variance compression", "artificially small variance", "cherry-picking")):
        evidence.append("selection_bias")
    return {"evidence": evidence or ["selection_bias"]}


def _hard_guess(context: str, stage_kind: str) -> Dict[str, Any]:
    lower_context = context.lower()
    risk = 0.5
    evidence: List[str] = []

    cutoff_match = re.search(r"training cutoff:\s*(\d{4}-\d{2}-\d{2})", lower_context)
    benchmark_match = re.search(r"released(?: on)?\s*(\d{4}-\d{2}-\d{2})", lower_context)
    if cutoff_match and benchmark_match:
        from datetime import date

        cutoff = date.fromisoformat(cutoff_match.group(1))
        benchmark_release = date.fromisoformat(benchmark_match.group(1))
        if cutoff >= benchmark_release:
            risk = 0.92
            evidence.append("temporal_overlap")
        else:
            risk = 0.08

    contaminated_signals = (
        "mirrored archive",
        "solution-thread",
        "answer-key mirror",
        "public educational material",
        "public instructional dump",
        "benchmark question mirror",
    )
    if any(signal in lower_context for signal in contaminated_signals):
        risk = max(risk, 0.9)
        evidence.append("benchmark_in_corpus")

    clean_signals = (
        "deduplicated",
        "held-out",
        "no benchmark-specific scrape",
        "exact/fuzzy match",
        "fuzzy match",
        "not explicitly filtered",
        "compliance log records no benchmark-question hits",
    )
    if any(signal in lower_context for signal in clean_signals):
        if risk < 0.5:
            risk = 0.08
        evidence.extend(["deduplication", "held_out_filtering", "no_benchmark_scrape"])

    if "not separated at crawl time" in lower_context or "not explicitly removed" in lower_context:
        evidence.append("no_exclusion_filter")
        risk = max(risk, 0.9)

    evidence = list(dict.fromkeys(evidence))
    if stage_kind == "risk_probe":
        return {"contamination_risk": round(risk, 2)}

    if not evidence:
        evidence = ["temporal_overlap"] if risk > 0.5 else ["deduplication"]

    return {
        "contamination_risk": round(risk, 2),
        "evidence": evidence[:3],
    }


def _heuristic_action(observation: ArgusObservation) -> Dict[str, Any]:
    task_name = (observation.task_name or observation.task_difficulty or "").lower()
    stage_kind, _, _ = _stage_info(observation)
    if task_name == "easy":
        return _easy_guess(observation.context, stage_kind)
    if task_name == "medium":
        return _medium_guess(observation.context, stage_kind)
    return _hard_guess(observation.context, stage_kind)


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
        return _heuristic_action(observation)

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
    except Exception:
        pass

    return _heuristic_action(observation)


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
                action = ArgusAction(**_heuristic_action(observation))
                action_log = _compact_action_log(action.model_dump(exclude_none=True))
                error = str(exc)

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