"""
Inference Script — ARGUS ML Research Integrity Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local Docker image (if using from_docker_image())

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

# Design decision: import from argus_env package, not relative imports,
# because inference.py runs standalone from root directory
from argus_env import ArgusEnv
from argus_env.models import ArgusAction

# ── Environment variables (MANDATORY) ───────────────────────────────────────
IMAGE_NAME = os.getenv("IMAGE_NAME")  # Docker image name for from_docker_image()
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Defaults reflect active inference setup
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ── Constants ───────────────────────────────────────────────────────────────
TASK_NAME = "argus-ml-integrity"
BENCHMARK = "argus_env"
MAX_STEPS = 3  # One step per task (missing_baseline, cherry_pick, contamination)
SUCCESS_SCORE_THRESHOLD = 0.5

# ── System prompt — instructs the LLM to respond as a research auditor ──────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML research auditor. Your job is to detect integrity
    violations in machine learning papers. Analyze the given scenario carefully
    and respond with ONLY valid JSON — no explanation, no markdown, no code blocks.

    Depending on the task_name in the prompt:

    For "missing_baseline" tasks — identify the well-known baseline absent from the table:
    {"missing_baseline": "<exact-method-name-from-the-known-baselines-list>"}

    For "cherry_pick" tasks — identify the cherry-picked variant and estimate true std:
    {"cherry_picked_variant": "<variant-name>", "estimated_std_low": 1.5, "estimated_std_high": 3.5}

    For "contamination" tasks — assess risk and list evidence:
    {"contamination_risk": 0.85, "evidence": ["signal1", "signal2", "signal3"]}

    Important rules:
    - For missing_baseline: look at the "Known strong baselines" list and compare
      with the methods shown in the table. Return the one that appears in the known
      list but NOT in the table.
    - For cherry_pick: the cherry-picked variant has suspiciously LOW std and HIGH mean.
    - For contamination: consider temporal overlap, data source known issues, and score plausibility.
    - Always respond with valid JSON only.
""").strip()


# ── Logging functions (EXACT mandatory format) ─────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line — exactly one at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line — one per step, immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Clean action string: remove newlines, cap length for readability
    action_clean = str(action).replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line — always emitted, even on exception, after env.close()."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call using OpenAI Client (MANDATORY) ───────────────────────────────

def get_model_action(client: OpenAI, task_name: str, prompt: str) -> str:
    """Call the LLM to analyze the research scenario and return JSON action.

    Design decision: temperature=0.0 for deterministic, reproducible outputs.
    max_tokens=500 gives enough room for evidence lists without wasting tokens.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {task_name}\n\n{prompt}"},
            ],
            max_tokens=500,
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        # Strip markdown code fences if model wraps response
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        print(f"[DEBUG] Model raw response: {raw[:300]}", flush=True)
        return raw
    except Exception as e:
        print(f"[DEBUG] Model call error: {e}", flush=True)
        return "{}"


# ── Main inference loop ─────────────────────────────────────────────────────

async def main() -> None:
    """Run one full episode: reset → 3 steps (one per task) → close."""

    # Initialize OpenAI client with mandatory env vars
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Support both Docker image and remote HF Space
    env = None
    if IMAGE_NAME:
        # from_docker_image() starts a local container
        env = await ArgusEnv.from_docker_image(IMAGE_NAME)
    else:
        # Connect to deployed HF Space
        env = ArgusEnv(base_url="https://uthandam-argus-env.hf.space")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment — returns first task observation
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            task_name = obs.task_name
            prompt = obs.prompt

            # Get LLM's analysis of the research scenario
            raw = get_model_action(client, task_name, prompt)

            # Parse JSON response into typed action
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[DEBUG] JSON parse failed, sending empty action", flush=True)
                data = {}

            # Construct typed action — unknown fields are silently ignored by Pydantic
            action = ArgusAction(**data)

            # Execute step — server grades the action and returns next task
            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = getattr(result, "last_action_error", None)

            rewards.append(reward)
            steps_taken = step

            # Emit [STEP] line immediately after env.step()
            log_step(step=step, action=raw, reward=reward, done=done, error=error)

            if done:
                break

        # Calculate normalized score in [0, 1]
        score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Always close environment and emit [END] — even on exception
        try:
            if env is not None:
                await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())