import asyncio
import os
import json
import textwrap
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = "argus-ml-integrity"
BENCHMARK = "argus_env"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML research auditor detecting integrity violations in ML papers.
    Respond with ONLY valid JSON. No explanation. No markdown. No code blocks.

    For missing_baseline tasks:
    {"missing_baseline": "<method-name>"}

    For cherry_pick tasks:
    {"cherry_picked_variant": "<variant-name>", "estimated_std_low": 1.5, "estimated_std_high": 3.5}

    For contamination tasks:
    {"contamination_risk": 0.85, "evidence": ["signal1", "signal2", "signal3"]}
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    action_clean = str(action).replace("\n", " ")[:200]
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_model_action(client, task_name, prompt):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {task_name}\n\n{prompt}"}
            ],
            max_tokens=300,
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        print(f"[DEBUG] Model response: {raw}", flush=True)
        return raw
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return "{}"


async def main():
    from argus_env.client import ArgusEnv
    from argus_env.models import ArgusAction

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with ArgusEnv(base_url="https://uthandam-argus-env.hf.space") as env:
            result = await env.reset()

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs = result.observation
                raw = get_model_action(client, obs.task_name, obs.prompt)

                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}

                action = ArgusAction(**data)
                result = await env.step(action)

                reward = result.reward or 0.0
                done = result.done
                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=raw, reward=reward, done=done, error=None)

                if done:
                    break

        score = sum(rewards) / MAX_STEPS
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())