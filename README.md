# ARGUS - ML Research Integrity Environment

> *Named after Argus Panoptes, the hundred-eyed giant of Greek mythology who sees everything.*

**ARGUS** is an RL environment for training agents to audit machine learning research integrity. It simulates tasks that human reviewers perform imperfectly at scale: detecting missing baselines, cherry-picked results, and benchmark contamination.

## Motivation

The ML reproducibility crisis is a documented, expensive, and active problem. arXiv receives 200+ ML papers per day. Human reviewers routinely miss:

- **Missing baselines** - Papers omit strong competing methods to make proposed approaches appear superior
- **Cherry-picked results** - Authors select the best seed from many runs, reporting artificially low variance
- **Benchmark contamination** - Training data overlaps with test benchmarks, producing inflated scores

No automated integrity checker good enough exists today. ARGUS provides a structured OpenEnv environment to train and evaluate agents that can detect these violations - filling a real gap in the RL/agent community.

## Tasks

### Task 1 - Missing Baseline Detection (Easy)

The agent receives a **results table** from a synthetic ML paper comparing 5 methods across 3 datasets. One well-known baseline from the literature is deliberately absent, making the proposed method appear state-of-the-art. The agent must identify which baseline is missing.

- **Difficulty**: Easy
- **Action**: `{"missing_baseline": "<method-name>"}`
- **Grading**: Exact string match (case-insensitive). Score: **0.0** or **1.0**
- **Domains covered**: Text classification, question answering, summarization, machine translation, code generation, image classification

### Task 2 - Cherry-Picked Seed Detection (Medium)

The agent receives an **ablation study** where one variant shows suspiciously high scores with unusually low variance across 5 seeds (cherry-picked from 20+ runs). The agent must identify which variant was cherry-picked and estimate the true standard deviation range.

- **Difficulty**: Medium
- **Action**: `{"cherry_picked_variant": "<variant-name>", "estimated_std_low": 1.5, "estimated_std_high": 3.5}`
- **Grading**: **0.6** for correct variant identification + **0.4** for std range overlapping with true range
- **Partial reward**: Yes — agent gets credit for variant even if std estimate is wrong

### Task 3 - Benchmark Contamination Assessment (Hard)

The agent receives a **full evaluation setup**: training data sources, benchmark name, model release date, and dataset update timeline. The setup contains injected contamination signals. The agent must output a contamination risk score (0.0–1.0) and list evidence signals.

- **Difficulty**: Hard
- **Action**: `{"contamination_risk": 0.85, "evidence": ["signal1", "signal2", "signal3"]}`
- **Grading**: Up to **0.5** for risk score accuracy (within 0.15 of truth) + up to **0.5** for evidence keyword matching
- **Novel reward design**: **-0.1 penalty** for false positives (flagging low-risk scenarios as high-risk). This tests calibration, not just recall.

## Reward Function Design

Each task rewards **partial progress**, not just binary success:

| Agent Skill Level | Expected Score | Why |
|-------------------|---------------|-----|
| Random/no-op      | ~0.0          | No valid answers provided |
| Weak agent        | ~0.2–0.3      | Gets easy task right occasionally |
| Good agent        | ~0.5–0.7      | Identifies variants, misses subtleties |
| Strong reasoning  | ~0.8–1.0      | Correct identifications + accurate estimates |

This **discriminative reward function** ensures the environment genuinely differentiates between agent capabilities.

## Action Space

```json
{
  "missing_baseline": "string (optional)",
  "cherry_picked_variant": "string (optional)",
  "estimated_std_low": "float (optional)",
  "estimated_std_high": "float (optional)",
  "contamination_risk": "float 0.0-1.0 (optional)",
  "evidence": ["string"] 
}
```

Fields are populated depending on the current task. Unused fields default to `null` / `[]`.

## Observation Space

```json
{
  "task_name": "string — current task identifier",
  "prompt": "string — the research scenario to analyze",
  "feedback": "string — grading feedback from previous step (null on first step)",
  "done": "boolean — whether the episode is complete",
  "reward": "float — reward from the previous action"
}
```

## State

```json
{
  "episode_id": "string — unique episode identifier",
  "current_task": "string — which task is active",
  "step_count": "int — steps taken in this episode",
  "total_reward": "float — cumulative reward"
}
```

## Procedural Data Generation

All scenarios are **generated programmatically** — no external datasets required:

- **6 ML domains** with 7 baselines each for Task 1
- **12 ablation variant templates** with randomized scores for Task 2
- **3 risk levels** (high/medium/low) with 9 model types, 7 benchmarks, 8 data sources for Task 3
- Every episode produces **unique, never-before-seen scenarios**
- Docker container is **fully self-contained** — graders never depend on external data

## Setup

### Install

```bash
pip install openenv-core
pip install git+https://huggingface.co/spaces/uthandam/argus-env
```

### Run Locally

```bash
# Clone and start server
git clone https://huggingface.co/spaces/uthandam/argus-env
cd argus-env
uv run server
```

### Use as Client

```python
import asyncio
from argus_env import ArgusEnv, ArgusAction

async def main():
    async with ArgusEnv(base_url="https://uthandam-argus-env.hf.space") as env:
        result = await env.reset()
        print(result.observation.task_name)   # "missing_baseline"
        print(result.observation.prompt)       # The research scenario

        action = ArgusAction(missing_baseline="DeBERTa-base")
        result = await env.step(action)
        print(result.reward)                   # 1.0 if correct

asyncio.run(main())
```

## Run Baseline Inference

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### With Docker

```bash
export IMAGE_NAME=argus-env:latest
export HF_TOKEN=your_token
python inference.py
```

## Baseline Scores

| Model | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Total Score |
|-------|--------------|-----------------|----------------|-------------|
| Qwen2.5-72B-Instruct | ~1.0 | ~0.6–1.0 | ~0.5–0.9 | ~0.70–0.97 |

> **Note**: Scores vary per episode due to procedural generation. The range reflects typical performance across multiple runs.

## Environment URL

**https://huggingface.co/spaces/uthandam/argus-env**

## Architecture

```
argus_env/
├── __init__.py              # Exports ArgusAction, ArgusObservation, ArgusEnv
├── models.py                # Typed Action, Observation, State (Pydantic)
├── client.py                # EnvClient implementation
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI server (create_app)
    ├── argus_env_environment.py  # Environment logic + graders
    ├── Dockerfile            # Container definition
    └── requirements.txt      # Server dependencies
```

## License

BSD-3-Clause