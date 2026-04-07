---
title: ARGUS - ML Evaluation Integrity Environment
emoji: 🔬
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
	- openenv
	- machine-learning
	- evaluation
---

# ARGUS

ARGUS is an OpenEnv environment for training and evaluating agents on ML research integrity checks. The task is intentionally non-toy: the agent reads paper-like artifacts and must detect the kinds of issues that regularly appear in ML writing and review workflows, including missing baselines, cherry-picked ablation results, and benchmark contamination.

This is a good fit for OpenEnv because it is:

- Realistic. Researchers, reviewers, and lab engineers already do this work.
- Deterministic. Every episode has a clear ground truth and reproducible scoring.
- Structured. The task is expressed through typed action, observation, and state models.
- Graded. The environment returns partial credit instead of a binary pass/fail.

## Environment Interface

ARGUS follows the standard OpenEnv pattern:

- `reset(seed=..., task=...)` starts a new episode and returns an initial observation.
- `step(action)` grades the submitted answer and returns the next observation with `reward` and `done`.
- `state()` returns the current episode state.

The environment supports explicit task selection through `reset(task=...)`.

- `task="easy"` selects missing baseline detection.
- `task="medium"` selects cherry-picked seed detection.
- `task="hard"` selects benchmark contamination assessment.

If `task` is omitted, the environment cycles through the three tasks in order.

## Action Space

`ArgusAction` is a typed Pydantic model with these fields:

- `missing_baseline: str | None` for the easy task.
- `cherry_picked_variant: str | None` for the medium task.
- `estimated_std_range: list[float] | None` for the medium task.
- `contamination_risk: float | None` for the hard task.
- `evidence: list[str] | None` for the hard task.

Only the fields relevant to the active task are used by the grader.

## Observation Space

`ArgusObservation` includes:

- `task_name`
- `task_instruction`
- `context`
- `task_difficulty`
- `case_id`
- `done`
- `reward`
- `metadata`

The observation context is synthetic, but it is written to look like a real ML paper excerpt, ablation table, or evaluation note.

## State

`ArgusState` exposes episode metadata for debugging and reproducibility:

- `episode_id`
- `step_count`
- `task_name`
- `task_difficulty`
- `case_id`
- `episode_index`
- `task_cursor`
- `last_reward`

## Tasks

### Easy: Missing Baseline Detection

The agent receives a comparison table from a paper and must identify the omitted baseline. This mirrors a common review problem: a paper can look stronger simply because one well-known baseline was left out.

Scoring:

- `1.0` for the exact missing baseline.
- Partial credit for family-level or near-match answers.
- `0.0` for clearly wrong answers.

### Medium: Cherry-Picked Seed Detection

The agent receives an ablation table with multiple variants and their run-to-run variance. One variant was cherry-picked from a much larger set of runs. The agent must identify the suspicious variant and give a plausible true standard-deviation range.

Scoring:

- `0.6` for identifying the correct variant.
- `0.4` for providing a range that overlaps the true variance band.
- Partial credit is preserved if only one part is correct.

### Hard: Benchmark Contamination Assessment

The agent receives a training-data and benchmark description with release dates, data-source hints, and possible contamination cues. The agent must estimate contamination risk and cite the evidence signals that support the estimate.

Scoring:

- Up to `0.5` for a risk estimate close to the ground truth.
- Up to `0.5` for matching evidence signals.
- A false positive on a clean case is penalized.

## Setup

Install dependencies with your preferred workflow:

- `uv sync`
- or `pip install -e .`

Run the environment locally:

```bash
uv run server
```

or:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Build the Docker image:

```bash
docker build -t argus-env:latest -f server/Dockerfile .
```

Run the container:

```bash
docker run --rm -p 8000:8000 argus-env:latest
```

Validate the submission:

```bash
openenv validate
```

## Baseline Inference

The root-level `inference.py` script runs all three tasks in order and prints the required OpenEnv log format:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>`

Environment variables used by the script:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `ENV_URL`
- `LOCAL_IMAGE_NAME` or `IMAGE_NAME`

Example:

```bash
python inference.py
```

If `HF_TOKEN` is present, the script uses the OpenAI client against `API_BASE_URL`. If the API is unavailable, it falls back to a deterministic local heuristic so the baseline still reproduces.

## Baseline Scores

The table below is the reproducible local baseline produced by `python -u inference.py` in fallback mode. The script still uses the OpenAI client when `HF_TOKEN` and `API_BASE_URL` are available; these numbers are from the offline deterministic path so they can be reproduced without external API access.

| Task | Score |
| --- | --- |
| Easy | 1.00 |
| Medium | 0.85 |
| Hard | 1.00 |
| Mean | 0.95 |

## Project Layout

- `models.py` contains the typed action, observation, and state models.
- `client.py` contains the typed OpenEnv client.
- `server/argus_env_environment.py` contains the grading and task logic.
- `server/app.py` creates the FastAPI app.
- `server/Dockerfile` builds the HF Space container.
- `inference.py` is the baseline runner.

## Why ARGUS

ARGUS is deliberately aimed at a real ML workflow. Missing baselines, cherry-picked ablations, and contamination checks are not game mechanics; they are part of the daily work of people reviewing and deploying ML systems. That makes the environment useful for evaluating agent reliability, calibration, and research integrity reasoning.
