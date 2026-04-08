---
title: ARGUS - ML Evaluation Integrity Environment
emoji: 🔬
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - machine-learning
  - evaluation
---

# ARGUS

ARGUS is an OpenEnv environment for training and evaluating agents on ML research integrity checks. The task is intentionally non-toy: the agent reads paper-like artifacts and must detect the kinds of issues that regularly appear in ML writing and review workflows, including missing baselines, cherry-picked ablation results, and benchmark contamination.

ARGUS now runs as a staged investigation instead of a one-shot classifier. Each episode unfolds across multiple clue reveals, so the agent must use the full `step()` / `reset()` / `state()` loop to refine the answer as new evidence appears.

This is a good fit for OpenEnv because it is:

- Realistic. Researchers, reviewers, and lab engineers already do this work.
- Deterministic. Every episode has a clear ground truth and reproducible scoring.
- Structured. The task is expressed through typed action, observation, and state models.
- Graded. The environment returns partial credit instead of a binary pass/fail.

## Environment Interface

ARGUS follows the standard OpenEnv pattern, but each episode is multi-step:

- `reset(seed=..., task=...)` starts a new episode and returns the first clue.
- `step(action)` grades the current stage, returns the next clue, and updates `reward` and `done`.
- `state()` returns the current episode and stage state.

The environment supports explicit task selection through `reset(task=...)`.

- `task="easy"` selects missing baseline detection over 2 stages.
- `task="medium"` selects cherry-picked seed detection over 3 stages.
- `task="hard"` selects benchmark contamination assessment over 3 stages.

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
- `stage_index`
- `stage_count`
- `stage_name`
- `stage_kind`
- `stage_weight`
- `next_focus`
- `episode_reward`
- `feedback`
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
- `episode_reward`
- `stage_index`
- `stage_count`
- `stage_name`
- `stage_kind`
- `last_feedback`

## Tasks

### Easy: Missing Baseline Detection

The agent receives a comparison table from a paper and must identify the omitted baseline. The first stage exposes only the family-level clue; the second stage reveals the exact citation. This mirrors a common review problem: a paper can look stronger simply because one well-known baseline was left out.

Scoring:

- `0.35` for the family-level clue stage.
- `0.65` for the exact missing baseline in the final stage.
- `0.0` for clearly wrong answers.

### Medium: Cherry-Picked Seed Detection

The agent receives an ablation table with multiple variants and their run-to-run variance. One variant was cherry-picked from a much larger set of runs. The episode unfolds across three stages: first identify the suspicious variant, then estimate the plausible true standard-deviation range, then cite the strongest evidence signals.

Scoring:

- `0.30` for identifying the correct variant.
- `0.35` for providing a range that overlaps the true variance band.
- `0.35` for matching the evidence signals.

### Hard: Benchmark Contamination Assessment

The agent receives a training-data and benchmark description with release dates, data-source hints, and possible contamination cues. The episode is staged so the agent must revise its answer as more evidence arrives. The final stage asks for a calibrated contamination risk and the evidence signals that support the estimate.

Scoring:

- Up to `0.25` for the initial risk probe.
- Up to `0.35` for refining the risk with evidence.
- Up to `0.40` for the final calibrated answer, with a penalty for confident false positives on clean cases.

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
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Build the Docker image:

```bash
docker build -t argus-env:latest .
```

Run the container:

```bash
docker run --rm -p 7860:7860 argus-env:latest
```

Validate the submission:

```bash
openenv validate
```

## Baseline Inference

The root-level `inference.py` script runs all three tasks in order, steps through the staged clues, and prints the required OpenEnv log format:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>`

Environment variables used by the script:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`
- `LOCAL_IMAGE_NAME` or `IMAGE_NAME` when using `from_docker_image()`

Example:

```bash
python inference.py
```

If `HF_TOKEN` or `OPENAI_API_KEY` is present, the script uses the OpenAI client against `API_BASE_URL`. If the API is unavailable, it falls back to a deterministic local heuristic so the baseline still reproduces. When `LOCAL_IMAGE_NAME` or `IMAGE_NAME` is set, the script connects through `ArgusEnv.from_docker_image()`; otherwise it uses the local service on port 7860.

## Baseline Scores

The table below is the reproducible local baseline produced by `python -u inference.py` in fallback mode. The script still uses the OpenAI client when `HF_TOKEN` and `API_BASE_URL` are available; these numbers are from the offline deterministic path so they can be reproduced without external API access.

| Task | Score |
| --- | --- |
| Easy | 0.990 |
| Medium | 0.895 |
| Hard | 0.714 |
| Mean | 0.866 |

## Project Layout

- `models.py` contains the typed action, observation, and state models.
- `client.py` contains the typed OpenEnv client.
- `server/argus_env_environment.py` contains the grading and task logic.
- `server/app.py` creates the FastAPI app.
- `Dockerfile` builds the HF Space container.
- `inference.py` is the baseline runner.

## Why ARGUS

ARGUS is deliberately aimed at a real ML workflow. Missing baselines, cherry-picked ablations, and contamination checks are not game mechanics; they are part of the daily work of people reviewing and deploying ML systems. That makes the environment useful for evaluating agent reliability, calibration, and research integrity reasoning.
