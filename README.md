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

**ARGUS** is an OpenEnv environment for training and evaluating agents on a real research workflow: auditing ML paper evaluation claims for integrity failures.

The target user is not a generic enterprise agent. It is a research-facing agent that helps reviewers, lab engineers, and post-training teams detect three recurring failures in machine learning evaluation:

- omitted baselines that make a new method look stronger than it is
- cherry-picked ablations that compress variance and overstate stability
- benchmark contamination risks caused by training-data overlap or filtering failures

This is a real task. People reviewing papers and model releases already do this work manually. ARGUS turns that workflow into a reproducible RL environment with deterministic graders and shaped rewards.

## Why This Environment Exists

A strong OpenEnv task should look like something a frontier lab would actually train against. ARGUS is designed around that standard.

- **Real-world utility**: evaluation integrity failures are common, expensive, and directly relevant to model release quality.
- **Partial-credit rewards**: the agent is rewarded for intermediate reasoning, not just final binary correctness.
- **Human-review strength**: the cases are staged and inference-driven, so the environment is not a thin wrapper around answer extraction.
- **Determinism**: every case has a fixed ground truth and reproducible scoring.

## What The Agent Does

Every episode is a staged investigation. The agent sees one clue at a time, responds with a typed action, receives reward and feedback, and then gets the next clue.

- `reset(task=..., seed=...)` starts a deterministic case
- `step(action)` grades the current stage and advances the episode
- `state()` exposes the current investigation state

The public interface stays stable:

- `ArgusAction`
- `ArgusObservation`
- `ArgusState`

## Task Families

ARGUS contains **12 total cases**: 4 easy, 4 medium, and 4 hard.

### 1. Missing Baseline Detection

The agent receives a comparison table and must infer which important baseline was omitted.

- Stage 1: infer the baseline family from indirect lineage clues
- Stage 2: infer the exact missing citation from architecture-level details

Domains covered:

- vision
- multilingual NLP
- speech
- multimodal retrieval

### 2. Cherry-Picked Seed Detection

The agent receives an ablation table where one variant reports implausibly low variance.

- Stage 1: identify the suspicious variant
- Stage 2: estimate the plausible true standard-deviation band
- Stage 3: explain the evidence for selection bias

The grader rewards partial progress and penalizes impossible variance bands or unsupported evidence.

### 3. Benchmark Contamination Assessment

The agent receives a training-data audit trail and must calibrate contamination risk.

- Stage 1: give a provisional risk estimate from release timing and source inventory
- Stage 2: reconcile risk with newly revealed audit evidence
- Stage 3: produce a final calibrated score with supporting signals

Cases include both contaminated and clean setups. Clean cases penalize overconfident false positives.

## Reward Design

The environment is explicitly shaped for learning instead of sparse pass/fail grading.

- **Easy task**: `0.35 + 0.65`
- **Medium task**: `0.30 + 0.35 + 0.35`
- **Hard task**: `0.25 + 0.35 + 0.40`

Additional design choices:

- perfect episodes now reach **exactly `1.00`**
- unsupported evidence can reduce score
- impossible variance ranges are penalized
- clean-case contamination overcalls are penalized
- contaminated-case severe undercalls are also penalized

The shaped reward matters because it makes the environment useful for RL, not just static evaluation.

## Action Space

`ArgusAction` exposes the full task surface:

- `missing_baseline: str | None`
- `cherry_picked_variant: str | None`
- `estimated_std_range: list[float] | None`
- `contamination_risk: float | None`
- `evidence: list[str] | None`

Only the fields relevant to the current stage are used by the grader.

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

The `context` is written like a paper excerpt, audit memo, appendix note, or governance log rather than a synthetic quiz prompt.

## State

`ArgusState` tracks:

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

## Short Episode Snapshots

These are abbreviated examples from the local deterministic smoke path.

### Easy

```text
Stage 1 context: masked-image-pretraining family is hinted, but no exact citation is shown
Action: {"missing_baseline":"BEiT"}
Reward: 0.35

Stage 2 context: base-width checkpoint with 16x16 patches is hinted
Action: {"missing_baseline":"BEiT-B/16"}
Reward: 0.65
```

### Medium

```text
Stage 1 action: {"cherry_picked_variant":"Adapter"}
Reward: 0.30

Stage 2 action: {"estimated_std_range":[0.88,1.24]}
Reward: 0.35

Stage 3 action: {"evidence":["twenty_run_audit","five_best_checkpoints","selection_bias"]}
Reward: 0.35
```

### Hard

```text
Stage 1 action: {"contamination_risk":0.92}
Reward: 0.25

Stage 2 action: {"contamination_risk":0.92,"evidence":["temporal_overlap","benchmark_in_corpus","no_exclusion_filter"]}
Reward: 0.35

Stage 3 action: {"contamination_risk":0.92,"evidence":["temporal_overlap","benchmark_in_corpus","no_exclusion_filter"]}
Reward: 0.40
```

## Setup

Install dependencies:

```bash
uv sync
```

or:

```bash
pip install -e .
```

Run the environment locally:

```bash
uv run server
```

or:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Build the container:

```bash
docker build -t argus-env:latest .
```

Run the container:

```bash
docker run --rm -p 7860:7860 argus-env:latest
```

Validate the environment:

```bash
openenv validate
```

Run the test suite:

```bash
python -m unittest discover -s tests
```

## Baseline Inference

The root-level `inference.py` script follows the required structured logging format:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

Environment variables:

- `API_BASE_URL` with default `https://router.huggingface.co/v1`
- `MODEL_NAME` with default `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN` required for submission-mode inference
- `LOCAL_IMAGE_NAME` optional when using `from_docker_image()`

### Submission Mode

This is the path intended for hackathon evaluation. It uses the **OpenAI client** against the Hugging Face router.

```bash
python inference.py
```

If `HF_TOKEN` is missing, the script fails immediately rather than silently switching to a local heuristic.

## Score Reporting

### HF-Router Baseline

Record these numbers from a real submission-mode run before final submission.

| Task | Score |
| --- | --- |
| Easy | pending live run |
| Medium | pending live run |
| Hard | pending live run |
| Mean | pending live run |

## Project Layout

- `models.py` defines typed actions, observations, rewards, and state
- `client.py` defines the typed OpenEnv client
- `server/argus_env_environment.py` contains the staged cases and grading logic
- `server/app.py` exposes the FastAPI server
- `inference.py` is the baseline runner
- `tests/test_argus_environment.py` contains deterministic regression tests

## Why ARGUS Is A Strong OpenEnv Submission

ARGUS is not a toy environment and not a thin reskin of an existing OpenEnv example. It models a workflow that frontier labs actually care about: whether evaluation evidence is complete, honestly reported, and uncontaminated.

That makes it useful for:

- post-training evaluation research
- agent reliability benchmarking
- release-readiness audits
- calibration-sensitive RL tasks where false positives should be penalized
