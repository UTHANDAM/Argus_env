# ARGUS  ML Research Integrity Environment

An OpenEnv environment for training agents to detect integrity violations in machine learning research papers.

## Motivation

The ML reproducibility crisis is real. Reviewers miss cherry-picked results, missing baselines, and benchmark contamination daily. ARGUS provides a structured RL environment to train and evaluate agents on these tasks.

## Tasks

Task 1  Missing Baseline Detection (Easy): Given a results table, identify which well-known baseline is absent to make the proposed method appear stronger. Reward: 1.0 for correct identification, 0.0 otherwise.

Task 2  Cherry-Pick Detection (Medium): Given ablation results across 5 seeds, identify which variant was cherry-picked from many runs and estimate the true standard deviation range. Reward: 0.6 for correct variant, 0.4 for correct std range.

Task 3  Benchmark Contamination Assessment (Hard): Given training data sources, model release date, and benchmark details, assess contamination risk and list evidence signals. Reward: 0.5 for accurate risk score, 0.5 for evidence signals found.

## Action Space
```json
{
  "missing_baseline": "string",
  "cherry_picked_variant": "string",
  "estimated_std_low": "float",
  "estimated_std_high": "float",
  "contamination_risk": "float (0.0-1.0)",
  "evidence": ["string"]
}
```

## Observation Space
```json
{
  "task_name": "string",
  "prompt": "string",
  "feedback": "string",
  "done": "boolean",
  "reward": "float"
}
```

## Setup
```bash
pip install openenv-core
```

## Run Baseline
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Baseline Scores

Model: Qwen2.5-72B-Instruct
Score: 1.0/1.0 (3/3 tasks correct)

## Environment URL

https://huggingface.co/spaces/uthandam/argus-env