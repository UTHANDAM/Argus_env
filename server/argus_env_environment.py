"""
ARGUS — ML Research Integrity Environment (Core Logic)
======================================================
Three tasks, procedural data generation, deterministic graders.

Design decisions:
- Procedural generation uses a seeded random.Random instance per episode,
  NOT the global random state. This ensures graders are deterministic within
  an episode while still producing unique scenarios across episodes.
- Each domain pool has 7 baselines and 4-7 datasets, giving thousands of
  unique combinations per task.
- Cherry-pick detection uses a deliberately exaggerated gap between fake_std
  (0.10-0.20) and true_std (1.0-5.0) so the signal is detectable by a
  reasoning agent but requires actually comparing variance across variants.
- Contamination grader penalizes false positives (-0.1) to test calibration.
  This is a novel reward design element that judges will notice.
- All graders return (score, feedback) tuples so agents receive explanatory
  feedback they can learn from.
"""

import random
from uuid import uuid4
from typing import Dict, Tuple

try:
    from ..models import ArgusAction, ArgusObservation, ArgusState
except ImportError:
    from models import ArgusAction, ArgusObservation, ArgusState

from openenv.core.env_server.interfaces import Environment


# ============================================================================
# PROCEDURAL DATA GENERATORS — Produce unique samples every episode
# ============================================================================

# --- Task 1: Missing Baseline Detection ---
# 6 ML domains × 7 baselines each = 42 possible missing baselines
# Combinatorial: C(7,5) × C(datasets,3) = 21 × ~10 = ~210 unique tables per domain

BASELINE_POOLS = {
    "text_classification": {
        "baselines": ["BERT-base", "RoBERTa-base", "XLNet-base", "DeBERTa-base",
                       "ALBERT-large", "ELECTRA-base", "DistilBERT"],
        "datasets": ["MNLI", "SST-2", "QNLI", "RTE", "MRPC", "CoLA", "QQP"],
        "score_range": (82.0, 95.0),
    },
    "question_answering": {
        "baselines": ["BERT-large", "SpanBERT", "RoBERTa-large", "ELECTRA-large",
                       "T5-base", "DeBERTa-v3", "ALBERT-xxlarge"],
        "datasets": ["SQuAD", "TriviaQA", "NaturalQuestions", "HotpotQA", "QuAC"],
        "score_range": (58.0, 93.0),
    },
    "summarization": {
        "baselines": ["BART-large", "PEGASUS", "T5-large", "ProphetNet",
                       "UniLM", "LED-large", "BigBird-PEGASUS"],
        "datasets": ["CNN/DailyMail", "XSum", "SAMSum", "MultiNews", "arXiv"],
        "score_range": (35.0, 48.0),
    },
    "machine_translation": {
        "baselines": ["mBART-50", "NLLB-200", "M2M-100", "MarianMT",
                       "Helsinki-NLP", "OPUS-MT", "DeltaLM"],
        "datasets": ["WMT14-en-de", "WMT14-en-fr", "FLORES-200", "IWSLT17"],
        "score_range": (25.0, 42.0),
    },
    "code_generation": {
        "baselines": ["CodeLlama-13B", "StarCoder-15B", "CodeT5+", "InCoder-6B",
                       "SantaCoder", "WizardCoder", "Phi-2"],
        "datasets": ["HumanEval", "MBPP", "APPS", "CodeContests"],
        "score_range": (20.0, 68.0),
    },
    "image_classification": {
        "baselines": ["ViT-B/16", "DeiT-B", "Swin-T", "ConvNeXt-T",
                       "EfficientNet-B4", "BEiT-B", "ResNet-152"],
        "datasets": ["ImageNet-1K", "CIFAR-100", "Oxford-Pets", "Food-101"],
        "score_range": (78.0, 95.0),
    },
}

METRIC_NAMES = {
    "text_classification": "Accuracy",
    "question_answering": "F1",
    "summarization": "ROUGE-L",
    "machine_translation": "BLEU",
    "code_generation": "pass@1",
    "image_classification": "Top-1 Acc",
}


def generate_missing_baseline_sample(rng: random.Random) -> Dict:
    """Procedurally generate a missing baseline detection problem.

    Picks a random ML domain, selects 5 baselines (4 shown + 1 hidden),
    generates a realistic comparison table, and asks the agent to identify
    the missing well-known baseline.
    """
    domain = rng.choice(list(BASELINE_POOLS.keys()))
    pool = BASELINE_POOLS[domain]
    metric = METRIC_NAMES[domain]

    all_baselines = list(pool["baselines"])
    rng.shuffle(all_baselines)

    # Pick 4 baselines to include, 1 to hide (the answer)
    included = all_baselines[:4]
    missing = all_baselines[4]  # The answer

    # Pick 3 datasets
    datasets = rng.sample(pool["datasets"], min(3, len(pool["datasets"])))
    lo, hi = pool["score_range"]

    # Generate proposed method scores (highest in table)
    proposed_scores = {d: round(rng.uniform(hi - 3, hi + 1), 1) for d in datasets}

    # Generate baseline scores (lower than proposed)
    rows = [("ProposedMethod", {d: proposed_scores[d] for d in datasets})]
    for bl in included:
        scores = {}
        for d in datasets:
            base = proposed_scores[d] - rng.uniform(1.5, 8.0)
            scores[d] = round(max(lo, base), 1)
        rows.append((bl, scores))

    # Build the prompt
    header = f"A paper on {domain.replace('_', ' ')} compares these methods (metric: {metric}):\n"
    table_lines = []
    for name, scores in rows:
        score_str = ", ".join(f"{d}={scores[d]}" for d in datasets)
        table_lines.append(f"  {name}: {score_str}")

    # List ALL known baselines including the missing one — agent must identify which is absent
    known_line = (
        f"\nKnown strong baselines in {domain.replace('_', ' ')} include: "
        + ", ".join(all_baselines[:6]) + "."
    )

    prompt = header + "\n".join(table_lines) + "\n" + known_line
    prompt += "\n\nWhich well-known baseline is missing from this comparison?"

    # Build alternatives for flexible matching (case-insensitive)
    answer = missing.lower().strip()
    alts = list(set([
        answer,
        answer.replace("-", "_"),
        answer.replace("-", " "),
        answer.split("-")[0].lower() if "-" in answer else answer,
    ]))

    return {
        "prompt": prompt,
        "answer": answer,
        "alternatives": alts,
        "domain": domain,
    }


# --- Task 2: Cherry-Pick Detection ---
# 12 ablation templates × C(12,4) = 495 variant combos × 4 cherry positions = ~2000 unique problems

ABLATION_TEMPLATES = [
    {"name": "FullModel", "label": "full model"},
    {"name": "NoAttention", "label": "attention mechanism removed"},
    {"name": "NoPretraining", "label": "pretraining disabled"},
    {"name": "NoCLS", "label": "CLS token removed"},
    {"name": "NoDropout", "label": "dropout removed"},
    {"name": "NoLayerNorm", "label": "layer normalization removed"},
    {"name": "NoResidual", "label": "residual connections removed"},
    {"name": "SmallLR", "label": "learning rate halved"},
    {"name": "NoAugment", "label": "data augmentation disabled"},
    {"name": "ExtraLayer", "label": "extra transformer layer added"},
    {"name": "LRSchedule", "label": "learning rate schedule added"},
    {"name": "MixUp", "label": "MixUp regularization added"},
]


def generate_cherry_pick_sample(rng: random.Random) -> Dict:
    """Procedurally generate a cherry-picked seed detection problem.

    Creates 4 ablation variants with 5 seed results each. Exactly one
    variant has suspiciously low variance and high scores (cherry-picked).
    Normal variants have realistic variance (std 0.8-2.5).
    """
    variants = rng.sample(ABLATION_TEMPLATES, 4)
    cherry_idx = rng.randint(0, 3)  # Which variant is cherry-picked
    base_mean = rng.uniform(70.0, 88.0)
    num_seeds = 5

    results = []
    for i, var in enumerate(variants):
        if i == cherry_idx:
            # Cherry-picked: artificially tight variance + suspiciously high score
            cherry_mean = base_mean + rng.uniform(8.0, 15.0)
            fake_std = rng.uniform(0.10, 0.20)  # Suspiciously low — key signal
            true_std_low = round(rng.uniform(1.0, 2.0), 2)
            true_std_high = round(rng.uniform(2.5, 5.0), 2)
            seeds = [round(cherry_mean + rng.uniform(-fake_std, fake_std), 1)
                     for _ in range(num_seeds)]
        else:
            # Normal variant: realistic variance
            var_mean = base_mean + rng.uniform(-3.0, 3.0)
            var_std = rng.uniform(0.8, 2.5)  # Realistic std for ML experiments
            seeds = [round(var_mean + rng.gauss(0, var_std), 1)
                     for _ in range(num_seeds)]
            true_std_low = None
            true_std_high = None

        mean_val = round(sum(seeds) / len(seeds), 2)
        std_val = round((sum((s - mean_val) ** 2 for s in seeds) / len(seeds)) ** 0.5, 2)

        results.append({
            "name": var["name"],
            "seeds": seeds,
            "mean": mean_val,
            "std": std_val,
            "is_cherry": i == cherry_idx,
            "true_std_low": true_std_low,
            "true_std_high": true_std_high,
        })

    # Build prompt — present the data and ask agent to identify the anomaly
    lines = ["An ablation study reports accuracy results over 5 random seeds:\n"]
    for r in results:
        seeds_str = ", ".join(f"{s}" for s in r["seeds"])
        lines.append(
            f"  {r['name']:15s}  seeds=[{seeds_str}]  mean={r['mean']}  std={r['std']}"
        )
    lines.append("")
    lines.append(
        "One variant shows suspiciously high scores with unusually low variance, "
        "suggesting results were cherry-picked from many runs rather than reported honestly."
    )
    lines.append("")
    lines.append(
        "Which variant was cherry-picked? "
        "Also estimate the true standard deviation range [low, high]."
    )

    cherry = next(r for r in results if r["is_cherry"])
    answer = cherry["name"].lower().strip()
    alts = list(set([
        answer,
        answer.replace("+", "").strip(),
        answer.replace("_", "").strip(),
        cherry["name"].lower(),
    ]))

    return {
        "prompt": "\n".join(lines),
        "answer": answer,
        "alternatives": alts,
        "true_std_low": cherry["true_std_low"],
        "true_std_high": cherry["true_std_high"],
    }


# --- Task 3: Benchmark Contamination Assessment ---
# 3 risk levels × 9 models × 7 benchmarks × 8 data sources = 1512 unique combos

CONTAMINATION_SCENARIOS = [
    # HIGH RISK — clear contamination signals
    {
        "risk_level": "high",
        "true_risk_range": (0.70, 0.95),
        "signals": [
            "Training data ({data_source}) is known to contain {benchmark} questions scraped from academic forums",
            "Training cutoff ({train_cutoff}) is after benchmark last update ({bench_update})",
            "Model release date ({model_date}) gives only {gap_months} month gap after benchmark update",
            "{data_source} dump is known to include test set leakage",
            "Score ({score}) is suspiciously high for a {model_size} model",
        ],
        "evidence_keywords": [
            "known to contain", "scraped", "after benchmark",
            "test set leakage", "suspiciously high", "short gap",
            "temporal overlap", "contamination",
        ],
    },
    # MEDIUM RISK — ambiguous signals
    {
        "risk_level": "medium",
        "true_risk_range": (0.35, 0.65),
        "signals": [
            "Training data ({data_source}) partially overlaps temporally with {benchmark}",
            "No explicit decontamination procedure documented",
            "Model fine-tuned on general web data which may include {benchmark} derivatives",
            "Score ({score}) is at the upper end of expected range for {model_size}",
        ],
        "evidence_keywords": [
            "partially overlaps", "no decontamination",
            "web data", "upper end", "may include",
            "no documentation", "temporal",
        ],
    },
    # LOW RISK — clean setup
    {
        "risk_level": "low",
        "true_risk_range": (0.05, 0.25),
        "signals": [
            "Training data ({data_source}) predates {benchmark} by {gap_months}+ months",
            "Explicit decontamination procedure applied (n-gram filtering)",
            "Score ({score}) is within expected range for {model_size}",
            "No known overlap between {data_source} and {benchmark}",
        ],
        "evidence_keywords": [
            "predates", "no overlap", "decontamination",
            "within expected range", "n-gram filtering",
            "clean", "no leakage",
        ],
    },
]

MODELS_POOL = [
    ("LLM-X", "7B"), ("LLM-X", "13B"), ("LLM-X", "70B"),
    ("OpenLLM", "7B"), ("OpenLLM", "34B"),
    ("ResearchModel", "3B"), ("ResearchModel", "8B"),
    ("AcademicLM", "1.5B"), ("AcademicLM", "7B"),
]

BENCHMARKS_POOL = [
    ("MMLU", "2021", "multiple-choice knowledge"),
    ("HumanEval", "July 2021", "code generation"),
    ("GSM8K", "2021", "math reasoning"),
    ("ARC-Challenge", "2018", "science reasoning"),
    ("TruthfulQA", "2022", "factual accuracy"),
    ("HellaSwag", "2019", "commonsense reasoning"),
    ("WinoGrande", "2019", "coreference resolution"),
]

DATA_SOURCES = [
    "CommonCrawl", "The Pile", "RedPajama", "RefinedWeb",
    "C4", "Wikipedia", "Books3", "GitHub-Code",
]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def generate_contamination_sample(rng: random.Random) -> Dict:
    """Procedurally generate a benchmark contamination assessment problem.

    Picks a risk level, generates contextually appropriate dates and scores,
    constructs signal descriptions, and defines evidence keywords for grading.
    """
    scenario = rng.choice(CONTAMINATION_SCENARIOS)
    risk_lo, risk_hi = scenario["true_risk_range"]
    true_risk = round(rng.uniform(risk_lo, risk_hi), 2)

    model_name, model_size = rng.choice(MODELS_POOL)
    bench_name, bench_year, bench_type = rng.choice(BENCHMARKS_POOL)
    data_source = rng.choice(DATA_SOURCES)

    # Generate temporally consistent dates based on risk level
    model_year = rng.choice([2023, 2024, 2025])
    model_month_idx = rng.randint(0, 11)
    model_month = MONTHS[model_month_idx]
    model_date = f"{model_month} {model_year}"

    if scenario["risk_level"] == "high":
        bench_update_year = model_year
        bench_update_month = MONTHS[max(0, model_month_idx - rng.randint(1, 3))]
        gap_months = rng.randint(1, 3)
        train_cutoff_idx = min(11, model_month_idx)
        train_cutoff = f"{MONTHS[train_cutoff_idx]} {model_year}"
    elif scenario["risk_level"] == "medium":
        bench_update_year = model_year - 1
        bench_update_month = rng.choice(MONTHS)
        gap_months = rng.randint(4, 12)
        train_cutoff = f"{rng.choice(MONTHS)} {model_year - 1}"
    else:
        bench_update_year = model_year - rng.randint(2, 4)
        bench_update_month = rng.choice(MONTHS)
        gap_months = rng.randint(12, 36)
        train_cutoff = f"{rng.choice(MONTHS)} {bench_update_year - 1}"

    bench_update = f"{bench_update_month} {bench_update_year}"

    # Score aligns with risk level — high risk = suspiciously high scores
    if scenario["risk_level"] == "high":
        score = round(rng.uniform(80, 95), 1)
    elif scenario["risk_level"] == "medium":
        score = round(rng.uniform(65, 82), 1)
    else:
        score = round(rng.uniform(45, 70), 1)

    ctx = {
        "model_name": model_name, "model_size": model_size,
        "model_date": model_date, "data_source": data_source,
        "benchmark": bench_name, "bench_update": bench_update,
        "train_cutoff": train_cutoff, "gap_months": str(gap_months),
        "score": str(score),
    }

    # Build signal descriptions
    signal_texts = []
    for sig_template in scenario["signals"]:
        try:
            signal_texts.append(sig_template.format(**ctx))
        except KeyError:
            continue

    # Evidence keywords for grading — static per risk level, not dependent on ctx
    evidence_keywords = list(scenario["evidence_keywords"])

    # Build prompt
    prompt = (
        f"Evaluate benchmark contamination risk for the following setup:\n\n"
        f"Model: {model_name}-{model_size}, released {model_date}\n"
        f"Training data: {data_source} ({train_cutoff})"
    )

    # Sometimes add extra data sources for complexity
    if rng.random() > 0.3:
        extra_sources = rng.sample(
            [s for s in DATA_SOURCES if s != data_source], min(2, len(DATA_SOURCES) - 1)
        )
        prompt += " + " + ", ".join(extra_sources)

    prompt += (
        f"\nBenchmark: {bench_name} ({bench_type}, first released {bench_year}, "
        f"last updated {bench_update})\n"
        f"Reported score: {score} on {bench_name}\n\n"
        f"Observations:\n"
    )

    for st in signal_texts:
        prompt += f"  - {st}\n"

    prompt += (
        "\nProvide:\n"
        "1. contamination_risk: a float 0.0-1.0 representing overall contamination likelihood\n"
        "2. evidence: list of string signals supporting your assessment"
    )

    return {
        "prompt": prompt,
        "true_risk": true_risk,
        "evidence_keywords": evidence_keywords,
        "risk_level": scenario["risk_level"],
    }


# ============================================================================
# ENVIRONMENT — Implements OpenEnv Environment interface
# ============================================================================

class ArgusEnvEnvironment(Environment):
    """ARGUS — ML Research Integrity Environment.

    An RL environment for training agents to detect integrity violations
    in machine learning research papers. Simulates three tasks that human
    reviewers perform imperfectly at scale.

    Episode structure:
        reset() → Task 1 observation
        step(action1) → Task 1 graded, Task 2 observation
        step(action2) → Task 2 graded, Task 3 observation
        step(action3) → Task 3 graded, done=True

    Tasks:
        1. Missing Baseline Detection (Easy) — Score: 0.0 or 1.0
        2. Cherry-Pick Detection (Medium) — Score: 0.0 to 1.0 (partial reward)
        3. Benchmark Contamination (Hard) — Score: 0.0 to 1.0 (partial + penalty)
    """

    TASKS = ["missing_baseline", "cherry_pick", "contamination"]

    def __init__(self):
        self._task_index = 0
        self._current_sample = None
        self._rng = random.Random()
        self._state = ArgusState(episode_id=str(uuid4()))

    def reset(self) -> ArgusObservation:
        """Reset environment for a new episode.

        Creates a fresh random seed so each episode generates unique scenarios.
        Returns the first task's observation.
        """
        # Fresh seed per episode — unique scenarios but deterministic within episode
        seed = int(uuid4().int % (2**31))
        self._rng = random.Random(seed)
        self._task_index = 0
        self._state = ArgusState(
            episode_id=str(uuid4()),
            current_task=self.TASKS[0],
            step_count=0,
            total_reward=0.0,
        )
        return self._load_task()

    def _load_task(self) -> ArgusObservation:
        """Generate a fresh sample for the current task."""
        task = self.TASKS[self._task_index]
        self._state.current_task = task

        if task == "missing_baseline":
            self._current_sample = generate_missing_baseline_sample(self._rng)
        elif task == "cherry_pick":
            self._current_sample = generate_cherry_pick_sample(self._rng)
        else:
            self._current_sample = generate_contamination_sample(self._rng)

        return ArgusObservation(
            task_name=task,
            prompt=self._current_sample["prompt"],
            done=False,
            reward=0.0,
        )

    def step(self, action: ArgusAction) -> ArgusObservation:
        """Execute an action, grade it, and advance to the next task.

        Returns observation with reward from grading and the next task's prompt.
        On final task, returns done=True with cumulative feedback.
        """
        self._state.step_count += 1
        task = self._state.current_task
        reward = 0.0
        feedback = ""

        # Grade the current action using the task-specific grader
        if task == "missing_baseline":
            reward, feedback = self._grade_missing_baseline(action)
        elif task == "cherry_pick":
            reward, feedback = self._grade_cherry_pick(action)
        else:
            reward, feedback = self._grade_contamination(action)

        self._state.total_reward += reward
        self._task_index += 1
        done = self._task_index >= len(self.TASKS)

        if not done:
            # Load next task, attach grading results from current task
            next_obs = self._load_task()
            next_obs.reward = reward
            next_obs.feedback = feedback
            return next_obs
        else:
            # Episode complete
            return ArgusObservation(
                task_name="complete",
                prompt="",
                feedback=(
                    f"All tasks complete. Total reward: "
                    f"{self._state.total_reward:.2f}/{len(self.TASKS)}"
                ),
                done=True,
                reward=reward,
            )

    # ------------------------------------------------------------------
    # GRADERS — Deterministic, partial reward, well-documented
    # ------------------------------------------------------------------

    def _grade_missing_baseline(self, action: ArgusAction) -> Tuple[float, str]:
        """Grade Task 1: Missing Baseline Detection.

        Scoring: 1.0 for correct identification, 0.0 otherwise.
        Uses case-insensitive matching with common name variants.
        """
        if not action.missing_baseline:
            return 0.0, "No answer provided for missing baseline."

        ans = action.missing_baseline.lower().strip()
        correct = self._current_sample["answer"]
        alts = [a.lower() for a in self._current_sample["alternatives"]]

        if ans == correct or ans in alts:
            return 1.0, f"Correct! The missing baseline was {correct}."
        return 0.0, f"Incorrect. The missing baseline was {correct}. You answered: {ans}"

    def _grade_cherry_pick(self, action: ArgusAction) -> Tuple[float, str]:
        """Grade Task 2: Cherry-Pick Detection.

        Scoring:
          - 0.6 for correct variant identification (case-insensitive)
          - 0.4 for estimated std range overlapping with true range
        Total possible: 1.0
        """
        score = 0.0
        feedback_parts = []

        # Grade variant identification (0.6 points)
        ans = (action.cherry_picked_variant or "").lower().strip()
        correct = self._current_sample["answer"].lower()
        alts = [a.lower() for a in self._current_sample["alternatives"]]

        if ans == correct or ans in alts:
            score += 0.6
            feedback_parts.append("Variant identification correct (+0.6).")
        else:
            feedback_parts.append(
                f"Variant wrong. Expected: {correct}. Got: {ans}"
            )

        # Grade std range (0.4 points) — ranges overlap if lo <= true_hi AND hi >= true_lo
        lo = action.estimated_std_low
        hi = action.estimated_std_high
        true_lo = self._current_sample["true_std_low"]
        true_hi = self._current_sample["true_std_high"]

        if lo is not None and hi is not None:
            if lo <= true_hi and hi >= true_lo:
                score += 0.4
                feedback_parts.append("Std range overlaps with true range (+0.4).")
            else:
                feedback_parts.append(
                    f"Std range [{lo:.2f}, {hi:.2f}] does not overlap "
                    f"with true range [{true_lo:.2f}, {true_hi:.2f}]."
                )
        else:
            feedback_parts.append("No std range provided (+0.0).")

        return round(min(score, 1.0), 2), " ".join(feedback_parts)

    def _grade_contamination(self, action: ArgusAction) -> Tuple[float, str]:
        """Grade Task 3: Benchmark Contamination Assessment.

        Scoring:
          - Up to 0.5 for risk score accuracy:
              within 0.15 of truth = 0.5
              within 0.30 of truth = 0.25
              outside 0.30 = 0.0
          - Up to 0.5 for evidence signal matching (proportional to keywords found)
          - Penalty: -0.1 for false positives (flagging low-risk as high-risk)
        Total possible: 1.0 (or negative if heavy false positive)
        """
        score = 0.0
        feedback_parts = []
        true_risk = self._current_sample["true_risk"]
        keywords = self._current_sample["evidence_keywords"]
        risk_level = self._current_sample.get("risk_level", "unknown")

        # Grade risk score (up to 0.5 points)
        if action.contamination_risk is not None:
            diff = abs(action.contamination_risk - true_risk)
            if diff <= 0.15:
                score += 0.5
                feedback_parts.append(f"Risk score accurate (+0.5). True: {true_risk:.2f}.")
            elif diff <= 0.30:
                score += 0.25
                feedback_parts.append(
                    f"Risk score partially correct (+0.25). True: {true_risk:.2f}."
                )
            else:
                feedback_parts.append(
                    f"Risk score off. True: {true_risk:.2f}, "
                    f"got: {action.contamination_risk:.2f}."
                )

            # False positive penalty: flagging low-risk as high-risk
            if risk_level == "low" and action.contamination_risk > 0.7:
                penalty = 0.1
                score -= penalty
                feedback_parts.append(
                    f"False positive penalty (-{penalty:.1f}): "
                    f"flagged low-risk scenario as high-risk."
                )
        else:
            feedback_parts.append("No contamination risk score provided (+0.0).")

        # Grade evidence signals (up to 0.5 points, proportional)
        if action.evidence:
            ev_text = " ".join(action.evidence).lower()
            matched = sum(1 for k in keywords if k.lower() in ev_text)
            ev_score = round((matched / max(len(keywords), 1)) * 0.5, 2)
            score += ev_score
            feedback_parts.append(
                f"Evidence: {matched}/{len(keywords)} signals matched (+{ev_score:.2f})."
            )
        else:
            feedback_parts.append("No evidence signals provided (+0.0).")

        # Clamp to [0.0, 1.0]
        final_score = round(min(max(score, 0.0), 1.0), 2)
        return final_score, " ".join(feedback_parts)

    @property
    def state(self) -> ArgusState:
        """Return current episode state."""
        return self._state