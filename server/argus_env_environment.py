import random
from uuid import uuid4

try:
    from ..models import ArgusAction, ArgusObservation, ArgusState
except ImportError:
    from models import ArgusAction, ArgusObservation, ArgusState

from openenv.core.env_server.interfaces import Environment

MISSING_BASELINE_DATASET = [
    {
        "prompt": (
            "A paper on text classification compares these methods:\n"
            "- ProposedMethod: MNLI=89.2, SST2=94.1, QNLI=91.3\n"
            "- BERT-base: MNLI=84.6, SST2=92.7, QNLI=87.4\n"
            "- RoBERTa-base: MNLI=86.4, SST2=93.1, QNLI=88.9\n"
            "- XLNet-base: MNLI=85.1, SST2=92.3, QNLI=87.1\n\n"
            "Known strong baselines in this area include: BERT-base, RoBERTa-base, "
            "XLNet-base, DeBERTa-base, ALBERT-large.\n\n"
            "Which well-known baseline is missing from this comparison?"
        ),
        "answer": "deberta-base",
        "alternatives": ["deberta", "deberta_base", "deberta-v3-base"]
    },
    {
        "prompt": (
            "A paper on question answering compares:\n"
            "- ProposedMethod: SQuAD=93.1, TriviaQA=71.2, NQ=52.3\n"
            "- BERT-large: SQuAD=90.9, TriviaQA=60.1, NQ=41.2\n"
            "- SpanBERT: SQuAD=91.8, TriviaQA=64.3, NQ=43.1\n\n"
            "Known strong baselines: BERT-large, SpanBERT, RoBERTa-large, ELECTRA-large, T5-base.\n\n"
            "Which well-known baseline is missing?"
        ),
        "answer": "roberta-large",
        "alternatives": ["roberta_large", "roberta large"]
    },
    {
        "prompt": (
            "A summarization paper compares:\n"
            "- ProposedMethod: ROUGE-1=45.1, ROUGE-2=21.3, ROUGE-L=41.2\n"
            "- BART-large: ROUGE-1=44.2, ROUGE-2=21.0, ROUGE-L=40.9\n"
            "- PEGASUS: ROUGE-1=44.6, ROUGE-2=21.2, ROUGE-L=41.1\n\n"
            "Known baselines: BART-large, PEGASUS, T5-large, ProphetNet, UniLM.\n\n"
            "Which well-known baseline is missing?"
        ),
        "answer": "t5-large",
        "alternatives": ["t5_large", "t5 large"]
    },
]

CHERRY_PICK_DATASET = [
    {
        "prompt": (
            "An ablation study reports results over 5 seeds:\n"
            "- FullModel:     [84.2, 83.9, 84.5, 84.1, 84.3]  mean=84.2  std=0.21\n"
            "- NoAttention:   [81.1, 80.8, 81.3, 81.0, 81.2]  mean=81.1  std=0.18\n"
            "- NoPretraining: [92.1, 91.8, 92.3, 91.9, 92.0]  mean=92.0  std=0.19\n"
            "- NoCLS:         [79.3, 78.9, 79.5, 79.1, 79.4]  mean=79.2  std=0.22\n\n"
            "One variant's results are suspiciously high and have unusually low variance "
            "suggesting cherry-picking from many runs.\n\n"
            "Which variant was cherry-picked? "
            "Also estimate the true std range: [low, high]."
        ),
        "answer": "nopretraining",
        "alternatives": ["no_pretraining", "no pretraining", "NoPretraining"],
        "true_std_low": 1.0,
        "true_std_high": 3.5
    },
    {
        "prompt": (
            "Ablation results over 5 seeds:\n"
            "- Baseline:    [71.2, 70.8, 71.5, 71.0, 71.3]  mean=71.2  std=0.26\n"
            "- +Augment:    [73.1, 72.9, 73.3, 73.0, 73.2]  mean=73.1  std=0.15\n"
            "- +Schedule:   [88.9, 88.7, 89.1, 88.8, 88.9]  mean=88.9  std=0.14\n"
            "- +Both:       [75.2, 74.9, 75.4, 75.1, 75.3]  mean=75.2  std=0.19\n\n"
            "One variant looks cherry-picked. Which one? Estimate true std range."
        ),
        "answer": "+schedule",
        "alternatives": ["schedule", "Schedule", "+Schedule"],
        "true_std_low": 1.5,
        "true_std_high": 4.0
    },
]

CONTAMINATION_DATASET = [
    {
        "prompt": (
            "Evaluate benchmark contamination risk:\n\n"
            "Model: LLM-X released March 2024\n"
            "Training data: CommonCrawl dump (Jan 2024), Books3, GitHub-2023\n"
            "Benchmark: MMLU (released 2021, last updated Dec 2023)\n"
            "Claim: Model scores 89.3 on MMLU 5-shot\n\n"
            "Known signals: CommonCrawl Jan 2024 is known to contain MMLU questions "
            "scraped from academic forums. The benchmark was updated Dec 2023, "
            "one month before the training cutoff.\n\n"
            "Rate contamination risk 0.0-1.0 and list evidence signals."
        ),
        "true_risk": 0.85,
        "evidence_keywords": [
            "commoncrawl", "jan 2024", "mmlu questions", "academic forums",
            "dec 2023", "one month", "training cutoff"
        ]
    },
    {
        "prompt": (
            "Evaluate benchmark contamination risk:\n\n"
            "Model: OpenLLM-7B released Jan 2023\n"
            "Training data: The Pile (2021 snapshot), Wikipedia-2020\n"
            "Benchmark: HumanEval (released July 2021)\n"
            "Claim: Model scores 67.1 on HumanEval pass@1\n\n"
            "Training data predates the benchmark. No known overlap. "
            "Score is within expected range for a 7B model.\n\n"
            "Rate contamination risk 0.0-1.0 and list evidence signals."
        ),
        "true_risk": 0.1,
        "evidence_keywords": [
            "predates", "no overlap", "2021 snapshot", "within expected range"
        ]
    },
]


class ArgusEnvEnvironment(Environment):

    TASKS = ["missing_baseline", "cherry_pick", "contamination"]

    def __init__(self):
        self._task_index = 0
        self._current_sample = None
        self._state = ArgusState(episode_id=str(uuid4()))

    def reset(self) -> ArgusObservation:
        self._task_index = 0
        self._state = ArgusState(
            episode_id=str(uuid4()),
            current_task=self.TASKS[0],
            step_count=0,
            total_reward=0.0
        )
        return self._load_task()

    def _load_task(self) -> ArgusObservation:
        task = self.TASKS[self._task_index]
        self._state.current_task = task

        if task == "missing_baseline":
            self._current_sample = random.choice(MISSING_BASELINE_DATASET)
        elif task == "cherry_pick":
            self._current_sample = random.choice(CHERRY_PICK_DATASET)
        else:
            self._current_sample = random.choice(CONTAMINATION_DATASET)

        return ArgusObservation(
            task_name=task,
            prompt=self._current_sample["prompt"],
            done=False,
            reward=0.0
        )

    def step(self, action: ArgusAction) -> ArgusObservation:
        self._state.step_count += 1
        task = self._state.current_task
        reward = 0.0
        feedback = ""

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
            next_obs = self._load_task()
            next_obs.reward = reward
            next_obs.feedback = feedback
            return next_obs
        else:
            return ArgusObservation(
                task_name="complete",
                prompt="",
                feedback=f"All tasks complete. Total reward: {self._state.total_reward:.2f}",
                done=True,
                reward=reward
            )

    def _grade_missing_baseline(self, action: ArgusAction) -> tuple:
        if not action.missing_baseline:
            return 0.0, "No answer provided."
        ans = action.missing_baseline.lower().strip()
        correct = self._current_sample["answer"]
        alts = [a.lower() for a in self._current_sample["alternatives"]]
        if ans == correct or ans in alts:
            return 1.0, f"Correct. The missing baseline was {correct}."
        return 0.0, f"Incorrect. The missing baseline was {correct}."

    def _grade_cherry_pick(self, action: ArgusAction) -> tuple:
        score = 0.0
        feedback_parts = []
        ans = (action.cherry_picked_variant or "").lower().strip()
        correct = self._current_sample["answer"].lower()
        alts = [a.lower() for a in self._current_sample["alternatives"]]

        if ans == correct or ans in alts:
            score += 0.6
            feedback_parts.append("Variant correct (+0.6).")
        else:
            feedback_parts.append(f"Variant wrong. Answer: {correct}.")

        lo = action.estimated_std_low
        hi = action.estimated_std_high
        true_lo = self._current_sample["true_std_low"]
        true_hi = self._current_sample["true_std_high"]

        if lo is not None and hi is not None:
            if lo <= true_hi and hi >= true_lo:
                score += 0.4
                feedback_parts.append("Std range correct (+0.4).")
            else:
                feedback_parts.append(f"Std range wrong. Expected [{true_lo}, {true_hi}].")

        return round(min(score, 1.0), 2), " ".join(feedback_parts)

    def _grade_contamination(self, action: ArgusAction) -> tuple:
        score = 0.0
        feedback_parts = []
        true_risk = self._current_sample["true_risk"]
        keywords = self._current_sample["evidence_keywords"]

        if action.contamination_risk is not None:
            diff = abs(action.contamination_risk - true_risk)
            if diff <= 0.15:
                score += 0.5
                feedback_parts.append("Risk score accurate (+0.5).")
            elif diff <= 0.30:
                score += 0.25
                feedback_parts.append("Risk score partially correct (+0.25).")
            else:
                feedback_parts.append(f"Risk score wrong. True risk: {true_risk}.")

        if action.evidence:
            ev_text = " ".join(action.evidence).lower()
            matched = sum(1 for k in keywords if k.lower() in ev_text)
            ev_score = round((matched / len(keywords)) * 0.5, 2)
            score += ev_score
            feedback_parts.append(f"Evidence: {matched}/{len(keywords)} signals found (+{ev_score}).")

        return round(min(score, 1.0), 2), " ".join(feedback_parts)

    @property
    def state(self) -> ArgusState:
        return self._state