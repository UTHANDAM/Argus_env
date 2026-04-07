from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import ArgusAction, ArgusObservation, ArgusState
except ImportError:  # pragma: no cover - direct source-tree execution
    from models import ArgusAction, ArgusObservation, ArgusState


def _normalize_text(value: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _normalize_aliases(values: Sequence[str]) -> set[str]:
    return {_normalize_text(value) for value in values}


def _safe_float_range(values: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    if not values or len(values) < 2:
        return None

    try:
        low = float(values[0])
        high = float(values[1])
    except (TypeError, ValueError):
        return None

    if low > high:
        low, high = high, low
    return low, high


def _contains_alias(text: str, aliases: Sequence[str]) -> bool:
    normalized_text = _normalize_text(text)
    return any(_normalize_text(alias) in normalized_text for alias in aliases)


@dataclass(frozen=True)
class TaskCase:
    case_id: str
    task_name: str
    difficulty: str
    instruction: str
    context: str
    answer_schema: str
    truth: Dict[str, Any]


_TASK_ORDER = ("easy", "medium", "hard")

_TASK_ALIASES: Dict[str, Tuple[str, ...]] = {
    "easy": ("easy", "missing_baseline", "missing-baseline", "task1", "baseline"),
    "medium": ("medium", "cherry_picked_seed", "cherry-picked-seed", "task2", "seed"),
    "hard": ("hard", "contamination", "benchmark_contamination", "task3", "risk"),
}

_EVIDENCE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "temporal_overlap": (
        "temporal overlap",
        "release gap",
        "training cutoff",
        "model released after the benchmark",
        "benchmark released before training ended",
    ),
    "benchmark_in_corpus": (
        "benchmark in the corpus",
        "benchmark questions appear",
        "benchmark terms appear",
        "benchmark solution threads",
        "benchmark documents",
        "public educational material",
    ),
    "no_exclusion_filter": (
        "not explicitly excluded",
        "no exclusion filter",
        "not excluded",
        "benchmark documents were not excluded",
    ),
    "deduplication": (
        "deduplicated",
        "dedupe",
        "exact and fuzzy matching",
        "duplicate filtering",
    ),
    "held_out_filtering": (
        "held-out",
        "held out",
        "removed with exact and fuzzy matching",
        "filtering against the benchmark",
    ),
    "no_benchmark_scrape": (
        "no benchmark-specific scrape",
        "benchmark-specific scrape was not included",
        "no benchmark scrape",
    ),
}

_EASY_CASES: Tuple[TaskCase, ...] = (
    TaskCase(
        case_id="vision-beit-omission",
        task_name="easy",
        difficulty="easy",
        instruction="Identify the omitted baseline from the comparison table and return its exact method name.",
        context=(
            "Table 2: Fine-tuning results on ImageNet-1k.\n"
            "Compared methods: Ours, ViT-B/16, DeiT-S, Swin-T, ConvNeXt-T.\n"
            "Related work discusses BEiT-B/16 as a standard baseline, but it is absent from the main table."
        ),
        answer_schema='{"missing_baseline":"string"}',
        truth={
            "missing_baseline": "BEiT-B/16",
            "observed_baselines": ["ViT-B/16", "DeiT-S", "Swin-T", "ConvNeXt-T"],
            "family_aliases": ["BEiT", "BEiT-B", "BEiT-B/16"],
        },
    ),
    TaskCase(
        case_id="nlp-xlmr-omission",
        task_name="easy",
        difficulty="easy",
        instruction="Identify the omitted baseline from the comparison table and return its exact method name.",
        context=(
            "Table 4: Cross-lingual intent classification on MASSIVE.\n"
            "Compared methods: Ours, mBERT, XLM-R-base, MiniLM, DeBERTa-v3.\n"
            "The appendix cites XLM-R-large as the stronger published baseline, but it is not included in the headline table."
        ),
        answer_schema='{"missing_baseline":"string"}',
        truth={
            "missing_baseline": "XLM-R-large",
            "observed_baselines": ["mBERT", "XLM-R-base", "MiniLM", "DeBERTa-v3"],
            "family_aliases": ["XLM-R", "XLM-R-large"],
        },
    ),
)

_MEDIUM_CASES: Tuple[TaskCase, ...] = (
    TaskCase(
        case_id="adapter-cherry-pick",
        task_name="medium",
        difficulty="medium",
        instruction="Identify the cherry-picked variant and estimate the plausible true standard deviation range as [low, high].",
        context=(
            "Ablation over 5 seeds on CIFAR-100.\n"
            "Full fine-tune: 84.1 ± 1.12\n"
            "LoRA: 84.7 ± 1.08\n"
            "Adapter: 86.0 ± 0.05\n"
            "Prompt tuning: 83.3 ± 1.15\n"
            "The authors sampled 20 runs for Adapter and reported the five best seeds in the paper."
        ),
        answer_schema='{"cherry_picked_variant":"string","estimated_std_range":[low,high]}',
        truth={
            "cherry_picked_variant": "Adapter",
            "true_std_range": (0.88, 1.24),
            "candidate_variants": ["Full fine-tune", "LoRA", "Adapter", "Prompt tuning"],
        },
    ),
    TaskCase(
        case_id="prefix-tuning-cherry-pick",
        task_name="medium",
        difficulty="medium",
        instruction="Identify the cherry-picked variant and estimate the plausible true standard deviation range as [low, high].",
        context=(
            "Ablation on a low-resource speech translation task.\n"
            "Baseline: 67.2 ± 0.91\n"
            "Retrieval-augmented: 68.5 ± 0.88\n"
            "Prefix tuning: 70.1 ± 0.04\n"
            "Distillation: 69.3 ± 0.95\n"
            "Prefix tuning was run 18 times and the paper kept only the five best checkpoints."
        ),
        answer_schema='{"cherry_picked_variant":"string","estimated_std_range":[low,high]}',
        truth={
            "cherry_picked_variant": "Prefix tuning",
            "true_std_range": (0.82, 1.09),
            "candidate_variants": ["Baseline", "Retrieval-augmented", "Prefix tuning", "Distillation"],
        },
    ),
)

_HARD_CASES: Tuple[TaskCase, ...] = (
    TaskCase(
        case_id="mmlu-contamination",
        task_name="hard",
        difficulty="hard",
        instruction="Estimate the benchmark contamination risk and list the strongest evidence signals you observed.",
        context=(
            "Model: ARGUS-7B (released 2024-03-01)\n"
            "Training data cutoff: 2023-09-15\n"
            "Sources: CommonCrawl, GitHub discussions, course transcripts, and a filtered dump of benchmark solution threads.\n"
            "Benchmark: MMLU-Pro (released 2024-02-10)\n"
            "The authors say benchmark documents were not explicitly excluded because they were treated as public educational material."
        ),
        answer_schema='{"contamination_risk":0.0,"evidence":["signal"]}',
        truth={
            "risk": 0.92,
            "evidence": ["temporal_overlap", "benchmark_in_corpus", "no_exclusion_filter"],
            "clean": False,
        },
    ),
    TaskCase(
        case_id="gpqa-clean",
        task_name="hard",
        difficulty="hard",
        instruction="Estimate the benchmark contamination risk and list the strongest evidence signals you observed.",
        context=(
            "Model: ARGUS-7B (released 2024-06-10)\n"
            "Training data cutoff: 2024-01-31\n"
            "Sources: CommonCrawl, technical blogs, arXiv abstracts, and code repositories.\n"
            "Benchmark: GPQA-Diamond (released 2024-03-11)\n"
            "The training corpus was deduplicated against the benchmark, held-out questions were removed with exact and fuzzy matching, and no benchmark-specific scrape was included."
        ),
        answer_schema='{"contamination_risk":0.0,"evidence":["signal"]}',
        truth={
            "risk": 0.08,
            "evidence": ["deduplication", "held_out_filtering", "no_benchmark_scrape"],
            "clean": True,
        },
    ),
)

_TASK_BANK: Dict[str, Tuple[TaskCase, ...]] = {
    "easy": _EASY_CASES,
    "medium": _MEDIUM_CASES,
    "hard": _HARD_CASES,
}


class ArgusEnvironment(Environment):
    """ARGUS - ML Evaluation Integrity Environment.

    The environment exposes real-world research integrity checks over synthetic
    but paper-like evaluation artifacts. Each reset chooses one of three task
    families (easy / medium / hard), and each family includes multiple
    deterministic scenario variants for reproducibility.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._episode_index = 0
        self._task_cursor = 0
        self._current_case: Optional[TaskCase] = None
        self._episode_done = False
        self._state = ArgusState(episode_id=str(uuid4()), step_count=0)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="argus_env",
            description="ML evaluation integrity environment for detecting missing baselines, cherry-picked ablations, and benchmark contamination.",
            version="1.0.0",
        )

    def _resolve_task_key(self, task_hint: Optional[str]) -> str:
        if not task_hint:
            return _TASK_ORDER[self._task_cursor % len(_TASK_ORDER)]

        normalized_hint = _normalize_text(task_hint)
        for task_key, aliases in _TASK_ALIASES.items():
            if normalized_hint == _normalize_text(task_key):
                return task_key
            if normalized_hint in _normalize_aliases(aliases):
                return task_key

        raise ValueError(f"Unknown ARGUS task '{task_hint}'. Expected easy, medium, or hard.")

    def _select_case(self, task_key: str, seed: Optional[int], episode_index: int) -> TaskCase:
        cases = _TASK_BANK[task_key]
        if seed is None:
            seed_value = episode_index * 97 + self._task_cursor * 13
        else:
            seed_value = int(seed)
        case_index = abs(seed_value) % len(cases)
        return cases[case_index]

    def _build_observation(
        self,
        case: TaskCase,
        reward: float,
        done: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArgusObservation:
        observation_metadata = {
            "task_name": case.task_name,
            "case_id": case.case_id,
            "answer_schema": case.answer_schema,
            "episode_index": self._state.episode_index,
        }
        if metadata:
            observation_metadata.update(metadata)

        return ArgusObservation(
            task_name=case.task_name,
            task_instruction=case.instruction,
            context=case.context,
            task_difficulty=case.difficulty,
            case_id=case.case_id,
            done=done,
            reward=reward,
            metadata=observation_metadata,
        )

    def _grade_easy(self, action: ArgusAction, case: TaskCase) -> Tuple[float, Dict[str, Any]]:
        submitted = _normalize_text(action.missing_baseline)
        target = _normalize_text(case.truth["missing_baseline"])
        observed = _normalize_aliases(case.truth["observed_baselines"])
        family_aliases = _normalize_aliases(case.truth.get("family_aliases", []))

        if not submitted:
            return 0.0, {"match": "empty"}

        if submitted == target:
            return 1.0, {"match": "exact"}

        if submitted in observed:
            return 0.25, {"match": "observed_baseline"}

        if submitted in family_aliases or any(alias in submitted or submitted in alias for alias in family_aliases):
            return 0.55, {"match": "family_alias"}

        similarity = SequenceMatcher(None, submitted, target).ratio()
        if similarity >= 0.8:
            return 0.45, {"match": "fuzzy"}

        return 0.0, {"match": "wrong"}

    def _grade_medium(self, action: ArgusAction, case: TaskCase) -> Tuple[float, Dict[str, Any]]:
        submitted_variant = _normalize_text(action.cherry_picked_variant)
        target_variant = _normalize_text(case.truth["cherry_picked_variant"])
        candidate_variants = _normalize_aliases(case.truth["candidate_variants"])

        variant_score = 0.0
        variant_match = "missing"
        if submitted_variant:
            if submitted_variant == target_variant:
                variant_score = 0.6
                variant_match = "exact"
            elif submitted_variant in candidate_variants:
                variant_score = 0.25
                variant_match = "candidate"
            elif target_variant in submitted_variant or submitted_variant in target_variant:
                variant_score = 0.4
                variant_match = "family"
            else:
                similarity = SequenceMatcher(None, submitted_variant, target_variant).ratio()
                if similarity >= 0.75:
                    variant_score = 0.2
                    variant_match = "fuzzy"

        range_score = 0.0
        range_match = "missing"
        std_range = _safe_float_range(action.estimated_std_range)
        if std_range is not None:
            low, high = std_range
            true_low, true_high = case.truth["true_std_range"]

            if low <= true_low and high >= true_high:
                range_score = 0.4
                range_match = "contains_true_range"
            elif high >= true_low and low <= true_high:
                range_score = 0.25
                range_match = "overlap"
            else:
                submitted_midpoint = (low + high) / 2.0
                true_midpoint = (true_low + true_high) / 2.0
                distance = abs(submitted_midpoint - true_midpoint)
                tolerance = max(true_high - true_low, 0.1)
                range_score = max(0.0, 0.15 - distance / (tolerance * 4.0))
                range_match = "near_miss" if range_score > 0.0 else "wrong"

        total = min(1.0, variant_score + range_score)
        breakdown = {
            "variant_score": round(variant_score, 3),
            "variant_match": variant_match,
            "range_score": round(range_score, 3),
            "range_match": range_match,
        }
        return total, breakdown

    def _grade_hard(self, action: ArgusAction, case: TaskCase) -> Tuple[float, Dict[str, Any]]:
        truth_risk = float(case.truth["risk"])
        submitted_risk = action.contamination_risk
        risk_score = 0.0
        risk_match = "missing"

        if submitted_risk is not None:
            distance = abs(float(submitted_risk) - truth_risk)
            if distance <= 0.15:
                risk_score = 0.5
                risk_match = "close"
            elif distance <= 0.25:
                risk_score = 0.35
                risk_match = "moderate"
            elif distance <= 0.35:
                risk_score = 0.2
                risk_match = "coarse"
            elif distance <= 0.5:
                risk_score = 0.05
                risk_match = "weak"

        evidence_score = 0.0
        evidence_matches: List[str] = []
        submitted_evidence = action.evidence or []
        truth_evidence = case.truth["evidence"]
        for canonical_signal in truth_evidence:
            aliases = (canonical_signal,) + _EVIDENCE_ALIASES.get(canonical_signal, ())
            matched = any(_contains_alias(submitted_item, aliases) for submitted_item in submitted_evidence)
            if matched:
                evidence_matches.append(canonical_signal)

        if truth_evidence:
            evidence_score = 0.5 * (len(evidence_matches) / len(truth_evidence))

        penalty = 0.0
        if not case.truth.get("clean", False) and submitted_risk is not None and float(submitted_risk) < 0.2:
            penalty += 0.05
        if case.truth.get("clean", False) and submitted_risk is not None and float(submitted_risk) > 0.5:
            penalty += 0.1

        total = max(0.0, min(1.0, risk_score + evidence_score - penalty))
        breakdown = {
            "risk_score": round(risk_score, 3),
            "risk_match": risk_match,
            "evidence_score": round(evidence_score, 3),
            "evidence_matches": evidence_matches,
            "penalty": round(penalty, 3),
        }
        return total, breakdown

    def _grade_action(self, case: TaskCase, action: ArgusAction) -> Tuple[float, Dict[str, Any]]:
        if case.task_name == "easy":
            return self._grade_easy(action, case)
        if case.task_name == "medium":
            return self._grade_medium(action, case)
        return self._grade_hard(action, case)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> ArgusObservation:
        task_hint = kwargs.get("task") or kwargs.get("task_name") or kwargs.get("difficulty")
        task_key = self._resolve_task_key(task_hint)
        episode_index = self._episode_index
        case = self._select_case(task_key, seed, episode_index)

        self._current_case = case
        self._episode_done = False
        self._task_cursor += 1
        self._state = ArgusState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=case.task_name,
            task_difficulty=case.difficulty,
            case_id=case.case_id,
            episode_index=episode_index,
            task_cursor=self._task_cursor,
            last_reward=0.0,
        )
        self._episode_index += 1

        return self._build_observation(
            case,
            reward=0.0,
            done=False,
            metadata={
                "phase": "reset",
                "episode_index": episode_index,
                "task_cursor": self._task_cursor,
            },
        )

    def step(self, action: ArgusAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ArgusObservation:  # type: ignore[override]
        if self._current_case is None:
            raise RuntimeError("ARGUS environment must be reset before step() is called.")

        if self._episode_done:
            return self._build_observation(
                self._current_case,
                reward=0.0,
                done=True,
                metadata={"phase": "complete", "warning": "step_called_after_episode_completed"},
            )

        self._state.step_count += 1
        reward, breakdown = self._grade_action(self._current_case, action)
        reward = max(0.0, min(1.0, float(reward)))
        self._state.last_reward = reward
        self._episode_done = True

        return self._build_observation(
            self._current_case,
            reward=reward,
            done=True,
            metadata={
                "phase": "step",
                "grade_breakdown": breakdown,
                "step_count": self._state.step_count,
            },
        )

    @property
    def state(self) -> ArgusState:
        return self._state
