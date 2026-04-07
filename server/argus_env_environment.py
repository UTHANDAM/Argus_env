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


def _stage(name: str, kind: str, instruction: str, context: str, weight: float) -> "StageSpec":
    return StageSpec(name=name, kind=kind, instruction=instruction, context=context, weight=weight)


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: str
    instruction: str
    context: str
    weight: float


@dataclass(frozen=True)
class TaskCase:
    case_id: str
    task_name: str
    difficulty: str
    stages: Tuple[StageSpec, ...]
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
        "release-window overlap",
        "release overlap",
        "published before training ended",
        "benchmark predated the training cutoff",
        "cutoff collision",
    ),
    "benchmark_in_corpus": (
        "answer-key mirror",
        "solution-thread archive",
        "public instructional dump",
        "benchmark question mirror",
        "reference solutions in the corpus",
    ),
    "no_exclusion_filter": (
        "not filtered at crawl time",
        "left in during crawl",
        "not separated at dedupe",
        "not explicitly removed",
        "no exclusion filter",
    ),
    "deduplication": (
        "deduplicated",
        "exact match removal",
        "fuzzy match pruning",
        "duplicate filtering",
    ),
    "held_out_filtering": (
        "held-out items removed",
        "held out filtering",
        "validation split filtered",
        "held-out questions pruned",
    ),
    "no_benchmark_scrape": (
        "no benchmark-specific scrape",
        "benchmark-free crawl",
        "no eval-question scrape",
        "never scraped the benchmark itself",
    ),
    "twenty_run_audit": (
        "sampled 20 times",
        "20 trials",
        "twenty-run audit",
        "full twenty-run sweep",
    ),
    "five_best_checkpoints": (
        "five best checkpoints",
        "top five checkpoints",
        "five highest runs",
        "kept only the top five",
    ),
    "selection_bias": (
        "selection bias",
        "variance compression",
        "artificially small variance",
        "truncated by cherry-picking",
    ),
}

_EASY_CASES: Tuple[TaskCase, ...] = (
    TaskCase(
        case_id="vision-beit-omission",
        task_name="easy",
        difficulty="easy",
        stages=(
            _stage(
                name="family-clue",
                kind="family_hint",
                instruction="Identify the missing baseline family from the comparison table. The exact citation will be revealed later.",
                context=(
                    "Table 2: Fine-tuning results on ImageNet-1k.\n"
                    "Compared methods: Ours, ViT-B/16, DeiT-S, Swin-T, ConvNeXt-T.\n"
                    "Reviewer note: the omitted line belongs to the BEiT family used in earlier masked-image-modeling experiments, "
                    "but the appendix has not yet named the exact variant."
                ),
                weight=0.35,
            ),
            _stage(
                name="exact-citation",
                kind="exact_missing_baseline",
                instruction="Now return the exact omitted baseline method name.",
                context=(
                    "Appendix D: The omitted baseline in Table 2 is BEiT-B/16. This exact model should have been compared against Ours."
                ),
                weight=0.65,
            ),
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
        stages=(
            _stage(
                name="family-clue",
                kind="family_hint",
                instruction="Identify the missing baseline family from the comparison table. The exact citation will be revealed later.",
                context=(
                    "Table 4: Cross-lingual intent classification on MASSIVE.\n"
                    "Compared methods: Ours, mBERT, XLM-R-base, MiniLM, DeBERTa-v3.\n"
                    "Reviewer note: the appendix references the stronger XLM-R family, but the exact citation is deferred."
                ),
                weight=0.35,
            ),
            _stage(
                name="exact-citation",
                kind="exact_missing_baseline",
                instruction="Now return the exact omitted baseline method name.",
                context=(
                    "Appendix C: The omitted baseline in Table 4 is XLM-R-large. It is the stronger published baseline that was left out of the headline table."
                ),
                weight=0.65,
            ),
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
        stages=(
            _stage(
                name="variant-probe",
                kind="variant_probe",
                instruction="Identify the suspiciously cherry-picked variant from the first ablation table.",
                context=(
                    "Ablation over 5 reported runs on CIFAR-100.\n"
                    "Full fine-tune: 84.1 ± 1.12\n"
                    "LoRA: 84.7 ± 1.08\n"
                    "Adapter: 86.0 ± 0.05\n"
                    "Prompt tuning: 83.3 ± 1.15\n"
                    "Reviewer note: one variant was later audited over twenty trials, but only the strongest checkpoints made the paper."
                ),
                weight=0.30,
            ),
            _stage(
                name="range-probe",
                kind="range_probe",
                instruction="Estimate the plausible true standard-deviation range for the suspicious variant as [low, high].",
                context=(
                    "Audit log: the suspicious variant was sampled 20 times. The paper kept only the five best checkpoints, so the reported std is artificially small.\n"
                    "The rerun notes indicate that the full spread is close to the other methods after selection bias is removed."
                ),
                weight=0.35,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="List the strongest evidence signals that justify your diagnosis.",
                context=(
                    "Reproducibility memo: the full twenty-run distribution centers near std ≈ 1.04 after removing the top-five selection bias.\n"
                    "The audit emphasizes that the variance compression was caused by cherry-picking, not by genuine stability."
                ),
                weight=0.35,
            ),
        ),
        answer_schema='{"cherry_picked_variant":"string","estimated_std_range":[low,high]}',
        truth={
            "cherry_picked_variant": "Adapter",
            "true_std_range": (0.88, 1.24),
            "candidate_variants": ["Full fine-tune", "LoRA", "Adapter", "Prompt tuning"],
            "evidence": ["twenty_run_audit", "five_best_checkpoints", "selection_bias"],
        },
    ),
    TaskCase(
        case_id="prefix-tuning-cherry-pick",
        task_name="medium",
        difficulty="medium",
        stages=(
            _stage(
                name="variant-probe",
                kind="variant_probe",
                instruction="Identify the suspiciously cherry-picked variant from the first ablation table.",
                context=(
                    "Ablation on a low-resource speech translation task.\n"
                    "Baseline: 67.2 ± 0.91\n"
                    "Retrieval-augmented: 68.5 ± 0.88\n"
                    "Prefix tuning: 70.1 ± 0.04\n"
                    "Distillation: 69.3 ± 0.95\n"
                    "Reviewer note: the suspicious row was audited across twenty trials, but the paper only reported the five top checkpoints."
                ),
                weight=0.30,
            ),
            _stage(
                name="range-probe",
                kind="range_probe",
                instruction="Estimate the plausible true standard-deviation range for the suspicious variant as [low, high].",
                context=(
                    "Audit log: Prefix tuning was run 18 times and the paper kept only the five best checkpoints.\n"
                    "The held-out runs show that the genuine spread is much closer to the baseline methods than the table suggests."
                ),
                weight=0.35,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="List the strongest evidence signals that justify your diagnosis.",
                context=(
                    "Reproducibility memo: the full distribution centers near std ≈ 0.95 after removing the top-five selection bias.\n"
                    "The audit flags cherry-picking because the reported variance is incompatible with the twenty-run envelope."
                ),
                weight=0.35,
            ),
        ),
        answer_schema='{"cherry_picked_variant":"string","estimated_std_range":[low,high]}',
        truth={
            "cherry_picked_variant": "Prefix tuning",
            "true_std_range": (0.82, 1.09),
            "candidate_variants": ["Baseline", "Retrieval-augmented", "Prefix tuning", "Distillation"],
            "evidence": ["twenty_run_audit", "five_best_checkpoints", "selection_bias"],
        },
    ),
)

_HARD_CASES: Tuple[TaskCase, ...] = (
    TaskCase(
        case_id="mmlu-contamination",
        task_name="hard",
        difficulty="hard",
        stages=(
            _stage(
                name="risk-probe",
                kind="risk_probe",
                instruction="Estimate the provisional contamination risk from the source inventory and timing clues.",
                context=(
                    "Model: ARGUS-7B (released 2024-03-01)\n"
                    "Training cutoff: 2024-03-15\n"
                    "Sources: CommonCrawl, GitHub discussions, course transcripts, and a mirrored archive of benchmark solution threads.\n"
                    "Reviewer note: public instructional material is present in the crawl, but the paper has not yet decided whether that material counts as contamination."
                ),
                weight=0.25,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="Refine the contamination estimate and list the evidence signals that support it.",
                context=(
                    "Benchmark dossier: MMLU-Pro was released on 2024-02-10. The source archive includes answer-key mirrors and a public educational dump that were not separated at crawl time.\n"
                    "The audit record suggests that benchmark-specific material entered the corpus through a mirrored discussion board rather than through a direct benchmark scrape."
                ),
                weight=0.35,
            ),
            _stage(
                name="final-calibration",
                kind="final_calibration",
                instruction="Provide the final calibrated contamination risk and the strongest evidence signals.",
                context=(
                    "Curator note: benchmark-specific material was not explicitly filtered during deduplication; the publication timeline shows the benchmark predates the training end date.\n"
                    "The review board wants a final calibrated score rather than a binary label."
                ),
                weight=0.40,
            ),
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
        stages=(
            _stage(
                name="risk-probe",
                kind="risk_probe",
                instruction="Estimate the provisional contamination risk from the source inventory and timing clues.",
                context=(
                    "Model: ARGUS-7B (released 2024-06-10)\n"
                    "Training cutoff: 2024-01-31\n"
                    "Sources: CommonCrawl, technical blogs, arXiv abstracts, and code repositories.\n"
                    "Reviewer note: nothing in the first pass points at benchmark-specific material yet."
                ),
                weight=0.25,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="Refine the contamination estimate and list the evidence signals that support it.",
                context=(
                    "Benchmark dossier: GPQA-Diamond was released on 2024-03-11. The corpus was deduplicated against held-out benchmark items, and the compliance log records no benchmark-question hits.\n"
                    "The crawl never used a benchmark-specific scrape."
                ),
                weight=0.35,
            ),
            _stage(
                name="final-calibration",
                kind="final_calibration",
                instruction="Provide the final calibrated contamination risk and the strongest evidence signals.",
                context=(
                    "Final audit note: no benchmark-specific scrape was included, exact/fuzzy match filtering removed overlap, and the compliance log contains no benchmark-question hits.\n"
                    "The board expects a calibrated low-risk score rather than a false alarm."
                ),
                weight=0.40,
            ),
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

    The environment exposes staged research integrity investigations over
    synthetic but paper-like evaluation artifacts. Each reset selects one of
    three task families (easy / medium / hard), and each family unfolds across
    multiple deterministic stages so the agent must use the full step() /
    reset() / state() loop.
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
            description=(
                "Multi-stage ML evaluation integrity environment for detecting missing baselines, "
                "cherry-picked ablations, and benchmark contamination over staged investigations."
            ),
            version="2.0.0",
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

    def _stage_for_index(self, case: TaskCase, stage_index: int) -> StageSpec:
        return case.stages[min(stage_index, len(case.stages) - 1)]

    def _build_observation(
        self,
        case: TaskCase,
        stage_index: int,
        reward: float,
        done: bool,
        feedback: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArgusObservation:
        current_stage = self._stage_for_index(case, stage_index)
        observation_metadata = {
            "task_name": case.task_name,
            "case_id": case.case_id,
            "answer_schema": case.answer_schema,
            "episode_index": self._state.episode_index,
            "stage_index": stage_index + 1,
            "stage_count": len(case.stages),
            "stage_name": current_stage.name,
            "stage_kind": current_stage.kind,
            "stage_weight": current_stage.weight,
            "next_focus": current_stage.instruction,
            "feedback": feedback,
            "episode_reward": self._state.episode_reward,
        }
        if metadata:
            observation_metadata.update(metadata)

        return ArgusObservation(
            task_name=case.task_name,
            task_instruction=current_stage.instruction,
            context=current_stage.context,
            task_difficulty=case.difficulty,
            case_id=case.case_id,
            stage_index=stage_index + 1,
            stage_count=len(case.stages),
            stage_name=current_stage.name,
            stage_kind=current_stage.kind,
            stage_weight=current_stage.weight,
            next_focus=current_stage.instruction,
            episode_reward=self._state.episode_reward,
            feedback=feedback,
            done=done,
            reward=reward,
            metadata=observation_metadata,
        )

    def _score_text_answer(
        self,
        submitted: Optional[str],
        target: str,
        aliases: Sequence[str],
        exact_weight: float,
        alias_weight: float,
        observed: Optional[Sequence[str]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        submitted_text = _normalize_text(submitted)
        target_text = _normalize_text(target)
        alias_set = _normalize_aliases(aliases)
        observed_set = _normalize_aliases(observed or [])

        if not submitted_text:
            return 0.0, {"match": "empty"}

        if submitted_text == target_text:
            return exact_weight, {"match": "exact"}

        if submitted_text in alias_set or any(alias in submitted_text or submitted_text in alias for alias in alias_set):
            return alias_weight, {"match": "alias"}

        if submitted_text in observed_set:
            return 0.0, {"match": "observed_baseline"}

        similarity = SequenceMatcher(None, submitted_text, target_text).ratio()
        if similarity >= 0.8:
            return min(alias_weight, exact_weight * 0.75), {"match": "fuzzy"}

        return 0.0, {"match": "wrong"}

    def _score_range_answer(self, submitted: Optional[List[float]], truth_range: Tuple[float, float], max_weight: float) -> Tuple[float, Dict[str, Any]]:
        std_range = _safe_float_range(submitted)
        if std_range is None:
            return 0.0, {"match": "missing"}

        low, high = std_range
        true_low, true_high = truth_range
        if low <= true_low and high >= true_high:
            return max_weight, {"match": "contains_true_range"}

        if high >= true_low and low <= true_high:
            return max_weight * 0.7, {"match": "overlap"}

        submitted_midpoint = (low + high) / 2.0
        true_midpoint = (true_low + true_high) / 2.0
        distance = abs(submitted_midpoint - true_midpoint)
        tolerance = max(true_high - true_low, 0.1)
        distance_score = max(0.0, 0.45 - distance / (tolerance * 3.5))
        if distance_score > 0.0:
            return min(max_weight * 0.5, distance_score), {"match": "near_miss"}

        return 0.0, {"match": "wrong"}

    def _score_evidence(self, submitted: Optional[List[str]], truth_evidence: Sequence[str], max_weight: float) -> Tuple[float, Dict[str, Any]]:
        submitted_evidence = submitted or []
        evidence_matches: List[str] = []
        for canonical_signal in truth_evidence:
            aliases = (canonical_signal,) + _EVIDENCE_ALIASES.get(canonical_signal, ())
            matched = any(_contains_alias(submitted_item, aliases) for submitted_item in submitted_evidence)
            if matched:
                evidence_matches.append(canonical_signal)

        if not truth_evidence:
            return 0.0, {"match": "missing"}

        score = max_weight * (len(evidence_matches) / len(truth_evidence))
        return score, {"match": "partial" if evidence_matches else "wrong", "evidence_matches": evidence_matches}

    def _score_risk(self, submitted_risk: Optional[float], truth_risk: float, max_weight: float) -> Tuple[float, Dict[str, Any]]:
        if submitted_risk is None:
            return 0.0, {"match": "missing"}

        distance = abs(float(submitted_risk) - truth_risk)
        if distance <= 0.08:
            return max_weight, {"match": "exact"}
        if distance <= 0.15:
            return max_weight * 0.85, {"match": "close"}
        if distance <= 0.25:
            return max_weight * 0.6, {"match": "moderate"}
        if distance <= 0.35:
            return max_weight * 0.35, {"match": "coarse"}
        if distance <= 0.5:
            return max_weight * 0.1, {"match": "weak"}
        return 0.0, {"match": "wrong"}

    def _feedback_text(self, case: TaskCase, stage: StageSpec, breakdown: Dict[str, Any]) -> str:
        if case.task_name == "easy":
            if stage.kind == "family_hint":
                if breakdown.get("match") in {"alias", "exact"}:
                    return "Family clue accepted; the exact citation is still hidden in the next stage."
                return "Family clue is still unclear; inspect the table and reviewer note again."
            if breakdown.get("match") == "exact":
                return "Exact baseline confirmed."
            if breakdown.get("match") == "alias":
                return "Close, but the exact citation is still needed."
            return "Exact omission not yet matched."

        if case.task_name == "medium":
            if stage.kind == "variant_probe":
                if breakdown.get("match") in {"exact", "fuzzy"}:
                    return "Suspicious variant identified; variance and evidence still need confirmation."
                return "Variant probe needs another look."
            if stage.kind == "range_probe":
                if breakdown.get("match") in {"contains_true_range", "overlap"}:
                    return "Range estimate recorded; evidence signals will determine the final grade."
                return "Variance range is still off."
            if breakdown.get("evidence_matches"):
                return "Evidence signals recorded."
            return "Evidence list is still incomplete."

        if stage.kind == "risk_probe":
            if breakdown.get("match") in {"exact", "close", "moderate"}:
                return "Risk calibration recorded; later stages will verify the evidence trail."
            return "Risk estimate needs refinement."

        if stage.kind == "evidence_probe":
            if breakdown.get("evidence_matches"):
                return "Evidence trail recorded; final calibration is next."
            return "Evidence trail is still incomplete."

        if breakdown.get("evidence_matches"):
            return "Final calibration recorded."
        if case.truth.get("clean", False) and float(stage.weight) > 0.0:
            return "The clean-case calibration should stay conservative."
        return "Final calibration needs stronger evidence."

    def _grade_easy_stage(self, action: ArgusAction, case: TaskCase, stage: StageSpec) -> Tuple[float, Dict[str, Any]]:
        target = case.truth["missing_baseline"]
        aliases = case.truth.get("family_aliases", [])
        observed = case.truth.get("observed_baselines", [])

        if stage.kind == "family_hint":
            score, breakdown = self._score_text_answer(
                action.missing_baseline,
                target,
                aliases,
                exact_weight=stage.weight,
                alias_weight=stage.weight,
                observed=observed,
            )
            if breakdown.get("match") == "observed_baseline":
                score = 0.0
            return score, breakdown

        score, breakdown = self._score_text_answer(
            action.missing_baseline,
            target,
            aliases,
            exact_weight=stage.weight,
            alias_weight=stage.weight * 0.55,
            observed=observed,
        )
        if breakdown.get("match") == "observed_baseline":
            score = 0.0
        return score, breakdown

    def _grade_medium_stage(self, action: ArgusAction, case: TaskCase, stage: StageSpec) -> Tuple[float, Dict[str, Any]]:
        target_variant = case.truth["cherry_picked_variant"]
        candidate_variants = case.truth["candidate_variants"]

        if stage.kind == "variant_probe":
            return self._score_text_answer(
                action.cherry_picked_variant,
                target_variant,
                candidate_variants,
                exact_weight=stage.weight,
                alias_weight=stage.weight * 0.5,
                observed=None,
            )

        if stage.kind == "range_probe":
            return self._score_range_answer(action.estimated_std_range, case.truth["true_std_range"], stage.weight)

        return self._score_evidence(action.evidence, case.truth["evidence"], stage.weight)

    def _grade_hard_stage(self, action: ArgusAction, case: TaskCase, stage: StageSpec) -> Tuple[float, Dict[str, Any]]:
        truth_risk = float(case.truth["risk"])
        truth_evidence = case.truth["evidence"]

        if stage.kind == "risk_probe":
            return self._score_risk(action.contamination_risk, truth_risk, stage.weight)

        if stage.kind == "evidence_probe":
            risk_score, risk_breakdown = self._score_risk(action.contamination_risk, truth_risk, stage.weight * 0.45)
            evidence_score, evidence_breakdown = self._score_evidence(action.evidence, truth_evidence, stage.weight * 0.55)
            score = risk_score + evidence_score
            breakdown = {
                "risk_match": risk_breakdown.get("match"),
                "evidence_matches": evidence_breakdown.get("evidence_matches", []),
            }
            return score, breakdown

        risk_score, risk_breakdown = self._score_risk(action.contamination_risk, truth_risk, stage.weight * 0.4)
        evidence_score, evidence_breakdown = self._score_evidence(action.evidence, truth_evidence, stage.weight * 0.6)
        penalty = 0.0
        if case.truth.get("clean", False) and action.contamination_risk is not None and float(action.contamination_risk) > 0.5:
            penalty += 0.1
        if not case.truth.get("clean", False) and action.contamination_risk is not None and float(action.contamination_risk) < 0.2:
            penalty += 0.05
        score = max(0.0, risk_score + evidence_score - penalty)
        breakdown = {
            "risk_match": risk_breakdown.get("match"),
            "evidence_matches": evidence_breakdown.get("evidence_matches", []),
            "penalty": penalty,
        }
        return score, breakdown

    def _grade_stage(self, case: TaskCase, stage: StageSpec, action: ArgusAction) -> Tuple[float, Dict[str, Any]]:
        if case.task_name == "easy":
            return self._grade_easy_stage(action, case, stage)
        if case.task_name == "medium":
            return self._grade_medium_stage(action, case, stage)
        return self._grade_hard_stage(action, case, stage)

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
            episode_reward=0.0,
            stage_index=0,
            stage_count=len(case.stages),
            stage_name=case.stages[0].name,
            stage_kind=case.stages[0].kind,
            last_feedback="Begin with the first clue.",
        )
        self._episode_index += 1

        return self._build_observation(
            case,
            stage_index=0,
            reward=0.0,
            done=False,
            feedback=self._state.last_feedback,
            metadata={
                "phase": "reset",
                "episode_index": episode_index,
                "task_cursor": self._task_cursor,
                "next_stage": case.stages[0].name,
            },
        )

    def step(self, action: ArgusAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ArgusObservation:  # type: ignore[override]
        if self._current_case is None:
            raise RuntimeError("ARGUS environment must be reset before step() is called.")

        if self._episode_done:
            final_stage_index = max(0, len(self._current_case.stages) - 1)
            return self._build_observation(
                self._current_case,
                stage_index=final_stage_index,
                reward=0.0,
                done=True,
                feedback="Episode already complete.",
                metadata={"phase": "complete", "warning": "step_called_after_episode_completed"},
            )

        stage_index = self._state.stage_index
        current_stage = self._current_case.stages[stage_index]

        self._state.step_count += 1
        reward, breakdown = self._grade_stage(self._current_case, current_stage, action)
        reward = max(0.0, min(float(current_stage.weight), float(reward)))

        self._state.last_reward = reward
        self._state.episode_reward = min(1.0, self._state.episode_reward + reward)
        self._state.last_feedback = self._feedback_text(self._current_case, current_stage, breakdown)

        next_stage_index = stage_index + 1
        done = next_stage_index >= len(self._current_case.stages)
        self._state.stage_index = min(next_stage_index, len(self._current_case.stages))
        self._state.stage_name = current_stage.name if done else self._current_case.stages[next_stage_index].name
        self._state.stage_kind = current_stage.kind if done else self._current_case.stages[next_stage_index].kind
        self._state.stage_count = len(self._current_case.stages)
        self._episode_done = done

        next_observation_stage = min(next_stage_index, len(self._current_case.stages) - 1)
        observation = self._build_observation(
            self._current_case,
            stage_index=next_observation_stage,
            reward=reward,
            done=done,
            feedback=self._state.last_feedback,
            metadata={
                "phase": "step",
                "grade_breakdown": breakdown,
                "step_count": self._state.step_count,
                "stage_completed": current_stage.name,
                "next_stage": None if done else self._current_case.stages[next_stage_index].name,
                "episode_reward": self._state.episode_reward,
            },
        )

        if done:
            observation.metadata["episode_complete"] = True
        return observation

    @property
    def state(self) -> ArgusState:
        return self._state