from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import ArgusAction, ArgusObservation, ArgusReward, ArgusState
except ImportError:  # pragma: no cover - direct source-tree execution
    from models import ArgusAction, ArgusObservation, ArgusReward, ArgusState


SCORE_CAP = 1.0
UNSUPPORTED_EVIDENCE_PENALTY = 0.03
IMPOSSIBLE_RANGE_PENALTY = 0.08
FALSE_POSITIVE_PENALTY = 0.10
FALSE_NEGATIVE_PENALTY = 0.05
MIN_REASONABLE_STD_WIDTH = 0.08
MAX_REASONABLE_STD = 2.50


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
        "benchmark predates the data freeze",
        "benchmark released before training ended",
        "release happened before corpus freeze",
        "evaluation set existed before training cutoff",
        "timeline overlap",
        "timing overlap",
        "published before the cutoff",
        "benchmark earlier than training stop",
    ),
    "benchmark_in_corpus": (
        "worked solution archive",
        "answer discussion mirror",
        "tutorial answers",
        "solved notebook dump",
        "forum mirror",
        "solution walkthroughs",
        "public answer threads",
        "benchmark-adjacent mirror",
    ),
    "no_exclusion_filter": (
        "filter rule missed mirrored sources",
        "policy gap",
        "cleanup skipped mirrors",
        "no rule for community mirrors",
        "filter did not cover derivatives",
        "removal policy was incomplete",
        "left in during review hold",
        "corpus rule did not exclude adjacent sources",
    ),
    "deduplication": (
        "deduplicated",
        "duplicate pruning",
        "exact and fuzzy filtering",
        "near-duplicate removal",
        "similarity filter",
    ),
    "held_out_filtering": (
        "held-out scrub",
        "held-out questions removed",
        "benchmark items filtered",
        "evaluation items pruned",
        "red-team exclusion list",
    ),
    "no_benchmark_scrape": (
        "no benchmark-specific crawl",
        "never ingested benchmark urls",
        "no direct benchmark scrape",
        "crawl excluded benchmark domains",
        "benchmark site was never collected",
    ),
    "twenty_run_audit": (
        "full rerun ledger",
        "all archived reruns",
        "20 replay jobs",
        "twenty reruns",
        "complete audit sweep",
    ),
    "five_best_checkpoints": (
        "rebuttal shortlist",
        "leaderboard slice",
        "retained only five snapshots",
        "best-five subset",
        "top-run shortlist",
    ),
    "selection_bias": (
        "spread re-expanded",
        "variance snapped back",
        "discarded runs restored",
        "dispersion widened after replay",
        "reported stability was not genuine",
    ),
}

_ALL_EVIDENCE_SIGNALS = tuple(_EVIDENCE_ALIASES.keys())

_EASY_CASES: Tuple[TaskCase, ...] = (
    TaskCase(
        case_id="vision-beit-omission",
        task_name="easy",
        difficulty="easy",
        stages=(
            _stage(
                name="family-clue",
                kind="family_hint",
                instruction="Identify the missing baseline family from the comparison table.",
                context=(
                    "Table 2: ImageNet-1k fine-tuning summary.\n"
                    "Reported rows: Ours, ViT-B/16, DeiT-S, Swin-T, ConvNeXt-T.\n"
                    "Reviewer margin note: the missing comparator comes from the masked-image-pretraining line that predicts discrete visual targets, "
                    "but the spreadsheet kept only family labels during drafting."
                ),
                weight=0.35,
            ),
            _stage(
                name="exact-citation",
                kind="exact_missing_baseline",
                instruction="Return the exact omitted baseline method name.",
                context=(
                    "Appendix reconstruction note: the omitted row used the base-width checkpoint from that family with 16x16 patches. "
                    "Internal shorthand described it as the family name plus the base/16 variant."
                ),
                weight=0.65,
            ),
        ),
        answer_schema='{"missing_baseline":"string"}',
        truth={
            "missing_baseline": "BEiT-B/16",
            "observed_baselines": ["ViT-B/16", "DeiT-S", "Swin-T", "ConvNeXt-T"],
            "family_aliases": ["BEiT", "BEiT base", "BEiT base/16", "BEiT-B/16"],
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
                instruction="Identify the missing baseline family from the comparison table.",
                context=(
                    "Table 4: Cross-lingual intent classification on MASSIVE.\n"
                    "Reported rows: Ours, mBERT, XLM-R-base, MiniLM, DeBERTa-v3.\n"
                    "Area-chair note: the omitted comparator is from the stronger multilingual RoBERTa family that the appendix references only at family level."
                ),
                weight=0.35,
            ),
            _stage(
                name="exact-citation",
                kind="exact_missing_baseline",
                instruction="Return the exact omitted baseline method name.",
                context=(
                    "Review meeting notes: the hidden row used the large-capacity checkpoint from that multilingual family, not the base model already shown in the headline table."
                ),
                weight=0.65,
            ),
        ),
        answer_schema='{"missing_baseline":"string"}',
        truth={
            "missing_baseline": "XLM-R-large",
            "observed_baselines": ["mBERT", "XLM-R-base", "MiniLM", "DeBERTa-v3"],
            "family_aliases": ["XLM-R", "XLM-R large", "XLMR large", "XLM-R-large"],
        },
    ),
    TaskCase(
        case_id="speech-hubert-omission",
        task_name="easy",
        difficulty="easy",
        stages=(
            _stage(
                name="family-clue",
                kind="family_hint",
                instruction="Identify the missing baseline family from the comparison table.",
                context=(
                    "Table 3: Low-resource ASR transfer.\n"
                    "Reported rows: Ours, wav2vec 2.0 Base, Whisper Small, data2vec Audio Base, Conformer.\n"
                    "Reviewer note: the missing comparator belongs to the masked-unit speech-pretraining family whose larger checkpoint was discussed in the lab notebook."
                ),
                weight=0.35,
            ),
            _stage(
                name="exact-citation",
                kind="exact_missing_baseline",
                instruction="Return the exact omitted baseline method name.",
                context=(
                    "Notebook excerpt: the omitted row was the 24-layer checkpoint from that speech family rather than its base counterpart."
                ),
                weight=0.65,
            ),
        ),
        answer_schema='{"missing_baseline":"string"}',
        truth={
            "missing_baseline": "HuBERT-Large",
            "observed_baselines": ["wav2vec 2.0 Base", "Whisper Small", "data2vec Audio Base", "Conformer"],
            "family_aliases": ["HuBERT", "HuBERT large", "HuBERT-Large"],
        },
    ),
    TaskCase(
        case_id="multimodal-clip-omission",
        task_name="easy",
        difficulty="easy",
        stages=(
            _stage(
                name="family-clue",
                kind="family_hint",
                instruction="Identify the missing baseline family from the comparison table.",
                context=(
                    "Table 5: Zero-shot retrieval benchmark.\n"
                    "Reported rows: Ours, ALIGN, BLIP-2, CLIP ViT-B/32, SigLIP So400m.\n"
                    "Meta-review note: the missing comparator belongs to the contrastive language-image pretraining line already represented by a smaller vision tower."
                ),
                weight=0.35,
            ),
            _stage(
                name="exact-citation",
                kind="exact_missing_baseline",
                instruction="Return the exact omitted baseline method name.",
                context=(
                    "Artifact log: the hidden row used the larger 14-patch image tower from that same contrastive family, rather than the B/32 configuration shown in the table."
                ),
                weight=0.65,
            ),
        ),
        answer_schema='{"missing_baseline":"string"}',
        truth={
            "missing_baseline": "CLIP ViT-L/14",
            "observed_baselines": ["ALIGN", "BLIP-2", "CLIP ViT-B/32", "SigLIP So400m"],
            "family_aliases": ["CLIP", "CLIP large", "CLIP ViT-L/14", "CLIP L/14"],
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
                instruction="Identify the suspiciously cherry-picked variant from the ablation table.",
                context=(
                    "Ablation on CIFAR-100 over the five runs reported in the paper.\n"
                    "Full fine-tune: 84.1 ± 1.12\n"
                    "LoRA: 84.7 ± 1.08\n"
                    "Adapter: 86.0 ± 0.05\n"
                    "Prompt tuning: 83.3 ± 1.15\n"
                    "Reviewer memo: one row looks too stable relative to its neighbours."
                ),
                weight=0.30,
            ),
            _stage(
                name="range-probe",
                kind="range_probe",
                instruction="Estimate the plausible true standard-deviation range for the suspicious variant as [low, high].",
                context=(
                    "Audit ledger: the published row came from a rebuttal shortlist rather than the full rerun ledger. "
                    "When archived jobs are reinstated, the dispersion sits a little above 1.0 and overlaps the neighbouring methods."
                ),
                weight=0.35,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="List the strongest evidence signals that justify your diagnosis.",
                context=(
                    "Post-mortem note: after replaying all archived reruns, the spread re-expanded and the apparent stability disappeared. "
                    "The table had preserved only a leaderboard slice for the rebuttal draft."
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
                instruction="Identify the suspiciously cherry-picked variant from the ablation table.",
                context=(
                    "Ablation on low-resource speech translation.\n"
                    "Baseline: 67.2 ± 0.91\n"
                    "Retrieval-augmented: 68.5 ± 0.88\n"
                    "Prefix tuning: 70.1 ± 0.04\n"
                    "Distillation: 69.3 ± 0.95\n"
                    "Reviewer comment: one method appears improbably stable."
                ),
                weight=0.30,
            ),
            _stage(
                name="range-probe",
                kind="range_probe",
                instruction="Estimate the plausible true standard-deviation range for the suspicious variant as [low, high].",
                context=(
                    "Replay summary: the paper snapshot came from a best-five subset. "
                    "The full audit sweep puts the genuine spread just under 1.0 and much closer to the surrounding rows."
                ),
                weight=0.35,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="List the strongest evidence signals that justify your diagnosis.",
                context=(
                    "Reproduction note: once discarded reruns are restored, variance snaps back into the same band as the rest of the ablation sheet. "
                    "The rebuttal draft had kept only the highest-performing snapshots."
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
    TaskCase(
        case_id="fusion-head-cherry-pick",
        task_name="medium",
        difficulty="medium",
        stages=(
            _stage(
                name="variant-probe",
                kind="variant_probe",
                instruction="Identify the suspiciously cherry-picked variant from the ablation table.",
                context=(
                    "Ablation on multimodal retrieval.\n"
                    "Late fusion: 72.8 ± 0.94\n"
                    "Cross-attention fusion: 74.1 ± 0.06\n"
                    "Shared projector: 73.4 ± 0.89\n"
                    "Mixture head: 73.2 ± 0.97\n"
                    "Internal reviewer note: one row has dispersion that does not look natural."
                ),
                weight=0.30,
            ),
            _stage(
                name="range-probe",
                kind="range_probe",
                instruction="Estimate the plausible true standard-deviation range for the suspicious variant as [low, high].",
                context=(
                    "Scheduler replay: after rehydrating the full rerun ledger, the suspicious row spans the high-0.8s to low-1.1s rather than staying near zero."
                ),
                weight=0.35,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="List the strongest evidence signals that justify your diagnosis.",
                context=(
                    "Stability review: the published table came from a leaderboard slice, not the complete audit sweep. "
                    "Once archived jobs were restored, the spread re-expanded and the original narrative no longer held."
                ),
                weight=0.35,
            ),
        ),
        answer_schema='{"cherry_picked_variant":"string","estimated_std_range":[low,high]}',
        truth={
            "cherry_picked_variant": "Cross-attention fusion",
            "true_std_range": (0.84, 1.16),
            "candidate_variants": ["Late fusion", "Cross-attention fusion", "Shared projector", "Mixture head"],
            "evidence": ["twenty_run_audit", "five_best_checkpoints", "selection_bias"],
        },
    ),
    TaskCase(
        case_id="router-lora-cherry-pick",
        task_name="medium",
        difficulty="medium",
        stages=(
            _stage(
                name="variant-probe",
                kind="variant_probe",
                instruction="Identify the suspiciously cherry-picked variant from the ablation table.",
                context=(
                    "Ablation on instruction tuning for a sparse router.\n"
                    "Dense baseline: 59.7 ± 1.01\n"
                    "Static routing: 60.1 ± 0.96\n"
                    "Router-LoRA: 61.4 ± 0.03\n"
                    "Shared experts: 60.8 ± 0.92\n"
                    "Program committee note: one result appears unnaturally tight."
                ),
                weight=0.30,
            ),
            _stage(
                name="range-probe",
                kind="range_probe",
                instruction="Estimate the plausible true standard-deviation range for the suspicious variant as [low, high].",
                context=(
                    "Ops note: the visible row was assembled from a best-five shortlist. "
                    "The full replay band sits around the surrounding one-point spread after the archived runs are restored."
                ),
                weight=0.35,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="List the strongest evidence signals that justify your diagnosis.",
                context=(
                    "Audit summary: the complete rerun ledger contradicts the paper's tiny error bars. "
                    "The stability claim was driven by a leaderboard slice and vanished when discarded runs were put back."
                ),
                weight=0.35,
            ),
        ),
        answer_schema='{"cherry_picked_variant":"string","estimated_std_range":[low,high]}',
        truth={
            "cherry_picked_variant": "Router-LoRA",
            "true_std_range": (0.79, 1.08),
            "candidate_variants": ["Dense baseline", "Static routing", "Router-LoRA", "Shared experts"],
            "evidence": ["twenty_run_audit", "five_best_checkpoints", "selection_bias"],
        },
    ),
)

_HARD_CASES: Tuple[TaskCase, ...] = (
    TaskCase(
        case_id="mmlu-pro-contaminated",
        task_name="hard",
        difficulty="hard",
        stages=(
            _stage(
                name="risk-probe",
                kind="risk_probe",
                instruction="Estimate the provisional contamination risk from the training timeline and source inventory.",
                context=(
                    "Model card excerpt: release 2024-03-01. Corpus freeze 2024-03-15.\n"
                    "Sources: CommonCrawl, lecture transcripts, forum discussions, and mirrored answer walkthroughs from public study boards.\n"
                    "The first review did not classify those mirrors as benchmark-specific material."
                ),
                weight=0.25,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="Refine the contamination estimate and list the strongest evidence signals.",
                context=(
                    "Benchmark dossier: MMLU-Pro became public on 2024-02-10, before the corpus freeze. "
                    "A later audit found that answer-discussion mirrors survived cleanup because the exclusion rule covered direct benchmark urls but not community copies."
                ),
                weight=0.35,
            ),
            _stage(
                name="final-calibration",
                kind="final_calibration",
                instruction="Provide the final calibrated contamination risk and the strongest supporting evidence.",
                context=(
                    "Governance note: duplicate removal ran, but benchmark-adjacent mirrors and worked solution threads were still present after the policy hold. "
                    "The review board wants a calibrated score, not a binary accusation."
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
        case_id="gsm8k-contaminated",
        task_name="hard",
        difficulty="hard",
        stages=(
            _stage(
                name="risk-probe",
                kind="risk_probe",
                instruction="Estimate the provisional contamination risk from the training timeline and source inventory.",
                context=(
                    "Training note: corpus freeze 2023-10-01.\n"
                    "Sources: CommonCrawl, educational blogs, solved notebook collections, and community math help threads.\n"
                    "The curation team treated mirrored solution notebooks as general educational material."
                ),
                weight=0.25,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="Refine the contamination estimate and list the strongest evidence signals.",
                context=(
                    "Benchmark dossier: GSM8K was already public years before the corpus freeze. "
                    "The audit found solution walkthroughs and notebook mirrors in the training mix, while the removal rule only blocked a narrow set of canonical urls."
                ),
                weight=0.35,
            ),
            _stage(
                name="final-calibration",
                kind="final_calibration",
                instruction="Provide the final calibrated contamination risk and the strongest supporting evidence.",
                context=(
                    "Red-team summary: worked solutions remained in derivative sources, and the timing makes accidental exposure plausible even without a direct benchmark crawl."
                ),
                weight=0.40,
            ),
        ),
        answer_schema='{"contamination_risk":0.0,"evidence":["signal"]}',
        truth={
            "risk": 0.89,
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
                instruction="Estimate the provisional contamination risk from the training timeline and source inventory.",
                context=(
                    "Model card excerpt: corpus freeze 2024-01-31.\n"
                    "Sources: CommonCrawl, arXiv abstracts, technical blogs, and open-source repositories.\n"
                    "No benchmark-specific domain appears in the source inventory."
                ),
                weight=0.25,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="Refine the contamination estimate and list the strongest evidence signals.",
                context=(
                    "Benchmark dossier: GPQA-Diamond became public on 2024-03-11, after the corpus freeze. "
                    "Compliance logs record a held-out scrub plus exact and near-duplicate filtering, and no direct benchmark crawl was performed."
                ),
                weight=0.35,
            ),
            _stage(
                name="final-calibration",
                kind="final_calibration",
                instruction="Provide the final calibrated contamination risk and the strongest supporting evidence.",
                context=(
                    "Final audit note: benchmark domains were never collected, exclusion lists removed held-out items, and duplicate pruning found no benchmark-question hits."
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
    TaskCase(
        case_id="mmmu-clean",
        task_name="hard",
        difficulty="hard",
        stages=(
            _stage(
                name="risk-probe",
                kind="risk_probe",
                instruction="Estimate the provisional contamination risk from the training timeline and source inventory.",
                context=(
                    "Training card: corpus freeze 2023-11-30.\n"
                    "Sources: web documents, image captions, textbook scans cleared for OCR, and code repositories.\n"
                    "No reviewer note suggests benchmark-specific ingestion."
                ),
                weight=0.25,
            ),
            _stage(
                name="evidence-probe",
                kind="evidence_probe",
                instruction="Refine the contamination estimate and list the strongest evidence signals.",
                context=(
                    "Benchmark dossier: MMMU was published after the corpus freeze. "
                    "The data team maintained a red-team exclusion list, ran exact and fuzzy filtering, and did not ingest the benchmark site or mirrors."
                ),
                weight=0.35,
            ),
            _stage(
                name="final-calibration",
                kind="final_calibration",
                instruction="Provide the final calibrated contamination risk and the strongest supporting evidence.",
                context=(
                    "Governance closeout: held-out scrub completed successfully, no benchmark-specific crawl was used, and similarity filtering reported zero benchmark-linked collisions."
                ),
                weight=0.40,
            ),
        ),
        answer_schema='{"contamination_risk":0.0,"evidence":["signal"]}',
        truth={
            "risk": 0.06,
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
    """ARGUS - ML Evaluation Integrity Environment."""

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
            version="3.0.0",
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
        seed_value = int(seed) if seed is not None else episode_index * 97 + self._task_cursor * 13
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
        reward_detail: Optional[ArgusReward] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArgusObservation:
        current_stage = self._stage_for_index(case, stage_index)
        observation_metadata: Dict[str, Any] = {
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
        if reward_detail is not None:
            observation_metadata["reward_detail"] = reward_detail.model_dump()

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

    def _range_penalty(self, submitted: Optional[List[float]]) -> float:
        std_range = _safe_float_range(submitted)
        if std_range is None:
            return 0.0

        low, high = std_range
        width = high - low
        if low < 0.0 or high <= 0.0 or high > MAX_REASONABLE_STD:
            return IMPOSSIBLE_RANGE_PENALTY
        if width < MIN_REASONABLE_STD_WIDTH:
            return IMPOSSIBLE_RANGE_PENALTY
        return 0.0

    def _score_range_answer(
        self,
        submitted: Optional[List[float]],
        truth_range: Tuple[float, float],
        max_weight: float,
    ) -> Tuple[float, Dict[str, Any]]:
        std_range = _safe_float_range(submitted)
        penalty = self._range_penalty(submitted)
        if std_range is None:
            return 0.0, {"match": "missing", "penalty": penalty}

        low, high = std_range
        true_low, true_high = truth_range
        if low <= true_low and high >= true_high:
            score = max_weight
            return max(0.0, score - penalty), {"match": "contains_true_range", "penalty": penalty}

        if high >= true_low and low <= true_high:
            score = max_weight * 0.7
            return max(0.0, score - penalty), {"match": "overlap", "penalty": penalty}

        submitted_midpoint = (low + high) / 2.0
        true_midpoint = (true_low + true_high) / 2.0
        distance = abs(submitted_midpoint - true_midpoint)
        tolerance = max(true_high - true_low, 0.1)
        distance_score = max(0.0, 0.45 - distance / (tolerance * 3.5))
        if distance_score > 0.0:
            score = min(max_weight * 0.5, distance_score)
            return max(0.0, score - penalty), {"match": "near_miss", "penalty": penalty}

        return 0.0, {"match": "wrong", "penalty": penalty}

    def _score_evidence(
        self,
        submitted: Optional[List[str]],
        truth_evidence: Sequence[str],
        max_weight: float,
    ) -> Tuple[float, Dict[str, Any]]:
        submitted_evidence = submitted or []
        evidence_matches: List[str] = []
        unsupported_items = 0

        for canonical_signal in truth_evidence:
            aliases = (canonical_signal,) + _EVIDENCE_ALIASES.get(canonical_signal, ())
            if any(_contains_alias(submitted_item, aliases) for submitted_item in submitted_evidence):
                evidence_matches.append(canonical_signal)

        for submitted_item in submitted_evidence:
            matched_any = any(
                _contains_alias(submitted_item, (signal,) + _EVIDENCE_ALIASES.get(signal, ()))
                for signal in _ALL_EVIDENCE_SIGNALS
            )
            if not matched_any:
                unsupported_items += 1

        if not truth_evidence:
            return 0.0, {"match": "missing", "unsupported_items": unsupported_items, "penalty": 0.0}

        raw_score = max_weight * (len(evidence_matches) / len(truth_evidence))
        penalty = min(max_weight * 0.35, unsupported_items * UNSUPPORTED_EVIDENCE_PENALTY)
        score = max(0.0, raw_score - penalty)
        return score, {
            "match": "partial" if evidence_matches else "wrong",
            "evidence_matches": evidence_matches,
            "unsupported_items": unsupported_items,
            "penalty": penalty,
        }

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
                if breakdown.get("match") in {"alias", "exact", "fuzzy"}:
                    return "Family clue accepted; now convert that lineage clue into the exact citation."
                return "Family clue is still weak; inspect the lineage hint instead of copying an observed row."

            if breakdown.get("match") == "exact":
                return "Exact omission confirmed."
            if breakdown.get("match") in {"alias", "fuzzy"}:
                return "You found the family, but the exact citation is still underspecified."
            return "The cited omission still does not match the hidden comparator."

        if case.task_name == "medium":
            if stage.kind == "variant_probe":
                if breakdown.get("match") in {"exact", "fuzzy"}:
                    return "Suspicious variant identified; now estimate its real spread."
                return "The suspicious row is still not isolated."
            if stage.kind == "range_probe":
                if breakdown.get("penalty", 0.0) > 0.0:
                    return "The variance band is implausible; use a realistic range."
                if breakdown.get("match") in {"contains_true_range", "overlap", "near_miss"}:
                    return "Range estimate recorded; finish by explaining why the table is misleading."
                return "The range estimate still misses the audited spread."
            if breakdown.get("unsupported_items", 0) > 0:
                return "Some evidence items are unsupported by the audit trail."
            if breakdown.get("evidence_matches"):
                return "Evidence trail recorded."
            return "The evidence list is still incomplete."

        if stage.kind == "risk_probe":
            if breakdown.get("match") in {"exact", "close", "moderate"}:
                return "Provisional risk recorded; now reconcile it with the evidence trail."
            return "The provisional risk estimate needs calibration."

        if stage.kind == "evidence_probe":
            if breakdown.get("unsupported_items", 0) > 0:
                return "Some evidence items are unsupported by the audit record."
            if breakdown.get("evidence_matches"):
                return "Evidence trail recorded; final calibration is next."
            return "The evidence trail is still incomplete."

        if breakdown.get("penalty", 0.0) >= FALSE_POSITIVE_PENALTY:
            return "This clean case is being overcalled; keep the final score conservative."
        if breakdown.get("penalty", 0.0) > 0.0 and case.truth.get("clean", False):
            return "The clean-case safeguards were stronger than your final claim."
        if breakdown.get("evidence_matches"):
            return "Final calibration recorded."
        return "Final calibration still needs stronger support."

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
        else:
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

        if stage.kind == "variant_probe":
            return self._score_text_answer(
                action.cherry_picked_variant,
                target_variant,
                [target_variant],
                exact_weight=stage.weight,
                alias_weight=stage.weight * 0.5,
                observed=None,
            )

        if stage.kind == "range_probe":
            return self._score_range_answer(action.estimated_std_range, case.truth["true_std_range"], stage.weight)

        return self._score_evidence(action.evidence, case.truth["evidence"], stage.weight)

    def _hard_final_penalty(self, case: TaskCase, action: ArgusAction) -> float:
        submitted_risk = float(action.contamination_risk) if action.contamination_risk is not None else None
        if submitted_risk is None:
            return 0.0

        if case.truth.get("clean", False) and submitted_risk > 0.55:
            return FALSE_POSITIVE_PENALTY
        if not case.truth.get("clean", False) and submitted_risk < 0.20:
            return FALSE_NEGATIVE_PENALTY
        return 0.0

    def _grade_hard_stage(self, action: ArgusAction, case: TaskCase, stage: StageSpec) -> Tuple[float, Dict[str, Any]]:
        truth_risk = float(case.truth["risk"])
        truth_evidence = case.truth["evidence"]

        if stage.kind == "risk_probe":
            return self._score_risk(action.contamination_risk, truth_risk, stage.weight)

        if stage.kind == "evidence_probe":
            risk_score, risk_breakdown = self._score_risk(action.contamination_risk, truth_risk, stage.weight * 0.45)
            evidence_score, evidence_breakdown = self._score_evidence(action.evidence, truth_evidence, stage.weight * 0.55)
            score = max(0.0, risk_score + evidence_score)
            return score, {
                "risk_match": risk_breakdown.get("match"),
                "evidence_matches": evidence_breakdown.get("evidence_matches", []),
                "unsupported_items": evidence_breakdown.get("unsupported_items", 0),
                "penalty": float(evidence_breakdown.get("penalty", 0.0) or 0.0),
            }

        risk_score, risk_breakdown = self._score_risk(action.contamination_risk, truth_risk, stage.weight * 0.4)
        evidence_score, evidence_breakdown = self._score_evidence(action.evidence, truth_evidence, stage.weight * 0.6)
        penalty = float(evidence_breakdown.get("penalty", 0.0) or 0.0) + self._hard_final_penalty(case, action)
        score = max(0.0, risk_score + evidence_score - penalty)
        return score, {
            "risk_match": risk_breakdown.get("match"),
            "evidence_matches": evidence_breakdown.get("evidence_matches", []),
            "unsupported_items": evidence_breakdown.get("unsupported_items", 0),
            "penalty": penalty,
        }

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

        if current_stage.kind == "range_probe":
            components = {
                "range_reward": reward,
                "penalty": float(breakdown.get("penalty", 0.0) or 0.0),
            }
        elif current_stage.kind in {"risk_probe", "evidence_probe", "final_calibration"}:
            components = {
                "step_reward": reward,
                "penalty": float(breakdown.get("penalty", 0.0) or 0.0),
            }
        else:
            components = {"step_reward": reward}

        reward_detail = ArgusReward(
            total=reward,
            stage_weight=current_stage.weight,
            components=components,
            matched_signals=list(breakdown.get("evidence_matches", [])),
            penalty=float(breakdown.get("penalty", 0.0) or 0.0),
            note=str(breakdown.get("match") or breakdown.get("risk_match") or "partial"),
        )

        self._state.last_reward = reward
        self._state.episode_reward = min(SCORE_CAP, self._state.episode_reward + reward)
        self._state.last_feedback = self._feedback_text(self._current_case, current_stage, breakdown)

        next_stage_index = stage_index + 1
        done = next_stage_index >= len(self._current_case.stages)
        state_stage_index = min(next_stage_index, len(self._current_case.stages) - 1)
        self._state.stage_index = state_stage_index
        self._state.stage_name = self._current_case.stages[state_stage_index].name
        self._state.stage_kind = self._current_case.stages[state_stage_index].kind
        self._state.stage_count = len(self._current_case.stages)
        self._episode_done = done

        next_observation_stage = min(next_stage_index, len(self._current_case.stages) - 1)
        observation = self._build_observation(
            self._current_case,
            stage_index=next_observation_stage,
            reward=reward,
            done=done,
            feedback=self._state.last_feedback,
            reward_detail=reward_detail,
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
