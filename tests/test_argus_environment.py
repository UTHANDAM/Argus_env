import re
import unittest

from models import ArgusAction
from server.argus_env_environment import ArgusEnvironment, _EASY_CASES, _HARD_CASES, _MEDIUM_CASES


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _run_episode(task: str, seed: int, actions: list[ArgusAction]):
    env = ArgusEnvironment()
    observation = env.reset(task=task, seed=seed)
    for action in actions:
        observation = env.step(action)
    return env, observation


class ArgusEnvironmentTests(unittest.TestCase):
    def test_oracle_episodes_reach_near_full_score(self) -> None:
        oracle_cases = [
            (
                "easy",
                0,
                [
                    ArgusAction(missing_baseline="BEiT"),
                    ArgusAction(missing_baseline="BEiT-B/16"),
                ],
            ),
            (
                "medium",
                0,
                [
                    ArgusAction(cherry_picked_variant="Adapter"),
                    ArgusAction(estimated_std_range=[0.88, 1.24]),
                    ArgusAction(evidence=["twenty_run_audit", "five_best_checkpoints", "selection_bias"]),
                ],
            ),
            (
                "hard",
                0,
                [
                    ArgusAction(contamination_risk=0.92),
                    ArgusAction(
                        contamination_risk=0.92,
                        evidence=["temporal_overlap", "benchmark_in_corpus", "no_exclusion_filter"],
                    ),
                    ArgusAction(
                        contamination_risk=0.92,
                        evidence=["temporal_overlap", "benchmark_in_corpus", "no_exclusion_filter"],
                    ),
                ],
            ),
        ]

        for task, seed, actions in oracle_cases:
            with self.subTest(task=task, seed=seed):
                env, observation = _run_episode(task, seed, actions)
                self.assertTrue(observation.done)
                self.assertAlmostEqual(observation.episode_reward, 0.99, places=6)
                self.assertLess(observation.episode_reward, 1.0)
                self.assertGreater(env.state.last_reward, 0.0)

    def test_wrong_trajectories_score_poorly(self) -> None:
        wrong_cases = [
            (
                "easy",
                1,
                [
                    ArgusAction(missing_baseline="mBERT"),
                    ArgusAction(missing_baseline="MiniLM"),
                ],
                0.0,
            ),
            (
                "medium",
                1,
                [
                    ArgusAction(cherry_picked_variant="Baseline"),
                    ArgusAction(estimated_std_range=[0.01, 0.02]),
                    ArgusAction(evidence=["unsupported signal"]),
                ],
                0.0,
            ),
            (
                "hard",
                2,
                [
                    ArgusAction(contamination_risk=0.95),
                    ArgusAction(contamination_risk=0.95, evidence=["temporal_overlap", "benchmark_in_corpus"]),
                    ArgusAction(contamination_risk=0.95, evidence=["temporal_overlap", "benchmark_in_corpus"]),
                ],
                0.1,
            ),
        ]

        for task, seed, actions, max_score in wrong_cases:
            with self.subTest(task=task, seed=seed):
                _, observation = _run_episode(task, seed, actions)
                self.assertTrue(observation.done)
                self.assertLessEqual(observation.episode_reward, max_score)

    def test_hard_partial_trajectory_gets_meaningful_credit(self) -> None:
        env = ArgusEnvironment()
        observation = env.reset(task="hard", seed=33)

        self.assertEqual(env.state.case_id, "gsm8k-contaminated")

        for action in [
            ArgusAction(contamination_risk=0.3),
            ArgusAction(contamination_risk=0.35, evidence=["solution_walkthroughs|notebook_mirrors"]),
            ArgusAction(contamination_risk=0.35, evidence=["solution_walkthroughs|notebook_mirrors"]),
        ]:
            observation = env.step(action)

        self.assertTrue(observation.done)
        self.assertGreater(observation.episode_reward, 0.25)

    def test_clean_case_false_positive_is_penalized(self) -> None:
        _, correct = _run_episode(
            "hard",
            2,
            [
                ArgusAction(contamination_risk=0.08),
                ArgusAction(
                    contamination_risk=0.08,
                    evidence=["deduplication", "held_out_filtering", "no_benchmark_scrape"],
                ),
                ArgusAction(
                    contamination_risk=0.08,
                    evidence=["deduplication", "held_out_filtering", "no_benchmark_scrape"],
                ),
            ],
        )
        env, false_positive = _run_episode(
            "hard",
            2,
            [
                ArgusAction(contamination_risk=0.90),
                ArgusAction(
                    contamination_risk=0.90,
                    evidence=["temporal_overlap", "benchmark_in_corpus", "no_exclusion_filter"],
                ),
                ArgusAction(
                    contamination_risk=0.90,
                    evidence=["temporal_overlap", "benchmark_in_corpus", "no_exclusion_filter"],
                ),
            ],
        )

        self.assertAlmostEqual(correct.episode_reward, 0.99, places=6)
        self.assertLess(correct.episode_reward, 1.0)
        self.assertLess(false_positive.episode_reward, 0.1)
        self.assertAlmostEqual(env.state.last_reward, 0.0, places=6)

    def test_case_text_no_longer_leaks_answers_or_canonical_labels(self) -> None:
        for case in _EASY_CASES:
            exact_target = _normalized(case.truth["missing_baseline"])
            final_context = _normalized(case.stages[-1].context)
            self.assertNotIn(exact_target, final_context)

        medium_labels = {"twenty_run_audit", "five_best_checkpoints", "selection_bias"}
        for case in _MEDIUM_CASES:
            combined_context = " ".join(stage.context for stage in case.stages).lower()
            for label in medium_labels:
                self.assertNotIn(label, combined_context)

        for case in _HARD_CASES:
            combined_context = " ".join(stage.context for stage in case.stages).lower()
            for label in case.truth["evidence"]:
                self.assertNotIn(label, combined_context)


if __name__ == "__main__":
    unittest.main()
