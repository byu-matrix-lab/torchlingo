"""Tests for the model diagnostics.

These checks exist to be trusted when something is already going wrong, so the
cases that matter most are the ones where a check could quietly report the
comfortable answer: a stale gradient making a dead parameter look live, a
detached graph, an empty loss history.
"""

import math
import unittest

import torch
from torch import nn

from torchlingo.diagnostics import (
    CheckResult,
    GradientReport,
    check_contamination,
    check_eval_mode,
    check_generalization,
    check_gradients,
    check_loss_moved,
    gradient_report,
    uniform_loss,
)


class PartlyUnused(nn.Module):
    """A model with a parameter the forward pass never touches.

    ``unused`` is a real parameter with ``requires_grad=True``; it simply never
    reaches the loss, so no gradient is computed for it. That is the "dead"
    bucket, and it is distinct from being frozen.
    """

    def __init__(self):
        super().__init__()
        self.used = nn.Linear(4, 2)
        self.unused = nn.Linear(4, 2)

    def forward(self, x):
        return self.used(x)


class CheckResultTests(unittest.TestCase):
    def test_truthiness_follows_passed(self):
        self.assertTrue(CheckResult("n", True, "d"))
        self.assertFalse(CheckResult("n", False, "d"))

    def test_str_is_tagged_and_carries_detail(self):
        self.assertIn("[PASS]", str(CheckResult("thing", True, "measured=1")))
        self.assertIn("[FAIL]", str(CheckResult("thing", False, "measured=1")))
        self.assertIn("measured=1", str(CheckResult("thing", True, "measured=1")))


class UniformLossTests(unittest.TestCase):
    def test_is_log_of_vocab_size(self):
        self.assertAlmostEqual(uniform_loss(125), math.log(125))

    def test_single_type_costs_nothing(self):
        self.assertEqual(uniform_loss(1), 0.0)

    def test_rejects_empty_vocabulary(self):
        with self.assertRaises(ValueError):
            uniform_loss(0)


class CheckLossMovedTests(unittest.TestCase):
    def test_flat_loss_fails(self):
        self.assertFalse(check_loss_moved([5.341, 5.339, 5.337]))

    def test_falling_loss_passes(self):
        self.assertTrue(check_loss_moved([5.223, 2.053, 0.265]))

    def test_rising_loss_fails(self):
        self.assertFalse(check_loss_moved([1.0, 2.0]))

    def test_threshold_is_respected(self):
        # Deliberately away from the exact boundary: a drop of 1.0 - 0.9 is
        # 0.09999999999999998 in binary floating point, so whether it clears a
        # threshold of exactly 0.1 is a fact about IEEE 754 rather than about
        # this check, and not a contract worth pinning.
        self.assertTrue(check_loss_moved([1.0, 0.9], min_drop=0.05))
        self.assertFalse(check_loss_moved([1.0, 0.9], min_drop=0.2))

    def test_vocab_size_adds_the_guessing_reference(self):
        detail = check_loss_moved([5.3, 5.3], vocab_size=125).detail
        self.assertIn("4.828", detail)

    def test_vocab_size_is_optional(self):
        self.assertNotIn("guessing", check_loss_moved([1.0, 0.5]).detail)

    def test_compares_first_to_last_not_the_minimum(self):
        # A run that improved and then diverged has not "learned"; reporting the
        # best epoch instead of the last would hide that.
        self.assertFalse(check_loss_moved([5.0, 1.0, 5.0]))

    def test_single_epoch_cannot_have_moved(self):
        self.assertFalse(check_loss_moved([5.0]))

    def test_rejects_empty_history(self):
        with self.assertRaises(ValueError):
            check_loss_moved([])


class GradientReportTests(unittest.TestCase):
    def test_healthy_model_is_all_live(self):
        model = nn.Linear(4, 2)
        report = gradient_report(model, model(torch.ones(1, 4)).sum())
        self.assertEqual(report.frozen, [])
        self.assertEqual(report.dead, [])
        self.assertEqual(sorted(report.live), ["bias", "weight"])
        self.assertTrue(report.all_live())

    def test_frozen_parameters_are_named(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        model[0].requires_grad_(False)
        report = gradient_report(model, model(torch.ones(1, 4)).sum())
        self.assertEqual(report.frozen, ["0.weight", "0.bias"])
        self.assertEqual(report.dead, [])
        self.assertFalse(report.all_live())

    def test_parameter_outside_the_graph_is_dead_not_frozen(self):
        model = PartlyUnused()
        report = gradient_report(model, model(torch.ones(1, 4)).sum())
        self.assertEqual(report.frozen, [])
        self.assertEqual(sorted(report.dead), ["unused.bias", "unused.weight"])
        self.assertFalse(report.all_live())

    def test_stale_gradients_cannot_disguise_a_dead_parameter(self):
        # The failure this guards against: an earlier training step left a
        # nonzero .grad on a parameter that no longer receives one. Without
        # clearing first, the check would report it live and the real bug would
        # stay hidden.
        model = PartlyUnused()
        for parameter in model.unused.parameters():
            parameter.grad = torch.ones_like(parameter)

        report = gradient_report(model, model(torch.ones(1, 4)).sum())
        self.assertEqual(sorted(report.dead), ["unused.bias", "unused.weight"])

    def test_all_zero_gradient_counts_as_dead(self):
        # Multiplying the output by zero keeps the parameter in the graph, so a
        # gradient is computed -- it is just uniformly zero, which teaches the
        # model nothing and should not read as healthy.
        model = nn.Linear(4, 2)
        report = gradient_report(model, (model(torch.ones(1, 4)) * 0.0).sum())
        self.assertEqual(sorted(report.dead), ["bias", "weight"])
        self.assertEqual(report.live, [])

    def test_rejects_non_scalar_loss(self):
        model = nn.Linear(4, 2)
        with self.assertRaises(ValueError):
            gradient_report(model, model(torch.ones(1, 4)))

    def test_detached_loss_is_reported_as_such(self):
        model = nn.Linear(4, 2)
        with self.assertRaises(ValueError) as caught:
            gradient_report(model, model(torch.ones(1, 4)).sum().detach())
        self.assertIn("detached", str(caught.exception))


class GradientReportSummaryTests(unittest.TestCase):
    def test_summary_counts_every_bucket(self):
        summary = GradientReport(frozen=["a"], dead=["b"], live=["c", "d"]).summary()
        self.assertIn("live=2", summary)
        self.assertIn("frozen=1", summary)
        self.assertIn("dead=1", summary)

    def test_summary_names_the_first_casualty(self):
        summary = GradientReport(frozen=["enc.weight"], dead=["dec.bias"]).summary()
        self.assertIn("enc.weight", summary)
        self.assertIn("dec.bias", summary)

    def test_healthy_summary_names_nobody(self):
        summary = GradientReport(live=["a", "b"]).summary()
        self.assertNotIn("first", summary)


class CheckGradientsTests(unittest.TestCase):
    def test_passes_on_a_healthy_model(self):
        model = nn.Linear(4, 2)
        self.assertTrue(check_gradients(model, model(torch.ones(1, 4)).sum()))

    def test_fails_and_names_the_frozen_parameter(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        model[0].requires_grad_(False)
        result = check_gradients(model, model(torch.ones(1, 4)).sum())
        self.assertFalse(result)
        self.assertIn("0.weight", result.detail)


class CheckGeneralizationTests(unittest.TestCase):
    def test_healthy_run_passes(self):
        # Validation below training is normal: dropout is off during validation.
        self.assertTrue(check_generalization([0.265], [0.099]))

    def test_memorizing_run_fails(self):
        self.assertFalse(check_generalization([0.540], [1.624]))

    def test_reads_the_last_epoch_not_the_first(self):
        self.assertFalse(check_generalization([5.0, 0.5], [5.0, 1.6]))

    def test_threshold_is_respected(self):
        self.assertTrue(check_generalization([1.0], [1.2], max_gap=0.3))
        self.assertFalse(check_generalization([1.0], [1.2], max_gap=0.1))

    def test_gap_sign_is_shown(self):
        self.assertIn("+1.084", check_generalization([0.540], [1.624]).detail)

    def test_rejects_empty_history(self):
        with self.assertRaises(ValueError):
            check_generalization([], [1.0])
        with self.assertRaises(ValueError):
            check_generalization([1.0], [])


class CheckContaminationTests(unittest.TestCase):
    def test_disjoint_sets_pass(self):
        self.assertTrue(check_contamination(["fresh one"], ["seen one"]))

    def test_any_overlap_fails(self):
        self.assertFalse(check_contamination(["a", "b"], ["b", "c"]))

    def test_counts_the_overlap(self):
        result = check_contamination(["a", "b", "c"], ["b", "c"])
        self.assertIn("2/3", result.detail)

    def test_quotes_offending_sentences(self):
        result = check_contamination(["seen one"], ["seen one"])
        self.assertIn("seen one", result.detail)

    def test_example_count_is_capped(self):
        shared = [f"s{i}" for i in range(10)]
        result = check_contamination(shared, shared, max_examples=2)
        self.assertIn("10/10", result.detail)
        self.assertEqual(result.detail.count("'s"), 2)

    def test_exact_match_only(self):
        # Documented limitation: near-duplicates are not caught. Pinned so the
        # docstring's warning stays true.
        self.assertTrue(check_contamination(["A sentence"], ["a sentence"]))

    def test_accepts_any_iterable(self):
        self.assertFalse(check_contamination(iter(["a"]), iter(["a"])))

    def test_empty_test_set_is_vacuously_clean(self):
        self.assertTrue(check_contamination([], ["a"]))


class CheckEvalModeTests(unittest.TestCase):
    def test_fresh_module_is_in_training_mode(self):
        self.assertFalse(check_eval_mode(nn.Linear(2, 2)))

    def test_eval_passes(self):
        model = nn.Linear(2, 2)
        model.eval()
        self.assertTrue(check_eval_mode(model))

    def test_train_explains_the_consequence(self):
        self.assertIn("reproduce", check_eval_mode(nn.Linear(2, 2)).detail)


if __name__ == "__main__":
    unittest.main()
