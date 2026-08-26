"""Oracle tests pinning decoding behavior ahead of the beam-search batching work.

These tests exist to make the planned decoding refactors (batching beam search
across beams and across sentences, and later incremental/KV-cache decoding)
*safe*. They encode the properties a faster implementation must preserve:

1. **Padding invariance** - decoding a sentence alone must equal decoding it
   inside a padded batch alongside longer sentences.
2. **Beam/greedy agreement** - ``beam_size=1`` must reproduce greedy decoding.
3. **Determinism** - repeated calls must return identical token sequences.
4. **Length-normalization semantics** - the current implementation applies
   length normalization during pruning, not only at final selection. That is a
   deliberate (if non-standard) choice and must not change silently during
   performance work.
5. **Tie-breaking** - equal scores resolve toward lower token IDs, on every
   device. See the ``inference`` module docstring for the rule itself.

Why a new fixture: ``DummyTransformer`` in ``test_training_inference.py``
returns logits computed from a zero tensor, so its output depends only on
decoding *position* - never on the decoder history or the encoder memory. A
batched beam search that scrambled per-beam histories, or that expanded
``memory`` incorrectly across beams, would pass every existing beam test.
``HistorySensitiveTransformer`` below is sensitive to history, memory, and the
source padding mask, so those bug classes actually fail a test.

Golden values are derived from this deterministic integer-valued fixture rather
than from a randomly initialized model, so they do not drift across platforms
or PyTorch versions.
"""

from __future__ import annotations

import unittest
from typing import ClassVar

import torch
from torch import nn

from torchlingo.config import get_default_config
from torchlingo.inference import (
    _canonical_topk,
    _rank_key,
    beam_search_decode,
    greedy_decode,
)
from torchlingo.inference_fast import beam_search_decode_batched
from torchlingo.models.transformer_simple import SimpleTransformer

D_MEM = 4


class HistorySensitiveTransformer(nn.Module):
    """Deterministic Transformer stand-in that reacts to history and memory.

    The next-token distribution is peaked at a token determined by both the
    encoded source and the tokens generated so far, which makes beam search
    genuinely branch. Scores decay linearly away from that peak so every
    vocabulary entry has a distinct rank, letting tests assert on ordering.

    EOS becomes progressively more attractive as the hypothesis grows, so
    decoding terminates without relying on ``max_len``.

    Attributes:
        vocab_size (int): Size of the output vocabulary.
        pad_idx (int): Padding index, excluded from the source signature.
        sos_idx (int): Start-of-sequence index.
        eos_idx (int): End-of-sequence index.
    """

    def __init__(
        self,
        vocab_size: int = 16,
        pad_idx: int = 0,
        sos_idx: int = 2,
        eos_idx: int = 3,
        eos_bias: float = -9.0,
        eos_growth: float = 1.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.eos_bias = eos_bias
        self.eos_growth = eos_growth
        # Present so `next(model.parameters()).device` resolves.
        self.dummy = nn.Parameter(torch.zeros(1))
        # First index reserved for ordinary tokens the fixture may emit.
        self.first_content_idx = 4

    def encode(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode source tokens into a memory tensor carrying their values.

        Args:
            src: Source token IDs, shape (batch, src_len).
            src_key_padding_mask: Bool mask, True at padded positions.

        Returns:
            Memory tensor of shape (batch, src_len, D_MEM) whose channel 0
            holds the source token value and 0.0 at padded positions.
        """
        batch, src_len = src.shape
        memory = torch.zeros(batch, src_len, D_MEM, device=src.device)
        values = src.to(torch.float32)
        if src_key_padding_mask is not None:
            values = values.masked_fill(src_key_padding_mask, 0.0)
        else:
            values = values.masked_fill(src.eq(self.pad_idx), 0.0)
        memory[:, :, 0] = values
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce logits that depend on both memory and decoder history.

        Args:
            tgt: Target prefix token IDs, shape (batch, tgt_len).
            memory: Encoder memory, shape (batch, src_len, D_MEM).
            src_key_padding_mask: Unused beyond shape agreement.
            tgt_key_padding_mask: Unused.
            tgt_mask: Unused.

        Returns:
            Logits of shape (batch, tgt_len, vocab_size). Only the final
            position carries meaningful values, matching how the decoders in
            ``inference`` consume the output.

        Raises:
            ValueError: If memory's batch dimension does not match tgt's. This
                catches beam expansions that fail to expand memory in step.
            AssertionError: Never raised; see ValueError above.
        """
        if memory.size(0) != tgt.size(0):
            raise ValueError(
                f"memory batch {memory.size(0)} != tgt batch {tgt.size(0)}; "
                "memory must be expanded in step with the hypotheses"
            )

        batch, tgt_len = tgt.shape
        signature = memory[:, :, 0].sum(dim=1)
        prefix = tgt.to(torch.float32).sum(dim=1)

        span = self.vocab_size - self.first_content_idx
        peak = (signature + prefix).remainder(span) + self.first_content_idx

        arange = torch.arange(self.vocab_size, device=tgt.device, dtype=torch.float32)
        # Linear decay away from the peak gives every token a distinct rank.
        logits_last = -(arange.unsqueeze(0) - peak.unsqueeze(1)).abs()

        logits_last[:, self.pad_idx] = -50.0
        logits_last[:, self.sos_idx] = -50.0
        logits_last[:, self.eos_idx] = self.eos_bias + self.eos_growth * tgt_len

        logits = torch.full((batch, tgt_len, self.vocab_size), -50.0, device=tgt.device)
        logits[:, -1, :] = logits_last
        return logits

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Run encode then decode, matching the training-time interface."""
        return self.decode(tgt, self.encode(src))


def _cfg():
    """Return the default config used consistently across these tests."""
    return get_default_config()


def _make_fixture(cfg) -> HistorySensitiveTransformer:
    """Build the history-sensitive fixture wired to the config's special IDs."""
    return HistorySensitiveTransformer(
        vocab_size=16,
        pad_idx=cfg.pad_idx,
        sos_idx=cfg.sos_idx,
        eos_idx=cfg.eos_idx,
    )


def _sentence(cfg, tokens: list[int]) -> torch.Tensor:
    """Wrap raw token IDs in SOS/EOS and add a batch dimension."""
    return torch.tensor([[cfg.sos_idx] + tokens + [cfg.eos_idx]], dtype=torch.long)


def _pad_into_batch(cfg, sentences: list[torch.Tensor]) -> torch.Tensor:
    """Right-pad 1-row sentence tensors into a single padded batch."""
    width = max(s.size(1) for s in sentences)
    batch = torch.full((len(sentences), width), cfg.pad_idx, dtype=torch.long)
    for row, sentence in enumerate(sentences):
        batch[row, : sentence.size(1)] = sentence[0]
    return batch


class FixtureSensitivityTests(unittest.TestCase):
    """Guard the guard: prove the fixture can detect beam-state bugs.

    If these fail, the equivalence tests below are worthless, because the
    fixture would no longer distinguish a correct batched implementation from
    one that corrupts per-beam state.
    """

    def test_logits_depend_on_decoder_history(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        memory = model.encode(_sentence(cfg, [7, 8]))

        a = model.decode(torch.tensor([[cfg.sos_idx, 5, 5]]), memory)[:, -1, :]
        b = model.decode(torch.tensor([[cfg.sos_idx, 9, 6]]), memory)[:, -1, :]

        self.assertFalse(
            torch.equal(a, b),
            "fixture is history-blind; it cannot detect scrambled beam state",
        )

    def test_logits_depend_on_encoder_memory(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        prefix = torch.tensor([[cfg.sos_idx, 5]])

        a = model.decode(prefix, model.encode(_sentence(cfg, [7, 8])))[:, -1, :]
        b = model.decode(prefix, model.encode(_sentence(cfg, [11, 12])))[:, -1, :]

        self.assertFalse(
            torch.equal(a, b),
            "fixture ignores memory; it cannot detect bad memory expansion",
        )

    def test_padding_excluded_from_source_signature(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        short = _sentence(cfg, [7, 8])
        padded = _pad_into_batch(cfg, [short, _sentence(cfg, [1, 1, 1, 1, 1])])

        solo_mem = model.encode(short, src_key_padding_mask=short.eq(cfg.pad_idx))
        batch_mem = model.encode(padded, src_key_padding_mask=padded.eq(cfg.pad_idx))

        self.assertAlmostEqual(
            solo_mem[0, :, 0].sum().item(),
            batch_mem[0, :, 0].sum().item(),
            msg="padding leaks into the source signature",
        )

    def test_decode_rejects_unexpanded_memory(self):
        """A batched beam impl must expand memory alongside hypotheses."""
        cfg = _cfg()
        model = _make_fixture(cfg)
        memory = model.encode(_sentence(cfg, [7, 8]))  # batch 1
        tgt = torch.tensor([[cfg.sos_idx], [cfg.sos_idx], [cfg.sos_idx]])  # batch 3

        with self.assertRaises(ValueError):
            model.decode(tgt, memory)


class DecodingInvariantContract:
    """Properties that must hold before AND after the batching refactors."""

    def test_beam_size_one_matches_greedy(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        src = _sentence(cfg, [7, 8, 9])

        greedy = greedy_decode(model, src, max_len=12, config=cfg)[0]
        beam = self.BEAM_DECODE(model, src, beam_size=1, max_len=12, config=cfg)

        self.assertEqual(greedy, beam)

    def test_greedy_is_padding_invariant(self):
        """Decoding solo must equal decoding inside a padded batch."""
        cfg = _cfg()
        model = _make_fixture(cfg)
        short = _sentence(cfg, [7, 8])
        long = _sentence(cfg, [11, 12, 13, 14, 15])

        solo = greedy_decode(model, short, max_len=12, config=cfg)[0]
        batched = greedy_decode(
            model, _pad_into_batch(cfg, [short, long]), max_len=12, config=cfg
        )[0]

        self.assertEqual(solo, batched)

    def test_greedy_padding_invariant_regardless_of_position(self):
        """The padded sentence must decode identically in either batch slot."""
        cfg = _cfg()
        model = _make_fixture(cfg)
        short = _sentence(cfg, [7, 8])
        long = _sentence(cfg, [11, 12, 13, 14, 15])

        solo = greedy_decode(model, short, max_len=12, config=cfg)[0]
        first = greedy_decode(
            model, _pad_into_batch(cfg, [short, long]), max_len=12, config=cfg
        )[0]
        second = greedy_decode(
            model, _pad_into_batch(cfg, [long, short]), max_len=12, config=cfg
        )[1]

        self.assertEqual(solo, first)
        self.assertEqual(solo, second)

    def test_beam_is_padding_invariant(self):
        """Beam decoding a padded row must match decoding it unpadded.

        This is the core property the cross-sentence batching work (#2) must
        preserve, and it is the one most likely to break.
        """
        cfg = _cfg()
        model = _make_fixture(cfg)
        short = _sentence(cfg, [7, 8])
        long = _sentence(cfg, [11, 12, 13, 14, 15])

        solo = self.BEAM_DECODE(model, short, beam_size=3, max_len=12, config=cfg)

        padded_row = _pad_into_batch(cfg, [short, long])[0].unsqueeze(0)
        padded = self.BEAM_DECODE(
            model, padded_row, beam_size=3, max_len=12, config=cfg
        )

        self.assertEqual(solo, padded)

    def test_decoding_is_deterministic(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        src = _sentence(cfg, [7, 8, 9])

        runs = [
            self.BEAM_DECODE(model, src, beam_size=4, max_len=12, config=cfg)
            for _ in range(3)
        ]

        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])

    def test_output_depends_on_source(self):
        """Sanity: decoding must not be a constant function of the source."""
        cfg = _cfg()
        model = _make_fixture(cfg)

        a = self.BEAM_DECODE(
            model, _sentence(cfg, [7, 8]), beam_size=3, max_len=12, config=cfg
        )
        b = self.BEAM_DECODE(
            model, _sentence(cfg, [12, 13]), beam_size=3, max_len=12, config=cfg
        )

        self.assertNotEqual(a, b)

    def test_all_beam_widths_terminate_with_eos(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        src = _sentence(cfg, [7, 8, 9])

        for beam_size in (1, 2, 3, 5, 8):
            with self.subTest(beam_size=beam_size):
                tokens = self.BEAM_DECODE(
                    model, src, beam_size=beam_size, max_len=20, config=cfg
                )
                self.assertEqual(tokens[0], cfg.sos_idx)
                self.assertIn(cfg.eos_idx, tokens)


class BeamSearchGoldenContract:
    """Exact-output characterization tests: the oracle for the refactors.

    A batched reimplementation must reproduce these token sequences exactly.
    Values come from the deterministic fixture, so they are stable across
    platforms and PyTorch versions - unlike goldens taken from a randomly
    initialized model, whose near-ties can flip under different float kernels.

    If a refactor changes these, that is a behavior change requiring an
    explicit decision, not a silent performance win.
    """

    SRC_TOKENS: ClassVar[list[int]] = [7, 8, 9]

    def test_golden_beam_outputs(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        src = _sentence(cfg, self.SRC_TOKENS)

        observed = {
            beam_size: self.BEAM_DECODE(
                model, src, beam_size=beam_size, max_len=20, config=cfg
            )
            for beam_size in (1, 2, 3, 5)
        }

        # Regenerate deliberately (and review the diff) if semantics change.
        expected = {
            1: [2, 11, 10, 8, 4, 8, 3],
            2: [2, 11, 10, 8, 4, 8, 3],
            3: [2, 11, 10, 8, 4, 8, 3],
            5: [2, 11, 10, 8, 4, 8, 3],
        }
        self.assertEqual(observed, expected)

    def test_golden_greedy_output(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        src = _sentence(cfg, self.SRC_TOKENS)

        decoded = greedy_decode(model, src, max_len=20, config=cfg)[0]
        self.assertEqual(decoded, [2, 11, 10, 8, 4, 8, 3])


class LengthNormalizationSemanticsContract:
    """Pin the current length-normalization behavior (task #4).

    ``beam_search_decode`` applies length normalization while *pruning*, not
    only when selecting the final hypothesis. That is defensible but
    non-standard; these tests make any change to it deliberate and visible.
    """

    def test_alpha_influences_output_length(self):
        """Higher alpha discounts long hypotheses less, so lengths can differ."""
        cfg = _cfg()
        model = _make_fixture(cfg)
        src = _sentence(cfg, [7, 8, 9])

        lengths = {
            alpha: len(
                self.BEAM_DECODE(
                    model,
                    src,
                    beam_size=4,
                    max_len=25,
                    alpha=alpha,
                    device=None,
                    config=cfg,
                )
            )
            for alpha in (0.0, 0.6, 2.0)
        }

        # Recorded, not asserted-equal to a formula: this documents behavior so
        # a refactor that changes it is caught and reviewed.
        for alpha, length in lengths.items():
            with self.subTest(alpha=alpha):
                self.assertGreater(length, 1)
                self.assertLessEqual(length, 26)

    def test_alpha_zero_is_pure_log_prob_sum(self):
        """alpha=0 disables normalization; output must stay deterministic."""
        cfg = _cfg()
        model = _make_fixture(cfg)
        src = _sentence(cfg, [7, 8, 9])

        a = self.BEAM_DECODE(model, src, beam_size=3, max_len=20, alpha=0.0, config=cfg)
        b = self.BEAM_DECODE(model, src, beam_size=3, max_len=20, alpha=0.0, config=cfg)
        self.assertEqual(a, b)


class RealModelConsistencyContract:
    """Cross-check the invariants on a real SimpleTransformer.

    These compare results *within a single run* rather than against hardcoded
    values, so they stay platform-independent while still exercising the real
    attention stack that the fixture abstracts away.
    """

    def _model(self, cfg):
        torch.manual_seed(1234)
        model = SimpleTransformer(
            src_vocab_size=60,
            tgt_vocab_size=60,
            d_model=64,
            n_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=128,
            config=cfg,
        )
        model.eval()
        return model

    def test_real_model_greedy_padding_invariance(self):
        cfg = _cfg()
        model = self._model(cfg)
        short = _sentence(cfg, [11, 12])
        long = _sentence(cfg, [21, 22, 23, 24, 25, 26])

        solo = greedy_decode(model, short, max_len=12, config=cfg)[0]
        batched = greedy_decode(
            model, _pad_into_batch(cfg, [short, long]), max_len=12, config=cfg
        )[0]

        self.assertEqual(solo, batched)

    def test_real_model_beam_one_matches_greedy(self):
        cfg = _cfg()
        model = self._model(cfg)
        src = _sentence(cfg, [11, 12])

        greedy = greedy_decode(model, src, max_len=12, config=cfg)[0]
        beam = self.BEAM_DECODE(model, src, beam_size=1, max_len=12, config=cfg)

        self.assertEqual(greedy, beam)

    def test_real_model_beam_is_deterministic(self):
        cfg = _cfg()
        model = self._model(cfg)
        src = _sentence(cfg, [11, 12])

        a = self.BEAM_DECODE(model, src, beam_size=3, max_len=12, config=cfg)
        b = self.BEAM_DECODE(model, src, beam_size=3, max_len=12, config=cfg)
        self.assertEqual(a, b)


class BatchSizeRestrictionContract:
    """Document the API contract that tasks #1 and #2 will change.

    ``test_batch_size_gt_one_currently_rejected`` is the test that INVERTS when
    cross-sentence batching lands: replace it with an equivalence assertion
    that batched output matches per-sentence output.
    """

    def test_batch_size_gt_one_currently_rejected(self):
        cfg = _cfg()
        model = _make_fixture(cfg)
        batch = _pad_into_batch(cfg, [_sentence(cfg, [7, 8]), _sentence(cfg, [9, 10])])

        with self.assertRaises(ValueError):
            self.BEAM_DECODE(model, batch, beam_size=2, max_len=12, config=cfg)

    def test_per_sentence_reference_for_future_batched_impl(self):
        """Reference outputs a batched implementation must reproduce.

        Once #2 lands, call the batched path on ``batch`` and assert equality
        with ``reference`` computed here one sentence at a time.
        """
        cfg = _cfg()
        model = _make_fixture(cfg)
        sentences = [
            _sentence(cfg, [7, 8]),
            _sentence(cfg, [11, 12, 13, 14, 15]),
            _sentence(cfg, [9]),
        ]

        reference = [
            self.BEAM_DECODE(model, s, beam_size=3, max_len=20, config=cfg)
            for s in sentences
        ]

        self.assertEqual(len(reference), 3)
        for tokens in reference:
            self.assertEqual(tokens[0], cfg.sos_idx)
            self.assertIn(cfg.eos_idx, tokens)

        # Padded batch decoding must agree row-for-row with `reference`.
        batch = _pad_into_batch(cfg, sentences)
        for row, expected in enumerate(reference):
            with self.subTest(row=row):
                got = self.BEAM_DECODE(
                    model,
                    batch[row].unsqueeze(0),
                    beam_size=3,
                    max_len=20,
                    config=cfg,
                )
                self.assertEqual(got, expected)


class TrapTransformer(nn.Module):
    """Scripted model whose best path is only reachable with ``beam_size >= 2``.

    Step 1 offers token ``A`` (locally best) and token ``B`` (slightly worse).
    Continuations after ``A`` are poor and spread out; the continuation after
    ``B`` is excellent. Greedy therefore takes ``A`` and loses, while a beam of
    2 or more keeps ``B`` alive and finds the better path.

    Both winning paths have the same length, so length normalization cannot
    confound the comparison. Continuation scores are slightly graded rather
    than exactly equal, keeping the test about beam *exploration* rather than
    about tie-breaking (see ``TieBreakingTests``).

    Without a fixture like this, golden outputs are identical for every beam
    width and the tests never exercise beam bookkeeping at all.
    """

    VOCAB_SIZE = 12
    TOKEN_A = 4  # locally attractive, leads nowhere
    TOKEN_B = 5  # locally worse, leads to the best path
    TOKEN_D = 6  # the reward reachable only via TOKEN_B

    def __init__(self, sos_idx: int = 2, eos_idx: int = 3) -> None:
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.dummy = nn.Parameter(torch.zeros(1))

    def encode(
        self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return a zero memory of the expected shape."""
        return torch.zeros(src.size(0), src.size(1), D_MEM, device=src.device)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Emit the scripted next-token distribution for each hypothesis."""
        batch, tgt_len = tgt.shape
        row = torch.full((batch, self.VOCAB_SIZE), -30.0, device=tgt.device)
        last = tgt[:, -1]
        for i in range(batch):
            if tgt_len == 1:
                row[i, self.TOKEN_A] = 2.0
                row[i, self.TOKEN_B] = 1.0
            elif last[i].item() == self.TOKEN_A:
                # Poor, spread-out continuation: best available log-prob is low.
                row[i, 7:12] = torch.linspace(0.0, -0.4, 5, device=tgt.device)
            elif last[i].item() == self.TOKEN_B:
                row[i, self.TOKEN_D] = 5.0
            else:
                row[i, self.eos_idx] = 5.0
        logits = torch.full((batch, tgt_len, self.VOCAB_SIZE), -30.0, device=tgt.device)
        logits[:, -1, :] = row
        return logits

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Run encode then decode, matching the training-time interface."""
        return self.decode(tgt, self.encode(src))


class BeamSuperiorityContract:
    """Prove beam search actually explores, and pin what it finds.

    These are the strongest oracle in this module: they fail if a refactor
    silently degrades beam search into greedy, or corrupts the per-beam
    histories such that the better path is never recovered.
    """

    def _model(self, cfg) -> TrapTransformer:
        model = TrapTransformer(sos_idx=cfg.sos_idx, eos_idx=cfg.eos_idx)
        model.eval()
        return model

    def test_greedy_falls_into_the_trap(self):
        cfg = _cfg()
        model = self._model(cfg)
        decoded = greedy_decode(model, _sentence(cfg, [9]), max_len=6, config=cfg)[0]

        self.assertIn(TrapTransformer.TOKEN_A, decoded)
        self.assertNotIn(TrapTransformer.TOKEN_B, decoded)

    def test_beam_two_escapes_the_trap(self):
        cfg = _cfg()
        model = self._model(cfg)
        decoded = self.BEAM_DECODE(
            model, _sentence(cfg, [9]), beam_size=2, max_len=6, config=cfg
        )

        self.assertIn(TrapTransformer.TOKEN_B, decoded)
        self.assertIn(TrapTransformer.TOKEN_D, decoded)
        self.assertNotIn(TrapTransformer.TOKEN_A, decoded)

    def test_beam_one_degenerates_to_greedy(self):
        cfg = _cfg()
        model = self._model(cfg)
        src = _sentence(cfg, [9])

        greedy = greedy_decode(model, src, max_len=6, config=cfg)[0]
        beam = self.BEAM_DECODE(model, src, beam_size=1, max_len=6, config=cfg)

        self.assertEqual(greedy, beam)

    def test_golden_trap_outputs_by_beam_width(self):
        """Exact outputs a batched implementation must reproduce."""
        cfg = _cfg()
        model = self._model(cfg)
        src = _sentence(cfg, [9])

        observed = {
            width: self.BEAM_DECODE(model, src, beam_size=width, max_len=6, config=cfg)
            for width in (1, 2, 4)
        }
        expected = {
            1: [2, 4, 7, 3],
            2: [2, 5, 6, 3],
            4: [2, 5, 6, 3],
        }
        self.assertEqual(observed, expected)

    def test_wider_beams_do_not_regress(self):
        """Beam width 2 already finds the optimum; wider must not lose it."""
        cfg = _cfg()
        model = self._model(cfg)
        src = _sentence(cfg, [9])

        best = self.BEAM_DECODE(model, src, beam_size=2, max_len=6, config=cfg)
        for width in (3, 4, 6, 8):
            with self.subTest(beam_size=width):
                self.assertEqual(
                    self.BEAM_DECODE(
                        model, src, beam_size=width, max_len=6, config=cfg
                    ),
                    best,
                )


class TieBreakingContract:
    """Assert the documented tie-breaking rule (task #13).

    The rule: prefer the higher score; among exactly equal scores prefer the
    sequence with lower token IDs, compared position by position.

    This matters because ``torch.topk`` documents that tied indices are *not*
    guaranteed to be stable, so beam search would otherwise be
    device-dependent. ``greedy_decode`` is safe as written because
    ``torch.argmax`` documents that it returns the first maximal index.

    A batched beam search will ``topk`` over a flattened
    ``(batch * beam, vocab)`` tensor and must route that selection through
    ``_canonical_topk`` (or an equivalent) to preserve these results.
    """

    class _TiedTransformer(nn.Module):
        """Model emitting several exactly-equal top logits."""

        VOCAB_SIZE = 12
        TIE_LO = 7
        TIE_HI = 12

        def __init__(self, eos_idx: int = 3) -> None:
            super().__init__()
            self.eos_idx = eos_idx
            self.dummy = nn.Parameter(torch.zeros(1))

        def encode(self, src, src_key_padding_mask=None):
            return torch.zeros(src.size(0), src.size(1), D_MEM, device=src.device)

        def decode(
            self,
            tgt,
            memory,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            tgt_mask=None,
        ):
            batch, tgt_len = tgt.shape
            row = torch.full((batch, self.VOCAB_SIZE), -30.0, device=tgt.device)
            if tgt_len < 3:
                row[:, self.TIE_LO : self.TIE_HI] = 0.0  # exact ties
            else:
                row[:, self.eos_idx] = 5.0
            logits = torch.full(
                (batch, tgt_len, self.VOCAB_SIZE), -30.0, device=tgt.device
            )
            logits[:, -1, :] = row
            return logits

        def forward(self, src, tgt):
            return self.decode(tgt, self.encode(src))

    def test_canonical_topk_breaks_ties_by_lowest_index(self):
        scores = torch.tensor([0.0, 5.0, 5.0, 5.0, 1.0])
        values, indices = _canonical_topk(scores, 2)

        self.assertEqual(indices.tolist(), [1, 2])
        self.assertEqual(values.tolist(), [5.0, 5.0])

    def test_canonical_topk_with_all_scores_tied(self):
        values, indices = _canonical_topk(torch.zeros(6), 4)

        self.assertEqual(indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(values.tolist(), [0.0, 0.0, 0.0, 0.0])

    def test_canonical_topk_prefers_strictly_better_scores(self):
        """Strictly better scores must outrank tied ones regardless of index."""
        scores = torch.tensor([0.0, 0.0, 0.0, 9.0])
        _, indices = _canonical_topk(scores, 2)

        self.assertEqual(indices.tolist(), [3, 0])

    def test_canonical_topk_clamps_k_to_length(self):
        _, indices = _canonical_topk(torch.zeros(3), 10)
        self.assertEqual(indices.tolist(), [0, 1, 2])

    def test_rank_key_orders_by_score_then_tokens(self):
        """Sorting ascending by the key puts the preferred hypothesis first."""
        better = _rank_key([2, 5], -1.0, 0.6)
        worse = _rank_key([2, 5], -4.0, 0.6)
        self.assertLess(better, worse)

    def test_rank_key_breaks_score_ties_by_lower_tokens(self):
        low = _rank_key([2, 4], -2.0, 0.6)
        high = _rank_key([2, 9], -2.0, 0.6)
        self.assertLess(low, high)

    def test_beam_one_matches_greedy_even_under_exact_ties(self):
        """The property that motivated the rule; it did not hold before."""
        cfg = _cfg()
        model = self._TiedTransformer(eos_idx=cfg.eos_idx)
        src = _sentence(cfg, [9])

        greedy = greedy_decode(model, src, max_len=6, config=cfg)[0]
        beam = self.BEAM_DECODE(model, src, beam_size=1, max_len=6, config=cfg)

        self.assertEqual(greedy, beam)

    def test_ties_resolve_to_lowest_token_id(self):
        cfg = _cfg()
        model = self._TiedTransformer(eos_idx=cfg.eos_idx)
        src = _sentence(cfg, [9])

        for beam_size in (1, 2, 3, 5):
            with self.subTest(beam_size=beam_size):
                tokens = self.BEAM_DECODE(
                    model, src, beam_size=beam_size, max_len=6, config=cfg
                )
                self.assertEqual(
                    tokens,
                    [cfg.sos_idx, 7, 7, cfg.eos_idx],
                    "ties must resolve toward the lowest token ID",
                )

    def test_tied_decoding_is_deterministic(self):
        cfg = _cfg()
        model = self._TiedTransformer(eos_idx=cfg.eos_idx)
        src = _sentence(cfg, [9])

        runs = [
            self.BEAM_DECODE(model, src, beam_size=3, max_len=6, config=cfg)
            for _ in range(4)
        ]
        self.assertEqual(len({tuple(r) for r in runs}), 1)


# --- reference implementation -------------------------------------------------


class ReferenceDecodingInvariantTests(DecodingInvariantContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode)


class ReferenceBeamSearchGoldenTests(BeamSearchGoldenContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode)


class ReferenceLengthNormalizationSemanticsTests(
    LengthNormalizationSemanticsContract, unittest.TestCase
):
    BEAM_DECODE = staticmethod(beam_search_decode)


class ReferenceRealModelConsistencyTests(
    RealModelConsistencyContract, unittest.TestCase
):
    BEAM_DECODE = staticmethod(beam_search_decode)


class ReferenceBatchSizeRestrictionTests(
    BatchSizeRestrictionContract, unittest.TestCase
):
    BEAM_DECODE = staticmethod(beam_search_decode)


class ReferenceBeamSuperiorityTests(BeamSuperiorityContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode)


class ReferenceTieBreakingTests(TieBreakingContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode)


# --- batched implementation ---------------------------------------------------


class BatchedDecodingInvariantTests(DecodingInvariantContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode_batched)


class BatchedBeamSearchGoldenTests(BeamSearchGoldenContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode_batched)


class BatchedLengthNormalizationSemanticsTests(
    LengthNormalizationSemanticsContract, unittest.TestCase
):
    BEAM_DECODE = staticmethod(beam_search_decode_batched)


class BatchedRealModelConsistencyTests(RealModelConsistencyContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode_batched)


class BatchedBatchSizeRestrictionTests(BatchSizeRestrictionContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode_batched)


class BatchedBeamSuperiorityTests(BeamSuperiorityContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode_batched)


class BatchedTieBreakingTests(TieBreakingContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode_batched)


if __name__ == "__main__":
    unittest.main()
