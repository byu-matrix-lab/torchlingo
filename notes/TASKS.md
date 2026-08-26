# TorchLingo — Session Task List

Captured 2026-08-22. Numbered for reference in conversation.

## Code — decoding performance

### Design decision (2026-08-22): reference and fast implementations live side by side

The optimized decoders are **added alongside** the simple ones, not layered into them.
The existing `greedy_decode` / `beam_search_decode` stay as the readable reference a
student can follow line by line; batching and caching go in separate, clearly named
implementations.

*Why:* readability is the reason this library exists. A batched, KV-cached beam search is
necessarily harder to read than the 85-line version — index bookkeeping across
`(batch x beam)`, cache invalidation, ragged completion. Folding that into the one
implementation trades away the thing the repo is for, to buy speed that only matters at
scales students often are not working at anyway.

*What this unlocks:* #3 (KV cache) was previously marked "decide whether to do it at all,
since it may compromise readability." That constraint is gone. The fast path can be as
dense as it needs to be, because the readable path is preserved. #3 moves from
questionable to straightforwardly worth doing.

*What this demands:* two implementations silently diverging is the obvious failure mode.
`tests/test_decoding_equivalence.py` already covers this — it was written as a
characterization oracle for a refactor, but the natural reading is now stronger:

> The simple implementation is the **specification**. The fast implementation must
> produce token-identical output on every fixture in that module.

Every fast variant should be run against the same fixtures as the reference, ideally
parameterized so adding an implementation automatically inherits the whole suite.

*Resolved 2026-08-26:*

**Module layout — a flat sibling module, `src/torchlingo/inference_fast.py`.**
`inference.py` keeps the reference decoders and the shared helpers (`_canonical_topk`,
`_rank_key`) and is not touched by the optimization work. Matches the repo's existing
flat-module convention (`config.py`, `training.py`, `evaluation.py`); subpackages are
reserved for places with several peers (`models/`, `preprocessing/`). Rejected: putting
both in `inference.py`, which would push it past 700 lines and defeat the split;
and an `inference/` subpackage, which makes a reader navigate a directory to find an
85-line function.

**`translate_batch` — mirrored, not switched.** The reference wrapper stays as-is;
`inference_fast.py` gets its own `translate_batch_fast`. This keeps the dependency arrow
one-way: **fast imports from reference, never the reverse.** A selector parameter or a
fast-by-default wrapper would force `inference.py` to import `inference_fast.py`,
coupling the module a student is meant to read to the one they are not.

**Guidance — three layers, because docs alone will not catch the failure case.**
1. `docs/docs/concepts/decoding.md`: reference vs fast, carrying the measurement
   (38.7x the `decode()` calls for 5.0x the math, every call at batch size 1).
2. Bidirectional docstring cross-references between each implementation and its
   counterpart.
3. A threshold-based, once-per-process `warnings.warn` when the reference path is used
   on a large input, naming `translate_batch_fast` and noting the output is identical.
   This is the layer that actually works: it fires at the moment of pain, whereas the
   student who most needs it is mid-experiment and not reading docs.

**Test structure — shared contract base class, one subclass per implementation.**
Lift the fixtures and invariants in `test_decoding_equivalence.py` into a
`DecoderContractTests` mixin with the decode callable supplied by each subclass:

```python
class DecoderContractTests:          # not a TestCase itself
    DECODE = None
    # ...every fixture and invariant...

class ReferenceBeamTests(DecoderContractTests, unittest.TestCase):
    DECODE = staticmethod(beam_search_decode)

class BatchedBeamTests(DecoderContractTests, unittest.TestCase):
    DECODE = staticmethod(beam_search_decode_batched)
```

Adding an implementation is one subclass and it inherits the whole suite; failures name
the implementation, so a divergence is unambiguous. The reference is the specification
and runs on every invocation, which is what keeps it from rotting into a museum piece.

### The 38.7x is two factors, not one

Worth stating plainly, because it was originally recorded here in a way that invited
over-reading. The measurement compared reference beam search against *greedy*, and greedy
was already batched across sentences. So it captured two independent inefficiencies
multiplied together:

```
reference beam vs greedy :  38.7x
  of which, beam axis    :   4.8x   <- one decode() call per beam, per step   (#1)
  of which, sentence axis:   8.0x   <- one sentence at a time                 (#2)
  product                :  38.7x
```

With `beam_size=5` and 8 sentences, 5 x 8 = 40 ~= 38.7 — the two factors are just the
beam width and the sentence count. **#1 recovers roughly `beam_size`; the rest needs #2.**
Neither task alone was ever going to deliver 38.7x, and quoting that figure against a
single one of them is misleading.

**#1 Batch beam search across beams** — *DONE (aabb260)*
Stack the live beams into one `(n_live, t)` tensor, expand `memory` as a view, issue one
`model.decode()` call per step instead of one per beam.
- Implemented as `beam_search_decode_batched` in `src/torchlingo/inference_fast.py`.
- Delivered: 125/125 token-identical across 25 sources x 5 beam widths; `decode()` calls
  121 -> 25 (4.8x fewer); wall clock 140.9 -> 39.1 ms/sentence (3.6x) on a small CPU model.
- The wall-clock gain trails the call-count gain because eliminating calls does not
  eliminate per-step Python bookkeeping, the per-row `_canonical_topk`, or `log_softmax`
  — and each surviving call now does 4.8x more work, so it is not free.
- Key enabling fact: every live beam always has the same length, so they stack with no
  padding at all.

**#2 Batch beam search across sentences**
Remove the batch-size-1 restriction in `inference_fast.py`; flatten to `(batch x k, t)`.
**This is where the remaining ~8x lives** (the sentence axis above) — and it scales with
the number of sentences decoded, so it matters more on a real test set than #1 does.
- Files: `src/torchlingo/inference_fast.py` (the raise, and the per-row loop in
  `translate_batch_fast`)
- Hard part is bookkeeping for ragged completion — sentences finishing at different steps.
- Needs a contract adapter: it takes a batch rather than one sentence, so it does not slot
  into the current `BEAM_DECODE` shape unchanged.
- `tests/test_training_inference.py:502`
  (`test_beam_search_decode_raises_on_batch_size_gt_one`) stays valid: under the
  side-by-side design the *reference* implementation keeps that restriction. The batched
  variant gets its own tests rather than inverting this one.

**#3 Incremental decoding / KV cache**
Removes the O(L^2) prefix recomputation. Independent of the two axes above: it reduces the
work *inside* each call rather than the number of calls.
- ~~DECISION NEEDED: may compromise the readability that makes this repo worth using for
  teaching. Consider stopping at #2 for an educational library.~~ **Resolved by the
  side-by-side design above:** the reference implementation stays readable regardless, so
  the fast path is free to be dense. Worth doing.

**#4 Resolve length-normalization semantics**
`inference.py:203` applies length normalization during *pruning*, not only at final
selection — comparing normalized scores across different lengths mid-search. Defensible
but non-standard. Preserve exactly during #1/#2 so perf work stays reviewable; raise as
a separate question.

## Code — other gaps

**#5 Add attention to the LSTM decoder**
`models/lstm_simple.py:117` notes attention masks are "currently unused." Blocks the
classic Bahdanau/Luong lesson and the with-vs-without ablation.

**#6 Multi-GPU training via DDP**
Not implemented. `config.py:663` states multi-GPU "requires custom DataParallel setup."

**#7 Address PyTorch deprecation warnings**
Surfaced while benchmarking on torch 2.13.0:
- "Support for mismatched key_padding_mask and attn_mask is deprecated" — raised from
  the decode path, will eventually break.
- Nested-tensor prototype warning from `nn.Transformer`.

## Evaluation / tooling

**#8 Verify Eole claims hands-on before syllabus use**
Specifically: COMET/MetricX integration in the training loop, and 7B-13B finetuning on a
single 24GB GPU. Both are from Eole's README, not from running it.
- Needs a separate venv: Eole requires Python >= 3.11 and torch >= 2.10, **< 2.13**.
  This repo's `.venv` has torch 2.13.0.

**#9 Run `pre-commit install`**
`.pre-commit-config.yaml` exists in the repo but hooks are not installed in this clone.

## Process decisions

**#10 Decide whether `notes/` is personal scratch or upstream-facing**
Now tracked on branch `notes/opennmt-eole-comparison`. Candid framing confirmed OK for
notes. Changes how future notes are written if they become public.

**#11 Decide push / PR for `notes/opennmt-eole-comparison`**
Branch is local only; commit `bd9cd9d` has not been pushed to
`byu-matrix-lab/torchlingo`.

## Possible tooling to productize

**#12 Decode benchmark harness**
The ad-hoc scripts written this session (greedy-vs-beam timing, `decode()` call counting
via monkeypatched instrumentation) would be worth a reusable
`scripts/bench_decode.py` — needed again to validate #1, #2, and #3, and useful as a
student exercise in its own right.

---

## Added 2026-08-22 (during test-suite review)

**#13 Specify decoding tie-breaking behavior** — *DONE (5b3e326)*
Under exactly-tied logits, `greedy_decode` (`argmax`) and `beam_search_decode`
(`topk` + Python sort on length-normalized scores) can select different tokens, so
`beam_size=1` is not guaranteed to equal greedy. A batched implementation will `topk`
over a flattened `(batch * beam, vocab)` tensor and will likely break ties differently
again. Decide the intended rule and assert it, before golden outputs are relied upon.
- **Resolved.** Rule adopted: *prefer the higher score; among exactly equal scores
  prefer the sequence with lower token IDs, position by position.* Enforced by
  `_canonical_topk()` and `_rank_key()` in `inference.py`; asserted by
  `tests/test_decoding_equivalence.py::TieBreakingTests` (9 tests).
- Matches prior CPU behavior, so no golden outputs changed. `beam_size=1` now provably
  equals greedy under exact ties, which was not previously guaranteed.
- **Carry-over for #1/#2:** a batched implementation must route its
  `(batch * beam, vocab)` selection through `_canonical_topk` or an equivalent, or it
  will break ties differently and silently change output.

**#14 Ruff version drift** — *DONE (branch `chore/ruff-modernization`)*

*Original diagnosis was wrong and is corrected here.* I first recorded this as "the repo
is not ruff-clean at HEAD," based on 354 fixes across 28 files. In fact the repo was
**already lint-clean** under the ruff version it pinned: `ruff 0.7.0 check src tests`
reported **zero** errors. The findings were an artifact of running 0.16.3, whose default
rule set is far wider.

The real defect was that nothing kept the versions in agreement, and nothing enforced
either one:
- `.pre-commit-config.yaml` pinned `v0.7.0`
- `pyproject.toml` `[dev]` asked for `ruff>=0.12`
- CI ran no lint step at all

So a fresh `pip install -e ".[dev]"` gave a ruff that disagreed with pre-commit on
hundreds of findings, and CLAUDE.md's "ALWAYS run ruff" then produced a 28-file diff
unrelated to the actual work — a trap for students and contributors.

**Resolved in five commits** on its own branch/PR, deliberately separable:
1. format under the pinned 0.7.0 (4 files had drifted), establishing a clean
   baseline so the modernization diff contains only real rule changes.
2. ruff 0.16 safe auto-fixes (358): PEP 604/585 annotations, import sorting.
3. implicit `Optional` made explicit (69 via RUF013).
4. findings needing judgment: narrowed bare `except Exception` to
   `ImportError` / `(AttributeError, TypeError)`, bound a loop variable in a lambda
   (B023), collapsed a nested `if` preserving short-circuit order, `ClassVar` on
   `_FIELD_VALIDATORS`, plus 13 more implicit `Optional`s that **RUF013 does not flag**
   (`config: Config = None`). RUF059 ignored under `tests/` — naming every element of a
   returned tuple documents the callee's contract.
5. pin `ruff==0.16.3` in both places with cross-referencing comments; add a
   CI `lint` job that reads the pinned spec out of `pyproject.toml` at runtime rather
   than hardcoding a third copy; `build` now needs `[tests, lint]`.

This work is split into its own PR (`chore/ruff-modernization`, branched from `main`)
so its 32-file mechanical diff does not bury the decoding work in review. The decoding
branch is stacked on top of it and should merge second.

`src` and `tests` are clean under both `ruff check` and `ruff format --check`. Full suite
unchanged throughout: 459 tests, OK (21 skipped).

**#15 Existing `DummyTransformer` is history- and memory-blind**
`tests/test_training_inference.py:16` computes logits from a zero tensor, so its output
depends only on decoding *position*. Verified: logits are byte-identical for different
decoder histories AND for different encoder memories. The three existing beam tests
therefore cannot detect scrambled beam state or bad memory expansion.
- Mitigated by the new `tests/test_decoding_equivalence.py`, but the old tests should
  eventually migrate to the history-sensitive fixture rather than sitting alongside it.
