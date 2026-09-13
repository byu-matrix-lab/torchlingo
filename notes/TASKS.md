# TorchLingo — Session Task List

Opened 2026-08-22, last updated 2026-09-10. Numbered for reference in conversation.

## Status

| | Task | State |
|---|---|---|
| #16 | Release pipeline broken — nothing ships | Open |
| #2 | Batch beam search across sentences (~8x, scales with test-set size) | Open |
| #3 | Incremental decoding / KV cache | Open |
| #4 | Resolve length-normalization semantics | Open |
| #6 | Multi-GPU training via DDP | Open |
| #7 | PyTorch deprecation warnings | Open |
| #8 | Verify Eole claims before syllabus use | Open |
| #9 | `pre-commit install` (still not installed) | Open |
| #12 | Decode benchmark harness | Open |
| #15 | Migrate history-blind `DummyTransformer` tests | Open |
| #21 | Beam search does not support LSTM models at all | Open |
| #22 | `examples/` and `scripts/` are outside the lint gate | Open |
| #26 | Broken doc links block `mkdocs --strict` | Open |
| #27 | Attention tutorial notebook | Open |
| #28 | Attention params skip `_init_weights` | Open |
| #29 | Recover the last 98 talks with a sentence aligner | Open |
| #30 | Nothing executes the tutorial notebooks | Open |
| #31 | Tutorial 03 soft-fails into a `NameError` | Open |

**Shipped to `main` but unreleased:** PR #7 (ruff pinned + CI lint gate, `3dca4d1`),
PR #8 (decoding oracle suite, tie-breaking rule, 3.6x batched beam search, `ff03631`),
and PR #9 (decoder naming schema, greedy default documented). See #16.

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
`inference_fast.py` gets its own `translate_batch`. This keeps the dependency arrow
one-way: **fast imports from reference, never the reverse.** A selector parameter or a
fast-by-default wrapper would force `inference.py` to import `inference_fast.py`,
coupling the module a student is meant to read to the one they are not.

**Guidance — three layers, because docs alone will not catch the failure case.**
1. `docs/docs/concepts/decoding.md`: reference vs fast, carrying the measurement
   (38.7x the `decode()` calls for 5.0x the math, every call at batch size 1).
2. Bidirectional docstring cross-references between each implementation and its
   counterpart.
3. A threshold-based, once-per-process `warnings.warn` when the reference path is used
   on a large input, naming `inference_fast.translate_batch` and noting the output is identical.
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
    DECODE = staticmethod(inference_fast.beam_search_decode)
```

Adding an implementation is one subclass and it inherits the whole suite; failures name
the implementation, so a divergence is unambiguous. The reference is the specification
and runs on every invocation, which is what keeps it from rotting into a museum piece.

### The 38.7x is a budget split across two levers

Batching offers ~38.7x fewer model calls, but as two independent levers whose effects
multiply — worth recording, because the figure was originally quoted here as if any one
task could deliver it:

| Lever | Worth | Task |
|---|---|---|
| Batch across beams | ~`beam_size` | #1 |
| Batch across sentences | ~`num_sentences` | #2 |

`beam_size=5` x 8 sentences = 40 ~= 38.7. #1 recovers roughly `beam_size`; the rest needs
#2, which is worth more the larger the test set.

Full explanation for students lives in `docs/docs/concepts/decoding.md` — keep it there
rather than duplicating it into code and notes.

**#1 Batch beam search across beams** — *DONE (aabb260)*
Stack the live beams into one `(n_live, t)` tensor, expand `memory` as a view, issue one
`model.decode()` call per step instead of one per beam.
- Implemented as `inference_fast.beam_search_decode` in `src/torchlingo/inference_fast.py`.
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
  `inference_fast.translate_batch`)
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

**#5 Add attention to the LSTM decoder** — *DONE (2026-09-10)*
The case was stronger than this entry originally recorded. `docs/concepts/what-is-nmt.md`
taught attention as a concept and then its summary table said
`| Attention | Built into Transformer layers |` — so the single most important idea in
modern NMT was the one core concept with **no readable implementation anywhere** in a
library whose entire reason for existing is readable implementations.

**Decisions (user, 2026-09-10):**
- **Both scorers, selectable** via `attn_type="dot"` (Luong 2015) / `"additive"`
  (Bahdanau 2014), so the 2014 → 2015 → 2017 progression is teachable directly.
- **`attention=False` by default.** Preserves existing behavior and checkpoints, and
  makes the ablation one visible flag rather than hidden history.
- **Weights are returned and plottable** — the payoff, not an extra.

**Delivered:**
- `src/torchlingo/models/attention.py`: `DotProductAttention` (parameter-free),
  `AdditiveAttention`, `build_attention`. Masked softmax uses `finfo.min`, not `-inf`,
  so a fully-padded row degrades to uniform instead of `NaN`.
- `SimpleSeq2SeqLSTM(attention=, attn_type=)`, plus `encode_source()` / `decode_step()`
  and `forward(..., return_attention=True)`.
- `src/torchlingo/visualization.py`: `plot_attention` (matplotlib) and
  `format_attention` (dependency-free shaded text grid).
- `matplotlib>=3.7` added to core `dependencies` at the user's direction.
- Config: `lstm_attention`, `lstm_attn_type`, with `ATTENTION_TYPES` as the single
  source of truth. It lives in `config.py`, not `models/attention.py`, because `Config`
  validates the field while it is still importing and `models` imports `Config` — the
  natural placement is circular. Caught at runtime, not guessed.
- `tests/test_attention.py`: 38 tests. Suite 527, OK (21 skipped).
- Docs: new `reference/models/attention.md` and `reference/visualization.md`, nav
  entries, and corrections to `reference/models/lstm.md` (see below) and
  `concepts/what-is-nmt.md`.

**`examples/attention_alignment.py`** trains on a task whose correct alignment is known
in advance — target is the source translated word-for-word and reversed, so the truth is
an anti-diagonal. That converts "did attention work?" into a measurable number:

| configuration | val loss | alignment accuracy |
|---|---|---|
| no attention | 1.1031 | n/a |
| dot (Luong) | 0.6801 | 98.4% |
| additive (Bahdanau) | 0.6533 | 93.6% |

Chance is ~12.5%. Synthetic on purpose — see #20, the repo has no usable parallel corpus.

**Two docs corrections worth noting**, because both were actively misleading:
- `reference/models/lstm.md` claimed "No attention: this simple model doesn't use
  attention ... This is not implemented in TorchLingo's simple model."
- The same page's greedy-decoding example hand-rolled the decoder loop from the final
  `(h, c)`, discarding encoder outputs. On an attention model that **silently disables
  attention** while appearing to work. Replaced with `greedy_decode` / `decode_step`,
  and a warning box explaining the trap.

**#23 The encoder consumed padding** — *DONE, found while doing #5*
A test asserting "trailing padding must not change the output" failed even with attention
masking correct. The leak was upstream of attention and predates it: `encode_source` ran
the encoder LSTM across `PAD`, so the `(h, c)` handed to the decoder described the padding
rather than the sentence — meaning a sentence decoded in a batch differed from the same
sentence decoded alone. Fixed with `pack_padded_sequence`. Masking attention weights alone
does not fix this; the leak is in the recurrence, not the alignment.

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

**#10 Decide whether `notes/` is personal scratch or upstream-facing** — *DONE*
Resolved: upstream-facing and tracked. Merged to `main` via PR #8. Candid framing
confirmed appropriate, and presented on its merits rather than hedged.

**#11 Decide push / PR for `notes/opennmt-eole-comparison`** — *DONE*
Resolved: pushed and merged as PR #8 (`ff03631`), approved by Coulson-Rich.

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

---

## Added 2026-09-10

**#16 The release pipeline is broken — nothing since Feb 2026 has shipped**

*Downgraded from BLOCKING on 2026-09-10:* nobody is installing from PyPI yet, so this is
a latent breakage rather than an active one. Still must be fixed before the first
classroom install, and the tag-vs-`pyproject` CI check should land **before** the next
tag so the mismatch fails loudly instead of silently for a third time.

Found while reviewing backlog status. `pyproject.toml` has said `version = "0.0.8"`
since February and is never bumped, so tagging a release builds a stale-version artifact:

```
pyproject.toml version   0.0.8      (unchanged since Feb 2026)
latest PyPI release      0.0.8      (uploaded 2026-02-18)
GitHub tags              v0.1.0, v0.1.1
  v0.1.0 assets          torchlingo-0.0.7-*.whl   <- tag says 0.1.0, artifact says 0.0.7
  v0.1.1 assets          (none)                   <- build or publish failed silently
```

PyPI rejects duplicate versions, so a build that produces `0.0.8` when `0.0.8` already
exists cannot upload. **Two tags have failed this way without anyone noticing**, because
the publish job's failure is not surfaced anywhere.

`main` is two merges ahead of `v0.1.1` (#7 and #8), so the beam search speedup, the
decoding contract suite, and the tie-breaking rule are all unreachable via
`pip install torchlingo`.

Fix should cover both halves:
- Bump `pyproject.toml` and cut a release that actually publishes.
- Make CI **fail** a tag build when the git tag and `pyproject.toml` disagree, so a
  mismatch is loud rather than silent. Same class of problem as #14 (ruff version drift):
  two sources of truth with nothing checking they agree.

**#17 Coulson's review points from PR #8** — *IN REVIEW (PR #9)*
Naming schema (mirrored names across `inference` / `inference_fast`) and the greedy
default documented. On branch `decoding/naming-and-defaults`, commit `5403271`. Opened
as PR #9 on 2026-09-10 with Coulson-Rich requested as reviewer.
- Also registered `inference_fast` in the package `__init__`, missed when it was added.

**#18 Answer Coulson's curriculum question** — *DONE (answered on PR #8, 2026-09-10)*
He asked on PR #8 (2026-08-26) whether the beam-search walkthrough belongs in the 270 or
312 curriculum, and explicitly asked for thoughts.
- Answered: the walkthrough is in **CS 479** for now — neither of the two courses he
  named. Whether beam search should be taught earlier than 479 is left open as a real
  question, not closed.
- The same reply pointed him at `5403271` for his other review point (naming schema and
  the greedy default), which is #17.

**#19 Josh's PR #1 will now fail CI** — *DONE (flagged on PR #1, 2026-09-10)*
`colab-checkpointing`, dormant since 2026-01-30, is behind `main` and will hit the
lint gate added in #7 — against a much wider ruleset than existed when it was written.
Not blocking anything, but a courtesy heads-up before he next picks it up.
- Also found while checking: GitHub reports the branch as **conflicting** with `main`,
  which the original entry did not capture. Both facts are in the comment.
- Left with @commanderjcc to act on; nothing further owed from this side.

---

## Added 2026-09-10 (while implementing #5)

**#20 `data/example.tsv` is not sentence-aligned** — *DONE (repaired 2026-09-10)*

**Resolved: the data was never bad, it was assembled wrong, and it is repaired in place.**
No replacement corpus needed. Jump to "The repair" below for the outcome; the diagnosis
is kept because the same failure mode can recur.

The shipped 100,000-row "parallel corpus" is not parallel. The first few dozen rows line
up, then `src` and `tgt` drift apart and never recover. Measured by checking how often a
row's English and Spanish sides share a proper noun or a number — quantities that survive
translation:

```
rows      0-  500:  8%
rows    500- 1000:  3%
rows   2000- 2500:  1%
rows   5000- 5500:  0%
rows  10000-10500:  0%
rows  20000-20500:  0%
rows  50000-50500:  0%
rows  90000-90500:  0%
```

Eyeballing confirms it: row 11000 pairs "So, the bacteria grows hair on the crab." with
"Lo llevo a terapia."

**It is not a fixable offset.** Sweeping shifts of -300..+300 against windows at rows 0,
5000 and 20000 finds no peak anywhere — the best shifted match rate is 0-1%, i.e. noise.
The two columns were built from sources that desynchronized progressively (plausibly TED
transcript vs. translation, which merge and split sentences at different rates), so there
is no single realignment that recovers it.

**Consequences:**
- Any model trained on it learns a mapping between unrelated sentences. Confirmed: a
  1-epoch run sat at loss 8.69 against `ln(vocab) = 8.69` — exactly chance.
- Every tutorial or exercise pointing a student at this file produces garbage, and the
  student has no way to tell the difference between "my model is wrong" and "the data is
  wrong." That is the worst possible failure mode for a teaching library.
- `data/multilingual_example/` is clean but is 5 unique phrases repeated ten times, so it
  is not a substitute.

### The repair

The "no fixable offset" finding was right but the conclusion drawn from it was too
pessimistic. Looking at the *structure* rather than the offset showed what actually
happened: **the two columns are independent documents that were zipped together.**

Each column is a concatenation of per-talk records — `url, description, tags, views,
title`, then the transcript — and each carries its own URL lines as boundaries. The
English side has **751 talks**, the Spanish side **735**. Because the talk lists differ,
the columns desynchronize almost immediately and the drift compounds. Row 150 shows it
plainly: `src` is at a talk's URL while `tgt` is already at that talk's description.

So the fix is to de-interleave rather than to re-align:

1. Read each column as its own document.
2. Segment both into talk records on their URL lines.
3. Match talks by the slug that appears in both URLs — **662 talks are in both streams,
   and in the same order.**
4. Emit pairs only where the two transcripts have the **same line count** (564 talks,
   85%), which means both sides kept TED's own sentence segmentation and a line-for-line
   pairing is sound. The 98 talks with differing counts are dropped rather than guessed
   at; recovering them needs a real sentence aligner (Gale-Church) and would add ~13k
   rows to the 73k already recovered — see #29.

Implemented as `scripts/realign_corpus.py`, a reusable tool rather than a one-off, since
this diagnosis applies to any TED-style dump assembled the same way.

**Result — two independent diagnostics, both decisive:**

| metric | before | after |
|---|---|---|
| rows | 100,000 | 73,083 |
| sentence-length correlation | **0.001** | **0.969** |
| rows sharing a proper noun / number | **1.4%** | **38.9%** |

0.97 length correlation is the textbook signature of genuinely parallel text. Random
sampled pairs are all correct translations.

**Training confirms it independently** — the same script and model on both files:

| | old corpus | repaired corpus |
|---|---|---|
| chance (`ln vocab`) | 8.69 | 9.16 |
| val loss, epoch 1 | 8.67 | 9.09 |
| val loss, epoch 2 | — | 8.53 |
| val loss, epoch 3 | — | **6.29**, still falling |

The old file sat *at* chance and stayed there, which is exactly what learning a mapping
between unrelated sentences looks like. The repaired file drops steeply and had not
plateaued when the run stopped. As a side effect the usable short-sentence subset nearly
doubled (12,107 → 22,337 pairs at 3-10 words per side), because both sides are now short
*together* rather than by coincidence.

**Guard added** — `tests/test_data_integrity.py`, 6 tests asserting the shipped corpus
clears minimum length-correlation (0.80) and anchor-agreement (0.25) thresholds, has no
empty rows, and is not the same language twice. Thresholds sit far above what the broken
corpus scored and well below what the repaired one achieves, so anything landing between
them is a real regression. **The guard was verified against the broken file** — it fails
on it with 2 failures — rather than only confirmed green on the good one.

Same class of defect as #14, #16 and #26: something nothing was checking. That is now
four instances, which is a pattern worth naming rather than four coincidences.

*Original assessment of fix options, kept for the record:* ship a different corpus
(Tatoeba en-es), realign with a sentence aligner, or delete it. The de-interleaving route
turned out to dominate all three — it keeps a real, domain-appropriate 73k-pair corpus at
no licensing or download cost.

**#21 Beam search does not support LSTM models at all**
`inference.py:295` raises unless the model exposes `encode`/`decode`, which only the
Transformer does. So `greedy_decode` works for both architectures but
`beam_search_decode` is Transformer-only, and the docs do not say so. Noticed while
wiring attention through the LSTM inference path. Now that the LSTM has attention it is a
real model rather than a toy baseline, which makes the gap more visible.
- Cheap partial fix: a clear error message naming the limitation.
- Real fix: route beam search through `encode_source`/`decode_step`, which #5 added
  precisely so the decoder need not be reimplemented per call site.

**#22 `examples/` is outside the lint gate**
CLAUDE.md and CI lint `src` and `tests` only. Running `ruff check examples` turns up 32
pre-existing errors across `train.py`, `evaluate.py`, `inference_ceb_cmn.py`,
`train_ceb_cmn_simple.py` and `multilingual_training_example.py` — unsorted imports,
unused imports and variables, `f`-strings with no placeholders, deprecated `typing.List`,
a blind `except Exception`.
- These are *examples*, i.e. the code students are most likely to copy, so they are
  arguably the worst place in the repo to let style rot.
- Not fixed here: it is unrelated to attention and a 5-file mechanical diff would bury
  the review, exactly the reasoning applied to the ruff split in #14.
- Suggested: fix under its own PR, then add `examples` to the lint scope so it stays
  fixed.
- **`scripts/` has the same gap**, found the same way: `generate_sentencepiece_models.py`
  trips EXE001 (shebang, not executable) and BLE001 (blind `except Exception`). Widen the
  scope to `src tests examples scripts` in one go.

---

## Added 2026-09-10 (loose ends and next steps)

**#24 Push `lstm/attention` and open its PR** — *NEXT*
Committed as `2ee6bdb`, stacked on `decoding/naming-and-defaults`. Not pushed; no PR.
Must merge **after** PR #9, the same stacking pattern used for ruff/decoding in #14.
Flag in the PR description:
- `matplotlib>=3.7` added to core `dependencies` (user's call). Reviewers may object to a
  new runtime dependency on a library; the rationale is that `plot_attention` should just
  work in a student environment, with the `ImportError` guard kept as a safety net.
- The `pack_padded_sequence` fix (#23) changes numeric output for **any** LSTM trained on
  padded batches, `attention=False` included. It is a correctness fix, not a regression,
  but it is a behavior change and should not arrive silently.
- The corpus repair (#20) is in the same branch. If review prefers, it separates cleanly
  into its own PR — `scripts/realign_corpus.py`, `tests/test_data_integrity.py` and
  `data/example.tsv` touch nothing the attention work touches.

**#25 Tutorial 02** — *DONE (2026-09-10)*

**The premise this task was filed under was wrong.** It claimed tutorial 02 "trains
students on the misaligned corpus." It does not. It builds a 12-phrase toy corpus inline
and trains on that; the only mention of `example.tsv` is one parenthetical in a markdown
cell. The original entry was written from a `grep` hit for the filename without checking
what the notebook actually does. Recorded here because the same shortcut would produce the
same error again.

What was actually wrong is worse, and unrelated to the corpus:

**The tutorial ran clean and produced nothing.** Executing it end to end gave empty
translations for every test phrase — `Hello world → ` — and the save/load demo printed
`'Hello world' -> ''`. At `num_epochs = 5` the final loss was 3.02 against
`ln(24) = 3.18`: barely below chance, so greedy decoding emitted EOS immediately. The
markdown promised the model "can fully memorize it in a few epochs" and the summary
claimed the reader had learned greedy decoding.

Nothing caught it because **nothing executes these notebooks**. `docs/mkdocs.yml` sets
`execute: false` *and* `allow_errors: true` for mkdocs-jupyter, and tutorials 02 and 03
carried **zero stored outputs**, so neither the docs build nor a reader would reveal it.
Tutorial 01 does store outputs, so the inconsistency was invisible too. Another instance
of the "nothing was checking" pattern — see #30.

**Fixed:**
- `num_epochs` 5 → 40. Final loss 3.56 → **0.667**, and all four test phrases now
  translate correctly, including the load-from-checkpoint round trip. Still runs in
  seconds. Verified **30 epochs suffices across 5 seeds (5/5 exact)**; 40 is margin.
- Added `torch.manual_seed(0)` so a reader's run matches the committed output.
- Executed the notebook and **committed its outputs**, matching tutorial 01's convention.
  `stderr` streams were stripped first: they carried a PyTorch nested-tensor deprecation
  warning (that is #7) embedding an absolute local venv path, which should not ship in
  published docs.
- Corrected the stale parenthetical: `example.tsv` is 73,083 pairs after #20, not 100k.

**Tutorial 03 was checked and is fine, but only because 02 is fixed.** It loads 02's
checkpoint, so before this fix it would have shown blank translations and BLEU 0
throughout. Verified end to end with the checkpoint present: BLEU 100, correct
greedy-vs-beam tables. Its own defect is filed as #31.

- `tests/test_sentencepiece.py` also reads `example.tsv`, but only to train a tokenizer,
  where alignment is irrelevant. That use was always fine and still passes.

**#26 Four broken doc links block `mkdocs build --strict`**
All four link to source files as though they were doc pages, so docs cannot be gated in
CI as-is:
`MULTILINGUAL_ANALYSIS.md` → `preprocessing/multilingual.py` and → `config.py`;
`MULTILINGUAL_QUICKSTART.md` → `examples/multilingual_training_example.py`;
`TESTING_GUIDE.md` → `preprocessing/sentencepiece.py#L102`.
Point them at the mkdocstrings reference pages or at GitHub URLs, then add a docs build to
CI. A fifth warning (missing return annotation in `visualization.py`) was introduced by #5
and fixed there.

**#27 Add an attention tutorial notebook**
Tutorials run 01-data-and-vocab, 02-train-tiny-model, 03-inference-and-beamsearch.
Attention now has reference docs and a runnable example but no tutorial, which is the
format the course actually uses. Cover the bottleneck, the one-flag switch, the ablation,
and reading an alignment heatmap. Now unblocked by #20: it can use real en-es pairs, with
the synthetic reversal task as the warm-up where ground truth is known.

**#28 Attention parameters skip `_init_weights`**
`SimpleSeq2SeqLSTM._init_weights` matches on `weight_ih` / `weight_hh` / `bias`, so
`AdditiveAttention`'s `W_dec`/`W_enc`/`v` and `attn_combine` keep PyTorch's default Linear
init. Defensible — they train well, additive reaches 93.6% alignment accuracy — but it is
currently implicit rather than chosen. Either extend `_init_weights` deliberately or leave
a comment saying the default is intended. Small, and worth settling while it is fresh.

**#29 Recover the last 98 talks with a sentence aligner**
#20 keeps only talks whose two transcripts have identical line counts (564 of 662). The
remaining 98 have differing counts — median delta 0, 90th percentile 1, max 19 — so their
segmentation diverged slightly rather than catastrophically. A Gale-Church length-based
aligner handling 1-1, 1-2, 2-1, 1-0 and 0-1 would recover roughly **13k additional pairs**
on top of the 73k already in hand.
- Explicitly *not* done in #20: guessing at alignment is how this corpus got into trouble
  in the first place, and 73k correct pairs beat 86k uncertain ones for a teaching library.
- Worth doing only if the extra data is actually wanted; it is a real aligner, not a
  one-liner, and `scripts/realign_corpus.py` is the natural place for it.

**#30 Nothing executes the tutorial notebooks**
`docs/mkdocs.yml` configures mkdocs-jupyter with `execute: false` **and**
`allow_errors: true`. Combined with tutorials 02 and 03 storing no outputs, a notebook can
rot completely — wrong results, or an outright exception — and neither the docs build nor
a reader will surface it. #25 is the proof: tutorial 02 shipped producing empty
translations, and tutorial 03 raises `NameError` on a clean run.
- Fix: execute the notebooks in CI. `jupyter nbconvert --to notebook --execute` is enough
  and takes seconds for all three.
- Execute in a scratch directory: both notebooks write `data/` and `checkpoints/`
  relative to their own location, and those paths are only gitignored at the repo root.
- Note that 03 depends on 02's checkpoint, so CI must run them in order or the job must
  make that dependency explicit.
- Fourth instance of "two things that must agree with nothing checking," after #14, #16,
  #20 and #26. That is a pattern, not a run of coincidences; worth one deliberate audit
  for the remaining cases rather than finding them one at a time.

**#31 Tutorial 03 soft-fails into a confusing `NameError`**
`03-inference-and-beamsearch.ipynb` cell 3 loads tutorial 02's checkpoint inside an
`if/else`. When the checkpoint is missing it prints a friendly
"⚠️ No checkpoint found. Please run Tutorial 2 first!" and continues — so `model`,
`src_vocab` and `tgt_vocab` are never bound, and the notebook dies three cells later with
`NameError: name 'model' is not defined`. The reader sees the crash, not the explanation.
- Fix: raise immediately with the guidance in the message, rather than printing and
  limping on.
- Then execute it and commit its outputs, as #25 did for 02. Verified working once the
  checkpoint exists: BLEU 100 on the toy corpus, greedy and beam agreeing.
- Also worth deciding: 03 hand-rolls its own `greedy_decode` and beam search rather than
  using `torchlingo.inference`. Defensible as a teaching exercise, but it can drift from
  the library — and #17 has just standardized the real decoders' naming and defaults, so
  the duplicate is now a second source of truth for how decoding works.
