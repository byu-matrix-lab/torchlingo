# TorchLingo — Session Task List

Opened 2026-08-22, last updated 2026-09-13. Numbered for reference in conversation.
Completed work is removed rather than marked done — git history is the record.

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
| #22 | `examples/` and `scripts/` are outside the lint gate | Open |
| #26 | Broken doc links block `mkdocs --strict` | Open |
| #28 | Attention params skip `_init_weights` | Open |
| #29 | Recover the last 98 talks with a sentence aligner | Open |
| #34 | Surface attention weights from greedy and beam decoding | Open |
| #44 | Gate the sdist on "no Git LFS pointer shipped" | Open |
| #45 | A stacked PR gets no CI at all | Open |

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

**#6 Multi-GPU training via DDP**
Not implemented. `config.py:663` states multi-GPU "requires custom DataParallel setup."

**#7 One PyTorch deprecation warning left**
On torch 2.13.0, "Support for mismatched key_padding_mask and attn_mask is deprecated",
raised from the decode path. It will eventually break. The decode path passes a boolean
`tgt_key_padding_mask` alongside a float `tgt_mask`; making both the same dtype should
settle it.

The other warning this entry used to list, the nested-tensor prototype notice from
`nn.Transformer`, is gone. It was a side effect of disabling the encoder's nested-tensor
fast path, which had to go because the op behind it is unimplemented on Apple's MPS
backend and made every library decoder raise `NotImplementedError` on Apple Silicon.
Worth knowing for the next device-specific bug: CI runners are x86 Linux, so nothing in
the matrix can reproduce that class of failure — the lab's Macs are the only place it
shows up, which is also where the students are.

## Evaluation / tooling

**#8 Verify Eole claims hands-on before syllabus use**
Specifically: COMET/MetricX integration in the training loop, and 7B-13B finetuning on a
single 24GB GPU. Both are from Eole's README, not from running it.
- Needs a separate venv: Eole requires Python >= 3.11 and torch >= 2.10, **< 2.13**.
  This repo's `.venv` has torch 2.13.0.

**#9 Run `pre-commit install`**
`.pre-commit-config.yaml` exists in the repo but hooks are not installed in this clone.

## Possible tooling to productize

**#12 Decode benchmark harness**
The ad-hoc scripts written this session (greedy-vs-beam timing, `decode()` call counting
via monkeypatched instrumentation) would be worth a reusable
`scripts/bench_decode.py` — needed again to validate #1, #2, and #3, and useful as a
student exercise in its own right.

---

## Tests

**#15 Existing `DummyTransformer` is history- and memory-blind**
`tests/test_training_inference.py:16` computes logits from a zero tensor, so its output
depends only on decoding *position*. Verified: logits are byte-identical for different
decoder histories AND for different encoder memories. The three existing beam tests
therefore cannot detect scrambled beam state or bad memory expansion.
- Mitigated by the new `tests/test_decoding_equivalence.py`, but the old tests should
  eventually migrate to the history-sensitive fixture rather than sitting alongside it.

---

## Release

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

**#44 Gate the sdist on "no Git LFS pointer shipped"**

`data/example.tsv` and `data/pretrained/model.pt` moved to Git LFS, and CI checks out
without LFS on purpose to keep runs light. That combination has a sharp edge: a build
that packages an LFS-tracked file in a no-LFS checkout ships 130 bytes of pointer text
under the name of a 17 MB corpus, with nothing in the build complaining. It would reach
PyPI looking fine and open as garbage.

`MANIFEST.in` now excludes both explicitly, so this is closed *by construction* rather
than *by check*. Two things that must agree with nothing checking they do, again:
a future `recursive-include` would reopen it silently.

- Add a release-job step that scans the built sdist and wheel for any member beginning
  `version https://git-lfs` and fails on a hit.
- Cheap, no LFS dependency, catches the whole class rather than today's two files.

Related, worth watching rather than acting on: LFS storage and bandwidth come out of the
org's quota. Two files at ~28 MB is nothing, but every clone by every student fetches
them. If a course section of 60 blows through the free tier, the fallback is to host the
corpus outside git and download it on first use.

## Inference gaps

**#34 Surface attention weights from greedy and beam decoding**
Raised by Coulson on PR #10: can we visualize alignments for beam search too?

Not today. Weights come only from a teacher-forced `model(src, tgt, return_attention=True)`,
which aligns a translation you already have. Both decoders compute weights and throw them
away — `inference.py:237` in greedy, and the LSTM beam path added in #12. So you can plot
the alignment of a *reference* translation but not of one the model generated, which is
the more interesting picture.
- Greedy is straightforward: accumulate the per-step weights.
- Beam is not. Weights belong to a hypothesis and hypotheses get pruned, so either carry
  per-beam weight history and filter to the winner, or re-run `decode_prefix` on the
  winning sequence once the search finishes. The second is cheaper and matches how the
  reference already re-scores prefixes.
- Shape it as an opt-in `return_attention=False` on both decoders so the default return
  type does not move — #9 has just standardized those, along with the contract tests.

## Lint and tooling gaps

**#45 A stacked PR gets no CI at all**

Found on #20, which reported "no checks reported on the branch" and has never triggered
a single workflow run. The workflow's `pull_request` trigger is filtered to
`branches: [main]`, and #20 targets `data/talk-ids-and-subwords` because it is stacked
behind #19. A PR that does not target main therefore runs nothing.

Checked all six open PRs: #15 through #19 target main and have checks; #20 is the only
stacked one and the only one with none. So the rule is exactly "stacked means unverified."

This matters because it is silent and it is backwards. The stacked PR is reviewed in that
state, so a reviewer sees no green checks and has no way to tell "not run" from "not
passing." CI only starts once the PR is retargeted to main, which happens *after* review.
And stacking is the normal mode here while PRs wait for acceptance, so this recurs.

Workaround used meanwhile: `gh workflow run tests_and_build.yml --ref <branch>`, since
`workflow_dispatch` is already enabled.

Fix is one line: drop `branches: [main]` from the `pull_request` trigger so a PR runs CI
regardless of base. The `push` trigger should keep its `branches: [main]`, which is what
stops every branch push from burning a run. Needs a decision because it changes CI
behavior for every PR in the repo.


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

## Docs and tutorials

**#26 Four broken doc links block `mkdocs build --strict`**
All four link to source files as though they were doc pages, so docs cannot be gated in
CI as-is:
`MULTILINGUAL_ANALYSIS.md` → `preprocessing/multilingual.py` and → `config.py`;
`MULTILINGUAL_QUICKSTART.md` → `examples/multilingual_training_example.py`;
`TESTING_GUIDE.md` → `preprocessing/sentencepiece.py#L102`.
Point them at the mkdocstrings reference pages or at GitHub URLs, then add a docs build to
CI. A fifth warning (missing return annotation in `visualization.py`) was introduced by #5
and fixed there.

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

