# TorchLingo — Session Task List

Captured 2026-08-22. Numbered for reference in conversation.

## Code — decoding performance

**#1 Batched beam search, Tier 1: batch across beams**
Stack the k beams into one `(k, t)` tensor, expand `memory` to `(k, src_len, d)`,
issue one `model.decode()` call per step instead of k.
- File: `src/torchlingo/inference.py:180-193`
- Measured payoff: beam currently makes 38.7x more `decode()` calls than greedy for
  only 5.0x more actual math, every call at batch size 1.0. Latency-bound on dispatch.
- Self-contained; no public API change.
- Oracle: output must be token-identical to current implementation.

**#2 Batched beam search, Tier 2: batch across sentences**
Remove the `src.size(0) != 1` restriction; flatten to `(batch x k, t)`.
- Files: `src/torchlingo/inference.py:159` (the raise), `:266` (the per-row loop in
  `translate_batch`)
- Hard part is bookkeeping for ragged completion — sentences finishing at different steps.
- `tests/test_training_inference.py:502`
  (`test_beam_search_decode_raises_on_batch_size_gt_one`) encodes the current limitation
  and is the first test to invert.

**#3 Batched beam search, Tier 3: incremental decoding / KV cache**
Removes the O(L^2) prefix recomputation.
- DECISION NEEDED: may compromise the readability that makes this repo worth using
  for teaching. Consider stopping at #2 for an educational library.

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

**#14 Repo is not ruff-clean at HEAD**
`ruff check --fix src tests` makes 354 fixes across 28 files, and `ruff format` reformats
5 more. This means CLAUDE.md's "ALWAYS run ruff after code changes" instruction produces a
large unrelated diff on any change. Either bring the repo to ruff-clean in one dedicated
commit, or pin the ruff version — otherwise every future PR is polluted.
- 124 errors remain that `--fix` cannot resolve automatically.

**#15 Existing `DummyTransformer` is history- and memory-blind**
`tests/test_training_inference.py:16` computes logits from a zero tensor, so its output
depends only on decoding *position*. Verified: logits are byte-identical for different
decoder histories AND for different encoder memories. The three existing beam tests
therefore cannot detect scrambled beam state or bad memory expansion.
- Mitigated by the new `tests/test_decoding_equivalence.py`, but the old tests should
  eventually migrate to the history-sensitive fixture rather than sitting alongside it.
