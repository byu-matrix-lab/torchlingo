# TorchLingo — Session Task List

Opened 2026-08-22, last updated 2026-09-20. Numbered for reference in conversation.
Completed work is removed rather than marked done — git history is the record.

Note: the Status table below has drifted — several rows marked Open have since merged,
and reconciling it against the merged history is its own pass, not done here.

## Status

| | Task | State |
|---|---|---|
| #16 | Release pipeline broken — nothing ships | Open |
| #2 | Batch beam search across sentences | **Descoped 2026-09-22** |
| #3 | Incremental decoding / KV cache | **Descoped 2026-09-22** |
| #4 | Resolve length-normalization semantics | Open |
| #6 | Multi-GPU training via DDP | **Descoped 2026-09-22** |
| #7 | PyTorch deprecation warnings | Open |
| #8 | Verify Eole claims before syllabus use | Open |
| #9 | `pre-commit install` (still not installed) | Open |
| #15 | Migrate history-blind `DummyTransformer` tests | Open |
| #22 | `examples/` and `scripts/` are outside the lint gate | Open |
| #26 | Broken doc links block `mkdocs --strict` | Open |
| #28 | Attention params skip `_init_weights` | Open |
| #29 | Recover the last 98 talks with a sentence aligner | Open |
| #34 | Surface attention weights from greedy and beam decoding | Open |
| #35 | Malformed tag `v.0.0.8` on the remote | Open |
| #36 | CI actions pinned to a deprecated Node runtime | Open |
| #37 | Stacked PRs fight the stale-review rule | Open |
| #38 | Colab checkpointing has never been run in Colab | Open — Coulson testing |
| #40 | Visualize the effect of decoding options | Open |
| #41 | Connect beam search back to prior coursework | Open |
| #42 | Lecture 7 assignment | Open — scope needed |
| #44 | Gate the sdist on "no Git LFS pointer shipped" | Open |
| #46 | Teach the corpus repair instead of doing it silently | Open |
| #47 | Docstring examples are not executed, and 30 fail | Open |
| #48 | Audit pedagogical value; write down sequencing and outcomes | In progress — `notes/CURRICULUM.md` |
| #49 | Pretrained checkpoint predates the enlarged corpus | Open |
| #50 | Tutorial 3 still teaches the wrong lesson about beam size | Open |
| #51 | The docs gate reports but does not block | Open — repo settings |
| #52 | Try Moore (2002) if more of the corpus is wanted | Open |
| #53 | Notebook gate runs 1 of 5 tutorials in CI, and looks green | Open |
| #54 | `val_losses` interleaved two measurements — fixed, needs PR | In review |
| #55 | Correct tutorial 5: the model was undertrained, not data-starved | Open |
| #56 | Correct #34's description and commit message | Open |
| #57 | Note on #27 that the recovered data buys no measurable BLEU | Open |
| #58 | `train_example_model.py` defaults to 20 epochs, which undertrains | Open |
| #59 | Nothing checks that a comparison controlled its variables | Open |
| #60 | Nobody is told when main goes red | Open |
| #61 | A PR closed itself during a merge and nobody noticed | Open |
| #62 | Check the open-PR set after every merge | Open |
| #63 | Three pages have no mkdocs nav entry | Open — blocked on PR #17 |
| #64 | Promote the tutorial 6 checks into `torchlingo.diagnostics` | Done in PR #45 |
| #65 | Tutorial 6 and `diagnostics` are two copies of the same checks | Open — blocked on PR #44 + #45 |
| #66 | Adopt `nltk.translate.gale_church`; split #29 into two jobs | Open |
| #68 | Cite `torcheck` as prior art in the diagnostics docs | Open |
| #69 | Position the project on the curriculum, not the architecture | Open — strategic |
| #70 | Print the sacreBLEU signature with every score | Open |
| #71 | Decide whether to report the Joey NMT breakage upstream | Open — Eric's call |
| #72 | Reconcile the Status table with the merged history | Open — after #73 |
| #73 | The review backlog is ten PRs deep | Open |
| #74 | A broken anchor and 93 unexplained warnings in the docs build | Open |

## Code — decoding performance

### Decision (2026-09-22): #2, #3 and #6 are descoped

**Eric's call, taking the recommendation from the competitive assessment.** Batched beam
search across sentences, KV caching, and multi-GPU training via DDP are not being built.

*Why.* Each is high-complexity, low-teaching-value, and duplicates what CTranslate2,
Marian and Joey NMT already do better than this repository ever would. The 2026-08-22
design decision below concedes the premise without drawing the conclusion: the fast path
"is necessarily harder to read than the 85-line version" and is kept separate *because*
it cannot be followed line by line. Something a student cannot read is not teaching them
anything, so the case for carrying it in a teaching library had to be made rather than
assumed — and on inspection it could not be.

The competitive assessment is what forced the question. Joey NMT covers this ground,
is actively maintained, and reaches 93.62 BLEU on its toy task in under four minutes on
CPU, so the decoder-performance work is the *least* differentiated thing we could spend
effort on. What is differentiated — diagnosis, empirical discipline, a corpus with a
documented repair history, documentation that executes — is where the time goes instead.
See #69 and `docs/docs/related-work.md`.

*What this does not touch.* #4 (length-normalization semantics) stays open and is
unaffected: it is a correctness-and-teaching question with real evidence behind it, not a
performance one. `inference_fast.py` stays as it is — already merged, already tested by
`tests/test_decoding_equivalence.py`, and still the faster path for anyone who wants it.
Nothing is being removed; we are declining to extend it.

*If this is ever reversed*, the technical notes below are kept deliberately intact — the
`(batch x k, t)` flattening, the ragged-completion bookkeeping, and the warning that a KV
cache leaves the call count unchanged and must be read on the positions-forwarded column
instead. That last one would cost a day to rediscover.

*Also worth telling students.* "We could make this faster and chose not to, because the
readable version is the point, and here is the toolkit to use when speed actually
matters" is a better lesson than a fast path nobody reads. Candidate for
`concepts/decoding.md`.

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

> **Superseded 2026-09-22.** This paragraph answered "may we build it?" and read the
> answer as "so we should." The descope decision above answers the question that was
> never asked: *should* we, given that it duplicates CTranslate2 and Marian and teaches
> nothing a student can read. The side-by-side design remains correct for the code that
> already exists — it is only the conclusion about #3 that is withdrawn.

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

**As of 2026-09-22 that remaining ~8x is being left on the table deliberately** — #2 is
descoped. #1's share is already merged. Kept here because the split is the thing worth
knowing: the figure was once quoted as if any single task could deliver all of it.

Full explanation for students lives in `docs/docs/concepts/decoding.md` — keep it there
rather than duplicating it into code and notes.

**#2 Batch beam search across sentences** — **DESCOPED 2026-09-22**, see the decision
above. Notes kept because they are the expensive part to rediscover.

Remove the batch-size-1 restriction in `inference_fast.py`; flatten to `(batch x k, t)`.
Validate it with `python scripts/bench_decode.py`: the sentence lever should show up as a
further drop in `decode()` calls with positions forwarded unchanged, and
`tests/test_bench_decode.py` will fail until the committed numbers are regenerated, which
is the intended prompt to update the docs.
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

**#3 Incremental decoding / KV cache** — **DESCOPED 2026-09-22**, see the decision above.

Removes the O(L^2) prefix recomputation. Independent of the two axes above: it reduces the
work *inside* each call rather than the number of calls.
- ~~DECISION NEEDED: may compromise the readability that makes this repo worth using for
  teaching. Consider stopping at #2 for an educational library.~~ **Resolved by the
  side-by-side design above:** the reference implementation stays readable regardless, so
  the fast path is free to be dense. Worth doing.
- `scripts/bench_decode.py` is the right instrument, but note it measures the wrong axis
  for this one: a KV cache leaves the **call count unchanged** and cuts *positions
  forwarded* instead. Read that column, not the call column, or the harness will make a
  real improvement look like no change at all.

**#4 Resolve length-normalization semantics**
`inference.py:203` applies length normalization during *pruning*, not only at final
selection — comparing normalized scores across different lengths mid-search. Defensible
but non-standard. Preserve exactly during #1/#2 so perf work stays reviewable; raise as
a separate question.

**Now has evidence, from #40.** Measured on the tutorial 5 model across five held-out
subsets, paired:

- `alpha=0.6`, the shipped default, is **indistinguishable from `alpha=0.0`**
  (−0.07 ± 0.03 BLEU). It is not doing the job it exists for.
- `alpha=1.0` is a real if small gain, +0.25 ± 0.06.
- The bias it targets is plainly present: mean output length falls monotonically with
  beam width, 12.61 tokens at greedy to 9.73 at beam 10, against references averaging
  11.62.

So the question is no longer whether the semantics are defensible in the abstract. It is
why a correction that measurably does nothing is on by default. Two candidate answers,
and the evidence does not distinguish them: the default is too weak, or normalizing
during pruning blunts it. Sweeping `alpha` with normalization applied only at final
selection would separate the two, and that is now a cheap experiment because
`scripts/sweep_decoding.py` exists.

## Code — other gaps

**#6 Multi-GPU training via DDP** — **DESCOPED 2026-09-22**, see the decision under
*Code — decoding performance*. The weakest of the three for a course and the highest
ongoing maintenance: the lab's students train on laptops and Colab, where there is one
GPU or none.

Not implemented. `config.py:663` states multi-GPU "requires custom DataParallel setup."
That sentence is now the honest final answer rather than a placeholder, and should be
left in place.

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


---

## The training-budget finding, and what it invalidated

Raised by Coulson on PR #34: "the BLEU scores are extremely low ... worth looking into
if it wasn't flagged before." The scores were expected and documented. Looking into them
anyway found a measurement error of mine.

**The root cause (#54).** `train_model` appended to `val_losses` in two places: the
periodic step-triggered validation from `config.val_interval`, and the epoch-end
validation. One list, two different measurements, and a docstring promising "per epoch".

```
              train_losses   val_losses   true epochs
  baseline         20            36            20
  new              36            72            36
```

I read `len(val_losses) == 36` off the baseline, concluded 36 epochs, and passed
`--epochs 36` to "match" it. The baseline had run **20**.

**What that did to #49.** The comparison gave one model 19% more data *and* 80% more
training, while its writeup claimed data was the only difference. Re-running with epochs
actually matched:

| data | epochs | BLEU | |
|---|---|---|---|
| 53,520 pairs | 20 | 4.96 | baseline as shipped |
| 53,520 pairs | 36 | **7.01** | control: same data, more epochs |
| 64,311 pairs | 36 | 7.32 | more data *and* more epochs |

- epochs 20 → 36, data fixed: **+2.05 BLEU**
- +20% data, epochs fixed: **+0.29 ± 0.22, 95% CI [−0.16, +0.71]**

The data effect's interval crosses zero. Training budget mattered roughly **7x** more
than the recovered data, and the recovered data bought nothing measurable.

So the published claim was wrong twice: ~88% of the +2.33 was training length, and the
residual is not significant. The diagnosis in tutorial 5 — "data-starved" — is also
wrong; the model was **undertrained**, which has a different fix.

**#55 Correct tutorial 5.** It currently teaches "more data" as the top lever, measured.
The honest version is the better lesson: the interesting hypothesis was wrong, the boring
one (you stopped training too early) was right, and only controlling the variable told
them apart. Numbers to use are in the table above; artifacts in
`docs/docs/_generated/checkpoint_comparison.json`, which also needs regenerating from the
controlled run.

**#56 Correct #34.** Its description and commit message both claim "same architecture,
same 36 epochs, same seed — the only thing that changed is the data." False. The
checkpoint itself is fine and worth shipping; only the explanation of why it is better
needs replacing.

**#57 Add a note to #27.** It claims "They are 18% more data", which is true, and makes
no BLEU claim, so nothing there is wrong. But the implicit case for the work is quality,
and the measured quality effect is indistinguishable from zero at this scale. Worth
saying so plainly, and restating the real justification: it is a correctness fix for data
being discarded for a fixable reason, it teaches Gale-Church, and it will matter at a
scale where the model is not the binding constraint.

**#58 The script's default undertrains.** `train_example_model.py` defaults to
`--epochs 20`, which is what produced the BLEU 4.96 checkpoint. 36 epochs gives 7.01 on
the same data, and by then validation loss has flattened (mean change over the last five
epochs: −0.0010/epoch). A default that stops well short of convergence teaches the wrong
thing about training, and it is what made "more data" look like the answer. Change the
default to 36, or add early stopping on the validation curve so the run ends when it
should rather than when a hardcoded count runs out.

**#59 Nothing checks that a comparison controlled its variables.** This is the general
version, and the reason the error survived review. `compare_checkpoints.py` pins the test
set and bootstraps the difference, which is why the *measurement* was sound; it never
looks at how the two checkpoints were trained. It has both checkpoint dicts in hand and
could refuse, or at least warn loudly, when `len(train_losses)`, `train_pairs`,
`model_config` or the seed differ — printing what differs alongside the BLEU delta so a
reader sees the confound next to the number. Same shape as every other finding on this
list: two things that must agree, with nothing checking they do.

**#60 Nobody is told when main goes red**

main broke on 2026-09-19 and stayed broken until someone happened to look.

The break itself is instructive: **no PR could have caught it.** #25 introduced
`preprocessing/alignment.py` carrying a docstring example that asserts
`looks_aligned()` on a single-row frame, which cannot pass. #32 added the
`--doctest-modules` gate that runs it. #32 could not have fixed the example,
because it branched from a main where the file did not exist yet. Both were
green on their own branches; the *combination* fails.

That is a merge-order interaction, and the only place it can surface is a
post-merge run on main. Which means the post-merge run is load-bearing and
currently nobody watches it:

- A PR's checks run against the *merge result*, so #39 was green while main was
  red. Green on your PR says nothing about the branch you are merging into.
- Nothing notifies on a failed push-to-main run. It sits in the Actions tab.
- The next person to open a PR inherits a red main and may reasonably assume
  their branch caused it.

Cheapest fix that would have caught this: notify on failure of the `push` to
main run — a GitHub Actions step on `if: failure()` posting to the Discord
channel the lab already uses, or simply enabling GitHub's own "Actions failure"
email for the repo. Neither needs new infrastructure.

Worth pairing with #51 and #53, which are the same family: a check that reports
but does not block, a check that runs a fifth of what it claims, and a check
nobody reads. Each is individually defensible and together they mean a green
tick carries less than it appears to.

**#61 A PR closed itself during a merge and nobody noticed**

#26 — the decoding sweep, carrying the #40 and #41 work — was **closed unmerged**
at 2026-09-19T00:19:55Z, one second after #24 was merged. It went unnoticed for a
day and was found only because it disappeared from a routine `gh pr list`.

Nothing was lost: `docs/decoding-option-effects` survived, the PR reopened with
its approval intact, and a rebase put it back on top of #34. But it was one
branch deletion away from being genuinely hard to recover.

**The cause is not established.** The timeline records `closed by ringger` at the
moment `gh pr merge 24 --squash --delete-branch` ran, and #26's base branch
(`data/retrain-on-enlarged-corpus`) was never deleted, so the usual
"base branch deleted closes the PR" explanation does not fit. This is the third
distinct way stacked PRs have misbehaved here, after #11/#12 auto-closing and the
rebase-dismissal problem.

Worth doing regardless of root cause:

- A check that the set of open PRs after a merge is the set expected before it,
  minus the one merged. Cheap, and would have caught this within seconds.
- Stop relying on noticing. Every stacked merge in this session needed a manual
  retarget, a manual rebase, and a manual look at what survived.

The deeper answer is the one #37 keeps pointing at: **stop stacking.** Every
mechanism that has bitten — auto-close, stale-review dismissal, this — is
specific to PRs whose base is another PR. Merging promptly and keeping stacks at
depth one avoids all three.

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

`main` is now eight merges ahead of `v0.1.1` (#7 through #14), so the beam search speedup,
the decoding contract suite, the tie-breaking rule, LSTM attention, the repaired corpus
and every tutorial fix are all unreachable via `pip install torchlingo`.

### Verified 2026-09-13, from the Actions history

An earlier guess recorded here — that the workflow might never fire on a tag, because
`tags:` sits under the same `push:` trigger as a `paths:` filter — is **wrong**. Every
`v*` tag has a run. Path filters do not suppress tag pushes:

```
v0.1.1    push   failure   2026-07-18
v0.0.8    push   success   2026-02-18
v.0.0.8   push   failure   2026-02-18   <- malformed tag name, see below
v0.1.0    push   failure   2026-02-18
v0.0.7    push   success   2026-01-30
v0.0.6    push   success   2026-01-30
```

So the pipeline runs; it fails at the end. Per-job results for the two failed releases:

| | v0.1.0 | v0.1.1 |
|---|---|---|
| tests 3.10-3.13 | pass | pass |
| Build wheels and sdist | pass | pass |
| Create GitHub Release | pass | **fail** |
| Publish to PyPI | **fail** | **fail** |

The publish log gives the cause outright:

```
ERROR  HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
```

which is what PyPI returns for a filename that already exists. That confirms the original
diagnosis: the build produced `0.0.8` because `pyproject.toml` says so, and `0.0.8` was
already on PyPI from February. The version collision is real and is the primary fault.

**Still unexplained:** why `Create GitHub Release` failed on v0.1.1 but succeeded on
v0.1.0. The step's own output is not in the archived log, so the cause is not recoverable
from here. It explains the "no assets" observation above, and it is a *second*,
independent failure — worth confirming before trusting the next tag, since fixing the
version collision alone would not have fixed v0.1.1.

Fix should cover both halves:
- Bump `pyproject.toml` and cut a release that actually publishes.
- Make CI **fail** a tag build when the git tag and `pyproject.toml` disagree, so a
  mismatch is loud rather than silent. Same class of problem as #14 (ruff version drift):
  two sources of truth with nothing checking they agree.
- Do the tag-vs-version check **first**, so the next tag cannot fail the same way.

**Do not test this by pushing a `v*` tag.** The `publish` job fires on any ref matching
`refs/tags/v*` and will attempt a real PyPI upload. The Actions history answers most
questions without that risk, which is how the table above was produced.

**#35 A malformed tag `v.0.0.8` exists on the remote**
Found while auditing the Actions history for #16. Someone typed `v.0.0.8` instead of
`v0.0.8`; it matches the `v*` trigger, ran, and failed. Both tags exist on origin today.
Harmless but confusing, and it is the kind of thing the tag-vs-version check in #16 would
have caught at push time. Decide whether to delete it or leave it as history.

**#38 Colab checkpointing has never been run in Colab**
PR #17 adds `training_checkpoint.py` with `is_colab()`, `mount_drive()` and a Drive-backed
default directory. None of it has ever executed in Colab. CI cannot cover it: GitHub
runners have no Drive to mount. Josh said the same of his original in PR #1, so this code
path has now been **written twice and run zero times**.

Asked Coulson on PR #17 to try it. What needs checking:
1. `mount_drive()` actually mounts, and `default_checkpoint_dir` lands under `MyDrive`
   rather than the runtime's own disk — a checkpoint on runtime disk dies with the
   runtime, defeating the purpose.
2. `latest.pt` and `best.pt` appear in Drive; the startup free-space line is sane.
3. Interrupt the runtime partway, re-run the same cell: it should print a resume line and
   train only the remaining epochs.

Item 3 is the one that matters. If it restarts from epoch 0 the feature does not work,
whatever the unit tests say.
- Open: whether to gate the #17 merge on this, or merge with the limitation documented,
  which it currently is in both the module docstring and the reference page.
- **Coulson accepted on 2026-09-16:** "I will return to this to review and test in Colab
  when I finish the other PRs." He has since reviewed everything else, so #17 is next in
  his queue and this is the one open item with a named owner.

**#36 CI actions are pinned to a deprecated Node runtime**
Every run now warns: `actions/checkout@v4`, `actions/setup-python@v5` and
`actions/download-artifact@v4` target Node 20, which GitHub deprecated, and are being
forced onto Node 24. It is a warning today and a hard failure whenever GitHub drops the
shim. Bump the action versions. Unrelated to anything in flight, and cheap.

**#37 Stacked PRs fight the repo's stale-review rule**
Not a code defect; a process one, recorded because it cost real time merging #9 through
#14 and will recur the next time work is stacked.

The `main` ruleset sets `dismiss_stale_reviews_on_push: true` and requires one approving
review. A stacked PR must merge its parent's changes in before it can land, and that merge
commit is a push, so it dismisses the approval it just earned. Every PR in the stack then
needed an admin override even though all six had been reviewed and approved on substance.

Two traps found the hard way, both worth avoiding next time:
- **Do not merge with `--delete-branch` while another PR is based on that branch.**
  GitHub auto-closes the dependents. Retarget them to `main` *first*, then delete.
  Recovering from it means pushing the deleted branch back temporarily, because GitHub
  refuses to reopen a PR whose base is missing and refuses to retarget a closed one.
- A PR retargeted to `main` after its base merged has **no status checks**, because no
  `pull_request` event with `base: main` ever fired for it. Closing and reopening the PR
  fires one without adding an empty commit.

Options, if this shape comes up again: keep branches independent off `main` where the work
allows; or ask for one re-approval pass after all branches are rebased; or accept admin
overrides as the normal cost of stacking.

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
- **Match on the first line, not anywhere in the file.** A `grep -rl` for the string
  flagged `tests/test_data_integrity.py` and `tests/test_sentencepiece.py`, which contain
  it as the literal the skip logic compares against. The detector would fail the build on
  the detector. Compare `head -c 23` instead.

Verified once by hand against the CI-built artifacts from run 34992013848, which is the
real case: a no-LFS checkout. The sdist carries only the small multilingual examples and
the wheel carries no data at all.

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
**Correction (2026-09-18): this is bigger than recorded, and the "cheap alternative"
above does not exist.** I claimed decode-then-teacher-force already yields the alignment,
making this ergonomics rather than a missing feature. That is true of the LSTM only:

```
SimpleSeq2SeqLSTM.forward(src, tgt, return_attention=False)   <- exists
SimpleTransformer.forward(src, tgt, ...)                      <- no such parameter
```

`SimpleTransformer` has **no attention-returning path at all**. There is nothing to
teacher-force into and no recipe to document. And the Transformer is the architecture the
tutorials train, the pretrained checkpoint uses, and a student is most likely to reach
for, so the gap is in the worse place.

Getting cross-attention out of `nn.Transformer` is also not a one-liner. PyTorch hardcodes
`need_weights=False` inside `TransformerDecoderLayer._mha_block`, so a forward hook on
`multihead_attn` captures `(output, None)`. The options are to wrap each layer's
`multihead_attn.forward` to force `need_weights=True` while capturing, or to subclass the
decoder layer and override `_mha_block`. The wrapper is less invasive and can be a context
manager, which also keeps the cost off the default path.

Revised shape:

1. A way to capture Transformer cross-attention at all. This is the real work and
   everything else depends on it.
2. Then the original item: surface it from greedy and beam decoding rather than only from
   a teacher-forced pass.

Worth doing because it is also a good lesson. Explaining *why* the weights are not simply
available — a fused fast path that discards them unless asked — teaches something true
about how these libraries are built.

Knock-on for #48: the audit lists "attention appears three times" under redundancy. The
sharper problem is that it appears three times for the **LSTM** and zero times for the
Transformer. Tutorial 4 teaches attention on an LSTM trained on a synthetic reversal task;
a student who moves to the Transformer cannot inspect attention on the model they are
using.

## Lint and tooling gaps

**Standing gotcha from #45, which is now fixed in #21**

A PR opened *before* #21 merged still shows no checks, because for `pull_request` events
GitHub reads the workflow from the PR branch rather than from main. Rebase such a PR onto
current main, or dispatch a run with
`gh workflow run tests_and_build.yml --ref <branch>`. PRs opened since #21 get checks
automatically, whatever branch they target.


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
- The gap keeps widening. `scripts/` has gained `bench_decode.py`, `execute_notebooks.py`,
  `train_example_model.py`, `sweep_decoding.py` and `diagnose_corpus.py`, all linted by
  hand on the way in and none of them gated. Hand-linting is exactly the thing that stops
  happening once whoever is doing it moves on.

---

## Course material

**#42 Lecture 7 assignment** — *placeholder, scope needed*
Captured so it is not lost. Not startable yet: what lecture 7 covers, which course it
belongs to, what students are meant to produce, and when it is needed are all unknown here.

**The assignment itself lives in the LMS, not in this repo.** So the work here is whatever
*supporting material* the assignment needs — a starter notebook, a script with gaps to
fill, a dataset slice — not the assignment text. That also means the deliverable may be
small or may be nothing at all, depending on what the assignment asks students to do.

`contributing.md` previously documented an `assignments/` directory that never existed.
Corrected when this was filed, and the page now says where assignments actually live.

Material an assignment could build on, all now on `main`:
- Tutorial 4 ends with an ablation and a measurable alignment accuracy, which is already
  close to an assignment shape.
- `scripts/bench_decode.py` measures decode call counts against wall clock; the original
  #12 entry flagged this as "useful as a student exercise in its own right", and the gap
  between the two numbers is a real lesson.
- `examples/attention_alignment.py` runs the same comparison at larger scale.
- The decoding contract tests demonstrate specification-by-test, if the assignment is
  about correctness rather than modelling.

## Visualization

All three raised by Coulson on Discord, 2026-09-14, after reviewing the open PRs:

> "if we can add visualization to any of the options that we present it could be useful
> for the students. I saw that it was added for attention, but did we add it for beam
> search as well? ... students should have learned about this in 312 ... but I think a
> reminder in this tool may be useful."

**The answer to his direct question was no.** That is now fixed: `visualization.py` gained
`format_beam_search` and `plot_beam_search`, and `beam_search_decode` takes an optional
trace recording candidates *before* pruning. The two below are what remains.

**#40 Visualize the effect of decoding options**
Distinct from #39: that shows how the search works on one run, this shows what the knobs
do across runs — `beam_size`, `alpha`, greedy versus beam.

Tutorial 3 already sweeps beam sizes 1, 2, 3, 5, 10 and prints a table where every row is
identical, because the toy model is decisive. That teaches nothing. On a model where the
answers differ, the sweep would show the diminishing returns past beam 3-5 that the docs
currently **assert in prose without evidence** — the same gap #12 closed for the
performance numbers.

Also worth showing `alpha`: its effect on output length is easy to demonstrate and hard to
intuit from the formula, and it connects to the open question in #4.

Smaller than #39; can reuse `scripts/bench_decode.py`'s structure for sweeping and
emitting JSON.

**#41 Connect beam search back to prior coursework**
The point stands regardless of which course number is right: students have likely met beam
search as a general search algorithm before meeting it as a decoder. The docs teach it
from scratch in NMT terms and never connect it to what they already know. A short framing
— best-first search with a fixed-width frontier, where the heuristic is the model's log
probability and pruning is what makes it tractable — lets them transfer understanding
instead of rebuilding it.

Cheap: a note box in `concepts/decoding.md` and a line in the tutorial. No code.

- **Open question for Coulson, not for us to settle:** which course. He says 312 and flags
  his own uncertainty; the answer recorded when he asked a related question on PR #8 was
  that the walkthrough sits in CS 479, with whether it should be taught earlier left open.
  Reference the concept rather than a number until that is confirmed.

## Docs and tutorials

**#49 The pretrained checkpoint predates the enlarged corpus**

`data/pretrained/model.pt` was trained on 73,082 pairs. After #29 the corpus holds
86,430, so the shipped model has never seen 18% of the data it is meant to represent.

Nothing is broken: the model's held-out talks are still whole talks, still held out, and
still present in the corpus, because #29 is a strict superset. It is stale rather than
wrong.

Worth retraining because everything downstream reads off this one checkpoint. Tutorial 5
shows its translations, #40 measures decoding options on it, and its BLEU of roughly 5 is
the number a student meets first.

- Rerun `scripts/train_example_model.py`; about 20 epochs.
- Regenerate `docs/docs/_generated/decoding_sweep.json` afterwards, since #40's numbers
  are measured on this checkpoint. `--rerender` is not enough; the sweep itself must
  re-run, which takes about an hour on CPU.
- Check whether BLEU actually moves. 18% more data on a small model may buy very little,
  and that is worth knowing either way. If it does not move, say so in tutorial 5 rather
  than quietly retraining.

**#50 Tutorial 3 still teaches the wrong lesson about beam size**

#40 put the real measurement in `concepts/decoding.md`, but tutorial 3 is untouched: it
still sweeps `beam_size` over 1, 2, 3, 5, 10 and prints a table where every row is
identical, because its toy model is decisive. A student runs it, sees no difference, and
draws the obvious and wrong conclusion.

The cell is not wrong to exist — a sweep is the right thing to show. It just has nothing
to show on that model.

- Minimum: say so in the notebook. "Every row is identical because this model is too
  small to be uncertain; see the measured version on a real model" costs two sentences
  and removes the misconception.
- Better: have the cell assert the rows are identical and explain why, so it becomes a
  deliberate demonstration of when a sweep tells you nothing.
- Tutorial 5 is where a real sweep belongs, since it has the model for it.

**#53 The notebook gate is weaker than its green check implies**

CI checks out without Git LFS on purpose, so `data/example.tsv` is a pointer there and
`scripts/execute_notebooks.py` skips tutorials 2 through 5. **The job passes having run
exactly one notebook out of five**, and the check mark on the PR looks the same either
way.

That was the accepted trade when LFS went in, and it is still the right one. The problem
is that nothing says so at the point where someone reads the green check.

Found concretely in #50. That change adds an assertion inside tutorial 3, whose whole
purpose is to fire when the model stops being decisive and the surrounding explanation
stops being true. It cannot fire in CI, because tutorial 3 does not run there. It was
verified by running the gate locally, twice, which is not a thing that keeps happening on
its own.

Options, roughly in order of cost:

- **Say it in the check.** The job already prints "1/1 notebooks executed cleanly" and
  lists what it skipped. Make the job summary carry that so it is visible on the PR
  without opening the log. Cheapest, and removes the false impression.
- **Fetch LFS for the notebook job only.** One `lfs: true` on one job, roughly 28 MB per
  run. This is the option the keep-CI-light decision ruled out, but the reasoning there
  was about every job on a four-version matrix, not about one job that is the only place
  tutorials execute at all. Worth revisiting on those narrower terms.
- **A scheduled full run.** Nightly or weekly with LFS, so drift is caught within a day
  without touching per-PR cost.

Related: #51, which is the same shape from the other direction. That gate runs and does
not block; this one blocks and does not run.

Partial relief from PR #44: tutorial 6 needs no LFS artifact, so it does run in CI. The
gate now executes 2 of 6 rather than 1 of 5. The false impression is unchanged — the
green check still does not say what it skipped.

**#63 Three pages have no mkdocs nav entry, all waiting on PR #17**

`docs/mkdocs.yml` belongs to PR #17, and we are not stacking, so three pages shipped
without a nav entry. A page absent from nav is INFO rather than a warning under
`--strict` — verified on each branch, the build stays clean — so none of this blocks.
But each page is reachable only by direct link until PR #17 lands.

| Page | From | Place it |
|---|---|---|
| `tutorials/06-diagnosing-failures.ipynb` | PR #44 | Tutorials, after `05-real-translations.ipynb` |
| `reference/diagnostics.md` | PR #45 | API Reference, after `config.md` |
| `related-work.md` | PR #46 | Top level, near Home |

One more thing to undo at the same time: `related-work.md` refers to
`torchlingo.diagnostics` as plain code text rather than linking to
`reference/diagnostics.md`, because linking a page that does not exist on `main` fails
`--strict`. Once PR #45 has merged, make it a link.

**#64 Promote the tutorial 6 checks into `torchlingo.diagnostics`**

Tutorial 6 defines its five checks inline so the notebook is self-contained and each one
is short enough for a student to copy. That is right for the notebook and wrong as the
permanent home: a student cannot `import` from a notebook.

- `check_learning` — did the loss move, against `ln(V)` as the "learned nothing" floor
- `gradient_report` / `check_gradients` — sorts every parameter into frozen / dead / live
  after one backward pass. The one with no existing equivalent in the library, and the
  only check that *names* the broken parameter rather than reporting that something is
  wrong. Needs no training, runs in under a second.
- `check_generalization` — the sign of the val−train gap
- `check_contamination` — train/test source overlap
- `check_eval_mode` — `model.training` before inference

Needs tests and an API reference page. Deliberately deferred so PR #44 stayed reviewable.

**Done in PR #45.** Shipped as `src/torchlingo/diagnostics.py` with `CheckResult`,
`GradientReport`, `uniform_loss`, and the five checks (`check_loss_moved` rather than
`check_learning`); 43 tests, 8 doctests, `mkdocs --strict` clean. What remains is #65.

**#65 Tutorial 6 and `torchlingo.diagnostics` are two copies of the same checks**

PR #44 defines the five checks inline in the notebook; PR #45 ships them as a module.
Until one sources from the other they can drift, and the notebook is the copy a student
reads.

- Only after **both** have merged — doing it in either PR would stack it on the other.
- Keep the student seeing the logic; the pedagogy depends on it. Import the functions and
  show the source (`inspect.getsource`), or keep a short annotated call, rather than
  silently calling a black box.

**#66 Adopt `nltk.translate.gale_church`; split #29 into two different jobs**

#29 conflates two goals that want different tools, and proposes hand-writing an algorithm
that is already a dependency away.

`nltk.translate.gale_church.align_blocks()` ships with exactly the priors #29 specifies:
(1,1)=0.89, (1,2)=(2,1)=0.089, (2,2)=0.011, (0,1)=(1,0)=0.0099, and
`VARIANCE_CHARACTERS=6.8`.

- **To recover the 98 talks (~13k pairs)** — use Vecalign or Bertalign. Embedding-based
  aligners measurably outperform length-based ones; an English–Slovak evaluation
  (*Scientific Reports*, 2023) puts Vecalign and Bertalign significantly ahead, with
  hunalign and Bleualign behind. Gale-Church is the wrong tool for the production job.
- **To teach alignment** — implement it, because the implementation *is* the lesson, but
  pin the output against NLTK's as a test oracle rather than shipping ours as the only
  word on it.

Same split applies to #52 (Moore 2002): still a good lesson, superseded in practice.

**#68 Cite `torcheck` as prior art in the diagnostics docs**

`pengyan510/torcheck` already does PyTorch sanity checking, including frozen-parameter
verification. PR #45 does not mention it, which implies more novelty than is warranted.

The framings genuinely differ and both are defensible: torcheck registers with the
optimizer and asserts *during* training that parameters do or do not change;
`torchlingo.diagnostics` inspects *after the fact* and sorts parameters into
frozen/dead/live, naming the culprit.

- Add a short prior-art note to `docs/docs/reference/diagnostics.md`: what it does, how it
  differs, when to reach for it instead.
- Check its current maintenance status first. It was found, not evaluated.

**#69 Position the project on the curriculum, not the architecture**

Established 2026-09-20 by running the competition rather than reading about it.

[Joey NMT](https://github.com/joeynmt/joeynmt) (Kreutzer, Bastings & Riezler, EMNLP 2019)
is actively maintained, explicitly targets novices, and covers RNN and Transformer, beam
search with length penalty, attention visualization, BPE/word/char and multilingual
training. Verified hands-on: the shipped `transformer_reverse` toy config trains in
3m52s on CPU and reaches **93.62 BLEU** on test.

So the model/decoder/attention code is the *least* defensible part of TorchLingo. What no
comparable project appears to teach:

- diagnosis as a subject — everyone else teaches the path where things work
- empirical discipline through MT: controlled comparison, contamination, `ln(V)`,
  significance testing
- a corpus with a documented repair history
- documentation that executes

`docs/docs/related-work.md` (PR #46) states this publicly. The task is to let it steer
effort: feed #48's curriculum audit. It has already had one concrete consequence —
#2, #3 and #6 were descoped on 2026-09-22 on the strength of it.

- Open question: does Joey NMT belong *in* the syllabus as a comparison point — "here is
  the same thing as a configured toolkit" — rather than only in related work?

**#70 Print the sacreBLEU signature with every score**

Adopted from the Joey NMT baseline run, which logs
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0` next to every BLEU.

We already use sacreBLEU, so the signature is available and we are throwing it away. A
BLEU number without it is not reproducible by someone who was not there — which is
precisely the discipline tutorial 6 and #59 are trying to teach. Currently we teach it
and do not practise it.

- `compute_bleu` returns a `sacrebleu.metrics.BLEU`; surface `.get_signature()`.
- Print it wherever a score is reported: `scripts/compare_checkpoints.py`, tutorial 5,
  `concepts/decoding.md`, and the generated `decoding_sweep.json`.

**#71 Decide whether to report the Joey NMT breakage upstream**

Their shipped quickstart does not run on current PyTorch: `joeynmt/builders.py` passes
`verbose=False` to `torch.optim.lr_scheduler.ReduceLROnPlateau`, which PyTorch removed,
so `scheduling: "plateau"` raises `TypeError` before the first step. The toy config also
sets `use_cuda: True` and `fp16: True`, which fail on any CPU-only machine.

Found while running their tutorial as a baseline (2026-09-20). Not reported — filing an
upstream issue is outward-facing and is Eric's call.

- One-line fix upstream; a courteous thing to send given we cite them favourably.
- If yes: report from a clean clone, not the patched scratch copy.

**#72 Reconcile the Status table with the merged history**

The table at the top of this file has drifted: several rows marked Open have merged, and
the file's own rule is that completed work is removed rather than marked done. Flagged in
place on 2026-09-20 but not fixed, because reconciling it from memory rather than from
the merge history is how it got wrong in the first place.

- Walk `git log main` and the closed-PR list, then delete what has landed.
- Worth doing once the current review backlog clears (#73), not before, or it drifts again.

**#73 The review backlog is ten PRs deep**

As of 2026-09-20: PRs #17, #26, #34, #37, #38, #42, #43, #44, #45, #46. Only #26 is
approved.

This is the condition that produced the stacking failures recorded in #37 — stacks
existed because PRs sat waiting, not because the work needed sequencing. The convention
now forbids stacking, which means the backlog converts into *blocked* work instead:
#63 waits on #17, #65 waits on #44 and #45 together.

- Merging promptly is the actual fix. #26 is approved and can go now.
- Related: #62 (check the open-PR set after every merge), #61 (a PR closed itself and
  nobody noticed).

**#74 A broken anchor and 93 unexplained warnings in the docs build**

Two pre-existing docs-hygiene items, both visible in every `mkdocs build --strict` run
and neither currently failing it:

- `reference/visualization.md` links to
  `#torchlingo.models.transformer_simple.capture_cross_attention`, and no such anchor
  exists on that page. Reported as INFO, so `--strict` does not catch it — the same class
  of rot as #26, which *was* caught only because those links were WARNINGs.
- 93 `Div ... unclosed ... closing implicitly` warnings, from the Material card grids.
  Believed benign; never actually diagnosed. Confirm, then either fix the markup or
  record why it is acceptable so the next person does not re-investigate.

**#51 The docs gate reports but does not block**

#26 added a `docs` job running `mkdocs build --strict`. It runs on every PR and takes 30
seconds, but it only *reports*: the repo ruleset decides what blocks a merge, and the job
is not in it.

A check nobody is required to pass is a check that gets ignored the first time it is
inconvenient. Add "Build docs strictly" to the required status checks. Same for the
notebook gate if it is not already there. Repository settings, not a code change.

**#52 Try Moore (2002) if more of the corpus is wanted**

Raised by Eric: Moore improved on Gale and Church for bitext alignment.
[Moore (2002)](https://aclanthology.org/2002.amta-papers.14/) aligns in two passes — a
length-based pass like the one #29 ships, whose confident pairs train an IBM Model 1
word-translation model, then a second pass scoring length *and* word correspondence, with
the search confined to segments the first pass found plausible.

What it would buy here, honestly: not much, and that is why #29 shipped length alone.
Gale-Church recovered 13,152 of a possible 13,305 pairs at a quality indistinguishable
from the talks that never needed repair. The 153 it gave up are the ceiling.

Where it would matter:

- The limitation pinned by `test_a_long_dropped_sentence_is_handled_worse`. Length treats
  a long deletion as so improbable that a poor one-to-one scores better; a dropped
  sentence shares no *words* with anything, which is precisely what lexical evidence
  sees.
- The 89 talks present in only one language stream. Length cannot help there and neither
  can Moore, so those are gone regardless.
- Any future corpus noisier than this one. The method is the transferable part.

Also a genuinely good teaching progression if #48 wants one: length alone, then why it
fails, then lexical evidence. Cited in `concepts/data-pipeline.md` and in the module
already, so the pointer exists whether or not the code follows.

**#47 Docstring examples are not executed, and 30 of them fail**

`pytest --doctest-modules src/torchlingo` reports **30 failed, 24 passed**. Nothing runs
doctests, so CLAUDE.md's "keep examples runnable and concise" is unenforced and the
examples students are most likely to copy have rotted.

Found the usual way: an example written in #46 asserted `looks_aligned()` on a
single-row frame, which cannot pass, because one row has no length variation to
correlate. It was wrong the day it was written and nothing noticed.

Failures span `preprocessing/base.py`, `multilingual.py`, `multilingual_helpers.py`,
`sentencepiece.py` and `training.py`, among others. Only `preprocessing/alignment.py` is
fixed so far, in #29.

- Fix in batches by module, since the failures are unrelated to each other.
- Then add `--doctest-modules` to the test job so it stays fixed. Fixing without gating
  just resets the clock, the same lesson as #26.
- Same shape as #14 and #16: two things that must agree, with nothing checking.

**#48 Audit what we have built for pedagogical value, and write down the sequencing**

Enough has accumulated that nobody can now say what a student is meant to learn, in what
order, or where the gaps are. The material was built task by task, each one justified on
its own, and never against a curriculum.

**Does something like this already exist?** Partly, and not enough.
`docs/docs/tutorials/index.md` has a "Learning Path" section, but it is student-facing
navigation over the five notebooks: a card per tutorial with a one-line description. It
states no outcomes, covers none of the concept pages or library modules, and predates
most of what exists now. It is a table of contents, not an audit.

What the artifact should carry:

- **Sequencing.** What depends on what. Some of this is already load-bearing and
  undocumented: tutorial 3 loads the checkpoint tutorial 2 trains, and #40's lesson only
  works on a model that is wrong often enough to be interesting, which is why it uses
  tutorial 5's checkpoint rather than tutorial 3's toy.
- **Learning outcomes per unit**, stated as what a student can *do* afterwards, not what
  was covered.
- **Coverage gaps**, which is the real output. Likely candidates on a first glance:
  training dynamics beyond "loss goes down", evaluation beyond BLEU, and anything about
  why a model fails rather than how it works.
- **Redundancy**, the other half. Beam search is now explained in `concepts/decoding.md`,
  reimplemented in tutorial 3, and visualized in two places.

Worth auditing against, since each was justified pedagogically when it was built:

| Where | What it teaches |
|---|---|
| Tutorials 1-5 | The end-to-end path, toy model through real translations |
| `concepts/decoding.md` | Greedy vs beam, cost, what the knobs buy (#40), search framing (#41) |
| `concepts/data-pipeline.md` | Loading, cleaning, alignment detection (#46) and repair (#29) |
| `concepts/vocabulary.md` | Words vs subwords |
| `concepts/models.md`, `training.md`, `what-is-nmt.md` | Architecture and training |
| `reference/visualization.md` | Attention maps, beam search traces |
| Generated measurements | `decode_bench`, `decoding_sweep`, `alignment_diagnosis`, `realign_report` |

**First pass written: `notes/CURRICULUM.md`.** What it found:

- Two load-bearing dependencies nobody had written down. Tutorial 3 cannot run without
  tutorial 2's checkpoint, and tutorial 3's model is too small to demonstrate the thing
  tutorial 3 teaches, which is why #40 had to measure on tutorial 5's model and why #50
  exists.
- Five coverage gaps, the largest being **why a model fails**. Everything teaches the
  machinery working; nothing teaches diagnosis, which is what a student actually hits.
  Others: evaluation beyond BLEU, training dynamics when training goes wrong, how much
  data is enough, and inference cost in practice.
- Beam search now appears four times and attention three. Defensible, but currently by
  accumulation rather than decision.

Still open, and genuinely instructor-owned: the outcomes in that file are reverse-
engineered from the material, so they describe what exists rather than what the course
needs. Four questions are listed at the bottom of it for you. #42 is the same shape.


**#46 Teach the corpus repair instead of doing it silently**

Coulson's suggestion on his approval of #19: the data cleaning "could be recorded and used
as an example for cleaning data. Instead of doing it quietly in the background we could
explain it to the students to reinforce the idea of clean data."

He is right, and right about its weak part too. The repair currently lives entirely in
`scripts/realign_corpus.py`, so the single most pedagogically loaded thing in the repo is
the one thing no student sees. What makes it teachable is that the diagnosis is already
quantified and the numbers are dramatic:

```
                      broken    repaired
length correlation     0.001       0.969
anchor agreement        1.4%       38.9%
```

That is a complete lesson in how to tell misaligned parallel data from your own modeling
mistake, which is exactly the confusion a student cannot resolve on their own.

- Use the alignment diagnosis as the spine, not the cleaning filters. Coulson notes the
  stage-direction removal "does almost nothing so it may be a poor example," and he is
  right: it removes exactly one line. It belongs as a footnote at most.
- The two checks are cheap enough to run live in a notebook on the shipped corpus, and
  the broken state can be reconstructed by re-zipping the columns, so students can see
  both numbers move.
- `tests/test_data_integrity.py` already encodes the thresholds and the reasoning. The
  tutorial and the test should quote the same source rather than restate the numbers.
- Open question for the author: standalone tutorial, or a section inside tutorial 1 where
  the corpus is first loaded.


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

