# Post-norm, the split is in, and one thing I am not promising

**2026-09-27.** Answering your baton. First hand-off under the new one-file-per-hand-off
layout — see `notes/handoff/README.md`; the old append-only logs are in `archive/` whole.

## 1. Pre-norm or post-norm: **post-norm.** Both your 8b slides stand

`SimpleTransformer` constructs `nn.Transformer` without passing `norm_first`, so PyTorch's
default of `False` applies. Confirmed on the built module rather than read off the
constructor:

```python
m = SimpleTransformer(...)
m.transformer.encoder.layers[0].norm_first   # False
```

That is post-norm — sublayer, then residual add, then LayerNorm — the ordering of the 2017
paper. So the encoder-block anatomy slide is right, and the residual-stream slide is right
that the channel gets rescaled at every block.

Worth saying how it was settled, because you will hit this class of question again: the call
site does not mention the argument, so no amount of reading `transformer_simple.py` answers
it. I built the model and read the attribute. If a slide ever depends on a PyTorch default,
that is the cheap way to be sure.

**Now a real option, and the default is pinned.** Eric's call the same day: TorchLingo should
also implement the modern pre-norm standard. So the answer above has changed shape slightly, in a
way that matters to your slides:

- **`norm_first` is now a parameter** on `SimpleTransformer` and on `Config`. `norm_first=True`
  gives pre-norm, `x + Sublayer(LayerNorm(x))`. Same parameter count, different arithmetic —
  verified, and it trains.
- **The default is still post-norm**, and now *asserted* rather than inherited. The test checks
  every encoder and decoder layer, and its failure message names your two slides and the docs
  page. A default that drifted used to be able to invalidate them with no diff anywhere near a
  slide; it no longer can.
- **`docs/docs/concepts/models.md`** shows both formulas side by side, says the default is the
  paper's, and explains why almost everything published since 2017 is pre-norm.

**What this changes for 8b:** your two slides still stand exactly as drawn, because the default
did not move. But there is now a legitimate half-sentence available if you want it — that the
arrangement you are diagramming is the paper's, that modern implementations invert it, and that
the library will do either. That is a genuine piece of the field's history rather than an
implementation detail, and students *will* meet pre-norm the moment they read any recent code.
Entirely your call whether 8b has room for it.

## 2. Your check is green. `8a` and `8b` parse, and three notebooks moved

**Done and in PR #127.** Lecture identifiers are **strings** now, because `8a` has no
integer form — that was the real cost of the scheme, and it is paid.

A normalizer means nothing else had to change: the integer `9` a notebook already held, your
schedule's `9`, and a filename's `09` all land on one key.

Your three moves are applied, each validated before it was written:

| notebook | now serves | role |
|---|---|---|
| `06-diagnosing-failures` | `[9]` | reading |
| `03-inference-and-beamsearch` | `[10]` | reading |
| `04-attention-and-alignment` | `["8a", "8b"]` | reading |

**The first was not optional, and this is the part worth your attention.** Lecture 8 stopped
existing, so tutorial 6 was pointing at nothing — the check said so by name. Had you
renumbered 9 onward instead, tutorial 6 would have kept claiming "8", the row would still
have existed, and **nothing anywhere would have complained** while it pointed at the wrong
lecture. The suffix scheme is the one the tooling can protect. Thank you for choosing it.

Two side effects you should know about rather than discover: tutorials 3 and 4 lost their
`note` fields. Both described the old pairings, so they are correctly gone, but that was a
consequence of restamping and not a judgement I made.

## 3. Your three notebooks are in, checked rather than trusted

`lecture-04-regex-refresher`, `lecture-10-comet-install`, `lecture-12-llm-context`. I
re-verified independently of your scan: zero outputs, zero execution counts, no tokens or
keys, no INSTRUCTOR copies, badges present, metadata sound. All three are in the mkdocs nav
now — without that the docs build carries pages nothing links to.

**Your roles read correctly to me**, so I have not changed them. `regex-refresher` as
`reference` at Lecture 4 is right given Learning Suite posts it on the Lecture 3 tab too.

**One gap this opened**, and it is now Task #154: the notebook gate runs `docs/docs/tutorials`
only. `docs/docs/course/` now holds seven notebooks and **none of them is executed by
anything.** That matters more for yours than for mine, because yours are opened in the room
on a clock. Several need Drive, a download or a HF token so they cannot all run in CI, but the
self-contained ones can, and the `needs` field already says which are which.

## 4. The A8 kickoff notebook is **yours to write**, and here is what I found trying

**Eric's call, 2026-09-27:** you write it, because you have full visibility into the
assignment details and I do not. I am making suggestions instead. **Plan the deck slide** —
the notebook is no longer waiting on me.

**Your structural argument was right and nobody is arguing with it.** Lectures 4, 5 and 6 each
had an activity that started the assignment; the largest assignment in the course has none.

I got far enough into a draft to hit the things you cannot see from the deck side. Eleven of
them, sharpest first. The first three change the design, not just the code.

**1. Cap sentence length BEFORE splitting, not after.** Your order is dedupe → split →
contamination → cap. The cap *removes pairs*, so capping after the split turns 100,000
training pairs into fewer than 100,000, and the student no longer meets the assignment's own
floor. Capping first makes the split sizes the sizes they submit.

**2. Dedupe-first makes contamination structurally impossible, which changes what that step
teaches.** Once source-side duplicates are gone, no split can put one source in two sets — so
the check cannot fail unless a cell was run out of order. Frame it as *verify, don't trust*,
and make it **raise** rather than print: a student who reads past a printed warning carries a
contaminated split into Assignments 9, 13 and 14.

**3. "Write the six files" is ambiguous and I could not resolve it from here.** TorchLingo
reads **TSV**, so the notebook needs three of those; six suggests `.src`/`.tgt` per split,
which is the parallel format A8 probably wants submitted. My draft wrote both — nine files,
one cell, no guessing. **Your call**, since you can see the handout.

**4. `train_losses[0]` is not the first loss.** It is the *average over epoch 1*, already well
below `ln(V)`. If the deck tells students to compare their first loss with `ln(V)` — and it
should — the notebook must surface the true first batch via `log_every`, or the comparison
looks like a failure when it is fine. This one will generate office-hours traffic.

**5. `uniform_loss(V)` already exists** in `torchlingo.diagnostics`. Use it rather than
`math.log`, so the notebook and the deck quote one implementation.

**6. Pass `num_workers=0` explicitly.** TorchLingo's own default is 4, applied to all three
loaders. Measured on the spawn start method: **~23 seconds of fixed overhead that never won at
any corpus size tested.** That is Task #158 and the library default has not changed yet, so
the notebook must override it.

**7. `split_data` cannot do this split.** It takes *ratios* and shuffles by row; A8 needs exact
counts (2,000 / 2,000 / the rest). Write the split inline in the notebook — which matches your
own design call that the notebook supplies the code — and seed it, or a re-run reshuffles the
test set and the student's BLEU stops being comparable with their own earlier BLEU.

**8. The word "instructor" is banned in this tree.** `notebook_meta.py --check` greps for it
case-insensitively, to keep worked solutions out of the public docs. Write "teaching staff".
It will fail your lint otherwise and the message will not be obvious.

**9. Course-notebook hygiene**, all enforced: no committed outputs, no `execution_count`, a
Colab badge, nbformat 4.0 with no cell `id`s. Stamp metadata with
`python scripts/notebook_meta.py --stamp <nb> --serves 8a --role activity --leads-to A8`.

**10. `leads_to: ["A8"]` validates cleanly** from a notebook serving 8a — A8 is due at Lecture
9 in the schedule, so the tooling reads it as a head start rather than a late reference.

**11. Make the missing-GPU branch stop them.** 60 epochs over 100K pairs on CPU is days, not
hours, and a student who misses that line loses the week.

**The gap I could not close, and you should decide how to handle:** steps 2 and 7 mount Drive
and write to it, so **this notebook cannot be executed here or in CI** — the executor only runs
`docs/docs/tutorials/`. Every other notebook in this repository is verified by running it. The
cheapest fix is that **Eric runs it once in Colab against a real A5 corpus before Wednesday**.
Without that it reaches eighteen students, in a room, on a twenty-minute clock, having never
run end to end.

## 5. Two answers to things you flagged

**#118 is still Coulson's and still unmeasured.** I have nothing to add except agreeing with
your framing: it now blocks a live decision, not a documentation claim. Raising the epoch
count on the 11.7M model is the more expensive path to the worse result *if* the 56.4M
configuration fits, and nobody knows whether it fits.

**A8's "What To Do" overlapping the kickoff notebook** is Task #155 and I have not acted on
it either — you were right that it is Eric's, because it changes an assignment students are
about to start.

## 6. Where your list and mine agree

Your items 1 to 6 map onto Tasks #138 (done), #152, #121, the three moves (done), your three
notebooks (done), and #153 with #118. **I have not found anything in your "waiting on the
repository session" list that is wrong in either direction.**

New on my side since your baton: **#154** the course notebook gate, **#155** the A8 handout
overlap, and **#153** which this entry closes.

## 7. Asking a third time: **A15 is still missing from the schedule**

This is a repeat, and I am flagging that it is a repeat rather than dressing it up as new. It was
Question 4 of my ninth entry, it came back unanswered, and that entry is now in `archive/` where
nobody will open it again. A question asked three times with no acknowledgement is a different
signal from one asked once, so I would rather say so plainly than have it quietly age out.

**Eric's call, 2026-09-27: this one is yours to fill in.** Filling the row needs the lecture
detail you hold — what A15 actually asks for and when it is due — and neither of us here can
invent it.

Where it now stands after your last update, so the ask is precise:

- **A1, A2, A3** — resolved, they arrived with your schedule change. Thank you.
- **A7** — confirmed deliberate. Nothing to do.
- **A15** — **still absent from the roadmap entirely.** Not inferred from a numbering gap; the
  string does not appear.

**Why it is no longer only bookkeeping.** `leads_to` validates notebook declarations against that
assignment table. A notebook that correctly says it jump-starts A15 is *rejected*, because the
schedule does not name A15. So the missing row can now block a correct declaration rather than
merely look untidy — which is what moved this from documentation to a gate.

If A15 does not exist and the numbering genuinely skips it, say that and I will close the task on
your word. Either answer ends it; silence is the only outcome that does not.

## 8. Eric wants an **NLLB spotlight lecture**

New, from Eric on 2026-09-27, and it is a lecture-design request so it lands with you:

> We should have an NLLB spotlight lecture and dig into what makes NLLB special as a model.
> Certainly the data is part of that, but they've also made some architectural decisions and
> curriculum decisions that we should understand. NLLB is generally state of the art and
> deserves this level of attention.

**The framing to take from that:** the course already treats NLLB as a data story — 200
languages, mined bitext. Eric's point is that the data is only one of three legs, and the
architecture and training-curriculum decisions are the ones a student never hears about.

Three axes, which is a natural three-act structure for a deck:

1. **Data.** Bitext mining with LASER3 rather than crawling alone; back-translation; and
   **FLORES-200** as the evaluation benchmark that made 200-language comparison possible at all.
   Also the toxicity and quality filtering, which is a rare chance to show that a state-of-the-art
   result rests on unglamorous data hygiene.
2. **Architecture.** A **sparsely gated mixture-of-experts** Transformer, so only some experts run
   per token — capacity without proportional compute. The regularization they added to stop
   low-resource pairs overfitting against high-resource experts is the interesting part, and it
   connects directly to Lecture 13. The published dense models students can actually run are
   *distillations* of the large sparse one, which is worth saying out loud because it explains why
   "NLLB-200" names several quite different things.
3. **Curriculum.** Language pairs are not all introduced at once, and the balance between
   high-resource and low-resource pairs is managed deliberately rather than left to corpus size.
   This is the leg with no coverage anywhere in the course right now.

**Where it fits.** Lectures 13 (low-resource) and 14 (multilingual, zero-shot) are where it
belongs — it is the worked example both of those lectures currently lack, and it would let 14 stop
being abstract. Your call whether it is a new slot or absorbs one.

**One caution, and I would rather say it than let it cause a wrong slide.** I have stated the
above from memory of the 2022 paper, not from a fresh read. The shape is right; **the specifics —
expert count, parameter counts, the exact regularization name, the precise curriculum schedule —
need checking against the paper before they reach a slide.** I can do that verification here if
you want numbers rather than structure; say so and it is a short job. Do not transcribe my
figures directly.

## 9. Housekeeping

- `notes/legacy-f2025/` is committed. Your README recording what last year's notebook got
  wrong is the most useful thing in it — it turned #121 from a guess into a scope.
- One correction to my own record: I dated yesterday's hand-off `2026-09-27` when it was the
  26th. Fixed, including one past heading, which is the only time I have edited an entry
  after the fact.
- The roadmap's 13,151 characters are indeed back, and the generated notebook map regenerated
  cleanly on top of your schedule changes.
