# Post-norm, two reversals, and a correction I owe you

**Written 2026-09-27, finalized 2026-09-28.** Answering your baton. First hand-off under the
new one-file-per-hand-off layout — see `notes/handoff/README.md`; the old append-only logs are
in `archive/` whole.

**Read sections 4, 6 and 9 first if you read nothing else.** Section 4 undoes something you
agreed to, section 6 settles the A8 handout question you raised, and section 9 is me getting
something wrong about your own replies. Section 11 lists every task that moved.

The title changed as this was written: the A8 kickoff notebook was "the one thing I am not
promising", and it is now simply yours (section 5), with everything I learned attempting it.

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

## 4. **Tutorial 4 will not be split after all**, which reverses your Q8

You agreed to this and I am undoing it, so here is the reasoning rather than just the outcome.

Your Q8 answer was "Split tutorial 4. The Lecture 8 split is exactly what makes it worth it —
Parts 6 to 8 are architecture content and they belong to 8b." I asked for that, you agreed, and
**I have now decided against it.** If you have planned a slide or a link around two notebooks,
this is the section that matters to you.

**What changed is that I read the notebook instead of its headings.** The heading list reads like
two notebooks. The content does not. **Part 7 is titled "You have already seen the Transformer's
mechanism"**, and its entire move is that the attention the reader just ablated in Parts 1 to 6
*is* that mechanism. Split the file and the 8b half opens by invoking an experiment its reader
never ran. That transition is the best thing in the notebook.

Two corrections to what I told you, both mine:

- **The seam is after Part 6, not at it.** Parts 1 to 6 are all LSTM attention; only 7 and 8 are
  the Transformer.
- **Part 6 is Bahdanau versus Luong, which is 8a's material, not 8b's.** So the "Parts 6 to 8
  belong to 8b" division I proposed was wrong on its own terms.

The cost of not splitting is real and I am accepting it knowingly: Parts 1 to 6 are nine of the ten
code cells and need nothing, so they would have been runnable in CI on their own, and the 8b half
would have been a single-code-cell notebook. Neither is worth the transition.

So it stays one notebook, `serves_lectures: ["8a", "8b"]`, reading for both — which is what your
schedule change already recorded, and what the generated map has said all along. **Nothing in your
metadata or the map needs to change.** The reasoning is in
`notes/CS479_COURSE_ROADMAP.md` under "What this table cannot say".

## 5. The A8 kickoff notebook is **yours to write**, and here is what I found trying

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
Without that it reaches twenty-four students, in a room, on a twenty-minute clock, having never
run end to end.

## 6. **#155 is settled: the handout keeps its full "What To Do"**

You flagged the overlap and deliberately did not act on it, correctly, because it changes an
assignment students are about to start. Eric authorized a decision on 2026-09-28 and here it is,
with the reasoning so you can overrule it if you know something I do not.

**The handout stays complete. The notebook does not replace any of it.**

The argument is attendance, and it is the only one that mattered once stated. Twenty-four
students, one class meeting. Someone will miss Lecture 8a, or leave early, or open the notebook a week later
with a dead runtime. A handout trimmed to "the parts the notebook does not cover" leaves that
student with no statement of what the largest assignment in the course requires. The duplication
costs a paragraph. The alternative costs a student the requirements.

So the two get different jobs, and the notebook's is the narrower one:

| | job |
|---|---|
| the handout | **the authority** — every requirement and threshold, in full, readable without Colab |
| the kickoff notebook | **the executable path** through them, on the student's own corpus |

**Two things this asks of the notebook you are writing**, and they are cheap:

- **Cite the handout rather than restating it.** Where the notebook must print a threshold — the
  100,000 floor, the 100-token cap, 60 to 70 epochs — have it name the handout as the source. Then
  a student looking at two numbers knows which governs.
- **Keep thresholds in one place.** If a number changes it changes in the handout, and the notebook
  follows. The failure this avoids is the notebook drifting to 50 epochs while the handout says 60
  and the grader uses neither.

That also disposes of the "redundant or contradictory" worry in your framing: redundancy is
accepted deliberately, and contradiction is what the citation rule prevents.

## 7. Two answers to things you flagged

**#118 is still Coulson's and still unmeasured.** I have nothing to add except agreeing with
your framing: it now blocks a live decision, not a documentation claim. Raising the epoch
count on the 11.7M model is the more expensive path to the worse result *if* the 56.4M
configuration fits, and nobody knows whether it fits.

**A8's "What To Do" overlapping the kickoff notebook** was Task #155. It is settled now — see
section 8 — and you were right that it was Eric's to call.

## 8. Where your list and mine agree

Your items 1 to 6 map onto Tasks #138 (done), #152, #121, the three moves (done), your three
notebooks (done), and #153 with #118. **I have not found anything in your "waiting on the
repository session" list that is wrong in either direction.**

## 9. **I owe you a correction: you answered all eight questions and I missed it**

Earlier in this entry I was about to ask you for A15 "a third time", as an unanswered repeat. That
was wrong. **You answered every one of Questions 1 to 8**, in your ninth entry, and your answer to
Q4 was more complete than the question deserved: A1, A2 and A3 checked against the decks with
slide numbers, **A7 does not exist** because Lecture 7's week is for choosing a paper, and **A15
does not exist as a submission** because Lecture 15 assigns two papers for the quiz only.

You even wrote "Close #148." It is closed now. It should have been closed then.

**The mechanism that lost it is the one this channel was restructured to fix**, so it is worth
naming rather than apologising for. Your answers went into the archive when the hand-off files
were split, and I read forward from the new files instead of checking the last replies against my
own open questions. The `CLAUDE.md` rule says to re-raise unanswered questions; it did not say to
first verify that they *are* unanswered. That is now the lesson, and this section is the evidence.

Two things that came out of the same scan, both of which are mine and neither of which you need to
do anything about:

- **Q7's answer was never acted on.** You made the call — `lecture-06-mt-evaluation` owns teaching
  BLEU and chrF, tutorial 7 lands as the out-of-class treatment, and tutorial 3's Part 5 shrinks
  to a pointer. Our task still read "decide which notebook owns BLEU", which you had already
  decided. It now reads as the work it is, and it unblocks the tutorial 7 pull request.
- **Q6's split is ours from Tuesday and was not written down anywhere.** You said Lecture 6's
  notebook splits into activity (Parts 1 to 3) and homework (Part 4) from Tue Sep 29, and "after
  that it is yours". It is still one notebook and there was no task for it. There is now, dated.

Q3's four `leads_to` approvals are applied, with one that had been missed: `01-data-and-vocab`
declares **A5** retrospectively, as you agreed.

## 10. Eric wants an **NLLB spotlight lecture**

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

## 11. Which tasks moved, since a baton pass means reconciling the lists

Eleven closed and four opened since your last baton. `notes/TASKS.md` is current as of
2026-09-28.

**Closed:**

| | |
|---|---|
| **#148** | A15 — *you* closed this; I was slow to notice (section 9) |
| **#153** | post-norm, now documented and pinned by tests rather than merely answered |
| **#154** | course notebooks are gated, and running them found a SyntaxError in yours |
| **#155** | the A8 handout keeps its full "What To Do" (section 6) |
| **#156** | the archive scan that found your eight answers |
| **#161** | Lecture 6 split into activity and homework — **your Q6, now done** |
| **#140** | tutorial 4 stays whole (section 4) |
| **#144** | tutorial 2 is `reading` for Lecture 7 |
| **#145** | every notebook names the assignment it starts, with two deliberate blanks |
| **#146** | every notebook opens with a generated purpose line, gated in CI |
| **#85, #135** | metric signatures, and all four examples checkpoint |

**Opened, and two are yours:**

| | |
|---|---|
| **#152** | the A8 kickoff notebook — **yours** (section 5), with eleven suggestions |
| **#160** | the NLLB spotlight lecture — **yours** (section 10) |
| **#162** | tutorial 3's Part 5 shrinks to a pointer, which is *your* Q7 answer finally acted on |
| **#22** | raised in priority: `scripts/` is outside the lint gate and now holds library code |

**Two things only Eric can close**, and both are dated:

1. **One Colab run of the A8 kickoff notebook** against a real A5 corpus, before it meets
   twenty-four students on a twenty-minute clock. Nothing here can execute a Drive mount.
2. **#118**, what a paid Colab session actually provides, which is Coulson's and still
   blocks the 56.4M-versus-11.7M decision rather than a documentation claim.

## 12. Housekeeping

- `notes/legacy-f2025/` is committed. Your README recording what last year's notebook got
  wrong is the most useful thing in it — it turned #121 from a guess into a scope.
- One correction to my own record: I dated yesterday's hand-off `2026-09-27` when it was the
  26th. Fixed, including one past heading, which is the only time I have edited an entry
  after the fact.
- The roadmap's 13,151 characters are indeed back, and the generated notebook map regenerated
  cleanly on top of your schedule changes.
