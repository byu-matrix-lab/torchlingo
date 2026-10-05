# Lecture 7 notebook: the student chooses the held-out phrase, predicts, and runs a contrast

**Baton to you, 2026-10-05.** One notebook, one change, decided by Eric today: "Let's fix the
Lecture 7 notebook now, since it's past." A7 has been graded, so nothing here affects a live
assignment; the change is for the notebook students meet next offering and for anyone who opens
it from the Content page now. `from-cowork/2026-09-30-8a-slide-reordered.md` is still live if you
have not picked it up; this file does not replace it.

## Why

Roadmap v10 (your copy is refreshed; the new section is "Learning outcomes, and who does the
work") rates `lecture-07-toy-model` **watch**: Part B's held-out phrase is fixed, the second
held-out experiment is optional, and a student can produce the Part C printout without reading
Part B. The fix: the student chooses the phrase, writes a one-line prediction before the cell
will run, and a second held-out phrase of the other kind runs as a required contrast. The report
then carries one decision and one contrast instead of a fixed printout. Same model, same
training, about one more minute of CPU.

The notebook's text rules stand: no dates, no weekdays, no due dates, and it must still run end
to end from its badge. The defaults below make it run unedited in CI (the default `HELD_OUT` is
the current one; `PREDICTION` is the only new required edit, and the assert that enforces it
must not fire in CI, so see the note at the end).

## The edits, cell by cell

Cell numbers are from `docs/docs/course/lecture-07-toy-model.ipynb` as of 2026-10-01 (25 cells,
index 0 to 24).

### md cell 1, the opening table: two rows change

Row B becomes:

> | **B. Outside class** | on your own | the same model with a phrase of your choice held out, a second phrase as the contrast, two scores that disagree, and a training run that survives a disconnect |

Row C becomes:

> | **C. Turn in** | on Learning Suite | your prediction, the printed report, and two sentences |

### md cell 17, the Part B opener: replace from "## Part B" to the end of the cell

Keep the "Report back, in class" paragraph and the rule above it. Replace everything from
`## Part B, outside class: hold one phrase out` with:

```
## Part B, outside class: hold one phrase out

Everything above scored the model on sentences it was trained on. That is not a test; it is a
memory check. So: choose one of the twelve phrases, remove it from the training set, train a
fresh model on the other eleven, and ask it to translate the one it never saw.

Choose with the corpus in front of you. The last four phrases share every word between them, so
a model that holds one of them out has seen every piece of it somewhere else. Most of the others
contain a word that appears nowhere else. Which kind you pick decides what the experiment can
show.

**Before you run anything, write your prediction in the cell below:** will the model produce the
right translation, and why? The cell refuses to run until you have.
```

### code cell 18: replace the whole cell

```python
# Choose any pair from BASE_PAIRS. The default is one of the four that share all their words.
HELD_OUT = ("The dog sleeps", "El perro duerme")

# Your prediction, in one sentence: will the model translate HELD_OUT correctly, and why?
PREDICTION = ""

assert HELD_OUT in BASE_PAIRS, "HELD_OUT must be one of the twelve pairs above."
assert PREDICTION.strip(), "Write your prediction before running this cell."

import sacrebleu


def hold_out(pair, label):
    """Train on the other eleven phrases; score the eleven (seen) and the one (unseen)."""
    train_pairs = [p for p in BASE_PAIRS if p != pair]
    assert len(train_pairs) == 11
    write_corpus(train_pairs, f"data/train_{label}.tsv")
    ds, loader, m = build(f"data/train_{label}.tsv")
    # One checkpoint folder per held-out phrase, so a different phrase trains a new model rather
    # than continuing an old one.
    name = "lecture-07-without-" + pair[0].lower().replace(" ", "-")
    train_model(m, loader, num_epochs=EPOCHS, device=device, config=config, gradient_clip=1.0,
                log_every=0, checkpointer=TrainingCheckpointer(name, checkpoint_dir=f"checkpoints/{name}", verbose=False))
    seen_src = [p[0] for p in train_pairs]
    seen_ref = [p[1] for p in train_pairs]
    seen_hyp = translate(m, ds, seen_src)
    unseen_hyp = translate(m, ds, [pair[0]])[0]
    # Exact match is the honest metric for three-word phrases. chrF works on short text too.
    # One hypothesis stream, one list of reference streams.
    return dict(
        pair=pair, unseen_hyp=unseen_hyp,
        exact_seen=sum(h == r for h, r in zip(seen_hyp, seen_ref)),
        chrf_seen=sacrebleu.corpus_chrf(seen_hyp, [seen_ref]).score,
        chrf_unseen=sacrebleu.corpus_chrf([unseen_hyp], [[pair[1]]]).score,
        bleu_seen=sacrebleu.corpus_bleu(seen_hyp, [seen_ref]).score,
        bleu_unseen=sacrebleu.corpus_bleu([unseen_hyp], [[pair[1]]]).score,
    )


def report(r):
    print(f"Held out {r['pair'][0]!r}")
    print(f"  the 11 phrases it trained on:  {r['exact_seen']}/11 exactly right,  chrF {r['chrf_seen']:5.1f}")
    print(f"  the 1 phrase it never saw:     chrF {r['chrf_unseen']:5.1f}   -> {r['unseen_hyp']!r}   (wanted {r['pair'][1]!r})")
    print(f"  BLEU, seen {r['bleu_seen']:.1f}   unseen {r['bleu_unseen']:.1f}")


mine = hold_out(HELD_OUT, "mine")
report(mine)
```

### code cell 19: replace the whole cell (it was the scoring; scoring now lives in `hold_out`)

```python
# The contrast: a phrase of the other kind. "Good night" shares no word with any other phrase
# (night -> noches appears nowhere else), so there is nothing to recombine from. If the phrase
# you chose is of that kind, the contrast is one of the four that share their words.
SHARED = {("The cat sleeps", "El gato duerme"), ("The dog runs", "El perro corre"),
          ("The cat runs", "El gato corre"), ("The dog sleeps", "El perro duerme")}
CONTRAST = ("Good night", "Buenas noches") if HELD_OUT in SHARED else ("The dog sleeps", "El perro duerme")

contrast = hold_out(CONTRAST, "contrast")
report(contrast)
```

### md cell 20: replace the "What to look at" paragraphs; keep the resume subsection

Replace everything above `### Also in Part B: watch a training run survive a disconnect` with:

```
**What to look at.** Both seen lines should read 11/11 and chrF 100: each model memorised its
training set, and a score measured on training data is not a measurement. The two unseen lines
are the only numbers that say anything about *translation*, and they should differ. A held-out
phrase whose every word appears elsewhere can be recombined: *El perro duerme* from *perro* in
one phrase and *duerme* in another. If the model produced something else (often *El perro
corre*), sixteen passes over eleven phrases were not enough to learn that *sleeps* is *duerme*
whichever animal is doing it. A phrase with a word the model never saw cannot come out right at
all, however long you train; look at what it produced instead.

Set the two lines against your prediction before reading on.

BLEU is 0.0 on every line because a three-word sentence has no 4-grams; that is the metric's
shape, not the model's quality, which is why this notebook reports chrF and exact match.
```

The resume subsection and code cell 21 are unchanged.

### md cell 22: delete the last paragraph

Remove the paragraph beginning `**Optional, five minutes:** change HELD_OUT to ("Good night", ...`.
It is now the required contrast. The three bullets above it stay.

### code cell 23, Part C: replace the whole cell

```python
print("=== Lecture 7 toy model: report ===")
print(f"Target vocabulary V = {V};  ln(V) = {uniform_loss(V):.3f}")
print(f"Final training loss, all 12 phrases:  {losses[-1]:.3f}")
print(f"Prediction: {PREDICTION.strip()}")
for r in (mine, contrast):
    print(f"Held out {r['pair'][0]!r}:  seen {r['exact_seen']}/11 exact, chrF {r['chrf_seen']:.1f};  "
          f"unseen chrF {r['chrf_unseen']:.1f}, produced {r['unseen_hyp']!r} (wanted {r['pair'][1]!r})")
print(f"BLEU on every line: {mine['bleu_seen']:.1f}")
print()
print("1. Was your prediction right? What does the difference between your two held-out phrases say about what the model learned?")
print("2. Assignment 8 trains on 100,000 pairs. Name one thing you will do differently because of this exercise.")
```

md cell 24 (the closing pointer to Lecture 8a) is unchanged.

## Running unedited: the assert and CI

Eric's rule is that every notebook runs end to end from its badge, and `PREDICTION = ""` with
an assert breaks that on purpose for a student. Two ways to keep CI green; your call:

1. Your execution harness sets `PREDICTION` before running (an environment variable the cell
   reads as a fallback: `PREDICTION = os.environ.get("CS479_PREDICTION", "")`), and the notebook
   stays a student-facing worksheet with one required edit, like the regex refresher's blanks.
2. The assert becomes a loud print ("No prediction written. Write one before you run Part B;
   the report will say it is missing.") and the report prints `Prediction: (none written)`. The
   notebook then runs unedited, and the grader sees the gap.

Eric's preference, if you want one without asking: option 2 keeps the badge rule intact with no
harness special case, and the empty line in the report is visible to whoever grades it.

## Course side, already done or noted

- The Lecture 7 deck's Part B slide now matches (choose, predict, contrast); the deck is past,
  so it was edited live.
- A7's Learning Suite text quotes the two questions; question 1's new wording goes in at the next
  offering, with the rest of the assignment texts. The transcript
  `torchlingo-private/notes/assignments/A07-directions.md` is yours to annotate.
- Roadmap: the Lecture 7 scope note and the notebook index row will say "a phrase of the
  student's choice, with a prediction and a required contrast" once you confirm the change is
  merged; until then v10's description of Part B is the old one.
