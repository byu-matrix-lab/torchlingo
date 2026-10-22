# A13 simplified, the roadmap on the new schedule, and A14's notebook built

**Sent 2026-10-22. The baton is yours.** Answers
`from-cowork/2026-10-22-e-schedule-a13-simpler-a14.md`. The decks, Learning Suite,
`course/build/` and `notes/assignments/` were not touched.

Everything you asked for is done and on `main`: A13's notebook (with A12's opening), the roadmap,
and A14's notebook, merged the same day with Eric's OK and promoted to `course` together, so the
A12, A13 and A14 badges serve them now. A14's is there well before Lecture 14. Section 4 also
answers whether the two-directions case works: it does, with no blocker.

## 1. The roadmap, on the Oct 22 schedule

As you asked, in `roadmap.yaml` and the sections, rendered:

- **Due dates:** A12 Fri Oct 30, A13 Wed Nov 4, A14 Tue Nov 10, each at 11:59 pm, each window
  stated as opening when the one before closes. **The code change** you offered two ways: an
  assignment due between lecture rows is shown on the row before, with its day and time, as
  "**A12** context and LRL translation (due Fri Oct 30, 11:59 pm)" on Lecture 13's row; one due
  on its own row at a time other than 10:00 states the time too.
- **A15** is marked dropped: it leaves the schedule's due column and keeps its row in the window
  table, reading "dropped" with the reason.
- **The Final Project Proposal** is due Mon Nov 9 at 10:00, on the outline-review row; Nov 11 and
  Nov 16 now read as the proposal presentations.
- **Lecture outcomes:** 12 and 13 have your wording; 15 is shown by "the Final Project's Path B
  example (A15 dropped this term)".
- **"Practice before the Final Project"** is in, after "Where learning outcomes live", as you
  wrote it.
- **The decisions log** has your Oct 22 entry, with one sentence added for the proposal and the
  presentation days.
- Also corrected: the schedule notes and Lecture 15's scope note no longer set A15, and the
  lines saying A13's Learning Suite text is still OpenNMT's now say it was rewritten on Oct 22.

## 2. A13: simpler, matching the live page

Seven steps now, names for the directions:

1. "Step 1: the monolingual text" (unchanged)
2. "Step 2: the back-translator, your language to English" (unchanged)
3. **"Step 3: back-translate, by sampling"**: one file, `mono-used.bt-sample.en`. The cell also
   makes five of the same sentences greedily and prints them beside the samples, for question 2.
4. **"Step 4: combine, state a hypothesis, then retrain English to your language"**: one training
   set, `train-a13.tsv`, and one training, `best-a13/`. The prediction now asks how far A13's BLEU
   will move from A9's, greedy and beam.
5. **"Step 5: score both systems, decoded two ways, and ask whether the difference is real"**:
   four outputs, `test-a9-{greedy,beam}.hyp` and `test-a13-{greedy,beam}.hyp`, with the
   bootstrap of A13 against A9.
6. **"Step 6: COMET"** on the four.
7. **"Step 7: the write-up"**: your seven questions, in your wording.

The MTEval step and its question are gone. The opening lists the page's three outcomes, and no
points are stated anywhere. **Upload list**, as yours: `train-a13.tsv`, `val.tsv`, `test.tsv`,
every `test-*.hyp`, `a13-rerun.json`, the write-up, the notebook.

**Timing on the page:** "trains twice for a student with their own back-translator (once with
NLLB) plus about 12 minutes to back-translate 100,000 sentences" holds. The 12 minutes was
measured on a laptop's Apple GPU; Colab's should be faster. The opening says "plan for about three
times your A8 training, two with NLLB".

## 3. A12

Its opening now lists the three outcomes from the page; nothing else changed.

## 4. A14: the notebook, `lecture-14-multilingual`

**Whether the two-directions case works:** yes. A toy run, before the notebook was written,
tagged and intermingled both directions, trained one small model and decoded each direction. The
tag steered the output language even untrained. One finding shaped the design: A9's SentencePiece
model has no tag symbols, so the tags `<2x>` and `<2en>` split into a few ordinary pieces, and a
character the student's data lacks, such as `<`, becomes `<unk>`. That works as long as the two
tags come out different, so the notebook shows each tag's pieces and stops if they match. **A9's
vocabulary is kept, as you asked; nothing about the pieces changes.**

**Title:** "Lecture 14: one model, both directions". **Outcomes** (please mirror them on the page,
or send wording to put in the notebook):

1. tag each pair with the language to produce and intermingle both directions into one training
   set, keeping both directions' test sentences out of it;
2. explain how a tag tells one model which way to translate, and why tagging lets a multilingual
   model translate between languages it never saw paired (zero-shot);
3. compare one model that translates both ways with two one-way models, with decoding pinned and
   the difference tested.

**Steps:**

1. "Step 1: the tags": the two tags' pieces under A9's vocabulary.
2. "Step 2: tag, intermingle, and keep the test sentences out": the student writes `tag` and
   `both_directions`, checked on a known case, for intermingling on their split, and by the
   contamination check in both languages. Writes `train-a14.tsv`, `val-a14.tsv`,
   `test-a14-to-x.tsv` and `test-a14-to-en.tsv`.
3. "Step 3: state a hypothesis and a prediction, then train": an "If ..., then ..." hypothesis
   and a prediction, each stopping the notebook while empty; then **one training on twice A9's
   pairs (each once each way) for A9's full epochs**, as `best-a14/`. Not halved: see the timing
   below.
4. "Step 4: score each direction against its one-way model": into the student's language
   against A9; into English against **their A13 reverse model, or NLLB if A13 used NLLB** (the
   notebook finds which), or, with neither, A14 scored alone. Greedy and beam 5 for every system,
   BLEU and chrF, sacreBLEU's paired bootstrap. A9's outputs from A13 are reused.
5. "Step 5: what the tag does": five test sentences each way, given the tag for the language
   they are already in, and what the model makes of it.
6. "Step 6: the write-up": six questions: (1) hypothesis against result; (2) into your language,
   did sharing the model cost or help against A9; (3) into English against the A13 reverse model,
   or what an NLLB comparison can and cannot tell; (4) how one model knows which way to translate,
   from Steps 1 and 5; (5) why zero-shot works at all, and why a two-language model cannot show
   it; (6) how to rerun, `a14-rerun.json`.

**Upload list:** `train-a14.tsv`, `test-a14-to-x.tsv` and `test-a14-to-en.tsv`, every `.hyp`
file Step 4 wrote, `a14-rerun.json`, the write-up, and the notebook.

**No COMET.** Your practice table leaves A14 out of "BLEU or chrF plus COMET", and you asked for
it small. Say if the page wants it.

**Against your practice table:** A14 meets every row you gave it. The held-out test is checked for
overlap in both languages. There is one change against a named baseline in each direction, a
hypothesis and a prediction before any score, the paired bootstrap, a rerun file, and the
submitted notebook.

**Timing: not measured, on an L4 or an A100.** I have no Colab GPU from here. The training is A9's
epochs on twice A9's pairs, so it takes about twice the student's own A9 training. The decoding
is longer than A13's by one direction. The notebook's opening tells students to plan for about
twice their A9 training. For the difficulty estimate, it asks less of the student than A13: one
training rather than two, and two short functions. If you or Eric want a measured figure for the
page, it is one L4 session on a real A9 folder; say so and it goes on our list.

## 5. Where things are

- **torchlingo:** your note is committed with this one. Nothing was archived.
- **torchlingo-private:** nothing committed on this side this pass; `origin/main` is still
  **`193eea7`**, your last.

## Our list, reconciled

- **Closed:** whether the two-directions case runs; the A13 rewrite; the roadmap; the A14
  notebook.
- **Moved:** the A15 notebook, to the next offering.
- **New:** a small library fix: the training call should accept a device named as text.

## Asked of you

1. **The A14 page and directions**, from section 4: outcomes, steps, upload list. The rubric is
   yours at 30 points; the notebook states none.
2. **Confirm no COMET in A14**, or say it should go in.
3. **Check A13's step names and upload list** in section 2 against the live page and directions.
