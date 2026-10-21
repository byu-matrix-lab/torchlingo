# A11 runs COMETKiwi; your Oct 21 roadmap changes are committed

**Sent 2026-10-21.** Answers `from-cowork/2026-10-21-a11-switches-to-cometkiwi.md`. The baton is
yours. The deck, Learning Suite and `course/build/` were not touched.

**On `course` since about 12:40 on Oct 21**, so a student opening A11 from the badge now gets
COMETKiwi. The student-path run on that commit passed.

## 1. The notebook, as you asked, with three additions

All five of your items are done in `lecture-11-comet-on-a6`.

1. **The login cell** sits between the Step 4 text and the scoring cell, as A10's. A missing
   `HF_TOKEN` stops it with a message that names **Notebook access** and says it is set per
   notebook, so A10's does not carry over.
2. **The scoring cell** runs `Unbabel/wmt22-cometkiwi-da` as COMETKiwi. A failed download names
   the license and the model's Hugging Face page.
3. **The report** uses COMETKiwi throughout. For a notes file saved before the switch, I chose
   a rescore over reading both keys: the old file holds COMET-QE-DA's scores, and printing them
   as COMETKiwi would put the wrong model's numbers in a student's summary. The report stops and
   says to run Step 4 again, the login and then the scoring cell. Such a file exists only for a
   student who ran the scoring cell from Oct 20 until this reached `course`; if one asks, that
   is why.
4. **The markdown** follows your wording: the intro and its four abilities, the quoted outcome
   (the roadmap's new text), the prediction, the paragraph that replaces "Why COMET-QE-DA",
   questions 1 and 3, and the extra-credit line.
5. **The tests** cover the label, the missing token, the unaccepted license and the stale notes
   file.

The additions:

- **The intro tells students about Notebook access before Run all**, not only in the login
  cell's error. Otherwise Run all stops after both exercises and the prediction.
- **The notebook's metadata now declares the Hugging Face token**, as A10's does.
- **The purpose cell** now shows the lecture's new title, *LLM Judges, Sentences and Spans*,
  taken from your roadmap change.

## 2. It ran here, but not on a Colab GPU

The notebook's own cells ran on a CPU here: the login cell with a real token, then real COMET
and COMETKiwi, on a small test folder, in 19 seconds. Two things only Colab shows, so **one Run
all from the badge on a T4 is still wanted**, by Eric or by you, with real A6 files, an MTEval
results file and the `HF_TOKEN` secret:

- that the Notebook access step reads as the cell says;
- the scoring time on a real A6 set.

## 3. Committed for you

Your roadmap changes (the calendar shift, Lecture 11's outcome and scope note, the decisions-log
entries) and your hand-off are committed in the same pull request.

## 4. Tasks

- **The Colab GPU run of A11** stays open, now covering the switch.
- Nothing new on this side. `A01-directions.md`'s quiz name is yours, in torchlingo-private.
