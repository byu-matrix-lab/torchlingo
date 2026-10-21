# A11 switches to COMETKiwi (please change the notebook today)

*From the Cowork session, Wed Oct 21 2026, about 10:55. Eric decided it in chat.*

## The decision

Assignment 11's estimator is now **COMETKiwi (`Unbabel/wmt22-cometkiwi-da`)**, the model A10
already runs, not COMET-QE-DA (`Unbabel/wmt20-comet-qe-da`). Eric's reason: one estimator across
A10 and A11, the newer and stronger one; every student set it up in A10. A11 was set today and is
due Mon Oct 26 at 10:00, and students are told to start tonight, so **the notebook should be on
`main` today**.

The in-class judges in Lecture 11 (the Be the Judge rounds) stay COMET-QE; nothing there changes.

## What to change in `docs/docs/course/lecture-11-comet-on-a6.ipynb`

1. **A Hugging Face login cell before the scoring cell**, the same as A10's
   (`lecture-10-comet-install`, cells 12 and 13): `google.colab.userdata.get("HF_TOKEN")`, a clear
   RuntimeError when it is missing that names the Colab key panel and **Notebook access** (the
   toggle is per notebook, so a student whose A10 worked still has to turn it on here), then
   `login(token=...)`.
2. **The scoring cell (now cell 15):** `("Unbabel/wmt20-comet-qe-da", "COMET-QE-DA", False)`
   becomes `("Unbabel/wmt22-cometkiwi-da", "COMETKiwi", False)`; wrap the download as A10's cell
   14 does, so a KeyError says the usual cause is the license not accepted on the token's account.
   The final print names COMETKiwi.
3. **The report cell (17)** and every label: "COMET-QE-DA" becomes "COMETKiwi" (three tuples).
   The saved `a11-notes.json` keys follow; a student who scored before the switch has
   "COMET-QE-DA" keys, so either read both or say in the cell that a rerun of the scoring is needed.
4. **Markdown:** cell 1 (the intro, the four "you will be able to" items, and the quoted Lecture
   11 outcome, now "Run COMET and COMETKiwi on their own A6 data; ..."); cell 12 (the
   prediction: "will COMETKiwi agree with COMET?"); cell 14 (replace the "Why COMET-QE-DA, not
   A10's COMETKiwi?" paragraph with one saying COMETKiwi is the estimator from A10, it needs the
   A10 Hugging Face setup, and COMETKiwi's scores run 0 to 1 but are still not comparable with
   COMET's); cell 18 (questions 1 and 3; the extra-credit line "Like COMETKiwi in A10" becomes
   "Like COMETKiwi").
5. **`tests/test_lecture_11.py`:** line 146's label tuple; fake `google.colab.userdata` and
   `huggingface_hub.login` for the new cell (check `colab_faked` in `notebook_cells`); any cell
   indices that shift with the inserted cell.

## Already done on the course side (uncommitted, please commit)

- **Learning Suite:** the Assignment 11 page (`jLj_`) names COMETKiwi and the A10 Hugging Face
  setup; questions 1 and 3 and the rubric row follow. The Lecture 11 page says "COMET and
  COMETKiwi" and "it needs a GPU and your Assignment 10 Hugging Face setup".
- **torchlingo-private:** `notes/assignments/A11-directions.md` matches the page (History line
  added). The Lecture 11 build script `course/build/l11_live_v5.py` (and `objectives_v1.py`, new,
  plus `l12_v2_shift.py`) carry today's deck changes.
- **torchlingo roadmap:** `roadmap.yaml` (Lecture 11 outcome, the notebook's purpose, the Hugging
  Face row), `sections/10-lecture-scope-notes.md` (Lecture 11 notebooks and sets) and
  `sections/16-decisions-log.md` (an "Oct 21, later" entry: this switch, "Lecture Quiz N", learning
  outcomes on every deck and content page); rendered with `render_roadmap.py --write`. These sit
  on top of the earlier uncommitted Oct 21 roadmap changes (the calendar shift).

## Also today, for the record

- Content quizzes are named "Lecture Quiz N" in Learning Suite (21 exams). `A01-directions.md`
  in torchlingo-private still says "Quiz - Lecture 1" (the page now says "Lecture Quiz 1").
- Each content page for Lectures 1 to 12 has a "Learning Outcomes" section after "Today's
  Lecture" (the content-page template now has four headings). Lectures 13 onward get theirs with
  their decks.
