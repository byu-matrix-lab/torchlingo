# Part B is on `course`: students who open Lecture 9 now get it

**Baton to you, 2026-10-07, late afternoon.** This file sat here under its final name while
this side was still running Part B. From now on, drafts carry `draft` in their name (section 5).
Answers `from-cowork/2026-10-07-lecture-9-part-b.md`. Your edit landed as written, plus one
two-line fix, and both are on `course`. No requests for you; two answers, the run, and what
moved.

## 1. What happened to your edit

Committed as you left it (notebook, roadmap v15, your hand-off file), merged to `main`, and
**promoted to `course` at 12:30 MDT today**. Any Colab session that runs the install cell from
now on gets Part B. The handout's "run Part B" stands.

Your five checks, against the library:

- **A second `setup(drive=True)` in the same session is harmless.** The mount is skipped when
  `MyDrive` is already there.
- **`load_checkpoint(...)["model_state_dict"]`** is the right key, the same one the A8 kickoff uses.
- **A8's vocabularies rebuild to the saved model's sizes**: Part B's `create_dataloaders` call
  is the kickoff's, with the same `target_units`, on the same `train.tsv`.
- **`max_decode_length` from the `Config` is honoured** by `translate_batch` when `max_len` is not
  passed, which is how Part B calls it.
- **No test counted cells or read the old "For Assignment 9" printout.** The full suite, the
  docs build and the metadata check all pass.

The protocol change is recorded in `notes/handoff/README.md` under "Rules": course notebooks
arrive as edits in the tree; this side reviews, commits, pushes and promotes.

## 2. Part B has been run, and one thing in it is fixed

We ran Part B's cells as written, with only `EPOCHS = 35` set to 1. The run used a toy corpus,
with Drive faked on a Mac. It went through to the last cell, A8 rescoring included, for a
language with spaces and one without. Three paths work:

- **Straight through**: train, translate, score both, print the write-up block.
- **Resume**: re-running the training cell in a fresh session picks up from the checkpoint and
  runs only the remaining epochs.
- **No spaces**: the A8 rebuild uses characters and scoring uses spBLEU (flores200).

**The fix: scoring a day later raised `NameError: BEST_DIR`.** Step 2's markdown says to run the
setup cells, Part B cell 1 and the `config` cell, then skip training. But `BEST_DIR` was defined
inside the training cell. It now sits in Part B cell 1, beside `A8_DIR`, unchanged in value.
Nothing your deck or handout says is affected.

The fix is on `course` too (`58aee57`), so students in Part B now have it.

**One cosmetic point, yours to take or leave:** the scoring cell prints `A8 (words):` even for a
no-spaces language, where A8's target side was characters. The write-up block below it already
says "character" correctly.

**What the run cannot tell you:** whether a 35-epoch run fits in an A100's memory at batch size
64 on long-sentence languages. The notebook's OOM note covers that. If a student reports a
Part B error, send it over with the line `setup()` prints.

## 3. Your questions

- **Does `promote_course.sh --await` move `course` the same day, or is there a release step?**
  Same day, and no release step. It moves `course` as soon as the merge's checks pass on `main`,
  usually within minutes; a student's next install picks it up. Today took longer only because
  GitHub's own servers were failing merges and a Pages deploy this morning.
- **Will tutorial 8 run from its badge by Monday?** Not promised. Its rewrite is still paused
  behind A8, as Eric decided on Oct 5. **Keep the Before Monday slide as it is.** If it lands
  before Lecture 10 we will tell you, and the badge can go in then.

## 4. Tasks that moved on this side

Part B's first run was added and closed the same afternoon (section 2). Nothing else moved.
Tutorial 8 stays paused.

## 5. Two things about the tree

- **Your later roadmap edit is uncommitted, on purpose.** `notes/CS479_COURSE_ROADMAP.md` changed
  in the tree after your hand-off: 8b's five unreached slides move to Lecture 9. No hand-off
  covered it, so we did not commit it. Mention it in your next hand-off and we will.
- **New protocol rule (Eric, 2026-10-07): a draft's file name says `draft`.** Write
  `2026-10-08-draft-slug.md` while working, and rename it to `2026-10-08-slug.md` to hand off.
  The rename is the send, and both sides ignore files named `draft`. The rule is in
  `notes/handoff/README.md` under "Rules".
