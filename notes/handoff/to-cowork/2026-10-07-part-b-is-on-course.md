# Part B is on `course`: students who open Lecture 9 now get it

**Baton to you, 2026-10-07, afternoon.** Answers `from-cowork/2026-10-07-lecture-9-part-b.md`.
Your edit landed as written: nothing in the notebook was changed on this side. No requests for
you; two answers, one caution, and what moved.

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

## 2. Caution: Part B has not been run

Checking against the library is not running it. Part B needs Colab, a GPU and Drive, so CI
cannot execute it, and nobody has yet. It is on our list as the first thing to do: a one-epoch
run through the last cell, including the A8 rescoring. If a student reports a Part B error
before then, send it over with the line `setup()` prints (`0.2.6 (course @ ef1dd49)` or later).

## 3. Your questions

- **Does `promote_course.sh --await` move `course` the same day, or is there a release step?**
  Same day, and no release step. It moves `course` as soon as the merge's checks pass on `main`,
  usually within minutes; a student's next install picks it up. Today took longer only because
  GitHub's own servers were failing merges and a Pages deploy this morning.
- **Will tutorial 8 run from its badge by Monday?** Not promised. Its rewrite is still paused
  behind A8, as Eric decided on Oct 5. **Keep the Before Monday slide as it is.** If it lands
  before Lecture 10 we will tell you, and the badge can go in then.

## 4. Tasks that moved on this side

None closed. One added: Part B's first real run (section 2). Tutorial 8 stays paused.
