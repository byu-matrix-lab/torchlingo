# Lecture 7 is live, 0.2.1 is released, and the roadmap needed two repairs

**Written 2026-09-28, 10:45 MDT.** Follows my `2026-09-28-lecture-7-is-in-review.md` from this
morning, which it supersedes on status. Self-contained. Ordered by what matters first.

## 1. Lecture 7 and 8a are on `main`, and the badge resolves

PR #140 merged at 09:47. The `main` badge on the `_v2` slide now opens the notebook, so Eric can
post the direct link rather than the Content tab's.

**One addition you did not write: step A5, "Connect Google Drive".** It sits at the end of
Part A, so it happens in the room. It mounts Drive, creates `MyDrive/CS479` (the exact folder
8a reads from), writes a check file there, and tells the student to **put their two A5 files in
that folder before Wednesday**. The reason: 8a mounts Drive in its first data step, and the
first mount in an account brings up Google's authorization dialog. Twenty-four of those at once
on Wednesday, with training runs waiting, is lost minutes. Part A's row in the intro table says
so. **If a Lecture 7 slide lists Part A's steps, it is now one step longer.**

Tested the way a student will run it: a fresh environment, the notebook's own install cell
pulling from PyPI, and Colab faked. It passed end to end on 0.2.1 in 15 seconds after the
install. The Drive mount itself is faked in that test; only a real Colab session exercises
Google's dialog.

## 2. Slide suggestions for Lecture 7, yours to take or leave

Eric asked what Lecture 7 should include, given that no student has installed TorchLingo yet.
The notebook covers the install. Three things belong on slides or in the spoken part:

- **Name the pipeline once, as the thing they will reuse from A8 to A14:**
  `Config → NMTDataset → DataLoader → SimpleTransformer → train_model → translate_batch`. The
  toy notebook runs exactly those six calls, and so does 8a. One slide saying "Wednesday is the
  same six calls on 100,000 pairs" turns 8a into a repeat rather than a new lesson.
- **Say why the GPU comes first:** changing the runtime type later wipes everything that has run.
  The notebook says it; hearing it once helps.
- **Have them print `torchlingo.__version__` when something goes wrong.** Both notebooks rely on
  0.2.1 (section 4), and the version is the first thing the teaching staff will need.

A resume demo on the toy model is also worth having before 8a introduces the checkpointer cold.
That one is ours, now Task #165, and goes into Part B rather than the timed part.

## 3. The roadmap needed two repairs. **One will recur unless your script changes**

Your 09:52 edit arrived five minutes after PR #140 merged: Lecture 11 now treats **LLM-as-judge**
as a method, using the five-slide GEMBA material. PR #147 carried it across verbatim.

**The same PR found that the repository's copy still opened with v5's title and its
"Changes from v4" list**, above v6's own introduction. Your refresh script replaces the file
from "Semester at a glance" down, so the title block above that heading is never touched. It
will go stale again at v7. Either extend the replaced region to the top of the file, or treat the
title block as part of what the script writes. After PR #147, your half of the repository copy
matches `Roadmap/CS479 Fall 2026 Roadmap_v6.md` byte for byte.

**Also: the generated map's footnote markers are now `[1]`, `[2]`, not superscripts.** The fixed
string of nine superscripts ran out the day tutorial 7 gained the tenth note, and Eric chose
brackets. If your script reads the map, or anything in your half cites a marker, check it.

## 4. torchlingo 0.2.1 is on PyPI

Released at 10:29, approved by Eric. It matters to 8a in two ways:

- **Resuming a run that stopped mid-epoch was broken on 0.2.0.** A periodic checkpoint recorded
  the epoch in progress as finished, so a resumed run skipped the rest of that epoch, trained
  fewer steps, and ended on the wrong learning rate. 8a tells students to re-run Step 7 after a
  disconnect, and at A8's scale nearly every disconnect lands mid-epoch. Fixed, with tests that
  fail on the old code.
- **`check_contamination` can now name the set it checked**, so 8a stops printing
  `val: 0/2000 test sources`.

A pull request, PR #152, makes 8a use both. Its install now asks for `torchlingo>=0.2.1`, so a
runtime holding an older copy upgrades. It is green, and waiting on Eric's merge.

## 5. Tutorials 4 and 5 now run from their Colab badges

Tutorial 4 is 8a/8b's reading. It had a commented-out install ("uncomment in Google Colab"), so a
student running straight through hit `ModuleNotFoundError` in the first code cell, and Part 8 read
`data/pretrained/`, which a pip install does not ship. Both are fixed (PR #149). The notebook now
installs itself and downloads the 11 MB it needs, and every output is unchanged. **Your 8b recap
that links tutorial 4 now points at something a student can actually run from the badge.**

Tutorials 1 and 3 still have the old pattern. That is Task #166, not urgent before Lecture 9.

## 6. Tutorial 7 landed, which was your Q7 call

PR #144: the evaluation tutorial, assigned reading for Lecture 6 leading to A6 and A8, with
`lecture-06-mt-evaluation` keeping the in-class teaching as you decided. It also unblocks
Task #162, tutorial 3's BLEU section shrinking to a pointer at it.

## Which tasks moved

`notes/TASKS.md` was reconciled in PR #151.

| | |
|---|---|
| **#88** | closed — tutorial 7, PR #144 |
| **#114** | closed — tutorials 4 and 5 from Colab, PR #149 |
| **#152** | the 8a notebook is merged; **only Eric's Colab run on a real A5 corpus remains** |
| **#157** | scheduler restore proven and the mid-epoch bug fixed; what remains is AMP-only |
| **#163** | library half in 0.2.1; notebook half is PR #152 |
| **#151** | relabelled `lib`: tutorial 4 already avoids it |
| **#162** | unblocked |
| **#165** | new — resume demo on the Lecture 7 toy model |
| **#166** | new — tutorials 1 and 3 still have the commented-out install |

The task list also gained lecture labels L1 to L13 today, replacing `10+`, which had meant both
"Lecture 10 onward" and "no lecture at all".

**Nothing here needs an answer from you** except the refresh-script question in section 3,
which is a change on your side rather than a question for me.
