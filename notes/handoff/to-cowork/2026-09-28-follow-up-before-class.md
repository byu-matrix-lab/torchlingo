# Follow-up to "Lecture 7 is live": three things moved in the last hour

**Written 2026-09-28, 10:55 MDT.** Read `2026-09-28-lecture-7-is-live.md` first; this corrects
it where it went stale and adds one thing you should know as a user of the stamping tool. Short
on purpose.

## 1. PR #152 is merged. The 8a notebook is final on our side

The earlier hand-off said it was waiting on Eric. It is in: Step 4 prints "val sources" for the
val set, and the install asks for `torchlingo>=0.2.1`. Only Eric's Colab run on a real A5 corpus
remains before Wednesday.

## 2. Tutorial 1 now runs from its badge; tutorial 3 has a second problem

Tutorial 1's commented-out install is fixed (PR #154), and it passes the way a student will run
it, on 0.2.1.

**Tutorial 3 cannot be fixed the same way**, which is new since the earlier hand-off. Beyond the
install, it loads the model *tutorial 2* saved, `checkpoints/tiny_model.pt`. In Colab each notebook
gets its own runtime, so that file is never there, and its "run Tutorial 2 first" error sends a
student somewhere that cannot help. It is Lecture 10's reading, so there is time. It is Task #166,
paired with Task #162, which reshapes the same notebook's Part 5. **Until then, a Lecture 10 slide
should not send students to tutorial 3's Colab badge.**

## 3. The stamping tool had a bug that could have corrupted a notebook. It is fixed

`notebook_meta.py --stamp` rewrote the `torchlingo` block with a comma after it, always. That is
right when the block is first in the notebook's metadata, which is where the tool puts it. But any
tool that re-serializes a notebook through `nbformat` (re-executing it, for one) sorts the keys
and moves the block last, and there the comma made the file invalid JSON. Tutorials 4 and 5 were
in that shape since this morning's re-execution, one restamp from breaking. Fixed in PR #154,
with a test for both positions.

**What it means for you:** nothing to undo. Your notebooks have the block first. But if a stamp
ever leaves a notebook that will not open, that was the cause, and it is fixed on `main` now.

## Which tasks moved since the earlier hand-off

`notes/TASKS.md` was reconciled again in PR #155. It now holds everything; the in-session list is
empty, at Eric's request.

| | |
|---|---|
| **#163** | closed — both halves merged (PR #146 in 0.2.1, PR #152) |
| **#113** | closed — every PR it was waiting on had merged |
| **#166** | narrowed to tutorial 3, now recording both of its failures, and paired with #162 |
| **#167** | new — prune merged branches and stale worktrees; deletions, so Eric decides |

Next for this side: **Task #121**, the Lecture 9 subword notebook, with #147 folded in. Lecture 9
is Oct 7 and A9 is due Oct 14. If you have scope for it from the Lecture 9 deck — what the
activity should leave a student holding — that is the question I will bring to the next baton.

Nothing here needs an answer except that last, optional one.
