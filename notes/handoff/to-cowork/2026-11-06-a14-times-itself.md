# A14 now times each stage, and it is on `course`

*From the repository session, Fri Nov 6. Answers `from-cowork/2026-11-06-m-stage-times.md`. The
baton goes back to Cowork with this file. One thing is asked of you: post the line you wrote for
A14's page.*

## 1. Post A14's page line

Your sentence fits what the notebook prints, so it can go up as you wrote it, under **What to
submit**. Step 8's markdown tells students the same thing, and adds that the table is for planning
and how long theirs took earns or costs nothing. Points, steps, and questions are unchanged.

A copy of the notebook opened from the badge before this promotion prints no table. Eric said on
Nov 5 that no student had started A14; if one has, the student gets the new notebook by opening
the badge again, or notes the times by hand.

## 2. What the last cell prints

This is the shape students get. The numbers are invented, to show a resumed stage; no real run
has made one yet (section 4):

```
=== Timing: paste this at the end of the write-up ===
Training steps: 3,000
Stage                               Time     Runs  GPU
Step 3: NLLB off the shelf          0:41:12  1     NVIDIA A100-SXM4-40GB
Step 4: fine-tuning                 2:18:40  2     NVIDIA A100-SXM4-40GB + NVIDIA L4
Step 5: translating                 ...
Step 5: bootstrap                   ...
Step 6: wrong code, third language  ...
Step 7: COMET                       ...
Total                               ...
```

- **The cut.** Your five stages, with Step 5 split in two, translating and the bootstrap, since one
  needs a GPU and the other does not, and next year's plan may treat them differently. The time
  to load NLLB and to install COMET is not counted: a few minutes, in cells every session repeats.
- **The time is cells running**, not a student's time between them. For planning a window, add
  the student's own work on top; the table cannot see it.
- **A dropped session keeps its time.** Each stage saves at its start, every minute while it runs,
  and at its end, so a drop loses at most the last minute. A stage run again adds to its time and
  counts as another run, so **Runs** above 1 means it was resumed. A stage can also add a run if a
  student reruns a cell that had finished, which adds little time unless that cell does real work.
- **The GPU is per stage**, because a resumed session may get a different one. A row with two GPUs
  cannot say how the time divides between them.
- **A timing failure never stops a student.** If a save to Drive fails, the stage carries on and
  the next save catches up.

## 3. The numbers come back without retyping

The same record goes into `a14-rerun.json`, under `stage_times`, in seconds, and that file is
already on the upload list. So when you and Eric plan next year's windows, the times can be read
from the uploaded files rather than from pasted tables. It is also pasted for question 7, so the
times appear twice in a write-up; that does no harm.

## 4. What was and was not tested

The notebook's own cells ran at toy size, with NLLB and COMET stubbed, including a stage run again
in a new session and the table after a dropped session. A separate test kills a process inside a
stage and checks the time survives. Nothing has run this on a real Colab GPU or real Drive; the
first real table will be a student's. The GPU name uses the same call as the line `setup` already
prints, so the risk is small, and a failure in timing cannot stop the assignment.

## 5. A13

The same timing in A13's notebook is a task for the next offering, on this side's list. With the
library piece written, it is a few lines. Your optional request on this term's page stands.

## 6. Your branch

Your hand-off went in through a pull request with this reply, and
`cowork/2026-11-06-m-stage-times` is deleted.

## Tasks

On this side, A14's timing is done, and A13's timing is a new next-offering task. On your list,
A14's page line is the one thing left of this, and the Colab menu label still waits on Eric.
