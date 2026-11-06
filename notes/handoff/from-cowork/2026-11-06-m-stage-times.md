# M: Time each stage of A14 (and later A13)

From the Cowork session, Fri 2026-11-06, just after midnight. Answers `to-cowork/2026-11-05-lecture-18-notebook-built.md`. Baton to the repository session.

## 1. Your reply: all four asks done

- **Lecture 6's deck:** slide 32 showed `sacrebleu.corpus_chrf(...)  # -> 78.4`, which is chrF++; it now says 77.0 (sacreBLEU's default gives 76.97). Promoted. Nothing else in the deck said the library's chrF was chrF++.
- **Learning Suite and the Final Project:** no page named chrF++. The Final Project page now says: *"Compute every chrF yourself with sacreBLEU's default (plain chrF), and do not compare it with a chrF from an earlier assignment, which may have been computed differently."*
- **"Runtime → Run after":** not checked yet; Eric will look the next time he opens the notebook in Colab.
- **A13's page:** done Nov 5, 23:40, word for word as you asked (the 25,000 sentences, the runtime as the notebook gives it, question 5's clause). Eric told students on Discord.
- **Lecture 18's notebook:** thank you. The deck will link it, and may show your Model 1 and attention tables.

## 2. New, asked of you: time each stage (Eric, Nov 5)

Eric: *"Students should be asked to log the amount of time involved in each stage for A13 and A14."* The point is planning: real per-stage times, on real Colab GPUs, for next year's windows and points (the next-offering notes already say A14 is heavier than estimated).

**A14 (urgent; no student has started it, and it is due Nov 13).** Eric chose to have **the notebook time each stage itself**, not students by hand:

- Record wall-clock time for each long stage, at least: Step 3 (NLLB off the shelf, both test sets, greedy and beam), Step 4 (fine-tuning), Step 5 (scoring, including the bootstrap), Step 6 (the wrong code and the third language), and Step 7 (COMET). Your judgment on the exact cut.
- Record the GPU name (`torch.cuda.get_device_name()`), and the training steps the student chose, since time depends on both.
- Save it to Drive with the other notes (in `a14-rerun.json`, or a sibling file), so a dropped session keeps the times already measured, and a resumed stage adds to its time rather than restarting it, or says it was resumed.
- Print it as a small table in the write-up cell, ready to paste.
- No change to points, steps, or questions.

When it is on `course`, tell me, and I add this to A14's Learning Suite page (written, not yet posted, so the page never asks for a table the notebook does not print):

> In **What to submit**, after the write-up: *"The timing table the last cell prints (each stage's time, your GPU, and your training steps), pasted at the end of your write-up."*

If it cannot be on `course` by about Sun Nov 8, say so in your reply, and the page asks students to note start and end times by hand instead.

**A13 (not urgent).** It is due today, so this term A13's page asks for the times as optional, no points (posted Nov 5, 23:58): *"about how long each stage took (the reverse model, if you trained one; back-translation; the retraining; scoring) and which GPU you used."* For the next offering, the same timing as A14 in A13's notebook would make the request a paste. A P4 task, at your pace.

## Tasks

On the course list: the A14 timing (yours, then Learning Suite) is new; the chrF line and the A13 page are done; the Colab menu label waits on Eric.
