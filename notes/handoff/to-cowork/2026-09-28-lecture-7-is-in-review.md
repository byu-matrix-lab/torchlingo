# Lecture 7 is in review, and both notebooks were run here

**Written 2026-09-28, 09:00 MDT.** Answering your `2026-09-28-baton.md`. Self-contained.

## 1. Lecture 7 and 8a are in PR #140, and it needs one review before 11:00

Your roadmap v6, both notebooks, Lecture 12's regenerated purpose cell, tutorial 2's `note`, and
nav entries for both notebooks. They travel as one PR because the generated map covers every
notebook in the tree. **Until it merges, the `main` badge on the `_v2` slide is a 404**, which
you had already made harmless by routing through the Content tab.

## 2. I ran Lecture 7 and it matches yours, with one wrong sentence

15 seconds on a CPU. 11/11 seen, *El perro corre* for the held-out phrase, and BLEU 0.0. The
model is seeded, so on CPU it gives the same result every time.

**The wrong sentence came from me.** Part A3 said `train_losses[0]` "is already below `ln(V)`"
and the first `log_every` line is "the true first number". Executed: the **epoch-1 mean is 3.560
against `ln(V)` = 3.178**, and the first logged line is 3.54, also above. Two reasons. A freshly
built model is not uniform, and leaning the wrong way costs more than not leaning, so it starts
*above* `ln(V)`. And `log_every` prints the mean of the last N steps, not one batch. My
suggestion #4 to you said otherwise and was wrong. Both notebooks are reworded: expect the first
line a little above `ln(V)`, and read the curve crossing it. **If a 7 or 8a slide says "first
loss ≈ ln(V)" or "below ln(V) after epoch 1", on the toy model that is not what students will
see.** Worth checking yours.

**One metadata change:** Lecture 7 declared `requires: pip`, but its install is Colab-gated and
sacrebleu and matplotlib are core dependencies. I cleared it, so CI now executes it on every
pull request, which your `note` already claimed.

## 3. The 8a notebook ran end to end here too, on a CPU copy against synthetic data

Every step works: the cap, dedupe, seeded split, raising contamination check, nine files, the
checkpointer, `save_dir`, reload through `load_checkpoint`, greedy decode and stream-shaped
BLEU. Every API it uses is in **PyPI 0.2.0**, which is what the Colab cell installs. Your
`str.split()` caveat is harmless as you said.

**One real bug, fixed in the PR.** The scoring cell read `result.best_checkpoint`, and `result`
exists only in the session that trained. A student returning tomorrow to a fresh runtime would
have hit a `NameError`. It now loads `OUT_DIR / "best" / "model_best.pt"` from Drive, and the
comment says to run Steps 0 to 6 and skip 7. I tested exactly that sequence in a fresh kernel,
and it reproduced the in-session BLEU to the digit.

**One wart it will show in the room**, which is now Task #163: the check prints
`val: 0/200 test sources also appear in training`. The library hard-codes "test". The fix needs
a PyPI release before the notebook can use it, so Wednesday will see the wart. If a student asks,
it is cosmetic.

**Eric's Colab run is still the thing.** Nothing here mounts Drive on a GPU.

## 4. A7, and the two moved due dates: reconciled

A7 is recorded in `TASKS.md` with its due date and Coulson's part. No task of ours depended on
the A12 or A16 dates, so the move changed nothing here beyond the purpose cell you already
regenerated.

## 5. Tutorial 5's placement is ours, and now has a task

Task #164. Either it moves earlier than Lecture 9, or it stops claiming A8 and names what it
really prepares. I will read the notebook before choosing, and tell you if the answer moves a
link on a deck.

## 6. Your 8a additions were right

The checkpointer and best-by-validation scoring are what the library exists to teach. They are
also why the fresh-session fix above was a two-line change rather than a redesign.

## Which tasks moved

| | |
|---|---|
| **#152** | written by you; **in review, PR #140**; done when it merges *and* Eric has run it in Colab |
| **#163** | new — the "test sources" label; L8a, needs a release |
| **#164** | new — tutorial 5 placement, ours |
| **#103** | closed — superseded by #154, which gates course notebooks on declared capabilities |
| **#142** | closed — your Q7 answered it; the work is #162 |
| **#85, #135** | rows deleted: my last hand-off reported them closed, and they were, in PR #132, but the rows had stayed |

Still open, as you listed: #121, #118, #160. No question of mine to you is unanswered.
