# Lecture 7 is live, 0.2.1 is released, and the roadmap needed two repairs

**Baton back to you, 2026-09-28, last updated 10:58 MDT.** Answers your `2026-09-28-baton.md`.
The one file for this pass: it replaces the two I wrote earlier today, which you have not seen.
Ordered by what matters first.

## 1. Lecture 7 and 8a are on `main`, and the badge resolves

PR #140 merged at 09:47, carrying roadmap v6, both notebooks, Lecture 12's regenerated purpose
cell, and a one-line `note` on tutorial 2 saying the course exercise was adapted from it. The
`main` badge on the `_v2` slide now opens the notebook, so Eric can post the direct link.

**One addition you did not write: step A5, "Connect Google Drive".** It sits at the end of Part
A, so it happens in the room. It mounts Drive, creates `MyDrive/CS479` (the exact folder 8a reads
from), writes a check file there, and tells the student to **put their two A5 files in that folder
before Wednesday**. 8a mounts Drive in its first data step, and the first mount in an account
brings up Google's authorization dialog; twenty-four of those at once on Wednesday, with training
runs waiting, is lost minutes. **If a Lecture 7 slide lists Part A's steps, it is now one longer.**

## 2. What I changed in your two notebooks, and why

- **The first-loss wording, in both — and the error was mine.** They said `train_losses[0]` is
  already below `ln(V)` and that the first `log_every` line is "the true first batch". Executed,
  Lecture 7's epoch-1 mean is 3.560 against `ln(V)` = 3.178, and `log_every` prints the mean of
  the last N steps, not one batch. A fresh model starts a little *above* `ln(V)`: it is not
  uniform, and leaning the wrong way costs more than not leaning. My suggestion #4 to you said
  otherwise. **If a 7 or 8a slide says "first loss ≈ ln(V)" or "below ln(V) after epoch 1",
  that is not what students will see on the toy model.**
- **Lecture 7 no longer declares `requires: pip`.** Its install is Colab-gated and its imports
  are core dependencies, so CI now executes it on every pull request, as your `note` said.
- **8a's scoring cell read `result.best_checkpoint`**, and `result` exists only in the session
  that trained. A student scoring the next day in a fresh runtime would hit `NameError`. It now
  loads the best model from Drive; run Steps 0 to 6, skip 7, then score. Tested in a fresh
  kernel: same BLEU to the digit.
- **8a's contamination check names the set it checked** (`name="val"`), and its install asks for
  `torchlingo>=0.2.1`, so a runtime holding an older copy upgrades (section 4).

## 3. How both were verified, and the one thing only Eric can do

Lecture 7 was run the way a student will run it: a fresh environment, the notebook's own install
cell pulling 0.2.1 from PyPI, and Colab faked so every Colab-only branch executes. It passes end to
end, 15 seconds after the install, and matches your run: 11/11 seen, *El perro corre*, BLEU 0.0.
The Drive mount is faked in that test; only a real Colab session exercises Google's dialog.

8a cannot run here, but a CPU copy against a synthetic corpus ran every step: cap, dedupe,
seeded split, raising contamination check, nine files, the checkpointer, `save_dir`, reload, greedy
decode, stream-shaped BLEU. Every API it calls is in 0.2.1. Your `str.split()` caveat is harmless as
you said. **Eric's Colab run against a real A5 corpus is still the only end-to-end test before
Wednesday.**

## 4. torchlingo 0.2.1 is on PyPI

Released at 10:29, approved by Eric. Two changes matter to 8a:

- **Resuming a run that stopped mid-epoch was broken on 0.2.0.** A periodic checkpoint recorded
  the epoch in progress as finished, so a resumed run skipped the rest of it, trained fewer steps
  and ended on the wrong learning rate. 8a tells students to re-run Step 7 after a disconnect, and
  at A8's scale nearly every disconnect lands mid-epoch. Fixed, with tests that fail on the old code.
- **`check_contamination` can name the set it checked**, so 8a stopped printing
  `val: 0/2000 test sources`.

## 5. Slide suggestions for Lecture 7, yours to take or leave

Eric asked what Lecture 7 should include, given that no student has installed TorchLingo yet:

- **Name the pipeline once, as the thing they reuse from A8 to A14:**
  `Config → NMTDataset → DataLoader → SimpleTransformer → train_model → translate_batch`. The toy
  notebook runs exactly those six calls, and so does 8a.
- **Say why the GPU comes first:** changing the runtime type later wipes everything that has run.
- **Have them print `torchlingo.__version__` when something breaks**; it is the first thing the
  teaching staff will need.

A resume demo on the toy model is ours, Task #165, for Part B rather than the timed part.

## 6. The roadmap needed two repairs. **One recurs unless your script changes**

Your 09:52 edit arrived five minutes after PR #140 merged: Lecture 11 now treats **LLM-as-judge**
as a method, with the five-slide GEMBA material. PR #147 carried it across verbatim.

**The repository copy also still opened with v5's title and its "Changes from v4" list**, above
v6's own introduction. Your refresh script replaces the file from "Semester at a glance" down, so
the title block above that heading is never touched, and it will go stale again at v7. Either
extend the replaced region to the top of the file, or have the script write the title block too.
After PR #147, your half matches `Roadmap/CS479 Fall 2026 Roadmap_v6.md` byte for byte.

**The generated map's footnote markers are now `[1]`, `[2]`, not superscripts.** The fixed string
of nine superscripts ran out at the tenth note, and Eric chose brackets. If your script reads the
map, or anything in your half cites a marker, check it.

## 7. The tutorials: 1, 4, 5 and 7 run from their badges; 3 does not yet

- **Tutorials 4 and 5** had a commented-out install and read `data/` that a pip install does not
  ship. Both now install themselves and download what they need, and every output is unchanged
  (PR #149). **Your 8b recap that links tutorial 4 now points at something a student can run.**
- **Tutorial 1**, the same install fix (PR #154).
- **Tutorial 7 landed** (PR #144), your Q7 call: Lecture 6 reading leading to A6 and A8, with
  `lecture-06-mt-evaluation` keeping the in-class teaching.
- **Tutorial 3 cannot be fixed the same way.** Beyond the install, it loads the model *tutorial 2*
  saved. In Colab each notebook gets its own runtime, so that file is never there. It is Lecture
  10's reading, so there is time: Task #166, paired with #162. **Until then, a Lecture 10 slide
  should not send students to tutorial 3's Colab badge.**

## 8. The stamping tool could corrupt a notebook. It is fixed

`notebook_meta.py --stamp` put a comma after the rewritten `torchlingo` block, always. Right when
the block is first, which is where the tool puts it; but re-serializing a notebook through
`nbformat` sorts the keys and moves the block last, where that comma is invalid JSON. Tutorials 4
and 5 were in that shape, one restamp from breaking. Fixed in PR #154 with a test for both
positions. **Nothing to undo on your side**: your notebooks have the block first.

## 9. Your baton, item by item

- **A7** is in `TASKS.md`, with Coulson's part. **A12 and A16's new dates** changed nothing here
  beyond the purpose cell you regenerated.
- **Tutorial 5's placement** is ours, Task #164: move it before Lecture 9, or stop it claiming A8.
- **Your two 8a additions** — the checkpointer, and best-by-validation scoring — were right, and
  are why the scoring fix in section 2 was two lines rather than a redesign.
- **Post-norm, tutorial 4 whole, #121, #118, #160**: acknowledged as you wrote them.

## Which tasks moved

`notes/TASKS.md` is reconciled (PRs #141, #143, #151, #155) and holds everything; the in-session
list is empty, at Eric's request. It also gained lecture labels L1 to L13, replacing `10+`, which
had meant both "Lecture 10 onward" and "no lecture at all".

| | |
|---|---|
| **#88** | closed — tutorial 7, PR #144 |
| **#114** | closed — tutorials 4 and 5 run from Colab, PR #149 |
| **#163** | closed — `name="val"`, PR #146 in 0.2.1 and PR #152 |
| **#103, #142, #113, #85, #135** | closed — each already done; the rows had stayed |
| **#152** | the 8a notebook is merged; **only Eric's Colab run remains** |
| **#157** | scheduler restore proven, mid-epoch bug fixed; what remains is AMP-only, `lib` |
| **#151** | relabelled `lib`: tutorial 4, the only student path, already avoids it |
| **#162** | unblocked; paired with #166 |
| **#164** | new — tutorial 5's placement, ours |
| **#165** | new — resume demo on the Lecture 7 toy model |
| **#166** | new — tutorial 3 cannot run from its badge, for two reasons |
| **#167** | new — prune merged branches and stale worktrees; deletions, so Eric decides |

**Next on this side: Task #121**, the Lecture 9 subword notebook, with #147 folded in. Lecture 9 is
Oct 7 and A9 is due Oct 14. **The one question for you:** from the Lecture 9 deck, what should
the activity leave a student holding? Everything else here is for information.
