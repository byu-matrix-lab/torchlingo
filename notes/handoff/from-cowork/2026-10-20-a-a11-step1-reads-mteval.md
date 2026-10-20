# A11 Step 1 reads the MTEval file; the final project on one page

**Sent 2026-10-20, 00:05.** Follows `from-cowork/2026-10-19-c-a6-sample-and-a12-folders.md`.

**Work in parallel.** You take A11 (below). Cowork keeps the Lecture 11 deck, which lives in the course folder, plus the build scripts `course/build/l11_*.py` and `deck_lib.py`: please leave those untouched until Lecture 11 is recorded. A11 is due **Mon Oct 26 at 10:00**, and students may start as soon as Lecture 11 is posted (by Wed Oct 21, 10:00), so this is the first thing in the queue.

## 1. A11 Step 1: read the student's own MTEval results file

The real format is in `torchlingo-private/course/data/a6-mteval-sample/` (one Fall 2026 A6 file, name removed; its README describes it). In short:

- **Header row:** `source<TAB><system 1><TAB><system 2>`. The system names are the student's own (the sample says `microsoft` and `nllb`), so read them from the header.
- **50 rows:** the English source and each system's rank, 1 or 2.
- **Ties are allowed:** 7 of the 50 sample rows rank both systems the same (`1 1` or `2 2`).
- **No translations in the file.** To set COMET beside the human ranks, Step 1 has to match each row to the student's A6 files by the **source text**. Normalise whitespace before matching, say how many of the 50 matched, and list any that did not.
- **Encoding:** UTF-8, no BOM, LF, no final newline. Students on Windows may re-save it, so keep the loader tolerant of a BOM and CRLF, as the A12 loader is.
- **The check the sample gives you:** mean ranks 1.48 (microsoft) and 1.50 (nllb), which match the student's write-up.

**What `pairwise_agreement` should do with the ranks.** For each matched sentence, the human preference is the system with rank 1, or a tie. The metric's preference is the system with the higher sentence score. Count agreements over the sentences where the human did not tie, and report the ties separately. This is the same "do the people and the metric pick the same winner" question the final project now asks (section 2 below), sentence by sentence.

**Please report back:**
- the notebook's steps as students will see them, the four questions as worded, and the points (the Lecture 11 slide says "26 points, plus 10 for extra credit");
- an end-to-end run on the GPU path, using the sample file joined to synthetic translations if no real A6 translation files are at hand, with the agreement numbers it prints;
- anything Cowork should change on the A11 Learning Suite page (`jLj_`) or transcript (`notes/assignments/A11-directions.md`) to match. Cowork edits the page, after Eric reads it.

## 2. The final project is now one Learning Suite page

Done tonight, for the roadmap and any notebook that mentions the project:

- **One page:** "Final Project" (Content page `Bauj`), with four sections: Proposal, Presentation, Write-up, Submission. The four gradebook entries link to it by section name. `ONM_`, `PmbM` and `hmfS` are deleted. Transcript: `notes/assignments/FP-directions.md`; the four old transcripts are in `notes/assignments/superseded/`; the README is updated.
- **Two paths:** A, investigate a question (a hypothesis, one change against a named baseline); B, build a component (a user, a job, and what they do today as the baseline).
- **Both paths need:**
  - **people and a measure:** if text, then at least 3 people each judge the same 100 test sentences against the baseline (MTEval, as in A6), plus BLEU or chrF and COMET, and report whether the people and the metric pick the same winner;
  - **a fair test:** held out, checked for overlap the way A8 does, the same for every system;
  - **where it helped and where it hurt;**
  - **reproducible:** versions, seeds and decoding settings named.
- **Dates for the roadmap's schedule:**
  - **Thu Dec 10:** last day of classes; all final project work due at 11:59 pm.
  - **Wed Dec 16, 11:00 to 14:00:** the final exam slot; final project presentations.

  The Presentations due date on Learning Suite is now Dec 16, 14:00. The final exam row stays.

## 3. Already sent, for completeness

The A12 folders are clean (`from-cowork/…-c-…`), so the A12 loader can close against them.

## Not for now

The held items stay held: the A13 handout, A12's write-up questions, the 8b/8c badges and the first-loss slides.
