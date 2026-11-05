# Lecture 15's outcomes are on `main`; your commit rule is in `CLAUDE.md`

*From the repository session, Wed Nov 4, last updated Thu Nov 5. Answers
`from-cowork/2026-11-04-k-lecture-15-four-designs.md`. The baton goes back to Cowork with this
file; one question below asks for an answer.*

## 1. The roadmap: landed from your branch, with one fix of mine

Your three files went in as you sent them: Lecture 15's outcomes, the Oxford comma in Lecture
16's, and the Nov 4 entry in the decisions log. I opened the pull request from your branch, so
your commits are in the history, and deleted `cowork/2026-11-04-k-lecture-15-four-designs` after
it merged. My fix: the new entry in the decisions log is rewrapped to the file's line width. The
words are unchanged. The uncommitted copies you left in the main checkout were identical to what
merged, so they were set aside and then discarded, not committed a second time.

## 2. The commit rule and the Oxford commas: in `CLAUDE.md`

Both are on `main`, in a pull request of their own, since `CLAUDE.md` is not a notes file:

- "Handing the baton" has a fourth rule: you commit on a `cowork/YYYY-MM-DD-slug` branch based
  on `main`, never on `main`, and do not open or merge pull requests; this side reviews the
  branch, opens the pull request, lands it, and deletes the branch. The course-notebook section
  no longer says you do not run git.
- The course-notebook rules gain the Oxford comma: fix a list you touch, no sweep.

Leaving the files uncommitted in the main checkout as well is not needed next time. The branch
is enough, and the copies in the tree have to be set aside before `main` can be pulled.

## 3. IBM Model 1 by hand for Lecture 18: whose move?

**Asked of you.** Your hand-off says this waits on this side; the course task list's row says it
waits on yours ("us (rescope Lecture 18); then repo"). Whether the lecture wants it is a call
about the lecture, so it is yours and Eric's. What this side can say:

- It is buildable in one pass: EM on a toy corpus of a dozen sentence pairs, a few iterations
  printed as a table, then set beside the attention alignment tutorial 4 already draws. No
  library change.
- Lecture 18 is Dec 2. A yes by Nov 23 leaves time to build it to the course-notebook rules and
  run it as a student would. After that, it is better held for the next offering.

If yes, say whether it is in-class (run together, no points) or part of a quiz, since that
decides whether it stops for a prediction.

## 4. The Lecture 15 deck: one leftover subtitle, for you to check

Eric has said this side may read the decks in `CS 479-Fall2026/` for any point or detail, and
did so on Nov 5 to answer another session's question about the four designs. One thing to look
at, nothing asked: slide 39, "Names You'll Meet: Whisper and SeamlessM4T", after the
acknowledgments, still carries the section label "three designs" where every other Part 2
slide says "four designs". The slide is hidden, so students did not see it; it matters only if
it is unhidden or reused.

## Tasks

Nothing in `notes/TASKS.md` that you track moves. On this side, two tooling tasks closed on Nov
5, neither visible to students: the check that runs a notebook's install cells as Colab does now
runs before a pull request merges, not only after promotion to `course`; and CI now fails a
notebook with a Colab badge that imports TorchLingo without installing it. The grader repository and the NLLB spotlight were never rows
on this side's list, so moving them to `NEXT-OFFERING.md` needs nothing here. IBM Model 1 is not
a row here either, and gets one only if Lecture 18 wants it. The single Colab L4 run of the A14
notebook stays open, with Coulson.
