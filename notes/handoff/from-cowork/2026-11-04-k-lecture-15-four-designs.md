# K: Lecture 15's outcomes after it ran; Oxford commas

From the Cowork session, 2026-11-04. Baton to the repository session.

## What changed in the working tree (yours to check, render, and commit)

- `notes/roadmap/roadmap.yaml`, `lecture_outcomes`:
  - Lecture 15's row now reads "explain when to choose among four speech translation designs; explain how a learned codebook turns sound into tokens" (was "how cascade, agent and speech-native designs differ" and "how a speech model turns sound into tokens").
  - Lecture 16's first outcome gains an Oxford comma: "timing, length, and voice".
- `notes/roadmap/sections/16-decisions-log.md`: a Nov 4 entry, newest first, recording the fourth design (the speech-to-text model: ASR and MT in one model, then TTS) and the Oxford-comma rule.
- `notes/CS479_COURSE_ROADMAP.md`: re-rendered with `scripts/render_roadmap.py --write`; `--check` passes.

## Why

Lecture 15 ran today and went well. Afterwards Eric asked for:

- the MATRIX demo at the front;
- slides on feature vectors, the VQ-VAE, and building the codebook;
- a fourth design, from Wesley's reading report, which Gemma 4's audio models also implement;
- outcomes that match the deck.

The deck and the Lecture 15 page in Learning Suite carry the same wording.

## A standing rule for both sides

Eric prefers Oxford commas ("A, B, and C") in all course text: slides, notes, Learning Suite pages, and notebooks. If a notebook or README you touch has a list of three or more without one, please add it as you go. No sweep is asked for.

## Tasks

Nothing in `notes/TASKS.md` moves. No questions are open from this side.

The course task list (`torchlingo-private/notes/course/TASKS.md`) was cleaned up the evening of Nov 4 (Eric): rows for lectures that have run and settled questions came off, and ideas for next year moved to `NEXT-OFFERING.md` ("Moved from the task list"). Two rows that carried your task numbers moved there: Task #120 (the grader repository: public, licensed, then the Lecture 4 and 5 decks) and Task #160 (the NLLB spotlight; Eric says Lecture 14 now has it, so it was dropped). Task #253 (IBM Model 1 by hand for Lecture 18) stays on the course list, waiting on this side to say whether Lecture 18 wants it.
