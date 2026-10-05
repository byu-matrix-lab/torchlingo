# Renumber the tutorials into course order

**Baton to you, 2026-10-05, third file today.** Eric's decision this morning: the tutorials are
renumbered so that their numbers follow the order the course meets them. Lecture alignments do
not change; only four numbers move. Course side flips its references (three slides, four
Content pages, the roadmap) the moment your merge lands, not before, so students see one
consistent set of names at any time.

## The mapping

| notebook | old | new | lecture (unchanged) |
|---|---|---|---|
| data-and-vocab | 01 | 01 | 4 (reference; also 9) |
| evaluating-translations | 07 | **02** | 6 |
| train-tiny-model | 02 | **03** | 7 |
| attention-and-alignment | 04 | 04 | 8a, 8b |
| transformer-attention | 08 | **05** | 8b |
| diagnosing-failures | 06 | 06 | 9 |
| real-translations | 05 | **07** | 10 |
| inference-and-beamsearch | 03 | **08** | 10 |

Why this order, for a reader who has never heard of the course: load data; learn to score before
you can train (tutorial 2 needs no model); train a toy; see attention on a task with a known
answer, then on real text; what to do when a model fails; what a real but undertrained model
produces; decoding as the last refinement. Real-translations stays at Lecture 10 rather than
moving to 8b (8b was getting too much), which also closes the roadmap's open question about its
placement.

## What to do

1. **Rename the four files** in `docs/docs/tutorials/`: `07-evaluating-translations` →
   `02-evaluating-translations`, `02-train-tiny-model` → `03-train-tiny-model`,
   `08-transformer-attention` → `05-transformer-attention`, `05-real-translations` →
   `07-real-translations`, `03-inference-and-beamsearch` → `08-inference-and-beamsearch`. Mind the
   ordering of the renames (02 and 05 are both vacated and reoccupied).
2. **Rewrite every cross-reference by number.** The notebooks cite each other in prose and in
   links ("Tutorial 3 warned you not to trust its BLEU of 100", "the checkpoint is the one
   Tutorial 5 reads", "[Tutorial 8](08-transformer-attention.ipynb)", the "What's next" cells,
   the `# Tutorial N:` titles). Also the docs index and nav, the concepts pages, the reference
   pages, `README.md`, `notes/NOTEBOOK_AUDIT.md`, the generated notebook map, the CI and
   `student_path.sh` lists, and the notebook metadata if it carries a number. A grep for
   `[Tt]utorial\s*\d` and `0\d-[a-z]` over the repository should find all of them; please
   list in your reply anything it found that you chose not to change.
3. **Keep the old filenames alive as redirect stubs** for the rest of the term: a one-cell
   notebook at each old path (`02-train-tiny-model.ipynb`, `03-inference-and-beamsearch.ipynb`,
   `05-real-translations.ipynb`, `07-evaluating-translations.ipynb`, `08-transformer-attention.ipynb`)
   whose single markdown cell says "This tutorial is now [N-name](N-name.ipynb)" with the Colab
   badge of the new file. Students who open a bookmarked badge or a Content page we have not
   flipped yet land somewhere useful. Remove the stubs after the term; note that in `TASKS.md`.
4. **Tell us when it is merged**, with the final filenames. Course side then flips, in one pass:
   - Lecture 8a deck: "A Learned Alignment, Seen" (slide 20) says "Tutorial 8" → 5.
   - Lecture 8b deck: "Where We Left Off" says tutorial 8 → 5.
   - Lecture 10 deck: the decoding slide says "Tutorial 3" → 8.
   - Learning Suite Content pages for Lectures 6, 7, 8a, 8b (and 9 and 10 when they are built):
     names and badge links.
   - Roadmap v11 and the audit's errata.

Unchanged references, for the record: tutorial 4 (8a, 8b decks and pages), tutorial 6 (8a's
loss-curve slide, 8b's A8 slide, Lecture 9), tutorial 1 (Lecture 9's bridge slide).

## Timing

Nothing here is urgent for Wednesday; the Lecture 7 and Lecture 9 hand-offs from this morning
come first. But the Lecture 9 and 10 decks are being rebuilt this week and next, and each
rebuild would otherwise bake in numbers that then change, so sooner is cheaper. If you can land
it with redirect stubs this week, the Lecture 10 rebuild goes straight to the new numbers.
