# Deck passes, the build library, and what is uncommitted in torchlingo-private

**Baton to you, 2026-10-12, evening.** From the Cowork session. This file sat under a `draft` name while it was written; the rename is the send. Nothing here needs a notebook change. Two things for you: commit what is in the `torchlingo-private` tree, and one question at the end.

## 1. What is in the tree, uncommitted

All in `torchlingo-private`, the private repository (course data, instructor material); nothing of this touches the public `torchlingo` tree.

- `course/build/` (new directory, 50 or so scripts): every python-pptx pass that produced a CS 479 deck since Oct 5, with a README. New since Oct 9: `deck_lib.py` (the shared library: unpack-to-pairs, question slides, citation cards, bold leads, title trim, `carry_rels` on every clone, `copy_slide_across`), the per-deck scripts `8a_v3.py`, `8b_v5.py`, `8c_v12.py`, `8c_claimfirst_v11.py`, `reorder_8_9.py`, `l9_*` (v6 through v16), `l10_v2.py`, `l1_v2.py`, `check_rids.py` (a lint for dangling relationship ids, which PowerPoint reports as a damaged file and LibreOffice never does; it runs on every deck before it leaves Cowork), `l9_fix_links_v7.py`.
- `course/figures/` (new directory): the figures the decks use (attention maps, loss curves, the vocabulary-growth PNG and CSV with `vocab_growth.py`, the whiteboard photo), plus `spm-enes-3000.model` / `.vocab` and `train_exercise_spm.py`: a 3,000-piece BPE SentencePiece model trained on NLTK's Europarl sample (English and Spanish together) for the "Subwords by Hand" exercise in Lecture 9. Nothing from the course data or the students.
- `notes/assignments/`: every transcript refreshed from Learning Suite on Oct 8 and after (the per-assignment Content pages); `A15-directions.md` is new, `A16` was removed when the assignment was.

Commit as you see fit; one commit is fine. The course folder's log (`Roadmap/CS479-LOG.md`) has the per-day record if you want commit messages that say more.

## 2. What happened on the course side since Oct 7, in one paragraph each

- Every deck from 8a on carries the same set of lessons now: at most two panels of body text a slide; the claim is the slide, with a question slide before its evidence and a citation card before each paper; figures full width with explanations in the notes; the lead phrase of every bullet bold; "if X, then Y" for conditionals; titles stop before a colon when the boxes carry the headings; no repeated subtitles on an unpacked pair; no self-acknowledgment. 8a `_v5`, 8b `_v6`, 8c `_v13`, Lecture 9 `_v16` and Lecture 10 `_v2` are in the course folder awaiting Eric's adoption and re-posting.
- Lecture 9 was taught on Oct 12 (it went well). Its deck gained: four morphology figure slides (inflection, derivation, compounding, agglutination), Koehn's BPE and SentencePiece paragraph examples rebuilt as native slides, a BPE-against-SentencePiece pseudocode slide, and the by-hand exercise with its answer slide (handout `Lecture 9 Subwords by Hand_v1.md` in the course folder). Eric will decide whether the exercise opens Lecture 10 or waits for next year.
- Cross-deck moves: "The Model Behind the Attention Maps" from 8b to 8a; the position pair and Deployment Challenges from 9 to 8c.
- Lecture 10 is rebuilt for Wednesday. Learning Suite: the Lecture 10 and Assignment 10 pages now link the repository's `docs/docs/course/lecture-10-comet-install.ipynb` (the Colab badge URL on `main`) instead of last year's Drive copy; Quiz - Lecture 10 has two new questions on the A10 papers; the syllabus names Level 1 (Reference) of the department AI policy, says the lecture quizzes are open book, and has a Colab Pro paragraph.

## 3. One question

**Tutorial 8 (`08-inference-and-beamsearch`) still cannot run from its badge** (it wants tutorial 3's checkpoint). The Lecture 10 deck and the Lecture 9 Content page say "it is a read" and link no badge. If the rewrite lands, tell us and the badge goes on the Lecture 10 page and the "Decode the Same Way" slide.

Nothing else is asked. The Cowork session holds the baton for the course folder and Learning Suite; you hold it for both repositories.
