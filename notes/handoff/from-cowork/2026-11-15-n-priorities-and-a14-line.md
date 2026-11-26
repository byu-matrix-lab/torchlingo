# N: A14's line is posted; the course list is ranked P1 to P4; outcomes done; two notebooks asked

From the Cowork session, Sun 2026-11-15, last updated 02:10. Answers `to-cowork/2026-11-06-a14-times-itself.md` and `to-cowork/2026-11-15-bleu-a14-and-priorities.md`. Baton to the repository session.

## 1. Why the Nov 9 ask went unanswered

The baton did not come to this session between Nov 6 and Nov 15, so your Nov 9 reply was first read today, with the Nov 15 one. Nothing was ignored; the repeat was right.

## 2. A14's page line: posted

Under **What to submit**, after the write-up, as a bullet: *"The timing table the last cell prints (each stage's time, your GPU, and your training steps), pasted at the end of your write-up. It is for planning and carries no points. If your notebook prints no timing table, note roughly how long each step took instead."* The page never asked for times by hand, so there was nothing to choose between. A14 has closed, so this serves late submissions and the record; the times also come back in `a14-rerun.json`, as you said.

## 3. The course list now uses P1 to P4

`torchlingo-private/notes/course/TASKS.md` is one table, ranked by your levels and sorted, and points to your "Priorities" section rather than copying it. **The P1 items:**

1. **Lecture 16's deck:** the Quiz 15 review slide, and the narrative and currency pass with Eric (then promote).
2. **Attendance Quiz 16:** one reading question, on Federico et al. only.
3. **Grading A11 and A12 BLEU** for languages written without spaces, with Eric: your section 3 note is on the list for whoever grades.
4. **Lecture 17's deck** (multimodal, then writing the report), with Quiz 17.
5. **Lecture 18's deck**, linking your IBM Model 1 notebook, with Quiz 18.
6. **Lecture 18's notebook:** "Runtime → Run after" still waits on Eric opening it in Colab.

None of them needs anything from your side. P2 includes Lecture 19, the closing lecture, the peer-feedback form for the final presentations, and the optional Final Project BLEU line (section 4).

## 4. The Final Project BLEU line: Eric's call

On the course list as P2, waiting on Eric. If he says yes, it goes in as you wrote it.

## 5. Your two fixes

Noted, and nothing needed here: A14's resume fix (Eric messaged students), and BLEU in A11 and A12 matching A9 (the grading note above).

## 6. New, asked of you: "What you need" as bullets in the notebooks (Eric, Nov 15)

Eric, tonight: *"The 'What you need' part of the project directions would be better as bullets. Same for every project."* The assignment pages now carry a **What you need** section of their own, a bulleted list, before the steps. Three course notebooks have the same paragraph in their opening cell, as prose: `lecture-06-mt-evaluation-homework`, `lecture-13-back-translation`, and `lecture-14-multilingual`. Please make each a bulleted list, with the same items, and use bullets for any notebook's "What you need" from now on (worth a line in `CLAUDE.md`'s house patterns). P2: something Eric asked for, no date.

Also changed tonight on every assignment page, in case a notebook or the roadmap quotes a heading: **"What to do" is now "Do the Following:"** and **"What to submit" is now "Submit the Following:"**; "Write-up questions" is a section of its own on A10, A12, A13, and A14. No notebook quotes either old heading today (checked). And Eric prefers absolute workload estimates to comparisons ("the largest assignment"): if a notebook says how long something takes, then give hours or a measured ratio, as A13's and A14's do.

## 7. A15's page is published, as a reference (Eric, Nov 15)

The roadmap said A15's page was unpublished; it never was. Eric chose to keep it visible as a reference for the Final Project's speech path, so the page now opens with *"Not assigned this term. This page is kept as a reference for the speech path of the Final Project (Path B) ... Nothing here is due, and it carries no points."* This branch corrects `roadmap.yaml`'s A15 row to say so, re-rendered (`--check` passes; the line may want rewrapping). The decisions log's Oct 22 entry, which says "unpublished", is history and is left as written.

## 8. Learning outcomes, leveled up: every page done (Eric, Nov 15 to 25)

Eric wanted every lecture's and assignment's outcomes more conceptual: what a student carries beyond the one lecture or assignment, with no bare deixis. **The review is complete** (Nov 25): every lecture and assignment page on Learning Suite carries its new outcomes. The record, with each page's old and new lists, is `Working Notes/CS479 Learning Outcomes Rewrite_v36.md` in the course folder (and `torchlingo-private/notes/course/`).

On this branch, every lecture's `can` row now matches its page: Lectures 1 to 19, one commit per lecture (re-rendered; `--check` passes). Two rules Eric added along the way, for anything you write that quotes them: name the technique (A13: "back-translated data", not "synthetic data"), and no bare deixis ("when sharing one model across languages hurts", not "when that sharing hurts").

**Asked of you, P1, in one pass: the notebooks' outcome quotes.** 26 notebooks quote a lecture outcome that has changed (checked by matching each `These serve Lecture X's outcome` quote against the new `can` rows):

- Course notebooks, Lectures 3 to 14: `lecture-03-word-embeddings`, `lecture-04-regex-refresher`, `lecture-04-tmx-cleaning`, `lecture-05-sentence-alignment`, `lecture-06-mt-evaluation`, `lecture-06-mt-evaluation-homework`, `lecture-07-toy-model`, `lecture-08a-a8-kickoff`, `lecture-08b-cross-attention`, `lecture-08c-attention-heads`, `lecture-09-subword-tokenization`, `lecture-10-comet-install`, `lecture-11-comet-on-a6`, `lecture-12-llm-context` and `lecture-13-back-translation` (these two paraphrase rather than quote; check them), `lecture-14-multilingual`.
- Tutorials 1 to 9, each quoting one outcome of Lecture 6, 8a, 8b, 9, 10, or 14.
- Only `lecture-18-ibm-model-1` still matches.

Where a notebook quoted an outcome that no longer exists in that form (tutorial 3's "start, checkpoint and resume a long run", for instance), quote the new outcome it serves, or say in your reply that none fits, and we'll take it to Eric.

## 9. New, asked of you: a tutorial on IBM Model 2, predict then reveal (Eric, Nov 25)

Eric wants a **tutorial** (not an extension of the course notebook) on IBM Model 2, and a node for it in the tutorial map (`charts.tutorials` in `roadmap.yaml`). Not tied to Wednesday's class: it is reading, with no points.

- **Its own purpose.** Model 2 adds what Model 1 ignores, word position: a learned table of how likely a source position is to align to a target position, which mostly favors the diagonal. The lesson: a fixed preference for word order helps where two languages keep the same order and misleads where they reorder, and attention avoids the trap by weighing content as well as position.
- **Predict, then reveal** (Eric). Before running Model 2, the student predicts whether position will help or hurt on a pair that reorders, such as an adjective and its noun, and why. The reveal shows the diagonal pulling the alignment the wrong way where the order flips, and helping where it does not.
- **Not a look-alike** (CLAUDE.md: students skip what looks like a repeat). The Lecture 18 notebook uses thirteen English-Spanish noun phrases; this tutorial needs different data and a different opening, though it may assume the Lecture 18 notebook (tutorials may assume course notebooks). Tutorial 4 measures attention as alignment in a trained model; this one is about word order before attention, so the two stay distinct.
- **The map.** Suggested: tutorial 10, read after the Lecture 18 notebook, leading to the Final Project ("word order in your language pair"). Your call on number, title, and arrow, by the map's rules.

## 10. Outcomes: Lectures 11 and 18 on this branch (Eric, Nov 25)

This branch now also changes **Lecture 11's** and **Lecture 18's** `can` rows to their pages' new outcomes, and Lecture 18's schedule title to *Open Problems, and What Earlier MT Methods Teach* (Eric: no lab research segment), with its scope note (re-rendered; `--check` passes). Two notebooks are affected:
- `lecture-11-comet-on-a6` quotes Lecture 11's old outcome. It's stale now and goes into your one pass over the notebooks' outcome quotes.
- `lecture-18-ibm-model-1`: its outcome quote is unchanged, but its generated purpose cell carries the old title, so `notebook_meta.py --normalize` refreshes it.

## 11. New, asked of you: Lecture 19's edit distance notebook (Eric, Nov 25), by Thursday Dec 3

Eric wants a course notebook for Lecture 19 (Mon Dec 7), the in-class activity for its outcome *Explain how edit distance measures post-editing effort* (the outcome under review reads *..., and what it misses*; the review is paused, so expect that wording). Please land and promote it by **Thursday Dec 3**, so the deck and the Lecture 19 page can link it on Friday.

- **Shape:** `family: course`, `role: activity`, `serves_lectures: [19]`, about 15 minutes in pairs, no points, CPU only (no `setup(gpu=True)`), `requires: [exercises]` with a `tests/notebook_cells.py` test that fills the answers in.
- **What the student does, in order:**
  1. **Post-edit** a few MT outputs into English themselves: fix each so it is correct, with as few changes as they can. English targets, so every student can do it whatever their languages.
  2. **Predict** (its own cell, stopping while empty): which output will need the most editing, by the number.
  3. **Compute** word-level edit distance between each MT output and their own post-edit, normalized by the post-edit's length (HTER's idea). The dynamic program is short and a CS student has met it, so it stays inline as the lesson; then the same with sacrebleu's TER (a dependency already) to see that TER counts a moved phrase as one edit.
  4. **Compare** with a professional post-edit of the same outputs, if the data has one: two correct post-edits, two different numbers.
  5. **Find what it misses:** a pair of cases where the number misleads. One wrong word that reverses the meaning (a dropped "not") costs one edit, as a harmless word swap does; and the number says nothing about the time or thought an edit took.
- **Data:** your call. A public post-editing set with source, MT, and professional post-edit (the WMT automatic post-editing or QE data, if the license allows) gives step 4 for free; otherwise a handful of fixed outputs from a model we already use. Either way, include at least one output with a meaning error that a one-word edit fixes, for step 5.
- **The write-up asks the concept** (CLAUDE.md): when is edit distance a fair measure of post-editing effort, and when not? It is also a measure a Final Project may use (the roadmap's `shown_by`), so the notebook can say so in one line.

## Tasks

On the course list: A14's timing line is done and off; the list is ranked; two new rows (the A11/A12 grading note, P1; the Final Project BLEU line, P2). On yours: one new P2 row, the notebooks' What you need as bullets (section 6). Also new on yours (P2): the IBM Model 2 tutorial and its map node (section 9); the purpose cell refresh for `lecture-18-ibm-model-1` (section 10). And one P1, for Lecture 19: the edit distance notebook, landed and promoted by Thursday Dec 3 (section 11). And the P1 one-pass update of the notebooks' outcome quotes (section 8).
