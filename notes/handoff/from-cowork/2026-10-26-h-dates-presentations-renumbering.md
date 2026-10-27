# Dates, two presentation days, lectures 15 to 19 renumbered, and A12's non-reasoning guard

*From the Cowork session, Mon Oct 26, evening. Answers `to-cowork/2026-10-26-charts-generated.md`
(all asks done; section 1 below). The baton comes back to Cowork when the roadmap edits are on
`main` and the A12 guard is merged, or when you need a decision.*

## Read this first: what to do, in order

| When | Section | What |
|---|---|---|
| **Before Mon Nov 2** (A12 due 10:00 am) | **2** | The A12 notebook's non-reasoning guard. |
| **Now** | **3** | Commit the roadmap edits in the working tree (nine files, listed there); they are rendered and `--check` passes. |
| **With 3** | **4** | One test and two comments that the renumbering breaks or stales. |
| When convenient | **5** | Due times for the Final Project parts, if you want `due_time`. |
| Nothing to do | **1** | Your Oct 26 asks are done. |

## 1. Your Oct 26 asks: done

- **Chart sources retired** in `torchlingo-private/course/figures/`; its README points to
  `notes/roadmap/generated/`. Cowork renders from there.
- **Every transcript has its `Due:` line** under the title (A15: "dropped this term"; FP: Dec
  10, 11:59 pm for parts 4 and 5, the rest in `roadmap.yaml`). So the standing check can go in.
- **Comparison by hand, today:** every transcript's `Due:` and **Total** match `roadmap.yaml`
  for A1 to A14 (A6, A12, A13, A14 with their stretch points).
- **A14 at 30 points** is confirmed (Eric, Oct 26).

## 2. A12: non-reasoning models only (Eric, Oct 26)

Eric: a reasoning model's chain of thought is a more powerful mechanism than in-context
learning, so it would confound what A12 measures. The A12 page and the Lecture 12 slides
already say it: "not trained for translation, and not a reasoning model (DeepSeek-R1 and its
distilled versions, Phi-4-reasoning, gpt-oss, and Qwen3 unless its thinking is turned off)."
In `lecture-12-llm-context`, please:

- pass `enable_thinking=False` to `apply_chat_template` wherever the template accepts it (Qwen3);
- refuse the known reasoning families (DeepSeek-R1 and its distills, Phi-4-reasoning, gpt-oss,
  QwQ) with a clear message naming the rule;
- flag any `<think>` in an output;
- record the check in `a12-rerun.json`.

The default, Qwen2.5-1.5B-Instruct, does not reason, so a student on the default sees no change.

## 3. Roadmap edits in the working tree, for you to commit

Edited and rendered on Eric's Mac (`render_roadmap.py --write`, then `--check`: current). Files:
`roadmap.yaml`, `CS479_COURSE_ROADMAP.md`, and in `sections/` `03-schedule-notes`,
`04-second-half`, `05-second-half-costs`, `06-through-line`, `10-lecture-scope-notes`,
`15-open-questions`, `16-decisions-log`. The decisions log has three new Oct 26 entries.

1. **A12 and A13 dates, overlapping windows (Eric).** A12 due Mon Nov 2 at 10:00 (no
   `due_time`, so the default); A13 Fri Nov 6 at 11:59 pm; A14 stays Tue Nov 10. Windows may now
   overlap, replacing Oct 22's "no two windows overlap": A13 Oct 28 to Nov 6, A14 Nov 2 to Nov
   10. If any notebook text names those days, please align it (Cowork found none).
2. **A1 to A3 due dates** corrected to Learning Suite's: Sep 9, 14, 16 (were Sep 2, 9, 14).
3. **Final project presentations over two days** (Steve Richardson: they need the last class
   day; Eric: make room on Dec 9 and give each student more time). Dec 9, the last class
   meeting, is day 1 (7 slots); the final exam slot, Dec 16, is day 2 (up to 18). Each slot is
   10 minutes: 7 to talk, 2 for questions, 1 to hand off. FP6's label says so.
4. **Lectures from 15 on renumbered, one number per session (Eric).** Last year's 18 (HAMT vs.
   MAHT) is dropped as a lecture; its post-editing stays as ten minutes of the new 19.

   | Date | New | From last year's |
   |---|---|---|
   | Wed Nov 4 | 15 Speech-to-Speech MT, with Automatic Dubbing and Interpretation | 15 and 16 |
   | Wed Nov 18 | 16 Multimodal NMT; then Writing the Final Project Report | 17, plus 21's writing slides |
   | Mon Nov 30 | 17 Research in the MATRIX Lab, and Open Problems in MT | 21's lab research, and 23 |
   | Wed Dec 2 | 18 Word Alignment, and What We Can Learn from Previous MT Methods | 19 and 20 |
   | Mon Dec 7 | 19 MT in Use: Applications, Jobs and Post-Editing | 22, with 18's post-editing |

   Schedule rows, the lecture outcomes (one per lecture now), the scope notes and the prose that
   named old numbers are updated. No notebook declares a lecture from 15 on (Cowork checked), so
   no `serves_lectures` changes. Lecture content is planned on the Cowork side; the yaml commits
   as edited.

Learning Suite already matches all of it: lecture pages, quiz names, calendar, gradebook.

## 4. What the renumbering breaks or stales on your side

- **`tests/test_notebook_meta.py::test_a_lecture_sharing_a_row_is_accepted`** declares
  `serves_lectures: [23]`, which no longer exists, and no schedule row now lists several ids
  (each session has one number). It will fail once the yaml lands; it needs a fixture, or a
  merged row from elsewhere (8a to 8c are separate rows too).
- **`scripts/notebook_meta.py` line 39** says tutorial 4 serves Lecture 19; that lecture is now
  18, and the tutorial's metadata says 8a and 8b.
- **`render_roadmap.py`'s comment near line 335** on "21 to 23 share one outcome".
- **Worth a look if a merged row comes back:** before the renumbering, the notebook index
  rendered the Dec 7 row (ids 21, 22, 23) as "21, 23", dropping 22. The schedule table above
  it was right.

## 5. Optional: Final Project due times

Learning Suite has FP1 and FP3 at 11:59 pm and FP6 and FP7 at 2:00 pm; the yaml's default is
10:00. Left as is unless you want `due_time` on them.

## Also, no action

- **A12's data link** on Learning Suite is now Steve Richardson's sharing link only (a student
  could not open Eric's). No notebook names either link (Cowork checked).
- **Lecture 12's deck** no longer names lecture numbers on its slides.
