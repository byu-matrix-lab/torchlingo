# Lecture outcomes in the roadmap, the A13 and A14 notebooks' wording, and Lecture 14 built

*From the Cowork session, Wed Oct 28, afternoon. Answers `to-cowork/2026-10-26-dates-landed-transcripts-checked.md`
(nothing was left open there; section 5 below closes its one suggestion). The baton comes back to
Cowork when these files are on `main` and the two notebooks are on `course`, or when you need a
decision.*

## Read this first: what to do, in order

| When | Section | What |
|---|---|---|
| **Now** (students are in A13, due Fri Nov 6) | **1** | Commit the A13 notebook's two wording changes and promote to `course`. |
| **Before A14 opens** (Mon Nov 2) | **2** | Commit the A14 notebook's outcome quote; promote with section 1. |
| **Now** | **3** | Commit the roadmap edits in the working tree; rendered, and both checks pass. |
| **When convenient** | **4** | Two facts in the Lecture 14 deck, quoted from the A14 page, for you to confirm. |
| **For information** | **5, 6** | What changed on the course side, and the Lecture Guide's new rules. |

The working tree holds exactly these six files, all from this side:

```
 M docs/docs/course/lecture-13-back-translation.ipynb
 M docs/docs/course/lecture-14-multilingual.ipynb
 M notes/CS479_COURSE_ROADMAP.md                    (rendered)
 M notes/roadmap/roadmap.yaml
 M notes/roadmap/sections/07-learning-outcomes.md
 M notes/roadmap/sections/16-decisions-log.md
```

`render_roadmap.py --check` and `notebook_meta.py --check` pass on them. Each notebook was
normalized with `notebook_meta.py --normalize` after its edit. Eric, Oct 28: both sessions
edit the roadmap; I edited it while holding the baton, and nothing else is pending on this side.

## 1. A13 notebook (`lecture-13-back-translation`)

Two markdown changes, no code:

- **Step 5** (Eric, Oct 27: "what the added data bought" is too cryptic): "separates what the
  added data bought from what the decoder bought" now reads "tells how much of a change came
  from the back-translated data and how much from the decoding method".
- **The opening cell's last line** said the three abilities "together" were Lecture 13's
  outcome. Lecture 13 now has five objectives (section 3), so it reads: "These serve Lecture 13's
  outcome, three of its five parts: what makes a language low-resource, how back-translation
  turns target-side text into training pairs, and how to separate the effect of data from that
  of decoding." It keeps the `These serve Lecture X's outcome` opening the check requires.

Please promote to `course` with section 2, since students are working in A13 now.

## 2. A14 notebook (`lecture-14-multilingual`)

The opening cell quotes Lecture 14's outcome verbatim. It now quotes the new one (section 3).
No other change.

## 3. The roadmap

- **`lecture_outcomes`, Lectures 13 to 19:** each `can` is now that lecture's objectives, one
  line each, "Explain ...", in the same words as its objectives slide and its Learning Suite page
  (Eric, Oct 28). 13 is as taught, with Eric's own rewording of the first ("Explain what makes a
  language qualify as low-resource."); 14 is as built for Monday; 15 to 19 are drafts from the old
  `can` text, which Eric will refine as each lecture is built. **Lecture 15's value is quoted**:
  it contains "translation: timing", and the colon broke the YAML until it was.
- **`lectures`, Lecture 13:** `deck: F2026` (was F2025); promoted Oct 28.
- **Section 07:** "Two or three per lecture" no longer held; it now says the outcomes from
  Lecture 13 on are the lecture's objectives, the same words as its slide and page.
- **Section 16, decisions log:** a new Oct 28 entry: outcomes as one-line objectives matching
  slide and page; lecture quizzes at most 8 questions, generic, following the current outcomes,
  this offering and the next; the in-class quiz review is one slide on one question; Lecture
  Quiz 12's Q5 thrown out.

## 4. Two facts to confirm, in the Lecture 14 deck

The deck's "Fine-Tuning Against Training From Scratch" slide quotes the A14 page: **about 56
million weights** for the A9 model, and **615 million** for the NLLB version A14 fine-tunes. And
"A Tag Names the Language to Write" says NLLB **puts the source code on the input and starts its
output with the target code**. If any of the three is wrong for what the notebook actually loads,
then tell me and I will fix the slide; the A14 page would need the same fix.

## 5. Your suggestion of Oct 26

Done before this hand-off: every transcript's italic header in torchlingo-private has lost its
dates and points, and keeps only the transcription note. `check_transcripts.py` was not re-run
here; please run it when the baton arrives, as you planned.

## 6. On the course side, for information (all on Learning Suite unless said)

- **Lecture 13** was taught today and its deck promoted to `_F2026`. Lecture Quiz 13 has eight
  new questions. Lecture Quiz 12's Q5 is thrown out: every student has its point.
- **Lecture 14's deck** `_F2026_v3` is built for Monday, from a per-lecture script on a shared
  `course/build/lecture_kit.py` in torchlingo-private (Eric: DRY). Eric is reviewing it.
- **Learning Outcomes** are on the Lecture 13 to 19 pages, matching section 3.
- **Paper links:** every embedded "Download" file widget, on 12 pages, is now a plain link on the
  paper's title, one paper per bullet.
- **The Final Project page** has a new section, "Projects from earlier offerings", moved there
  from Lectures 15, 16 and 19.
- **The A12 language list** (Efik, Kosraean, Palauan, Pohnpeian, Telugu, Yapese) is not the A13
  language: A13 stays in each student's A8 and A9 language. The Lecture 13 deck now says so, and
  points students who want their A12 language to the Final Project.

## Tasks

Nothing moves on your list from this side. Ours: the hand-off I row closes with this file; the
Lecture 14 deck row is Eric's review, then promotion; the quiz rows for Lectures 14 to 19 stay
open.
