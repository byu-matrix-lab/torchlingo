# Research questions in the assignment notebooks, and A14 due Friday Nov 13

*From the Cowork session, Mon Nov 2, afternoon. Answers
`to-cowork/2026-10-28-outcomes-landed-lecture-14-facts-hold.md` (nothing was open there). The
baton comes back to Cowork when these files are on `main` and the notebooks are on `course`, or
when you need a decision.*

## Read this first: what to do, in order

| When | Section | What |
|---|---|---|
| **Now** (students are in A12, A13 and A14) | **1** | Commit the ten notebooks' research questions and promote to `course`. |
| **Now** | **2** | Commit the roadmap edits: A14's new due date and the Nov 2 decisions entry. |
| **When convenient** | **3** | Consider a standing rule in CLAUDE.md's house patterns. |

The working tree holds exactly these thirteen files, all from this side:

```
 M docs/docs/course/lecture-03-word-embeddings.ipynb
 M docs/docs/course/lecture-06-mt-evaluation-homework.ipynb
 M docs/docs/course/lecture-07-toy-model.ipynb
 M docs/docs/course/lecture-08a-a8-kickoff.ipynb
 M docs/docs/course/lecture-09-subword-tokenization.ipynb
 M docs/docs/course/lecture-10-comet-install.ipynb
 M docs/docs/course/lecture-11-comet-on-a6.ipynb
 M docs/docs/course/lecture-12-llm-context.ipynb
 M docs/docs/course/lecture-13-back-translation.ipynb
 M docs/docs/course/lecture-14-multilingual.ipynb
 M notes/CS479_COURSE_ROADMAP.md                    (rendered)
 M notes/roadmap/roadmap.yaml
 M notes/roadmap/sections/16-decisions-log.md
```

*Added Tue Nov 3, before the repository picked this up:* the roadmap now also carries Lecture
15's refined outcomes (Eric: "Explain how spoken language differs from the text MT is trained on"
replaces "where errors compound") and a Nov 3 decisions entry (research questions in the
notebooks; the Final Project proposal states its research question). Same files, re-rendered;
both checks pass.

*And later on Nov 3:* Lecture 15 split in two (Eric). Lecture 15 is Speech Translation (Nov 4);
a new Lecture 16, Automatic Dubbing and Live Interpretation (Nov 18), takes simultaneous MT from
the multimodal lecture, which becomes Lecture 17 (Nov 30); research and open problems merge with a
condensed word alignment and earlier methods as Lecture 18 (Dec 2). `roadmap.yaml` (lectures and
lecture outcomes 15 to 18), sections 03, 05, 10, 15 and 16, all re-rendered; both checks pass. No
notebook quotes those outcomes. Learning Suite's pages and Lecture Quiz 15 already follow.
Later still: Lecture 15's outcome 2 is now "Explain how cascade, agent and speech-native designs differ.", and Lecture 16 has a fourth, "Explain how a live system uses a script, when it has one." (`roadmap.yaml`, re-rendered).

Each notebook was normalized with `notebook_meta.py --normalize` after its edit;
`notebook_meta.py --check` and `render_roadmap.py --check` pass. Preflight was not run here.

## 1. Research questions in the assignment notebooks

Eric, Nov 2: every assignment states its research questions, not only its experiments, and
the assignment notebooks open with them. Each of the ten notebooks gains one block in its
opening markdown cell, just before "By the end you will be able to:", markdown only, no code:

```
**Assignment N's research questions:**

1. ...
2. ...
```

The questions are word for word those on the assignment's Learning Suite page (a new Research
Questions section before What to do) and in its transcript in torchlingo-private
`notes/assignments/`. Eric approved the wording. A1, A4, A5 and the Paper Review have none by
his decision (reading and skills assignments), and A2 has no notebook. The Lecture 3 activity
carries A3's questions, since it is A3's dry run, and the Lecture 6 homework carries A6's.

Please promote to `course` with this commit: A12 is due today, and students are in A13 and A14.

## 2. The roadmap

- **A14 due Friday Nov 13, 11:59 pm** (Eric, Nov 2), not Tuesday Nov 10: the A14 entry and its
  window note in `roadmap.yaml`. This overrides the Oct 22 rule that A14 end before the Nov 11
  proposal presentations; the Final Project proposal now overlaps A14's last days, and Eric
  accepted that. The Learning Suite due date, the A14 page and the Lecture 14 deck already say
  so; the A14 transcript's `Due:` line is updated, so `check_transcripts.py` should pass.
- **Section 16, decisions log:** a Nov 2 entry: the A14 date; the Mixture-of-Experts slides
  moved to Lecture 14's hidden appendix; "released", not "set", for assignments; and research
  questions stated on every assignment.

## 3. A suggestion for CLAUDE.md

A house pattern for course notebooks: *an assignment notebook opens with the assignment's
research questions, word for word from its Learning Suite page, before "By the end you will be
able to:"*. A new notebook for A15 or later would then follow it without a hand-off. Your file,
your call.

## On the course side, for information

- **Lecture 14** was taught today; its deck is promoted (`_F2026`, v9) and posted. Two
  assignment slides now state their research questions in the subtitle (the Final Project idea,
  and A14).
- **The Final Project page:** the proposal's first slide now states the research question in
  one sentence, on either path.
- **Lecture pages:** "Please do not distribute without permission." is its own bullet on every
  one.

## Tasks

Ours: the research-questions row closes with this file, and the Lecture 14 deck row closed
today. Nothing moves on your list from this side, except that A14's date in any task row there
should read Nov 13.
