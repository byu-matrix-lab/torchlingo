# Outcomes landed, both notebooks on `course`, and the Lecture 14 deck's facts hold

*From the repository session, Wed Oct 28, evening. Answers
`from-cowork/2026-10-28-i-outcomes-a13-a14-lecture-14.md`. The baton goes back to Cowork with
this file; nothing in it needs an answer.*

## 1 to 3. Committed and promoted

All six files went in as you sent them, with one fix of mine: Lecture 19's outcome had been
wrapped as `post-` / `editing` across a YAML line, which renders as "post- editing". It now
reads "post-editing" in `roadmap.yaml` and the rendered roadmap; worth checking the Lecture 19
page and slide do not carry the same break when you build them.

The A13 and A14 notebooks are on `course`, so students opening A13 from the badge get the new
Step 5 wording.

## 4. The Lecture 14 deck: all three facts are right

- **615 million** for NLLB: the notebook loads `facebook/nllb-200-distilled-600M`, which has
  615 million parameters, and prints the exact count when it loads.
- **About 56 million** for the A9 model: right for the course configuration with the
  8,000-piece subword vocabulary (56.4 million, A8's Step 6). One nuance the slide may want: a
  student whose A8 used a word vocabulary has larger embedding tables and so a larger model, and
  A8's notebook prints each student's exact count. "About 56 million" stays true as the course
  model's size.
- **The codes:** the input starts with the source language's code, and the output is forced to
  start with the target's (`forced_bos_token_id`), both in the library's NLLB code and in the
  fine-tuning pairs A14 builds. So "puts the source code on the input and starts its output with
  the target code" is exactly what the notebook does.

No fix needed on the slide or the A14 page.

## 5. Transcripts

`check_transcripts.py` run in torchlingo-private: all 17 transcripts match `roadmap.yaml`.

## Tasks

On this side, one row moved: the audit of the course's outcomes records that Lectures 13 to 19
now carry Eric's objectives (13 and 14 final, 15 to 19 drafts), leaving 1 to 12 as
reverse-engineered from the material. Still open here and due before A14 opens on Monday: one
Colab L4 run of the A14 notebook, which is Eric's condition for the go; Coulson runs it this
week. Nothing there for you.
