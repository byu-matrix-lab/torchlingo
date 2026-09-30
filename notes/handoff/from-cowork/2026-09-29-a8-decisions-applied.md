# The A8 decisions are on the slides; the directions you asked for; the notebook audit is yours

**Baton back to you, 2026-09-29.** Answers `to-cowork/2026-09-28-b-lecture-9-answers-and-8a.md`.
Ordered by what matters first: 8a is tomorrow. Section 7 is the largest item: the notebook
audit with per-notebook directions, all assigned to you, lecture-08a first.

## 1. Your question: the Project Directions

They are not a separate document. For this course "the handout" is the assignment's own slides
in the lecture deck, and Learning Suite carries a copy of that text as the assignment
description. So I have started `notes/assignments/`, one transcript per assignment, taken from
the slides that are the authority:

- `A08-directions.md`: the three A8 slides as of today, **with section 5 applied** (56.4M
  configuration, "about 35 epochs, and keep going if validation loss is still falling", A100/L4/G4
  not T4). It ends with a table of the constants the kickoff quotes, so the thresholds cell can be
  checked line by line. The line that said "60 to 70" is gone from the deck; it is only in the
  roadmap's history now.
- `A07-directions.md`: the three-part exercise.
- **Update, later the same day: every assignment is now transcribed** from Learning Suite,
  `A01` through `A16`, the paper review and the four final-project items, with a README table
  giving each one's Learning Suite due date, roadmap due date and status. A8 carries both the
  current deck text and the stale Learning Suite text; A9, A13 and A14 are marked stale
  (OpenNMT). Every Learning Suite due date from A9 on is still the pre-shift date.

**Caveat you should know:** nobody has confirmed that Learning Suite's "Lecture 8 Assignment"
description has been replaced; it may still be last year's OpenNMT text. That is on the course-side task list (`CS479-TASKS.md`)
list for Eric and Coulson, not something either session can fix.

## 2. Section 5, applied to the decks and the roadmap

- **8a**: step 4 now reads "about 35 epochs; keep going if validation loss is still falling",
  with the 512 / 8 / 6 + 6 configuration and the best-checkpoint reasoning beneath it; the kickoff
  slide says "Runtime > Change runtime type: A100, L4 or G4, not the free T4" and mentions
  bucketing; "Reading Your Loss Curve" says "a few hours", not two.
- **8b**: the A8 reminder's clock is now your Colab numbers (150 to 360 ms per batch at worst-case
  lengths on an A100, so two to five hours for 35 epochs, less with bucketing) and the T4 warning.
  The capacity table is unchanged; it is the reason for the decision.
- **Roadmap v6**: 8a's Sets line, the measurement section (now recording the decision and the
  colab-memory numbers), and a Sep 28 evening entry in the decisions log. Your copy is refreshed;
  the tail is byte-identical to `HEAD`. About the refresh script: yesterday's version matched the
  tail marker inside its own generated header and duplicated the body once; I repaired the file
  and the script anchors on the last line-start occurrence now. `notebook_meta.py --check` is
  green.
- **`CS479-TASKS.md`** (the course-side task list, renamed from Open Items today): the model-size decision closed; the Colab run on a real corpus recorded as
  Coulson's (#152); tutorial 5's placement and the Learning Suite A8 text recorded as Eric's.

## 3. Sections 1 and 2: taken, and they change the Lecture 9 slides

Same files, same split, three settings; score what `translate_batch` returns. That is exactly the
shape A9's slide needs, and it is simpler than what I had. The Lecture 9 deck is the next rebuild
(it still teaches OpenNMT on two slides) and will quote your notebook's printed settings rather
than restate them; `max_decode_length` gets its own line, since a student who leaves it at 100
gets a BLEU drop for a reason unrelated to subwords.

## 4. Sections 3 and 7: no deck shows a notebook's first cells

Checked: none of the F2026 decks carries a screenshot of any notebook cell, so the `setup(...)`
change has nothing to invalidate on this side. The Lecture 7 deck's Part A already says "A5:
connect Google Drive"; the resume step in Part B is optional and the deck does not enumerate Part
B's steps, so nothing changes there either. No weekdays in notebook text: noted for anything I
write.

Tutorial 5 pointing at A11 is fine by the roadmap as it stands (read at Lecture 10). If Eric
moves it to 8b, the 8b deck's reading slide changes; I will do that when he says.

## 5. Two things only Eric can do, both on the course-side task list

- The Learning Suite text for A8 (section 1's caveat), and the A7 assignment, which still needs
  creating there.
- Tutorial 5's placement.

## 6. Roadmap v7, and four purpose cells to regenerate

Eric asked for a coherence pass; it is `Roadmap/CS479 Fall 2026 Roadmap_v7.md` and your copy is
refreshed from it. No schedule change. Because the Lecture 6 and Lecture 9 scope notes were
rewritten (tutorial 7 named at Lecture 6; the subword notebook and the three-setting A9 at
Lecture 9), `notebook_meta.py --check` now reports four stale purpose cells:
`07-evaluating-translations`, `lecture-06-mt-evaluation`, `lecture-06-mt-evaluation-homework`,
`lecture-09-subword-tokenization`. Please run `--purpose`; I did not touch the notebooks.

No questions for you this pass. Next on this side: the Lecture 9 deck, against your notebook.

## 7. Notebook audit, with implementation directions: all yours

Eric asked for an audit of all 18 notebooks from a student's seat (clear goal and outcome;
brief enough). It is `notes/NOTEBOOK_AUDIT.md` (course-side copy: `Roadmap/CS479 Notebook
Audit_v3.md`). The second half, "Implementation directions", is written so you can carry out
every change without asking: six rules that apply to all notebooks (a standard opener, one
closing section, prose ceilings, marked setup cells, project history told once in tutorial 6,
the text rules already in force), then per-notebook directions with replacement text where
the wording matters, and an order of work.

Eric's governing principle, in his words: **the job of each notebook is to teach.** Every cell
either teaches the student something they need for this course or it moves out to the
reference, changelog or concept pages. Apply that test wherever the directions leave room.

Three decisions from Eric today, which the directions already reflect:

- Every notebook must run end to end from its Colab badge with Run all: nothing to uncomment,
  no dependency on another notebook's output. Tutorial 3 stops needing tutorial 2's checkpoint
  (#166), and lecture-12-llm-context becomes a runnable notebook with a built-in smoke-test
  sample, not a TODO scaffold.
- You do all of it, including lecture-07-toy-model and lecture-08a, which were mine.
- No project history in notebooks, not even as a one-sentence aside (rule 5 in the audit). Bugs
  the library shipped and experiments that went wrong belong in CHANGELOG.md and
  concepts/when-it-fails.md; a notebook states the habit and does not say where it came from.
  Tutorials 5, 6 and 7 each lose a passage to this; the directions say which.

Order: lecture-08a this week (students are running it); tutorial 6 before Lecture 9; tutorial 3,
lecture-10-comet-install and tutorial 5 before Lecture 10; lecture-12 before Lecture 12, once
Eric supplies one A12 language's files or their format (the one item only he can provide; it is
on the course-side task list). The rest when convenient. Five notebooks need no change.

No new dates, no roadmap change.

## 8. Remove the INSTRUCTOR check from `notebook_meta.py`

Eric, today: he never banned the word, and he wants the check gone. `scripts/notebook_meta.py`
(around line 529) flags any notebook whose text contains "INSTRUCTOR", case-insensitive, as a
guard against solution notebooks landing in the tree. Please remove it, and its test in
`tests/test_notebook_meta.py`. The guard it was standing in for is CLAUDE.md's rule that the
Lecture 3 and 4 solution notebooks live in `torchlingo-private`; if you want a mechanical check
for that, match on the filename, not on the word. The notebooks may say "ask the instructor".

I had read the check as a tree-wide rule and reworded the `notes/assignments/` transcripts to
satisfy it. That is undone: the transcripts are verbatim again and their headers no longer
mention a lint rule.
