# The audit is under way: 8a and tutorial 6 done; the INSTRUCTOR check is gone; one A8 slide to reorder

**Baton back to you, 2026-09-29.** Answers `from-cowork/2026-09-29-a8-decisions-applied.md`.
Every question in our last hand-off was answered there; thank you for the assignment
transcripts, which were checked against the kickoff the same evening (section 1). Ordered by what
matters first.

## 1. Your A8 slides: the cap should come before the split

`A08-directions.md` has "What to do" in this order: 1 split, 2 deduplicate and
verify, **3 drop pairs over 100 tokens**. The kickoff caps *first*, on purpose, and its Step 1
says why: capping after the split removes pairs from a training set already sized to the
100,000 floor, so a student who follows the slide's order submits fewer than 100,000 training
pairs without knowing it. The notebook's order is cap, deduplicate, split, verify.

**Please reorder steps 1 to 3 on the 8a "What To Do" slide** (and the Learning Suite text when it
is replaced), or add "before splitting" to step 3. Everything else in the table at the foot of
that file matches the kickoff's thresholds cell line for line: 100,000 / 2,000 / 2,000, the
100-token cap, about 35 epochs, the 512 / 8 / 6 + 6 / 2048 model, and decoding stated in the
write-up.

## 1b. A correction you need before the Lecture 9 deck: A9's call did not work

My Sep 28 hand-off told you the A9 path was "8a's Step 6 with two added arguments",
`use_sentencepiece=True, sp_model_path=...`. **That was read, not run, and it was wrong.** Run end
to end on 2026-09-29, the call loaded the *target* vocabulary from a default path that no Colab
runtime has, and failed with file-not-found: every student would have stopped at A9's first
training cell. Two fixes, in review: PR #189 fixes the library (needs a release), and PR #190
makes the Lecture 9 notebook print a third argument that works on the released version:

```
use_sentencepiece=True,
sp_model_path=str(OUT_DIR / 'spm.model'),
sp_tgt_model_path=str(OUT_DIR / 'spm.model'),
```

The same file twice, because one subword model serves both languages. **A9 is still three settings
in substance** (subwords on, the model, the decode length), but a slide that quotes the printout
should show both path lines. With both fixes the whole A9 path runs: same split, both models
trained, both scored on decoded text.

**Lecture 8b also has a question for you and Eric** (Task #183): its only notebook material is
tutorial 4's Part 7, and the audit's directions, applied in PR #187, mark Parts 6 to 8 optional.
Either tutorial 4 names Part 7 as 8b's reading, or 8b gets an activity of its own, or it stays as
it is. Your call with Eric.

## 2. The assignment directions moved to `torchlingo-private`

Eric, 2026-09-29: they belong in the private repository, not the public one. They are now
`torchlingo-private/notes/assignments/`, committed there byte for byte as you wrote them, and
they never reach this repository's `main`. Three references on your side point at the old path:

- the roadmap's **Dates** paragraph and its **Learning Suite's assignment texts** open question
  ("transcribed in the repository at `notes/assignments/`"), and the Sep 29 decisions-log entry;
- `NOTEBOOK_AUDIT.md`'s lecture-12 direction ("Verify against `notes/assignments/A12-directions.md`").

Please write the new location into your next refresh of the roadmap and the audit. Write new
transcripts to `torchlingo-private/notes/assignments/` from now on.

## 3. A new rule from Eric: notebooks carry no due dates at all

Eric, 2026-09-29: each notebook is self-contained and carries no due dates; Learning Suite is
where they live. This tightens the 2026-09-28 rule, which allowed "due before Lecture 8a". Now
none of: "due before Lecture N", a timing column keyed to a lecture, or the purpose cell's
"(due at Lecture N)", which `notebook_meta.py` stops writing in **PR #186**. Naming a lecture
for what it is ("Lecture 8a explains why") and "turn in on Learning Suite" are fine. The Lecture
7 notebook's five instances are gone in **PR #185**, which also applies your audit directions
for it (1,343 to 925 words; Eric made it a priority). Please write new notebook text this way,
and read the audit's standard opener with it in mind.

## 4. Section 7, the notebook audit: three done, the rest scheduled

**PRs #182 to #186 are merged to `main`** (2026-09-29, evening); PR #181, this hand-off, stays
open while the baton is on this side.

- **PR #181** takes in your hand-off: roadmap v7 and the audit. v7's two
  renamed lectures made **six** purpose cells stale, not four: `01-data-and-vocab` and
  `lecture-06-mt-evaluation-homework` also name Lecture 9 or 6. All six regenerated, with the
  roadmap's generated map.
- **PR #182, `lecture-08a-a8-kickoff`**: your directions applied as written, plus trims to the
  opener and Steps 1, 2, 5 and 7. Prose 1,355 to 1,067 words; the rest is in the two cells you
  said to keep. **One correction you may want on the slides too:** the opener said the T4 "ran
  out of memory". No T4 was ever run. `reports/colab-memory.md` compares the A100's measured
  peaks with a T4's nominal 15 GiB, and one configuration (50K vocabulary, length 180) exceeds
  it. The notebook now says a T4 has too little memory for this model with a large word
  vocabulary. "Not the free T4" on the slide is still right; "it ran out of memory" would not be.
  It also no longer quotes 9.6 against 35.8 GiB in Step 1, since the deck carries it.
- **PR #184, tutorial 6**: every direction applied; the frozen-encoder note moved to
  `concepts/when-it-fails.md`. **A defect the audit did not list:** it had a Colab badge and no
  install cell, so Run all from the badge stopped at `import torchlingo`. It now opens with
  tutorial 2's two cells. Prose 2,330 to 2,081 words: about 1,930 without the final table,
  against your 1,600. What is left is each question's Recognize and Fix, and cutting it would
  cut the lesson. Executed end to end in 93 seconds on a CPU; every quoted number matches.

Next, in your order: tutorial 3 (with Tasks #166 and #162, which reshape the same notebook),
`lecture-10-comet-install`, tutorial 5, all before Lecture 10. `lecture-12-llm-context` waits on
Eric's A12 files. The rest when convenient.

**Rule 4 and `CLAUDE.md` pull against each other on tutorial 6's `Vocab`.** `CLAUDE.md` lists
"tutorial 6's `Vocab`" among code that *is* the lesson and stays written out; your rule 4 says to
mark it "no need to read". I did the second, which also keeps the code written out. If Eric
meant the first, the marker comes off in one line.

## 5. Section 8: the INSTRUCTOR check is removed

**PR #183** removes the check and its test. Nothing replaces it; the rule it stood in for is in
`CLAUDE.md`.

## 6. Which tasks moved

`notes/TASKS.md` is reconciled in PR #181.

| | |
|---|---|
| **Task #118** | closed: the A8 slides now say "about 35 epochs; keep going if validation loss is still falling" |
| **Tasks #172 to #177** | new: the audit, one task per group that shares a deadline |
| **Task #172** | 8a kickoff, done in PR #182 |
| **Task #173** | tutorial 6, done in PR #184 |
| **Task #166** | now also carries the audit's tutorial 3 directions |
| **Task #176** | `lecture-12-llm-context` rebuild, **blocked on Eric** for one A12 language's files |
| **Task #178** | new and done: the INSTRUCTOR check, PR #183 |
| **Task #179** | new: tutorial 6 is the only tutorial with no committed outputs, so the docs site shows its code with no results |
| **Task #177** | its Lecture 7 part done in PR #185, at Eric's priority; tutorials 1, 2, 4, 7, regex and Lecture 6 remain |
| **no number** | notebooks carry no due dates: PR #186 (purpose cells, `CLAUDE.md`). Its commit on `main` ends "(Task #22)" by mistake: that was a working-list number, and `TASKS.md`'s Task #22 is the unrelated lint-gate item |
| **Task #170** | corrected: mixed precision is **not** "8a after the next release", as the Sep 28 hand-off said. On A100, L4 and G4 it runs in bfloat16 with no loss scale to lose, so no release is needed; it is now an A9 question, pending a Colab timing. Until then the decks' two-to-five-hour estimate stands |
| **Task #152** | unchanged: Coulson's real-corpus Colab run of 8a, no report yet |

**Audit rule 6's "the banned word stays banned"** meant "instructor", and Eric confirms it is
not banned: PR #183 settles it, and rule 6's clause is void.

**Question for you:** **the 8a step order** (section 1). Will you reorder the slide, or should
the notebook say in Step 1 that it deliberately differs from the slide?
