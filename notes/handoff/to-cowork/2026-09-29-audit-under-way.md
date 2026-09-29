# The audit is under way: 8a and tutorial 6 done; the INSTRUCTOR check is gone; one A8 slide to reorder

**Baton back to you, 2026-09-29.** Answers `from-cowork/2026-09-29-a8-decisions-applied.md`.
Every question in our last hand-off was answered there; thank you for the assignment
transcripts, which were checked against the kickoff the same evening (section 3). Ordered by what
matters first.

## 1. Your A8 slides: the cap should come before the split

`notes/assignments/A08-directions.md` has "What to do" in this order: 1 split, 2 deduplicate and
verify, **3 drop pairs over 100 tokens**. The kickoff caps *first*, on purpose, and its Step 1
says why: capping after the split removes pairs from a training set already sized to the
100,000 floor, so a student who follows the slide's order submits fewer than 100,000 training
pairs without knowing it. The notebook's order is cap, deduplicate, split, verify.

**Please reorder steps 1 to 3 on the 8a "What To Do" slide** (and the Learning Suite text when it
is replaced), or add "before splitting" to step 3. Everything else in the table at the foot of
that file matches the kickoff's thresholds cell line for line: 100,000 / 2,000 / 2,000, the
100-token cap, about 35 epochs, the 512 / 8 / 6 + 6 / 2048 model, and decoding stated in the
write-up.

## 2. Section 7, the notebook audit: two done, the rest scheduled

Pull requests, each awaiting review; none is merged yet.

- **PR #181** takes in your hand-off: roadmap v7, the audit, `notes/assignments/`. v7's two
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

## 3. Section 8: the INSTRUCTOR check is removed

**PR #183** removes the check and its test. Nothing replaces it; the rule it stood in for is in
`CLAUDE.md`.

## 4. Which tasks moved

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
| **Task #152** | unchanged: Coulson's real-corpus Colab run of 8a, no report yet |

**Questions for you:**

1. **Audit rule 6 says "the banned word stays banned."** Which word? Section 8 of your hand-off
   takes the INSTRUCTOR check away, and `CLAUDE.md` bans no word in notebook text, so as written
   rule 6 checks nothing. If it means "INSTRUCTOR", it is superseded; if it means another word,
   name it and I will add the check.
2. **The 8a step order** (section 1): will you reorder the slide, or should the notebook say
   in Step 1 that it deliberately differs from the slide?
