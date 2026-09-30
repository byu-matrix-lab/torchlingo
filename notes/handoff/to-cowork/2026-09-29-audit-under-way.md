# The audit is mostly done; A9's call was broken and is fixed; 8b has a notebook; one 8a slide to reorder today

**Baton back to you, 2026-09-30, early morning.** Answers `from-cowork/2026-09-29-a8-decisions-applied.md`.
Every question in our last hand-off was answered there; thank you for the assignment
transcripts, which were checked against the kickoff the same evening (section 1). Ordered by what
matters first: section 1 is for **today's Lecture 8a**, section 2 for the Lecture 9 deck you are
rebuilding next.

## 1. Before Lecture 8a today: the cap should come before the split

`A08-directions.md` has "What to do" in this order: 1 split, 2 deduplicate and verify, **3 drop
pairs over 100 tokens**. The kickoff caps *first*, on purpose, and its Step 1 says why: capping
after the split removes pairs from a training set already sized to the 100,000 floor, so a
student who follows the slide's order submits fewer than 100,000 training pairs without knowing
it. The notebook's order is cap, deduplicate, split, verify.

**Please reorder steps 1 to 3 on the 8a "What To Do" slide** (and the Learning Suite text when it
is replaced), or add "before splitting" to step 3. Everything else in the table at the foot of
that file matches the kickoff's thresholds cell line for line: 100,000 / 2,000 / 2,000, the
100-token cap, about 35 epochs, the 512 / 8 / 6 + 6 / 2048 model, and decoding stated in the
write-up.

**One correction you may want on the 8a slides too:** the kickoff's opener used to say the T4 "ran
out of memory". No T4 was ever run. `reports/colab-memory.md` compares the A100's measured peaks
with a T4's nominal 15 GiB, and one configuration (50K vocabulary, length 180) exceeds it. The
notebook now says a T4 has too little memory for this model with a large word vocabulary. "Not the
free T4" on the slide is still right; "it ran out of memory" would not be.

## 2. Before the Lecture 9 deck: A9's call did not work, and is fixed

My Sep 28 hand-off told you the A9 path was "8a's Step 6 with two added arguments",
`use_sentencepiece=True, sp_model_path=...`. **That was read, not run, and it was wrong.** Run end
to end on 2026-09-29, the call loaded the *target* vocabulary from a default path that no Colab
runtime has, and failed with file-not-found: every student would have stopped at A9's first
training cell. Now fixed three ways, all merged:

- **PR #190**: the Lecture 9 notebook prints a third argument, which works on every release:

  ```
  use_sentencepiece=True,
  sp_model_path=str(OUT_DIR / 'spm.model'),
  sp_tgt_model_path=str(OUT_DIR / 'spm.model'),
  ```

  The same file twice, because one subword model serves both languages.
- **PR #189**: the library fix, so `sp_model_path` alone works too. **Released as torchlingo 0.2.4**
  (2026-09-30), which also carries resume restoring the random generators and the AMP loss scale.
- **PR #195**: a test that runs the A9 path end to end in CI, and fails on the code before the fix.

**A9 is still three settings in substance** (subwords on, the model, the decode length), but a slide
that quotes the printout should show both path lines, which work on any release; with 0.2.4
either form works. Both runs are scored on decoded text, never pieces.

## 3. Lecture 8b now has a notebook of its own

Eric's decision (PR #193, Task #183). Tutorial 4's Part 8, a pretrained Transformer's attention on
real text, is now `tutorials/08-transformer-attention.ipynb`: reading for 8b and a head start on
A8. It was the only part needing the 11 MB model download. Tutorial 4 keeps Parts 1 to 7 **in the
same file**, so every link and badge still works, and it now says "Part 6 is optional ... Part 7
is not optional: Lecture 8b starts from it." (The audit's "Parts 6 to 8 are optional" had made
8b's reading optional.) **Please name tutorial 8 on the 8b deck's reading slide.**

## 4. Your half of the roadmap: four things are now stale

Please correct these at your next refresh; the generated map in the repository's half is already
current.

1. **`notes/assignments/`** is named in the Dates paragraph, the "Learning Suite's assignment
   texts" open question and the Sep 29 decisions-log entry. The directions now live in
   `torchlingo-private/notes/assignments/` (section 6). The same path is in `NOTEBOOK_AUDIT.md`'s
   lecture-12 direction.
2. **8a and 8b's notebooks**: 8a's line says tutorial 4 "(reading, Parts 1 to 6)", 8b's says "Part
   7", and the index table has no tutorial 8. Tutorial 4 is now Parts 1 to 7, read between 8a and
   8b, and tutorial 8 is 8b's.
3. **A9's settings** (Lecture 9's section and the Sep 28 decisions-log entry): the printout now
   passes `sp_tgt_model_path` as well; section 2.
4. **The tools table's "0.2.3 current"** is now 0.2.4, which also fixed A9's call.

## 5. The notebook audit: done except Lecture 10's three and Lecture 12

Everything below is merged to `main`. Where I departed from your directions, it says so.

| Notebook | PR | Prose words | Beyond the directions |
|---|---|---|---|
| `lecture-08a-a8-kickoff` | #182 | 1,355 → 1,067 | the T4 correction (section 1) |
| `lecture-07-toy-model` | #185 | 1,343 → 925 | Eric's priority; also no due dates (section 7) |
| tutorial 6 | #184, #188 | 2,330 → 2,081 | **had no install cell**, so Run all from its badge failed; outputs now committed |
| tutorial 4 | #187, #193 | 1,709 → 1,031 | Part 8 split out (section 3) |
| `lecture-04-regex-refresher` | #192 | 737 → 553 | |
| `lecture-06-mt-evaluation` | #192 | 818 → 702 | |
| tutorial 1 | #192 | 233 → 250 | |
| tutorial 2 | #192 | 306 → 306 | `data/example.tsv` is 86,430 pairs, not 73,083 |
| tutorial 7 | #192 | 1,839 → 1,590 | **had no install cell either**; found by the first student-path run |

**Four places the audit's wording was wrong or incomplete**, so you can correct your copy:

- **Lecture 6's Part 2 cell** is not a "70-line helper" but a data fixture (twenty references and
  two systems' outputs). `CLAUDE.md` keeps fixtures visible, so it stayed, with a first line
  saying to skim it.
- **Tutorial 1**: A5's output is not "exactly what `load_data` expects". A5 produces two
  sentence-aligned text files, which go through `parallel_txt_to_dataframe` first, as the A8
  kickoff does. The opener says that.
- **Tutorial 7** had a third passage of project history the directions did not list (Part 2: "a
  test here once compared two BLEU scores that were both 0.0"); it is gone under rule 5.
- **The regex refresher is a fill-in worksheet** (`requires: blanks`), which cannot meet rule 6's
  "runs end to end with no edits". I applied your directions and left it a worksheet. If Eric
  wants it runnable, it needs its answers filled in and the exercises turned into "change this
  and rerun", which is a rewrite rather than an edit.

**The word ceilings**: I treated them as targets, not limits, and stopped short of them where what
remained was teaching (tutorials 6 and 7). Eric's principle, that each notebook's job is to
teach, came first.

**Left, in your order:** tutorial 3 (with Tasks #166 and #162, which reshape the same notebook),
`lecture-10-comet-install`, tutorial 5, all before Lecture 10. `lecture-12-llm-context` waits on
Eric's A12 files.

## 6. Two rules from Eric, and the assignment directions

- **Notebooks carry no due dates at all** (Eric, 2026-09-29). Not "due before Lecture N", not a
  timing column keyed to a lecture, not the purpose cell's "(due at Lecture N)", which
  `notebook_meta.py` no longer writes (PR #186). Naming a lecture for what it is ("Lecture 8a
  explains why") and "turn in on Learning Suite" are fine. Please write new notebook text this
  way, and read the audit's standard opener with it in mind.
- **"Instructor" is not a banned word.** Audit rule 6's "the banned word stays banned" meant it,
  and Eric confirms it is not banned: PR #183 removed the check, and that clause of rule 6 is void.
- **Tutorial 6's `Vocab` is setup**, marked "no need to read", as your rule 4 said; `CLAUDE.md` no
  longer lists it as lesson code (PR #191).
- **The assignment directions moved to `torchlingo-private`** (Eric, 2026-09-29): they belong in the
  private repository. They are `torchlingo-private/notes/assignments/`, byte for byte as you wrote
  them, and never reach this repository's `main`. Write new transcripts there. That repository now
  has a private GitHub remote (`byu-matrix-lab/torchlingo-private`), so it is backed up.

## 7. Your notebooks, run the way a student runs them

The student-path workflow (every notebook from a fresh PyPI install, Colab faked) ran for the first
time on 2026-09-29. After a fix to the harness itself (PR #194: its `!pip` installed outside its
own environment, a false failure for four notebooks), **your Lectures 3, 4 (TMX cleaning), 5, 6,
7 and 12 all pass**, as do tutorials 1, 2, 4, 5 and 6. Tutorial 7 failed for real (the missing
install cell, now fixed); tutorial 3 is the known failure (Task #166). Notebooks needing Drive, a
GPU or a token are skipped. It now runs weekly and on every release tag.

## 8. Which tasks moved

`notes/TASKS.md` is reconciled in PR #181, the pull request carrying this hand-off.

| | |
|---|---|
| **Task #118** | closed: the A8 slides say "about 35 epochs; keep going if validation loss is still falling" |
| **Tasks #172, #173, #179** | done: the 8a kickoff, tutorial 6, tutorial 6's outputs |
| **Task #177** | the notebooks already read: done in PRs #185, #187 and #192 |
| **Tasks #174, #175, #166** | Lecture 10's three, open, in that order with tutorial 3 first |
| **Task #176** | `lecture-12-llm-context` rebuild, **blocked on Eric** for one A12 language's files |
| **Task #178** | done: the INSTRUCTOR check, PR #183 |
| **Task #183** | done on this side (PR #193); left: tutorial 8 on your 8b deck |
| **Task #97** | the A9 path: fixed, released in 0.2.4, and tested (PR #195) |
| **Task #150** | done: `torchlingo-private` has a remote |
| **Task #170** | corrected: mixed precision is **not** "8a after the next release", as the Sep 28 hand-off said. On A100, L4 and G4 it runs in bfloat16 with no loss scale to lose; it is now an A9 question, pending a Colab timing. Until then the decks' two-to-five-hour estimate stands |
| **Task #180** | new: a per-PR check that a notebook with a Colab badge installs TorchLingo, after tutorials 6 and 7 both shipped without |
| **Task #152** | Coulson's real-corpus Colab run of 8a: no report yet; Eric follows up this morning |
| **no number** | notebooks carry no due dates: PR #186. Its commit on `main` ends "(Task #22)" by mistake: that was a working-list number, and `TASKS.md`'s Task #22 is the unrelated lint-gate item |

**Question for you:** **the 8a step order** (section 1). Will you reorder the slide before class
today, or should the notebook say in Step 1 that it deliberately differs from the slide?
