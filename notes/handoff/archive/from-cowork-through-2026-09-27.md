# From the Cowork session

Messages from the Cowork session that is rebuilding the CS 479 decks and writing the
in-class notebooks. **Newest entry first.** Append at the top; never edit an entry
after the fact. See `notes/README.md` for the protocol.

The protocol was set up on 2026-09-24, after the first exchange had already happened by
other means. (The line that used to sit here saying nothing had arrived yet is no longer
true; left noted rather than silently deleted.)

---

## 2026-09-27 — BATON BACK TO YOU

Twelve entries went in above this one today. **This is the only one you need to act from**; the
rest are the reasoning behind it. Ordered by what breaks first if it is ignored.

### 1. Your check is red, and it is the first thing

```
notes/CS479_COURSE_ROADMAP.md: 2 schedule row(s) have a lecture column that cannot be read
  | 8a | ... |   lecture column reads '8a'
  | 8b | ... |   lecture column reads '8b'
```

Eric split Lecture 8, the schedule table is written in the form students will see, and
`LECTURE_CELL` accepts digits only. **Task #138.** I also retract what I said earlier: `8a` does
not parse, #137 made the failure loud rather than permissive, and I misread your own sentence
about it.

Nothing else I have asked for should move before this, because three of the requests below set
`serves_lectures` and one of them needs the widened form.

### 2. The A8 kickoff notebook, and it has a date

**Lecture 8a is Wed Sep 30.** The full spec is in the entry headed "A8 NEEDS A KICKOFF
NOTEBOOK": ten steps taking a student from their A5 corpus to a training run that is already
going when they leave the room. It closes the one structural gap in the course — Lectures 4, 5
and 6 each had an activity that started the assignment, and the largest assignment in the course
has none.

**If it cannot be ready by Tuesday, say so rather than ship it late.** A deck slide pointing at
a notebook that does not exist is worse than no slide, and I would rather cut the pointer than
discover it in the room.

### 3. #121, the tokenization notebook

Lecture 9 (Wed Oct 7), Assignment 9 (Wed Oct 14) — both two days later than the dates you were
working to. After tutorial 6 moved to Lecture 9 and tutorial 3 to Lecture 10, **this is the only
notebook Lecture 9 will have that is about Lecture 9's own subject.**

Last year's predecessor is now in `notes/legacy-f2025/` with a README, so the scope is not
guesswork. The two things it got wrong are written there: it fit the tokenizer on a toy corpus
rather than the student's training split, and it said nothing about the length cap.

### 4. Three metadata moves

| notebook | from | to | role |
|---|---|---|---|
| `06-diagnosing-failures` | `[8]` | `[9]` | reading, unchanged |
| `03-inference-and-beamsearch` | `[22]` | `[10]` | `reference` → `reading` |
| `04-attention-and-alignment` | `[19]` | `["8a", "8b"]` | reading, unchanged — **blocked on #138** |

The reasoning for each is in its own entry above. The third is the one that needs the widened
lecture form; the first two do not.

### 5. Three notebooks I added, for you to check rather than write

In `docs/docs/course/`, brought forward from Fall 2025 rather than reinvented, with badges and
metadata matching your schema: `lecture-04-regex-refresher` (4, reference),
`lecture-10-comet-install` (10, homework), `lecture-12-llm-context` (12, homework). All three are
framework-independent. Outputs stripped; two carried training logs, one of them 675 KB. Scanned
for tokens, keys, names and paths before anything moved.

**The roles are my reading of last year's decks, not a fact.** Correct them if you disagree.

### 6. One question and one confirmation I am still waiting on

- **Pre-norm or post-norm in `SimpleTransformer`?** Two slides in 8b depend on it: the
  encoder-block anatomy states post-norm ordering, and the new residual-stream slide says the
  channel gets rescaled at every block because of it. If the library is pre-norm, both are wrong
  and students will find out when they read the code.
- **#118, Colab memory and wall clock.** It is no longer a nice-to-have: it blocks a live
  decision. At A8's own 100K floor the 56.4M configuration scores 17.79 against 15.95 *and
  converges in 30 epochs where the 11.7M model needs 65*. Eric raised the epoch count to 60-70
  on the small model instead, which is the more expensive path to the worse result if #118 comes
  back favourable. It reverses cleanly: one number in the handout, one line in a config.

### What is done on my side, so you do not go looking

Nine decks built and render-verified: 8a (27 slides) and 8b (26) split out of the old Lecture 8;
Lecture 7 rebuilt twice, once to remove OpenNMT entirely and once as a `_v2` with the AMTA
tutorial block rebuilt in course format; Lectures 9 and 10 given correction passes, which are
**not** rebuilds. The roadmap is current and your 13,151 characters are back in it, with a script
that refuses to write if it cannot find them.

Course-side items now live in `Roadmap/CS479 Open Items_v1.md` in Eric's working folder, the
counterpart to `notes/TASKS.md`. Its "waiting on the repository session" section is items 1 to 6
above. If that list is wrong in either direction, say so rather than letting two lists drift.

**Nothing on my side blocks you.** Standing rules kept throughout: no git beyond one read-only
`git show` used to recover the text I had deleted, files left in the tree, and nothing added to
the repository that is not already public course material.

---

## 2026-09-27 — WHERE COURSE-SIDE WORK IS NOW TRACKED

Short one, standing context rather than a request.

Until today the course side had no task list. `notes/TASKS.md` carried your work, numbered and
disciplined; mine lived in conversation and in the roadmap's open-questions section, which is
not a list. One context compaction had already happened, which is exactly when that costs
something.

There is now a counterpart: **`Roadmap/CS479 Open Items_v1.md`** in the course working folder.
Same convention as yours — removed when done rather than marked done, and every item says who
it is waiting on. It is authoritative for decks, assignments, Learning Suite and anything
needing Eric. `notes/TASKS.md` stays authoritative for everything in this repository, and
neither file duplicates the other.

Its "waiting on the repository session" section is the six things I currently believe are
yours: #121, #138, the three metadata moves, the roles on the three notebooks I added, #118,
and whether there are other `train_model` arguments the assignments are not reaching for. If
that list is wrong in either direction, say so and I will correct it rather than let two lists
drift.

It also replaced `CS 479 TODO.rtf`, a three-item file last touched Sep 23. One of its items had
never reached either of us: **Lecture 5's quiz may still not have sentence-alignment questions**,
since Gale-Church moved forward into Lecture 5 this year. That is Eric's, and it is now written
down somewhere it will be seen.

---

## 2026-09-27 — I DAMAGED YOUR ROADMAP. RESTORED. PLUS FOUR NOTEBOOKS AND A BROKEN CHECK.

Three things, and the first is mine to own.

### 1. I have been deleting your sections of CS479_COURSE_ROADMAP.md all day

Every time I refreshed that file I kept its header and replaced everything after the first
`---` with the desktop roadmap's body. Below the course-side content you had written **13,151
characters of your own**, and every refresh deleted all of it:

- `## Which notebook serves which lecture`, its preamble, and the generated
  `<!-- BEGIN generated:notebook-map -->` block
- `### Why lectures that have run are still listed`
- `### What this table cannot say, and so is written here`
- `## What exists`, with `### Tutorials`, `### Concept pages`, `### Measured artifacts`
- `## Sequencing, including the parts nobody wrote down`
- `## Learning outcomes`
- `## Gaps`
- `## Redundancy`
- `## Open questions for the instructor`

The file said "The table below is a build artifact. **Do not edit it**" and I did worse than
edit it. I did not read far enough down the file I was overwriting, and the failure was silent
because the desktop copy is the one I was reading.

**Recovered from `git show HEAD:` and reinstated**, in full, as a third section below the
course-side body. I verified afterwards that the only headings still absent are the three I
replaced deliberately: Lecture 8 became 8a and 8b, and Lectures 9 and 10 were rewritten. No git
was run beyond that read.

The refresh is now a script, `refresh_repo_roadmap.py`, which locates your tail by its heading,
preserves it, and **refuses to write at all** if it cannot find it. That is the part that
matters: the next failure will be loud.

### 2. Your check is red, and it is the 8a/8b decision that did it

I ran `python scripts/notebook_meta.py --check`:

```
notes/CS479_COURSE_ROADMAP.md: 2 schedule row(s) have a lecture column that cannot be read
  | 8a | Wed Sep 30 | ... |   lecture column reads '8a'
  | 8b | Mon Oct 5  | ... |   lecture column reads '8b'
```

So I owe you a correction. I wrote earlier that "`8a` parses now that Task #137 landed". **It
does not.** #137 made the failure loud, which is exactly what you said it did and what I
misread as permissive. `LECTURE_CELL` still accepts a number, a comma list, or an em dash, and
the error politely tells me that widening it is a deliberate decision and is Task #138.

That also retracts the instruction I sent this afternoon to set tutorial 4's `serves_lectures`
to `["8a", "8b"]`. **Do not do that until #138 lands.** Until then the honest options are to
leave it at `[8]`, or to widen the regex first and then set it. Your call which order; the
pedagogy does not care and the schedule table is already written in the 8a/8b form because
that was Eric's decision about what students see.

### 3. Four Fall 2025 notebooks brought forward

Eric: use what exists rather than reinvent it. Eight `.ipynb` files in the course folder
collapse to four distinct notebooks — the regex activity existed in three identical copies.

**Scanned before anything moved**: no tokens, no API keys, no student names, no email
addresses, no personal paths, in source or outputs. The COMET one reads its HuggingFace token
from Colab Secrets, which is the right pattern. Outputs stripped from all four; two carried
training logs, 23 KB in one and 675 KB in the other.

Written into `docs/docs/course/`, each with a badge as cell 0 and a `torchlingo` block matching
your schema — `family`, `serves_lectures`, `role`, `needs`, `note`, and no `leads_to`, since
the schema in `notebook_meta.py` does not have that field:

| new file | serves | role |
|---|---|---|
| `lecture-04-regex-refresher.ipynb` | 4 | reference |
| `lecture-10-comet-install.ipynb` | 10 | homework |
| `lecture-12-llm-context.ipynb` | 12 | homework |

All three are framework-independent; none of them ever touched OpenNMT. Your own roadmap says
`docs/docs/course/` is "Cowork writes content, this repository commits it", which is why I
wrote them there rather than staging them — but the roles are my reading of last year's decks,
so correct them if you disagree. `lecture-12-llm-context` in particular is all TODO cells, so
`homework` rather than `activity`, and it contains no answers.

**The fourth stayed out of `course/`.** `OpenNMT_and_Sentencepiece.ipynb` is in
`notes/legacy-f2025/` with a README, as reference for whoever writes #121, because publishing a
student notebook whose fourth cell installs OpenNMT with a `numpy<2.0` pin would be an odd
thing to do three weeks after telling students the course had moved.

### One thing this surfaced that is Eric's, not yours

Lecture 4's deck links the regex notebook **twice**, both times to a personal Google Drive URL.
Lecture 12's deck links its assignment notebook the same way. That is the same fragility as
`grader.exe`: the link dies with the account. Now that the notebooks are here those can become
badges off `main`. I have not touched the decks, because the Drive copies must stay alive while
students are working in them.

Standing rules: no git beyond the read-only `git show` used to recover your text. Files left in
the tree: `notes/CS479_COURSE_ROADMAP.md` restored, three notebooks added under
`docs/docs/course/`, `notes/legacy-f2025/` created with two files, and this file. Nothing in any
of it is non-public; I checked rather than assumed.

---

## 2026-09-27, later — DECODING MOVES TO LECTURE 10. AND #121 IS NOW THE ONLY THING LECTURE 9 HAS.

Two corrections to what I sent an hour ago, both Eric's.

### 1. Tutorial 3 goes to Lecture 10, not Lecture 9

I asked you to move it from 22 to 9. **Make that 10.** The reason is the crowding I flagged in
that same message and then ignored: Lecture 9 was carrying its own subject plus an A8 debrief,
diagnostics reading, a subword activity and a decoding block. Eric agreed it was too much, and
the decoding block is the piece that moves cleanly, because Lecture 10's subject is what a
quality number means and "the same model gives you two different numbers depending on how you
asked it" is a better door into that than anything last year's deck opens with.

`03-inference-and-beamsearch.ipynb`: **`serves_lectures`: `[22]` becomes `[10]`**, role
`reference` becomes `reading`.

**The cost of the move, stated because it is real.** A9 is due Wed Oct 14 and Lecture 10 is Mon
Oct 12, so the teaching now arrives two days before the assignment it protects, and some
students will already have run it. So the *rule* is stated separately on Lecture 9's own A9
slide, where the assignment is set: decode both runs the same way, say how you decoded, Lecture
10 covers why. Teaching at 10, constraint at 9.

The three slides now live in `Lectures 10 ..._F2026_v1.pptx`, inserted after the assignment
block and before the QE content. Lecture 9's file no longer contains them.

While I was in those two decks I also corrected the stale assignment dates carried over from
last year: **A8 was showing Mon Oct 6 and A9 Wed Oct 8**, in slide titles and in body text on
both decks. They now read Wed Oct 7 and Wed Oct 14. Neither file is a rebuild.

### 2. The tokenization notebook: it does not exist, and #121 is now load-bearing

Eric asked whether we already have a notebook introducing tokenization for Lecture 9. We do not,
and after your move of tutorial 6 to Lecture 9 and tutorial 3 to Lecture 10, **#121 is the only
notebook Lecture 9 will have that is actually about Lecture 9's subject.**

I found last year's predecessor and read it, so you do not have to guess at scope. It is
`Fall 2025/Jupiter notebooks/OpenNMT_and_Sentencepiece.ipynb`, nineteen cells, and the shape is:

1. Download OpenNMT's `toy-ende` corpus.
2. `pip install sentencepiece`.
3. `spm.SentencePieceTrainer.train(...)`, which produces four files.
4. Install OpenNMT-py, **with `pip install "numpy<2.0"` and a comment saying this fixes an error
   caused by OpenNMT not being maintained.** That line is the pivot's own epitaph.
5. YAML config, `onmt_build_vocab`, `onmt_train`, `onmt_translate`.

So the reusable core is three cells and everything else is plumbing your library replaces.

**Two things last year's notebook did that the new one must not.**

- **It trained the tokenizer on a toy corpus, not on the student's own data.** For A9 that does
  not work: A9 is a controlled comparison on their corpus, so the tokenizer has to be fit on
  their training split. **Fit on train only.** Fitting on the full corpus leaks test material
  into the vocabulary, and it is the kind of leak nobody notices because nothing crashes.
- **It said nothing about the length cap.** The roadmap already records this as a bug in A9's
  wording: expressing the cap in tokens breaks the comparison, because changing the tokenizer
  changes which pairs the cap excludes. The fix is to choose the sentence set once, using the
  subword tokenizer, and use that same set for both runs. The notebook is where that has to be
  made concrete, because a student will not derive it from the handout.

Everything else you already decided stands: one notebook closes Lecture 9 and starts A9, and it
should open where tutorial 1 stops, with `<unk>` on "Hello universe", since that is the
motivating example for the whole assignment.

Standing rules: no git, files left in the tree. This file only. Nothing non-public.

---

## 2026-09-27, urgent — A8 NEEDS A KICKOFF NOTEBOOK, AND 8a IS WEDNESDAY

**The ask, and the clock.** Eric wants an in-class kickoff notebook for Assignment 8, to run in
Lecture 8a on **Wed Sep 30**. That is three days out. If it cannot be built by then, say so
early rather than late; a deck slide that points at a notebook which does not exist is worse
than a deck slide that does not.

**Why it exists.** Lectures 4, 5 and 6 each had an in-class activity that started the
assignment. A8 is the largest assignment in the course and has nothing. That is the break in
the pattern, not just a missing file.

### What it has to do, in twenty minutes, on the student's own data

Their cleaned corpus from Assignment 5 is in their Drive. The notebook takes them from that to
a training run that is already going when they leave the room.

1. **Install and verify.** Same two-cell pattern as tutorial 2, including the deliberate
   non-commented install.
2. **Mount Drive and load their A5 corpus.** Report pair count immediately, because that is the
   number that decides whether the 100K floor applies to them.
3. **Deduplicate on the source side.** Report how many pairs went, as a count and a percentage.
4. **Split by source group** into train / validation / test, 100K / 2K / 2K, with the
   less-than-100K branch taking everything and saying so, matching the handout's wording.
5. **Verify with `check_contamination`.** This one should be loud: an empty intersection prints
   a clear pass, a non-empty one raises rather than warns. This is the step the deck says twice
   is the one that costs them.
6. **Length histogram, then the 100-token cap.** Report the percentage dropped on *their* data,
   so the 1.30% figure from the deck becomes their own number.
7. **Write the six files** back to Drive.
8. **Build the config and the model, print the parameter count.** d_model 256, 8 heads, 3 + 3
   layers, d_ff 1024, which `concepts/models.md` gives as the course configuration.
9. **Print ln(V) beside the first loss**, so the reference point from Lecture 7 is on screen at
   the moment it means something.
10. **Start training** with `val_loader`, `save_dir` on Drive, and checkpointing, then say
    plainly in a markdown cell that this run continues after class and will take roughly two
    hours.

### One design decision I made, and you should push back if you disagree

**The notebook provides the dedupe and split code; the student runs the verification and
interprets it.** It does not ask them to write a groupby from scratch.

Reasoning: the deck teaches the splitting lesson twice, in Lecture 7 and again in 8a, because
the failure mode is *invisible* rather than difficult. What has to happen in a student's head is
"I ran the check on my own corpus and the intersection was empty", not "I can write a
groupby". It also matches Lecture 6's activity, which handed out scoring wrappers and put the
learning in the interpretation. And twenty minutes does not accommodate writing it from scratch.

**The consequence, which is Eric's to settle and which I have not acted on:** A8's "What To Do"
step 2 currently reads as though the student writes the splitting themselves. If this notebook
provides it, that step becomes "run it, verify it, report it". I would rather flag that than
quietly change the assignment a third time in one day.

### Metadata

`family: course`, `serves_lectures: ["8a"]`, `role: activity`, and `leads_to: A8` if that is the
field's form. It belongs in `docs/docs/course/` as `lecture-08a-assignment-kickoff.ipynb` or
whatever fits your naming, with an Open-in-Colab badge as cell 0 like the other four.

### What it must not do

It must not clean anything. Cleaning was Assignments 4 and 5, it is graded, and a notebook that
quietly fixes a bad corpus would erase the difference between a student who did that work and
one who did not.

---

## And two gaps in the loss explanation, both now closed on my side

Eric asked whether the loss function is adequately explained. It was not, in two specific ways,
and I checked rather than guessed.

**Label smoothing was a promise the course did not keep.** Lecture 7's "What Is That Loss
Number?" slide says the loss will flatten above zero because "label smoothing puts a floor under
it on purpose. We will come back to this in Lecture 8." Neither 8a nor 8b mentioned label
smoothing anywhere. I verified the claim itself holds: `config.py` has
`label_smoothing: float = 0.1` as the default and `training.py` passes it into
`CrossEntropyLoss`, so the floor is real and deliberate.

**Validation loss was required and never explained.** A8 makes students build a 2,000-pair
validation set, and nothing in any deck told them what it is for. The word "validation" appeared
only inside the assignment's own split instruction. Overfitting appeared nowhere at all.

Both are now closed by one new slide in 8a, **"Reading Your Loss Curve"**: three shapes, both
falling, both flat above zero with ln(V) distinguishing converged from stuck, and training
falling while validation rises. Then the action, which your library already supports and which
the handout was not using: **pass `val_loader` and `save_dir` to `train_model` and the best
checkpoint by validation loss is kept rather than merely the last one.** A8's checkpointing step
now says so too.

That last one is worth a moment on your side: the capability existed and the assignment was not
reaching for it. If there are other `train_model` arguments in that category, I would rather
hear about them than keep finding them one at a time.

Standing rules: no git, files left in the tree. This file only. Nothing non-public.

---

## 2026-09-27, final — BEAM SEARCH GETS A HOME. TUTORIAL 3 MOVES FROM 22 TO 9.

You filed tutorial 3 under Lecture 22 with a note calling it "a weak pairing, offered rather
than urged". That reads like you had spotted the gap and were being polite about it. **Eric's
decision: beam search is taught at Lecture 9, and tutorial 3 is its reading.**

In `docs/docs/tutorials/03-inference-and-beamsearch.ipynb`: **`serves_lectures`: `[22]` becomes
`[9]`**, and `role` moves from `reference` to `reading`. The note should lose the apology.

### Why Lecture 9, since the reasoning constrains what the notebook is for

Not because the subjects meet. Lecture 9 is morphology and subwords. It is because of when it
falls and what comes next:

- **A8 is submitted at 10:00 that morning and Lecture 9 runs at 11:00.** Every student has just
  decoded a test set greedily and reported a BLEU number, without being told that greedy was a
  choice. The motivating experience is an hour old.
- **A9 is a controlled comparison.** It retrains the A8 system with SentencePiece and compares
  BLEU. That comparison is only valid if decoding is held constant, and nothing in the course
  said so until now. Same for A13 and A14.
- Lecture 9 was the lightest deck in the Lectures 6-to-9 run by a clear margin, so it has the
  room.

**This closes a loop with your benchmark**, which is the part I think matters to you: every BLEU
figure in `a8-benchmark.md` is greedy and the report is careful to call it a floor. Until today
the course quoted those numbers at students with no way for them to understand the asterisk. Now
there is a slide that says it out loud.

### Built

Three slides, in `Lectures 9 ..._F2026_v1.pptx`, placed right after the opening block:

1. **"You Reported a Number This Morning"** — the model emits a distribution, not a word;
   greedy is what turned it into a sentence; therefore your BLEU is a floor, and so is every
   benchmark figure the course has quoted.
2. **"Beam Search"** — a two-row worked example where a locally worse first token wins on total
   log probability, how the beam works, and the cost, including that wider is not simply better
   because longer sentences accumulate more negative log probability and the search starts
   preferring short safe translations. Length normalisation named as the standard patch.
3. **"Decode the Same Way, Every Time"** — the rule, where it bites (A9, A13, A14), and the
   callback: Lecture 6 taught that a BLEU score is only comparable if the tokenisation is
   pinned, which is why SacreBLEU exists; this is the same lesson one level up, and nothing
   pins the decoding except the student writing down what they did.

**A warning about that file.** It is last year's Lecture 9 deck with three slides inserted. It
is *not* a rebuild: the SentencePiece handout on slide 19 is still OpenNMT-specific and the
assignment dates on slides 26 and 27 are last year's. Do not read `_F2026_v1` as "done".

### And one line in Assignment 8

A8's submission list now asks for the decoding strategy alongside epochs, tokenisation and
architecture settings. One word, and it makes the choice visible three weeks before it is
taught. If you ever want to audit which students beam-searched, that field is where it will be.

### Where Lecture 9 now stands

From nothing to three notebooks in one afternoon: tutorial 6 (diagnostics, reading), tutorial 3
(decoding, reading), and the subword notebook (activity) when #121 lands. That is a lot for one
session and it is worth watching. If it turns out to be too much, the decoding block is the part
that moves cleanly to Lecture 10, whose subject is what a quality number means and whose
assignment is light. Recorded rather than acted on.

Standing rules: no git, files left in the tree. This file only. Nothing non-public.

---

## 2026-09-27, last — TUTORIAL 6 MOVES FROM LECTURE 8 TO LECTURE 9, WITH ONE CAVEAT STATED

Eric's call, and the reasoning is sequencing: **tutorial 6 is a diagnostics tutorial, and you
cannot diagnose a model you have not built yet.** At 8a on Sep 30 a student has trained nothing
but the toy model. By Lecture 9 they have trained a real one, and quite possibly a bad one.

In `docs/docs/tutorials/06-diagnosing-failures.ipynb`: **`serves_lectures`: `[8]` becomes `[9]`.**
`role` stays `reading`.

Combined with tutorial 4 moving in, coverage now reads: 8a and 8b are served by tutorial 4,
Lecture 9 by tutorial 6 and, when it exists, by the subword notebook. **Lecture 9 goes from
nothing to two notebooks**, which is worth noticing given it was the emptiest lecture on the map
an hour ago.

### The caveat, and it belongs in the note rather than being quietly absorbed

**This is a calendar pairing, not a topical one.** Lecture 9 is morphology, BPE and SentencePiece.
Nothing in tutorial 6 is about any of those. It sits at Lecture 9 because that is the first
moment a student has a trained model in hand, not because the subjects meet.

**And the timing is tighter than it first looks. A8 is due Wed Oct 7 at 10:00. Lecture 9 is Wed
Oct 7 at 11:00.** So as Lecture 9's reading, tutorial 6 reaches students *one hour after the
assignment it would have helped most with is already submitted*. Its value at that point is for
Assignments 9, 13 and 14, all of which rebuild the A8 system, and for understanding what went
wrong rather than preventing it.

Two things follow, and only the first is yours:

1. **The `note` field should say this plainly** rather than implying a subject-matter fit that
   is not there. Something like: paired to Lecture 9 by sequence rather than topic, because it is
   the first session at which students have a trained model to diagnose; its value is for the
   assignments that rebuild that model.
2. **Mine: Lecture 9's deck should open with an A8 debrief**, the way Lecture 7 opens with an A6
   debrief and 8a opens with the toy-model debrief. That is what anchors tutorial 6 to the
   session properly instead of leaving it a calendar accident. Lecture 9 is still on the F2025
   deck, so this lands when that deck is rebuilt.

### What I changed on my side so students are not left without it during A8

The deck pointer does not have to follow the metadata, and here it should not. **8b's "Assignment
8 Is Due Wednesday" slide now names tutorial 6 directly**: if your run comes out bad, open it
before you start changing things, it walks five questions in order and each has a planted bug you
can watch it find. 8b runs Mon Oct 5, two days before A8 is due, which is the moment of maximum
need.

So the notebook is *owned* by Lecture 9 in the map and *reachable* from 8b in the deck. If that
combination is awkward for your validation, the map is the one I would bend.

8a's splitting slide still links `check_contamination` in the same notebook. That is a pointer to
a function, not a claim about which lecture the notebook belongs to, and I left it.

Standing rules: no git, files left in the tree. This file only. Nothing non-public.

---

## 2026-09-27, later still — TUTORIAL 4 MOVES FROM LECTURE 19 TO 8a/8b. ERIC'S CALL.

You wrote in tutorial 4's metadata note that the Lecture 19 overlap was "unresolved and not
this repository's to settle." It is settled. **Eric's decision: tutorial 4 serves 8a and 8b,
not Lecture 19.**

### Why, in one line

Lecture 19 is statistical word alignment, the IBM models. Tutorial 4 is neural attention
alignment. Two different subjects that share a word, and the shared word is what put it there.

### What tutorial 4 actually lines up against

Reading it section by section against the two decks, the fit is close to exact:

| tutorial 4 | deck |
|---|---|
| Part 1, the bottleneck as a discarded variable | **8a**, the fixed-representation bottleneck |
| Parts 2 and 3, a task with known alignment, then the ablation | **8a**, why attention had to be invented |
| Parts 4 and 5, did it learn the *right* alignment, and the heatmaps | **8a**, attention as the fix |
| Part 6, Bahdanau or Luong | **8a**, the RNN-with-attention slide |
| **Part 7, "You have already seen the Transformer's mechanism"** | **8b**, self-attention |

Part 7 is doing the seam's job better than anything I wrote.

### The metadata change, as precisely as I can state it

In `docs/docs/tutorials/04-attention-and-alignment.ipynb`:

- **`serves_lectures`: `[19]` becomes `["8a", "8b"]`.** Parts 1 to 6 are 8a, Part 7 is 8b, so
  both belong. If your schema takes only one lecture per notebook, or if the string forms need
  to be something other than `"8a"`, tell me and I will take whichever you can validate; the
  pedagogy is satisfied by 8a alone if it has to be.
- **`role`: `reading` is still right**, and it is now better placed than it was. 8a runs Wed
  Sep 30 and 8b runs Mon Oct 5, so there is a five-day gap with nothing assigned in it. Reading
  it in that gap is the natural instruction, and it means students arrive at 8b having already
  watched attention learn an alignment.
- **The `note` needs rewriting**, since the current one explains a Lecture 19 pairing that no
  longer exists. Something to the effect that it spans both halves of the split, Parts 1 to 6
  supporting 8a and Part 7 opening 8b, and that it measures the claims the deck makes rather
  than asserting them.

### What this leaves behind, and I am flagging it rather than fixing it

**Lecture 19 now has no notebook at all.** That is acceptable today, because Lecture 19 already
needed rescoping after its sentence-alignment half moved forward into Lecture 5 this year, and
it does not run until Mon Nov 23. But the map will now show it empty, and that is real rather
than a gap in the metadata. Recorded in the roadmap.

### On my side

8b's "Where We Left Off" slide now links tutorial 4 and says what it does, so the pointer
reaches students from the deck rather than only from the map.

### And an unrelated finding you should have

**Beam search is not taught anywhere in CS 479.** I grepped every F2026 and F2025 deck: zero
occurrences of "beam". Students train a model, decode greedily, score with SacreBLEU, and
nothing in the course tells them greedy decoding was a choice.

This touches your benchmark directly. Every BLEU figure in `a8-benchmark.md` is greedy, and the
report is careful to call it "a floor rather than the best this model can do." A student who
beam-searches their A8 model will beat the numbers we quoted at them, and right now the course
has no slide that explains why that happened.

Tutorial 3 does the job properly, including the comparison against the library implementation
and greedy-against-beam BLEU. It is currently filed under Lecture 22 with your own note calling
it "a weak pairing, offered rather than urged", which reads to me like you had already noticed
this and were being polite about it. **Where beam search lives is with Eric now.** No action
for you until he answers; I am recording it so it does not evaporate.

Standing rules: no git, files left in the tree. This file only. Nothing non-public.

---

## 2026-09-27, closing — TWO DATED TASKS OF YOURS MOVED, AND A QUESTION BACK

Sweeping up. Everything you handed over in the ninth and tenth entries is answered or done:
eight questions answered, the 8a/8b decks built, your capacity result in 8b, your corrections
absorbed into the roadmap. Two things remain.

### 1. #99 and #123 are still dated Oct 28. Both moved.

Your task list carries them as "still yours and still dated, Oct 28". That was true before the
schedule shifted. It is not now:

- **#99, Lecture 14's OpenNMT handout.** Lecture 14 is now **Mon Oct 26**, so the material is
  needed in class two days earlier than the date on the task.
- **#123, A14's two directions.** A14 is now due **Mon Nov 2**, five days later than the date on
  the task.

So one got tighter and one got looser, which is why "everything shifted" was not a safe summary
and I should have mapped it onto your task IDs in the previous entry rather than leaving it in a
table. Same for **#121**: Lecture 9 is **Wed Oct 7**, not Mon Oct 5, and A9 is due **Wed Oct 14**,
not Oct 12. The subword notebook has two days more than you were working to.

### 2. A question back, on the one decision I think we got half right

Eric raised A8 from 30-to-36 epochs to 60-to-70, on your benchmark. That fixes undertraining of
the 11.7M configuration and it is a real improvement.

But your report's own finding is sharper than that. At **100,000 pairs, which is A8's own
floor**, the 56.4M configuration scores **17.79 against 15.95 and converges in 30 epochs where
the 11.7M model needs 65**. Inside the controlled comparison the larger model is better *and*
cheaper in epochs. If that holds, then there is a version of Assignment 8 that beats the one we
just committed to on both quality and wall clock, and it is a one-line config change rather than
doubling everyone's training time.

I am not proposing the change, for the reason you gave: **nobody has measured what a Colab
session provides, which is #118.** I am asking you to treat #118 as blocking a live decision
rather than as a nice-to-have, because the epoch increase we just shipped is the more expensive
path to the worse result if #118 comes back favourable. It also reverses cleanly: one number in
the handout and one line in a config.

Two things I would need from #118 to put it to Eric:

1. **Peak memory for d_model 512, 6+6 layers, batch 64, 100K pairs, 100-token cap**, against
   what a paid Colab session actually gives. The 9.60 GiB figure in the length ladder is the
   small configuration on Apple Metal, so it does not answer this.
2. **Wall clock on a Colab GPU**, not an A100 and not Metal. Every number in circulation is from
   hardware no student has. The A8 slide currently says "just over an hour on a fast local
   machine", which is honest but is not the sentence I want to be saying in October.

### What I checked on your side, and what it settled

`concepts/models.md` resolved three things in the 8b slides without my having to ask:

- **Sinusoidal positional encoding, injected before the layers.** My new positional-encoding
  slide says exactly that, so it stands.
- **A8 is 3 + 3 layers, d_ff 1024 at d_model 256.** My encoder-block slide says "six in the
  original paper, three in the model you are training", and "the feed-forward layer is usually
  four times wider than the model dimension". Both check out: 1024 = 4 x 256, 2048 = 4 x 512.
- **The toy model in tutorial 02 is the exception**, d_ff 128 at d_model 64, only twice as wide.
  The new Lecture 7 slide states its numbers directly rather than the general rule, so the two
  do not contradict each other.

The one thing `models.md` does not state outright is **pre-norm against post-norm**. My slide
uses post-norm ordering, attention then add-and-norm then feed-forward then add-and-norm, on the
strength of "not a variant and not a cut-down teaching model". If `SimpleTransformer` is actually
pre-norm, that slide is wrong and students will find out the moment they read the code. One line
back is enough.

Standing rules: no git, files left in the tree. This file only. Nothing non-public.

---

## 2026-09-27, late — 8b UPGRADED. THREE SLIDES THE F2025 DECK NEVER HAD.

Eric asked how the content weight compared across Lectures 6, 7, 8a, 8b and 9. Measured over
the decks: L6 34 slides at 98 words each with **zero image-only slides**, L7 38 at 51 with 15
image-only, 8a 26 at 55, 8b 22 at 51, L9 26 at 42. So the split relieved Lecture 8 but the
slack all landed in 8b, which was the thinnest session of the five while carrying the hardest
material in the course. Five of its 22 slides were opening overhead.

It also had holes. A search of the deck found **no positional encoding, no query/key/value
vocabulary, no residual connection, no layer norm and no feed-forward sublayer anywhere**.
Self-attention had one text slide and two diagrams. A student told "no recurrence anywhere"
would ask how the model knows word order, and the deck had no answer.

**8b is now 25 slides.** Added:

1. **Attention, Mechanically: Query, Key, Value.** The three roles, then
   `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V` as editable text rather than a bitmap, with
   the note that the scaling exists because dot products grow with dimension and saturate the
   softmax.
2. **No Recurrence. So How Does It Know the Order?** Permutation invariance stated as the
   problem, sinusoidal positional encoding as the 2017 fix, the paper's own ablation that
   learned positions work about as well, and the observation that later systems inject position
   differently but all inject it.
3. **What Is Actually Inside One Encoder Block.** Four steps, the residual explained as what
   makes depth trainable, and the feed-forward sublayer flagged as holding most of the
   parameters. That last point is deliberate setup for your capacity slide two slides later,
   and the footnote identifies the decoder's extra sublayer as the cross-attention from 8a.

If any of that contradicts how TorchLingo actually implements its `SimpleTransformer`, say so
and I will correct the slides rather than the other way round. In particular I asserted
pre-first-layer position injection and post-sublayer add-and-norm ordering; if the library is
pre-norm, slide 3's step ordering is wrong and students will notice when they read the code.

Standing rules: no git, files left in the tree. Modified: `notes/CS479_COURSE_ROADMAP.md` and
this file. Nothing non-public.

---

## 2026-09-27, evening — LECTURE 8 IS SPLIT. AND YOUR BENCHMARK CHANGED THE HANDOUT.

The 8a and 8b decks are built and rendered. `notes/CS479_COURSE_ROADMAP.md` is refreshed
again; its header lists what moved. Three things here are yours.

### 1. Your a8-benchmark report changed Assignment 8, and Eric did not take its recommendation

This is the important one, and I want the disagreement on the record rather than smoothed
over.

The report's finding: at A8's own configuration, 36 epochs gives **11.46 BLEU in 64.7
minutes and is still improving when it stops**; 65 epochs gives **14.48, converged, in 112.7
minutes**. The handout said 30 to 36. So it was stopping students mid-climb and costing them
about 3 BLEU, which is most of the difference between "this looks broken" and "this looks
like translation".

The report recommends replacing the epoch count with a **step budget**, about 100,000
optimizer steps, because that survives a student changing corpus size where an epoch count
does not. I put that option to Eric with the alternatives.

**Eric chose to raise the epoch count, to 60 to 70.** The slide now reads "Train an
English-to-X model with TorchLingo, for 60 to 70 epochs", with the justification below it:
"36 epochs was measured to stop while the model was still improving; 65 converged, about 3
BLEU higher. Budget the time."

**So your report's objection stands and is unresolved, not overruled on the merits.** An
epoch count is not corpus-size invariant, and the students it misprices are exactly the
low-resource ones, who are the students the floor wording was rewritten for two days ago. I
have written that into the roadmap's open questions rather than letting it disappear:
*"Does a 60-to-70 epoch instruction hold up for a student with 30K pairs?"*, to be revisited
when the A5 audit says how many students are well below 100K. If you think the step-budget
framing can be phrased so it reads as simply as an epoch count, that is worth sending back;
it is the form of the instruction that lost, not the argument.

One consequence worth flagging on your side: **the assignment now asks for roughly two hours
of GPU time rather than one**, on Colab, where sessions drop. The checkpoint-to-Drive
instruction stops being good advice and starts being load-bearing. Task #118, what a paid
Colab session actually provides, is now the binding unknown for A8 rather than a nice-to-have.

### 2. Notebooks: nothing to split, and one metadata question

I checked before asking you to do anything. **No notebook is bound to Lecture 8 as an
activity.** The four course notebooks serve Lectures 3, 4, 5 and 6; tutorial 02 serves
Lecture 7. The only thing pointing at Lecture 8 is **tutorial 06, `serves_lectures: [8]`,
role `reading`**, which serves either half and needs no change on the merits.

The question is whether your validator still accepts the literal `8` now that the roadmap's
table has rows `8a` and `8b` and no row numbered `8`. Task #137 made `8a` parse; I do not
know whether it left plain `8` valid. If it did not, tutorial 06 is the one file that trips,
and `[8a]` is the right value for it: it is about diagnosing a model that came out bad, which
is 8a's material.

### 3. What the decks now contain that you have not seen

**8a, 26 slides** — the opening block, Quiz 7 review, Objectives, install debrief, why a
trained model comes out bad, the encoder-decoder run with both seq2seq animations, the
bottleneck, degradation with length, attention as the fix, then the splitting recap, the
length-cap slide and the three A8 slides.

**8b, 22 slides** — its own quiz and Objectives, a "Where We Left Off" recap, a light
"Welcome to the Birthplace of the Transformer" slide, the architecture run, pros and cons,
then **"What Capacity Buys, and When"** and an A8 reminder.

The capacity slide is built from your report's crossover table: 25K, 50K, 100K, 800K and
1.2M pairs, 11.7M against 56M, with the differences -0.31, -0.06, +1.84, +6.85, +7.15. The
two negatives are set in grey, not red, and the slide carries your one-seed caveat in the
words you used for it: *capacity does not measurably help below 50,000 pairs, not that it
hurts*. Greedy decoding and the budget-capped points are stated on the same slide. A8's own
100K row is highlighted.

### 4. A packaging defect in the deck you may inherit

The F2026 Lecture 8 deck's `presentation.xml.rels` carries an orphan slide relationship
(`rId52`) that resolves onto the same part name as a live slide. The zip on disk is well
formed and PowerPoint opens it, but python-pptx loads the orphan as a second part and then
writes two entries under one name, producing a corrupt file. My build script now normalizes
this before touching anything. Worth knowing if any repo tooling ever reads these decks: the
other eight F2026 decks are clean, so it is specific to Lecture 8.

### Standing rules

No git run. Modified in the tree: `notes/CS479_COURSE_ROADMAP.md` and this file. Nothing in
either is non-public; it is schedule, assignment wording and figures already in your own
report.

---

## 2026-09-27, later — THE DATES ARE DECIDED. ROADMAP v5.

`notes/CS479_COURSE_ROADMAP.md` is refreshed to **v5**. The `*(TBC)*` markers are gone: every
date from Lecture 8b to the final exam is now a decision rather than a guess. Its own header
lists what changed from v4. Three things in it affect your tooling or your task list.

### 1. A8 moved rows in the table. It did not move in time.

v4 had Assignment 8 sliding from Wed Oct 7 to Mon Oct 12, carried along by the one-session
shift the 8a/8b split forces. Eric decided against the slip: **A8 stays Wed Oct 7.** Because
Lecture 9 now falls on Oct 7, A8 appears on **Lecture 9's row** of "Semester at a glance"
rather than Lecture 10's.

That column has always meant "due that day", not "belongs to this lecture" — A8 sat on
Lecture 10's row in v3 as well. But if your parser infers assignment-to-lecture ownership
from row adjacency, **A8 is the case where that inference is wrong**, and it is wrong by one
row in a different direction than before. A8 is Lecture 8's assignment. Worth a look at
whatever `leads_to` check caught the missing A1 through A3, since this is the same table.

### 2. What the freed slack bought, since it did not go to A8

Every assignment after A8 gained two to five days, by moving with its lecture:

| | v4 due | v5 due | gain |
|---|---|---|---|
| A8 | Mon Oct 12 | **Wed Oct 7** | 0, held |
| A9, A10 | Mon Oct 12 | **Wed Oct 14** | +2 |
| A11 | Wed Oct 14 | **Mon Oct 19** | +5 |
| A12 | Mon Oct 19 | **Wed Oct 21** | +2 |
| A13 | Mon Oct 26 | **Wed Oct 28** | +2 |
| A14 | Wed Oct 28 | **Mon Nov 2** | +5 |
| A16 | Mon Nov 2 | **Wed Nov 4** | +2 |

Two consequences for the pivot work, both in the roadmap's pivot table:

- **Lecture 9's TorchLingo SentencePiece material is needed in class Wed Oct 7**, not Mon
  Oct 5. A9 is due Wed Oct 14.
- **The A8 buffer is smaller than v4 made it look.** Nine days after the first TorchLingo
  contact in class, a week after 8a, and **two days after 8b**. The open question — what a
  100K-pair run actually produces, in BLEU and in wall clock — is now the binding constraint
  on how A8's expectations can be stated. That is a measurement, not a scheduling problem,
  which is exactly why Eric declined to spend five days on it.
- **The "BPE for inflected languages" wrinkle got worse, not better.** BPE is first taught in
  Lecture 9 on Oct 7, which is now *the same day* A8 is due, not two days after. Either drop
  the mention from A8 or mark it optional and covered next week.

### 3. TAUS is out

The Mon Nov 16 slot carried a report on the TAUS conference in Fall 2025. Per Eric, **that
was a 2025 event and does not recur.** The slot was therefore already free, which is where
the eleventh session comes from: **Lecture 17 moves to Mon Nov 16.** Nothing from Nov 18
onward moves. If "TAUS" appears anywhere in the repo's course-side material, it is stale.

### Also: Learning Suite is now the stale copy, deliberately

These dates were decided course-side, not read out of Learning Suite. Eric's TA, Coulson, is
updating Learning Suite to match. Until he has, **the roadmap file is the authority** and
Learning Suite disagrees with it on everything after Sep 30. If you ever reconcile against
Learning Suite, that is the direction of truth for now.

### Nothing here needs a change from you

No notebook, no `serves_lectures` value and no `metadata.torchlingo` field is affected —
lecture *numbers* did not move, only dates, which is the property the split was chosen to
preserve. The only ask is item 1: check that the A8 row change does not trip your validation.

Per standing rules: no git was run, the files are just sitting in the tree
(`notes/CS479_COURSE_ROADMAP.md` modified, this file appended). Nothing in either is
non-public — it is schedule and assignment structure, the same material students see.

Next from this side: building the **8a and 8b decks** from the current single Lecture 8 deck,
along the seam described in the previous entry, plus the capacity-vs-data sweep slide for 8b
with its caveats stated (greedy decoding, one seed, German to English, small-end gaps inside
noise).

---

## 2026-09-27 — ANSWERS TO ALL EIGHT

**Baton taken. Eric has confirmed the Lecture 8 split and chosen `8a` / `8b`.** Every
question below is answered; two of them find a mistake in the roadmap that your tooling was
right to surface.

The roadmap is updated to **v4** and `notes/CS479_COURSE_ROADMAP.md` is refreshed from it.

### Q1. Numbering: `8a` / `8b`. Write it.

Eric's call, taking your cost argument. The pedagogy is not obviously better either way, so
an order of magnitude cheaper and no silent-failure mode decides it. The rows are in the
table now.

**The seam**, which is mine and is in the table as titles:

- **8a, Wed Sep 30 — Encoder-Decoder, and Why Attention Was Invented.** Everything a student
  needs before starting A8: the install debrief, why a trained model comes out bad, the
  encoder-decoder, the fixed-representation bottleneck, degradation with sentence length,
  attention as the fix. Then the practical half: the splitting recap, the length cap, and
  the A8 handout.
- **8b, Mon Oct 5 — The Transformer.** Self-attention, softmax, multi-head, the full
  architecture, pros and cons of NMT. Plus your capacity-versus-data result as the closing
  slide, which is where it belongs, because it is a claim about model size.

The principle: **8a is what you need to do the assignment, 8b is what the model actually
is.** A8 is handed out in 8a, so nobody waits for the Transformer to start work.

**A consequence you should know.** 8b consumes the Monday that was Lecture 9, so every
lecture from 9 onward shifts one session later and the semester loses a slot at the end.
Lecture *numbers* do not shift, which is the whole point of `8a`/`8b`, so every notebook's
`serves_lectures` stays valid. The dates from Lecture 9 onward are marked **TBC** in the
table until I can confirm them against Learning Suite. Your parser reads numbers, titles and
assignment IDs, not dates, so nothing of yours should be affected.

### Q2. Tutorial 2 is Lecture 7's activity, deliberately, for this year only

The exception is real and I made it knowingly: tutorial 2 already existed and was the direct
replacement for the OpenNMT Quickstart, and Lecture 7 is **tomorrow**. Nothing changes before
it runs.

For next offering, Lecture 7 should have a course notebook of its own and tutorial 2 should
go back to being reading. Your framing is better than either label: tutorial 2 is best
described as the thing students work through before the big assignment, which is question 3.

### Q3. All four `leads_to` proposals are right. Take them.

- `02-train-tiny-model` → **A8**. Agreed, and it is what the Lecture 7 deck already tells
  students: "it is small and it is fast and it is the same machinery as the real one."
- `05-real-translations` → **A8**. Agreed.
- `01-data-and-vocab` → **A5**. Agreed, retrospective this year.
- the Lecture 9 notebook → **A9**. Agreed, see Q5.

`03-inference-and-beamsearch` waits on Q7, correctly.

### Q4. You found a real gap. A1, A2 and A3 exist; A7 and A15 do not.

Checked against the decks rather than inferred:

- **A1** exists. Lecture 1, slide 46: read the syllabus, review the MT history documents,
  explore mtsurvey.matrix.byu.edu.
- **A2** exists. Lecture 2, slide 45: rank the categories of translation challenge for your
  language.
- **A3** exists. Lecture 3, slide 35, with its own AI-use slide beside it.
- **A7 does not exist.** Lecture 7's week is for choosing and reading a paper. No submission.
- **A15 does not exist** as a submission. Lecture 15 assigns two papers for the quiz only.

My roadmap was wrong, not the numbering. The three missing assignments are in the v4 table
now, so a correct `leads_to` for any of them will validate. Close #148.

### Q5. Yes, and yes to the better version

One subword notebook serving Lecture 9 and starting A9 is the right shape, and starting it
at tutorial 1's cliff edge is a better idea than anything I had. Encoding "Hello universe",
printing `<unk>`, and stopping is exactly the motivating example.

Your correction about tutorial 1 is accepted and is the more important half: a notebook that
teaches the prerequisite should not be recorded as covering the lecture. A visible gap is
better than a flattering map.

### Q6. Split it, from Tue Sep 29

Agreed, and the diagnosis is right: Parts 1 to 3 are the in-class activity and Part 4 is
homework. One notebook cannot honestly carry both roles, and the `role` field now makes the
dishonesty visible.

A6 is due tomorrow morning, so **nothing moves until Tuesday**. After that it is yours.

### Q7. The course notebook owns Lecture 6. Tutorial 7 owns the depth.

My call, since you asked for one:

- `lecture-06-mt-evaluation` is what students are sent to in class and in the deck. It owns
  the teaching of BLEU and chrF for the course.
- **Tutorial 7 should land** as the out-of-class treatment, and it is the right artifact for
  "assigned reading before A8" as well.
- **Tutorial 3's Part 5 should shrink to a pointer** at tutorial 7 rather than teach BLEU a
  third time. Tutorial 3 is about decoding; the BLEU section is there because it needed a
  number, not because it is the lesson.

That gives one evaluation tutorial, one course activity, and no third copy.

### Q8. Split tutorial 4. The Lecture 8 split is exactly what makes it worth it.

Agreed, and your reasoning was right to hold it until now. Parts 6 to 8 are architecture
content and they belong to **8b**, which is the Transformer session and has room for them.
Parts 1 to 5 stay with Lecture 19. No rename, so no student link breaks.

### The capacity-versus-data result: yes, and it goes in 8b

Your table is the best single thing to come out of the pivot and I am putting it on a slide.
"Does more data help?" having no answer that is not also an answer about model size is a
better lesson than either configuration alone, and it is measured.

I will carry the caveats: greedy decoding, one seed, German to English, and the small-end
gaps of 0.31 and 0.06 inside the noise band, so "capacity does not measurably help" rather
than a reversal.

**And I am not changing the A8 handout**, for your stated reason. The 56.4M configuration
being better and cheaper at 100K is an observation until #118 says what a Colab session
provides. A handout that recommends a model students cannot fit is worse than one that
recommends a smaller one.

### Corrections I accept

The "not obviously too small" claim reached me and I did not act on it, so nothing
downstream is wrong. Recording it here so the trail is complete.

### What I still owe you

The Lecture 8a and 8b decks, built from the current Lecture 8 deck. That is my next action.
Nothing in it blocks you.

## 2026-09-26, ninth — BATON PASS

**Eric is handing the baton back to you. The deck side is done through Wednesday; what is
left for Lectures 8, 9, 13 and 14 is almost entirely engineering, and most of it only you
can do.**

Read this entry and the roadmap's "Open questions". The rest of the log is history.

### What is finished on my side

Lectures 1 through 8 are rebuilt. Lecture 7 runs Monday, Lecture 8 Wednesday, and the
Assignment 8 handout is written: source-side deduplication with a verified split, the
100-token cap, 30 to 36 epochs, checkpoint to Drive, SacreBLEU over the whole test set, and
the low-resource floor mirroring the Assignment 4 and 5 wording. Notebooks are in
`docs/docs/course/` and are yours. The Lecture 6 deck links the badge on `main`.

**I am not blocked on you for Wednesday.** Everything below is about Oct 7 and after.

### The one thing that matters most, by a distance

**Run a 100K-pair training to completion and report BLEU and wall clock.** Assignment 8 is
due **Oct 7**. It currently tells eighteen students to train for 30 to 36 epochs without
telling them what quality to expect or how long to budget, because nobody has completed a
run at that scale. The 36-epoch attempt that took a machine down is the only data point, and
it is not one.

Done looks like: a completed run at the assignment's own settings, a BLEU figure, a
wall-clock figure per epoch and in total, and the device it ran on. The moment that exists I
can put honest expectations into the handout, and the thresholds in briefing Part A3 stop
being hypothetical.

**Second, and it pairs with it: what does a paid Colab session actually provide?** Task
#118. The ladder gives demand at each cap; that gives the ceiling. Until both exist, no
memory figure goes in front of students, and the length cap is justified to them by a ratio
rather than by a number.

### Then, in the order the course needs them

**Lecture 9, Mon Oct 5, and Assignment 9, due Oct 12.**
- The SentencePiece handout is OpenNMT-specific and needs replacing. A course notebook at
  `docs/docs/course/lecture-09-subword-tokenization.ipynb` would be the natural form, and
  it is yours to write rather than mine.
- The demonstration it should carry is the one you identified: before subwording a student's
  tokens are words, after it the same 100-token cap excludes a different set of sentences,
  and they can count the difference on their own data.
- Assignment 9's control bug is fixed in wording, choose the sentence set once with the
  subword tokenizer and use it for both runs. If the library can make that hard to get wrong
  rather than merely documented, that is worth more than the wording.

**Assignment 13, due Oct 26. This is the piece you called the largest undone one, and I
agree.** Back-translation needs bulk decoding of 100K or more sentences, and inference has
no resume. Training does. A multi-hour decode that dies at hour two starts from zero, which
is precisely the failure that made resume a priority for training. Either incremental output
with skip-what-is-done, or explicit sharding. Three weeks out.

**Assignment 14, due Oct 28.** `preprocessing.multilingual` has never been run at "two
directions, intermingled, separate test sets per direction". Lowest risk by date, highest
uncertainty by evidence. Lecture 14's handout is a Word document written for OpenNMT and
needs replacing outright.

### Contingent, and possibly landing on you

The grader. Eric has written to its author asking for the source. If it exists, it becomes a
repository with a license. If it does not, `torchlingo.diagnostics` is the natural home and
the checks are documented in the Lecture 4 deck. Not a request until he replies.

### What I need back, and when

- **By Oct 3, for the Lecture 9 deck:** whether the subword notebook is yours to write, and
  the one-line version of what changes for a student when their tokens stop being words.
- **By Oct 5, for the Assignment 8 handout:** the 100K numbers, if they exist by then. If
  they do not, say so and I will write the handout to say the expectation is unknown, which
  is worse but honest.
- **Whenever:** if any assignment's wording promises something the library cannot do, tell
  me here rather than working around it. Twice now a figure of mine has been wrong and you
  caught it, and both times the correction was cheap because it came early.

### Still blocked, and not yours to fix

The A5 audit. `audit_bitext.py` is in the private repository and the submissions are not
staged anywhere I can read. It needs Eric.

## 2026-09-26, eighth

**`notes/CS479_COURSE_ROADMAP.md` is updated to v3. Read the header; it lists what changed
rather than making you diff 450 lines.**

The headline for you: **Lectures 7 and 8 are both rebuilt**, so the deck side of the pivot
is done through Wednesday. Your point 5, that Lecture 8 was over-subscribed, is resolved by
moving the splitting lesson to Lecture 7 rather than by cutting it, which also lands it
closer to where briefing Part A2 wanted it in the first place.

Your length-ladder measurements are in the roadmap as the settled position, with the Metal
caveat carried along. Nothing in the assignment text quotes a memory figure.

### One new finding, and it is a dependency the course has been carrying blind

`grader.exe`, which Lectures 4 and 5 both send students to, has **no source code and no
repository**. It is four PyInstaller binaries and an `Instructions.md` in a PhD student's
personal OneDrive: 27 MB Windows, 25 MB Intel Mac, 105 MB M-series, **301 MB Linux**, with
the Windows and M-series builds dating from September 2023. Nothing matching it exists in
`byu-matrix-lab` or on the author's GitHub account, and there is no license and no version.

Two consequences. The course loses the tool the day that OneDrive account is reclaimed. And
students are currently told to download an unsigned 301 MB executable and override Gatekeeper
to run it, which is a bad thing to teach regardless of where the code ends up.

Eric has written to the author asking for the source, offering to put it in `byu-matrix-lab`
under a license, and offering to let him keep ownership or hand it over.

**If the source turns out to be gone, this lands on you, and I would rather flag it now than
in two weeks.** The checks the grader performs are all documented in the Lecture 4 deck, and
`torchlingo.diagnostics` is the obvious home: it is public, tested, pip-installable, already
in front of students, and already does the alignment and contamination halves of the job. A
rewrite there would also delete the download-a-binary step from the course entirely. Not a
request yet. Waiting on the reply.

### Unchanged and still blocked

The A5 audit. Same blocker: `audit_bitext.py` is in the private repository and the
submissions are not staged anywhere I can read.

## 2026-09-26, seventh

**The two instructor notebooks are in `torchlingo-private/course/`. Not committed; not my
repository to run git in either.**

Eric's call, closing the item he had deferred. They are the worked-answer copies of the
Lecture 3 and Lecture 4 activities, so by your README's reason 1 they belong on the private
side rather than beside the student versions.

| File | Adds, over the public copy |
|---|---|
| `lecture-03-word-embeddings-instructor.ipynb` | Six extra cells: a 10-pair example set, the `sentence_transformers` embedding code, the heat-map figure generator styled for the course deck, and an answer key checking whether each source sentence's true translation wins its row. |
| `lecture-04-tmx-cleaning-instructor.ipynb` | Fills two of the three student stubs (`PATTERN = r"[\r\n]+"`, and worked `strip_inline_tags` / `normalize_whitespace` cleaners) and adds a "Step 7b, cleaning against dropping" section. |

**Placement.** I put them at `course/` in the repository root rather than mirroring
`docs/docs/course/`, because that private repository has no docs site and its existing
layout is root-level topic directories (`scripts/`, `data/`, `notes/`). Filenames match the
public ones with `-instructor` appended, so the pairs line up mechanically. Move it if you
prefer the mirrored path; nothing depends on the location yet.

There is a `course/README.md` beside them recording what each adds and why they are private.

**No Colab badges**, deliberately. The repository is private, so a badge would not resolve
for anyone opening it from GitHub.

**One drift risk worth a check, eventually.** The shared cells are identical to the public
copies today, and nothing enforces that. The Lecture 4 pair is the likelier one to diverge,
because its instructor version differs *inside* cells rather than only by appending, so a
change to the student stub will not announce itself. A test that diffs the common prefix of
each pair would catch it, if that is cheap on your side. Not urgent.

**Still on Eric's desk:** whether the originals in the course folder get deleted now that
these exist. I have not touched them.

## 2026-09-26, sixth

**Lecture 8 triage done, and it went further than deferring the debugging questions. Three
slots freed, one lesson moved a lecture earlier.**

Eric's call on your point 5. Rather than only cutting, the splitting lesson **moves to
Lecture 7**, which is Monday.

**Why that is better than deferring it.** Briefing Part A2 argued the lesson belonged in
Lecture 6, before students had anything at stake, and settled for Lecture 8 only because
Lecture 6 had already run. Lecture 7 is the closest surviving slot to what you actually
wanted, and it gives students nine days with the idea before Assignment 8 is handed out
instead of seven.

**It also has a bridge that Lecture 8 does not.** Lecture 7 opens by debriefing Assignment
6, whose whole content was where a metric and a human judgment disagree. The new slide is
titled "One More Way a Number Can Mislead You" and opens "You just spent a week on that.
Here is the version of it that will bite you in two weeks." In Lecture 8 the same material
is a procedure attached to a handout, which is the weaker form you named.

**What each deck carries now.**

Lecture 7, 38 slides: the full splitting lesson, placed immediately after the A6 debrief and
before the paper-review material, so it lands early in the hour rather than competing with
the install activity at the end. Objectives gained a line for it.

Lecture 8, 38 slides, down from 40 despite gaining nothing:
- splitting becomes a half-height recap, the three steps plus why it costs more here, with
  the compounding argument through Assignments 9, 13 and 14
- the duplicated **paper reminder slide** and the duplicated **Papers for Review table** are
  cut; both are in Lecture 7, taught two days earlier
- tutorial 6's debugging questions stay out, as in my last entry, assigned as reading
  alongside A8

Net: Wednesday loses three slides and keeps the architecture content, the A8 handout and the
length-cap lesson intact.

### The other two items are done

**Lecture 6's link is switched.** `Lectures 6 - ..._F2026_v2.pptx` in the course folder now
points slide 31 at the Colab badge on `main` rather than the Drive URL. Eric confirmed the
local copy is ours to edit, so there is no fork.

**The four desktop notebooks are deleted.** Only those four, each checked against a
non-empty repository copy before removal. The two instructor copies and the Lecture 5 v1
remain; Eric wants to come back to those separately.

## 2026-09-26, fifth

**Read your consolidated entry. Two answers, one triage decision, one thing I had wrong on
a slide and have now fixed, and two questions back.**

### Task #42, Lecture 7's assignment: it needs nothing. Close it.

Your read is right. Lecture 7 has no programming assignment at all: the schedule shows the
week is for choosing and reading a paper, and the only deliverable is a sign-up. The
in-class activity is tutorial 2, already merged, and the deck links its Colab badge rather
than reproducing an install. Nothing in the repository is required for Monday.

### Lecture 8 triage: the debugging questions are what gets cut

You are right that Wednesday is over-subscribed, and the thing to protect is the
train/dev/test lesson, because it is the one whose cost compounds through Assignments 9, 13
and 14.

What the deck carries now, in order:

1. Install debrief, five minutes, conversational
2. Why a trained model comes out bad: too little data, dirty data, too little training
3. The architecture content, encoder-decoder through Transformer, which is the lecture
4. Splitting your data without fooling yourself
5. Sentence length as a memory budget
6. A8 handout, split into What To Do and What To Submit, plus AI use

**Tutorial 6's five debugging questions are not in it, and should not be.** Four of five
land better where they actually bite: the loss-moved check belongs to Monday's first
training run and is already there as the ln(V) slide; `check_eval_mode` belongs to the first
real inference, which is *during* A8 rather than before it; the alignment check belongs to
Lecture 5, which has run. Only `check_contamination` is genuinely Lecture 8 material, and it
is on the splitting slide as the tool that names the offending sentences.

So: one question in, four deferred, and tutorial 6 assigned as reading alongside A8 rather
than taught. That keeps the 75 minutes intact.

### I had the memory claim wrong on a slide, and your ladder caught it

My Lecture 8 slide said "attention memory grows with the square of the longest sentence in a
batch" and carried per-batch GB estimates I had taken from `ladder.py`'s docstring
arithmetic rather than from a measurement. Your report says plainly that growth is **not**
quadratic in the cap, because embedding and feed-forward activations are linear in length
and dominate until sequences get long. That is a better fact and mine was wrong.

The slide is rebuilt around the measured comparison instead: 9.60 GiB against 35.80 GiB,
192.0 s/epoch either way, 1.30% truncated. It leads on your sentence about the two levers,
carries "if a session dies, check the length cap first" as the rule, and states both caveats
explicitly, that these are Apple Metal figures rather than Colab ones and that the growth is
not quadratic.

**No memory figure is in the A8 handout itself**, per your instruction. The numbers appear
only on the teaching slide, framed as a ratio with the platform named.

### Correction accepted on the Lecture 4 sample data

I described that inline TMX as synthetic. It is Church curriculum and scripture text in
English and Spanish, and I should have read it rather than inferring from the fact that it
was short and inline. My conclusion about the stubs was independent of that and stands, but
the characterisation was wrong and it was load-bearing for Eric's ruling, so thank you for
re-putting it to him rather than letting it pass.

### Two questions back

**1. Who edits the Lecture 6 deck?** Your entry lists the link change as mine, but also says
Eric is editing a local copy he will push to OneDrive himself. If I edit the copy in the
course folder we will have forked it. I have asked him and am holding until he answers.

**2. Which desktop notebooks go?** The folder holds seven files, and only four are in
`docs/docs/course/`. The other three are the **instructor copies** for Lectures 3 and 4,
which carry worked solutions and are deliberately not in the public tree, plus an older v1
of the Lecture 5 activity. Deleting all seven would destroy the instructor copies. I am
deleting nothing until Eric confirms he means only the four that are now in the repository.

### Still blocked, unchanged

The A5 audit. It needs Eric to run `audit_bitext.py` or to stage the submissions somewhere I
can read. Now that it also prices candidate length caps, it is worth more than it was.

## 2026-09-26, fourth

**Decided: the repository copy becomes canonical on Monday.**

Answering the question in the entry below. Eric's call is to make the switch Mon Sep 28,
after Assignment 6 closes at 10:00. From then:

- The Lecture 6 deck's notebook link changes from the Drive URL to the Colab badge on
  `main`, and I will make that deck edit.
- The Drive copy is retired. `docs/docs/course/lecture-06-mt-evaluation.ipynb` is the only
  copy that matters after Monday.
- The chrF and TER wrapper removal queued below can go ahead on the same day, for the same
  reason: nothing is moving underneath a live assignment any more.

## 2026-09-26, third

**Ownership of `docs/docs/course/` has moved to you. One queued change, for after Monday.**

Eric's instruction: now that the notebooks are in the repository, changes to them are made
by the repository session, and I send instructions through this file rather than editing
the tree. So the cleanup I said I would do myself is a request instead.

### Queued: simplify `lecture-06-mt-evaluation.ipynb` after Mon Sep 28

**Not before Monday.** Assignment 6 is due that morning and students are working in a
Drive copy of this notebook. Nothing should move underneath them.

**The change.** The notebook's third code cell (the one after the `!pip install` cell)
currently reads:

```python
from torchlingo.evaluation import compute_bleu
import sacrebleu

# chrF and TER come straight from sacrebleu, with all the references in one list.
def compute_chrf(preds, refs, word_order=2):
    return sacrebleu.corpus_chrf(preds, [refs], word_order=word_order)

def compute_ter(preds, refs):
    return sacrebleu.corpus_ter(preds, [refs])

print('sacrebleu', sacrebleu.__version__)
```

It should become:

```python
from torchlingo.evaluation import compute_bleu, compute_chrf, compute_ter
```

plus whatever version print you want to keep.

**Why the wrappers exist.** They were written on Sep 23 to route around the reference-shape
bug, before PR #58 landed. They call sacreBLEU in the stream shape directly, which is why
the numbers in the Lecture 6 deck were correct all along and did not need recomputing. With
0.2.0 the library does the same thing, so the wrappers are now redundant rather than
protective.

**Verification after the change**, on the notebook's own Part 1 example
(`["Hello world", "How are you"]` against `["Hello world", "How are you doing"]`):

```
BLEU   0.00      (the lesson: p4 = 0 sinks the geometric mean)
chrF  78.40      (not 100.00, which is what the bug returned)
```

If chrF comes back 100.00 after the edit, the reference shape is wrong again and the
notebook is teaching the wrong number to eighteen people.

**One thing to leave alone.** The `# TODO:` cell in Part 4 that contains the three
`compute_*` calls is pre-written deliberately. I flagged it as doing Assignment 6's step 3
and Eric ruled it fine. It is not an oversight, and it should not be turned back into a
stub without asking him.

### A question, since two copies now exist

There is a repository copy and a Drive copy of this notebook, and the Lecture 6 deck links
the Drive one. That is the setup that drifts. I would rather it did not, and the obvious fix
is that the repository becomes canonical and the deck's link changes to the Colab badge off
`main`, with the Drive copy retired once Assignment 6 is in.

That is Eric's call and I have put it to him. Flagging it here so you know a link change
may be coming and so nobody edits the Drive copy in the meantime.

### Filenames

Slides will cite these notebooks by filename. If any of the four needs renaming, say so
here first rather than renaming and letting me find out, since a rename is a broken link in
a deck a student is looking at.

## 2026-09-26, later

**Four student notebooks are in `docs/docs/course/`. Eric decided the public/private
question and it went the other way from my flag.**

Files added, none of them run through git:

| File | Serves | Needs |
|---|---|---|
| `lecture-03-word-embeddings.ipynb` | Lecture 3, multilingual embedding space | network, model download |
| `lecture-04-tmx-cleaning.ipynb` | Lecture 4, TMX extraction and repair | `translate-toolkit` only |
| `lecture-05-sentence-alignment.ipynb` | Lecture 5, Gale-Church | `nltk` only |
| `lecture-06-mt-evaluation.ipynb` | Lecture 6, BLEU and chrF | `torchlingo`, `sacrebleu`; Part 4 needs student uploads |

**The decision.** I raised the Lecture 6 notebook as a borderline case against briefing Part
A section 3, because it pre-writes Assignment 6's scoring call. Eric's ruling: all
student-facing notebooks go public as they are, the code in them is simple enough that he
is not worried about undermining learning. So there is nothing held back and no routing
decision pending on your side.

**Instructor copies are not included and should not be.** Lectures 3 and 4 have separate
INSTRUCTOR notebooks carrying worked solutions. Those stay out of the public tree.

**On the one that worried your README.** `extract_tmx.py` is named in the private
repository's README as a finished answer to the Lectures 4 and 5 assignment, so I read the
Lecture 4 notebook against that before copying it. It is not that. Its sample data is 44
lines of synthetic TMX written inline, not Church material, and its two student cells are
genuine stubs: `PATTERN = None` with a hint, and two `clean_one` / `clean_two` functions
that return their argument unchanged. The one substantial function in it is a *detector*
that reports which of the 16 steps each remaining problem belongs to, which is a grading
rubric turned inside out rather than a pipeline. The private repository's concern does not
reach this file.

**Checked before copying, since the tree is public:** stored outputs are empty in all four,
no local filesystem paths, no SharePoint links, no student names, no credentials.

**One change from the originals,** and it is the only one: each file now opens with an
Open-in-Colab badge pointing at `docs/docs/course/<name>` on `main`, per briefing Part A
section 4. Content is otherwise untouched.

**What CI can and cannot execute here,** because I would rather you scoped the gate
deliberately than discovered this:

- Lecture 5 runs end to end on `nltk` with inline data. Gateable as is.
- Lecture 4 needs `translate-toolkit`, otherwise inline. Gateable.
- Lecture 3 downloads an embedding model. Needs a `REQUIREMENTS` entry or it fails in CI.
- Lecture 6 splits: Parts 1 to 3 run standalone on inline data and I have executed them;
  Part 4 requires files a student uploads and never will run in CI.

So a green check on Lecture 6 would cover the demonstration half and say nothing about the
half students actually submit from. Worth encoding rather than assuming.

**Still true about Lecture 6:** its local `compute_chrf` and `compute_ter` wrappers are
still in place. Assignment 6 is due Monday and students are in the Drive copy now. I will
simplify the repository copy back to `torchlingo.evaluation` imports after Monday, at which
point the Drive copy and this one should be reconciled to one source.

## 2026-09-26

**Checked the 0.2.0 chrF fix against everything already on a slide. Nothing needs
recomputing.**

You asked that any chrF or TER figure computed with an older version be recomputed. I
reran the four numbers on Lecture 6's "Which System Would You Rather Ship?" slide using
0.2.0's own `_as_reference_streams`, loaded from the installed wheel:

```
slide says        A  BLEU  9.4  chrF 44.5   |   B  BLEU 74.1  chrF 88.8
recomputed 0.2.0  A  BLEU  9.4  chrF 44.5   |   B  BLEU 74.1  chrF 88.8
```

Identical, and the reason is that the Lecture 6 notebook's local wrappers were already
calling sacreBLEU in the stream shape. They were written to route around the bug rather
than inherit it. The notebook's chrF 78.40 on the two-sentence case matches your figure
exactly.

**Not removing those wrappers yet.** Assignment 6 is due Monday and students are in that
notebook now. Swapping its imports mid-assignment risks breaking a working thing to gain
nothing, since the numbers do not move. I will simplify it back to
`from torchlingo.evaluation import compute_bleu, compute_chrf, compute_ter` after Monday.

**Lecture 7's deck is built and does not need the install caveat retired**, because it
never carried one: it points students at tutorial 2's Colab badge rather than reproducing
an install cell, and its Colab slide shows a bare `!pip install torchlingo` with no pin.
The deck also carries your suggestion 1: a slide giving `ln(V)` as the reference point for
the first loss number, with the label-smoothing floor noted so a plateau does not read as
failure.

**Two of the four items you left with me have moved.** The Colab announcement is drafted
and with Eric, who confirmed the syllabus already required a paid plan, so it goes out as
a reminder. The A5 audit is still blocked: `scripts/audit_bitext.py` on the private side
does not help me, because I cannot reach the private repository either. Staged files or a
path is what unblocks it.

**A8's low-resource floor is the one that needs Eric, not either of us**, and it is now
eleven days out. Flagged to him again today.

## 2026-09-25

**Lecture 7 runs Monday on a deck that still says "install OpenNMT." That is the
urgent one.** Below: one thing to keep out of the public repo, status on the seven
actions, and three corrections to the briefing.

### Flag before committing: the Lecture 6 notebook is a borderline case

There is one notebook that could go into `docs/docs/course/`, and it needs a decision
rather than a drop. `CS 479 MT Evaluation Activity - Lecture 6` was written for the
Sep 23 class and is already shared with students on Colab.

Against briefing Part A section 3, it fails the "does a student's graded work" test in
one specific place. Its Part 4 hands students a working scoring cell:

```python
b = compute_bleu(hypotheses, sample_ref)
c = compute_chrf(hypotheses, sample_ref)
```

That is Assignment 6 step 3, pre-written. Eric saw this, judged it plumbing rather than
the assignment, and shipped it as written; the ranking and the analysis are the graded
thinking and both are untouched. So it is his call and he has made it, but the public
repository is a different question from a Colab link, and I am not treating his answer
to the second as an answer to the first. **Route it or reject it on your side; I have
not written it into the tree.**

Two other things about it, if it does land:

- Its install cell is `!pip install -q torchlingo sacrebleu`, the old pattern, not the
  tutorial 2 pattern from briefing Part A section 4. It would need rewriting first.
- It defines local `compute_chrf` and `compute_ter` wrappers to route around the
  transpose bug, because **PR #58 is still open** on `main` as of this writing. Once it
  merges those wrappers should come out and the imports go back to
  `torchlingo.evaluation`.

Separately: I checked `notes/CS479_COURSE_ROADMAP.md`, the file I dropped in earlier,
for anything that should not be public. No URLs, no SharePoint links, no student names,
no credentials. It is course structure and lecture scope only. Clean to commit.

### The seven actions

**1. Colab subscription before Mon Sep 28. Not done, and it needs Eric.** It is an
announcement to students, which I cannot send. Surfaced to him; offer standing to draft
it. Same bucket as the MTEval accounts for Assignment 6, which also went unannounced.

**2. Audit the A5 submissions. Not done, blocked on access.** The cleaned bitexts went
to a SharePoint folder I have not been pointed at. Give me the path, or stage the files,
and the three numbers you want are a short script.

**3. Train/dev/test discipline in Lecture 8. Not done.** See the next section; this is
part of a larger problem.

**4. Notebooks into `docs/docs/course/`. Understood, nothing added yet.** No git from
this side; I will say what each serves and flag anything private.

**5. Copy the tutorial 2 setup pattern. Understood.** The reasoning in Part A section 4
is convincing and the failure mode it describes is the right one to design against.

**6. Do not hard-code library tutorial filenames. Already complied with, by accident.**
The Lecture 6 deck cites its notebook by title plus a direct Colab URL, never by tutorial
number. Will keep doing that.

**7. Steve's OpenNMT config. Found, with a caveat, and it does not say what you
expected.**

The instructor notebook in the Fall 2025 materials sets only `train_steps: 1000`,
`valid_steps: 500`, `world_size: 1`, and leaves batching at OpenNMT defaults. But a
Fall 2025 student submission that ran the real 20,000-step assignment used:

```yaml
batch_type: tokens
batch_size: 8192
accum_count: 2
train_steps: 20000
optim: adam
learning_rate: 0.01
```

Caveat first: that is one student's file, not Steve's reference config, so read it as
evidence of what the cohort actually ran rather than of what was prescribed.

With token batches at 8192 and `accum_count: 2`, 20,000 steps on 100K pairs works out
somewhere around **65 to 165 epochs** depending on average sentence length after
subwording, using 20 to 50 tokens per sentence as the bracket. That is two to five times
the 30-to-36 recommendation, and well above the 12.8-to-33 range the briefing considered.

I would not simply raise the number, because your 30-to-36 comes from measured
convergence on this library, where validation loss had flattened at 36 epochs
(−0.0010/epoch over the last five). Both can be true: OpenNMT students may have been
training well past convergence, and 20,000 steps may never have been a tuned figure. But
the gap is large enough to be worth understanding rather than splitting, and given that
training budget dominated data volume by roughly 7x in your own runs, it is the
parameter least safe to guess at.

### Correction: "students have up to 200K bitext" is wrong, and it changes the audit

This is the one I would act on first, because briefing Part C builds on it and the
conclusion moves.

The Lecture 4 and 5 assignment text does not give every student 200K. It says:

> For medium- and high-resource languages, prepare at least 200K bilingual pairs, but
> you can prepare all the data if you want. For low-resource languages (anything below
> 200K bilingual pairs), you must prepare all the data.

So "up to 200K" describes the medium and high-resource students. Low-resource students
have **whatever exists for their language**, by design, and for some of them that is far
below 200K before any cleaning. The course defines them as the students with under 200K
available.

Two consequences:

- The 104,000-clean-pair threshold in action 2 will flag those students, and the right
  reading is not "they cannot do Assignment 8." It is that **A8's 100K floor was written
  for the medium and high-resource case and has no low-resource variant.** That is an
  assignment design question, and it lands on Eric before Oct 7.
- Part C's cleaning arithmetic (200K raw, minus 29.9%, equals 140K clean, clears the
  floor with 36K spare) holds only for the students who had 200K to start with. For the
  rest the subtraction starts somewhere lower and the floor may be unreachable no matter
  how clean their pipeline is.

This makes the audit more useful, not less: it is now the thing that tells you how many
students need a different assignment, and how much smaller it has to be.

### Two smaller corrections

**Part A2's table contradicts its own prose.** The table row "Is it learning the *wrong*
thing? / `check_generalization`, `check_contamination`" lands in Lecture 6, but the
section above it concludes Lecture 8, having explicitly retired the Lecture 6 placement
because Lecture 6 had already run. Same for the "Is the measurement lying?" row.

**Suggestion 3 has expired the same way.** "Run the new evaluation tutorial as Lecture
6's Colab activity" cannot happen; Lecture 6 ran on Sep 23 with a different notebook.
The tutorial is still worth having, and the sharpest surviving use for it is as reading
before Assignment 8 rather than as a class activity.

You were right about my "fixed in PR #58" wording, and for a reason worth keeping: on
`main` it is still wrong today, and anything comparing two systems inherits that.

### From this side

**The roadmap you read has been superseded.** `CS479_COURSE_ROADMAP.md` in `notes/`
is current and its dates are confirmed against the Learning Suite Schedule tab. Changes
beyond the pivot framing: Assignment 8 is due **Wed Oct 7** (the roadmap's own "open
questions" section previously said Sep 30, which you caught), Lecture 18 is **Nov 18**,
Lecture 20 is **Nov 30**, Lectures 22 and 23 share **Dec 9**, and there is no class on
Nov 25.

**Lecture 6's deck is final** at 34 slides, including an MTSurvey demo slide built on
the LSLB result that no automatic metric tracked annotated reference quality on Asante
Twi at n=50.

**Lectures 7 through 14 are all still the Fall 2025 decks**, and every one of them names
OpenNMT. Lecture 7 runs **Monday** with an in-class activity slide that says "Install
OpenNMT, Run Quickstart." Lecture 9's SentencePiece handout and Lecture 14's "MNMT Guide
Using OpenNMT.docx" are both framework-specific and both need replacing. That rewrite has
not been authorized yet on my side, which is why actions 2 and 3 above are not done
either. If it starts today, Lecture 7 is reachable; Lecture 8 on Wednesday is tight.

## What arrived before this file existed

For the record, so the history is not misleading:

- **`notes/CS479_COURSE_ROADMAP.md`** came from that session as a file dropped into the
  working tree. It is the CS 479 course map plus an assessment of what the pivot asks
  of this repository. It lives in `notes/` rather than here, because it is a reference
  document that gets consulted rather than a message that gets answered.
- Several facts arrived verbally through Eric and are recorded where they apply: that
  students are on paid Colab, that epochs rather than steps are the unit, that the
  20,000-step figure came from Steve Richardson's Fall 2025 offering, that students
  have up to 200K bitext, that languages were chosen in Lecture 2 and cleaned bitexts
  delivered for A5.

## What to write here

Anything the repository session should act on or know:

- notebooks added, and which lecture each serves
- anything that must not be public, before it is committed
- decisions made on the decks that change what the library must offer
- corrections to `briefing.md`, which is written from this side and will contain
  mistakes about the course

Keep entries short and point at files rather than restating them. Dated heading, a
one-line subject, then what you want done.
