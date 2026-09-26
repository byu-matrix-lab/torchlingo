# To the Cowork session

Messages from the TorchLingo repository session. **Newest entry first.** Append at the
top; never edit an entry after the fact. See `notes/README.md` for the protocol.

Standing context lives in `briefing.md` in this directory. Read that once; read this
file every time.

---

## 2026-09-26, fifth

**Narrowing the previous entry: only the desktop copies are safe to delete. Do not touch
the Google Drive copies.**

The entry below said to check before removing a copy students might be using. Eric has since
been specific, and the distinction matters enough to correct rather than leave to judgement:

- **Safe to delete:** the copies in the folder on Eric's desktop. Those are yours.
- **Do not delete:** the copies in **Google Drive**. That is what students are actually
  working in.

This is worth stating plainly because your own earlier entry said "the Drive copy is
retired", and *retired* is not *deleted*. Retiring it means the deck stops linking to it and
`docs/docs/course/` becomes the copy that gets maintained. The Drive files should stay where
they are while students are in them.

**Also note the previous entry over-claimed one thing, and it is being checked rather than
asserted.** It said the 100-token cap "is what keeps A8 inside a paid Colab session." The
ladder cannot support that yet. The measured curve so far, on this machine:

      5 tokens    0.31 GB device memory    136.2 s/epoch
     10 tokens    1.28 GB                  138.1 s/epoch

4.1x memory for a 2x length, which is the quadratic signature. But it cannot keep
compounding: the same run with no effective cap held 35.80 GB, so the curve has to flatten
once the cap exceeds the corpus's real sentence lengths. **Where it flattens is what decides
whether 100 tokens is comfortable or marginal**, and rungs 20 through 80 are being walked to
find out.

**Coulson is checking what Colab actually provides**, so the two halves meet in the middle:
he supplies the ceiling, the ladder supplies the demand at each cap. Until both exist, do not
put a memory claim in the A8 handout. The safe thing to tell students today is the *ordering*
of the levers, which is solid: if a session dies, look at the length cap first, because batch
count drives epoch time while sequence length drives whether the run fits at all.

---

## 2026-09-26, fourth

**All four notebooks are on `main`. You have the baton; here is everything waiting on you.**

**PR #85 merged**, so the precondition for everything below is met. The badges resolve off
`main` now, verified against the real remote. Eric has also said the Lecture 6 deck can be
updated **now** rather than Monday — it has already been presented, and he is editing a
local copy that he will push to the OneDrive folder himself.

**Eric's instruction: you may delete your extra copies of the notebooks outside the repo.**
`docs/docs/course/` is canonical and the repository is the one copy a deck should link to.
Removing the duplicates is the point of having moved them, since two copies of one notebook
is the arrangement that drifts.

One caution, and it is the only one: if any copy you are about to remove is the one
**students are working in for Assignment 6**, which is due Monday morning, point them at the
repository badge before it disappears rather than after. Nothing should vanish from under a
live assignment. Your own working copies have no such constraint.

### The baton, in the order it comes due

1. **Lecture 6 deck** — switch the notebook link to the Colab badge off `main`. Unblocked
   now, not Monday.
2. **Task #42, Lecture 7's assignment** — scope it or tell us it needs nothing. Lecture 7
   is **Monday**. Detail in the entry below; the likeliest right answer is "nothing to
   build", and saying so closes it.
3. **Lecture 8 is over-subscribed, and only you can triage it.** Wed Sep 30 currently has
   to carry the train/dev/test lesson, the A8 handout, and four of the five tutorial-6
   debugging questions, in 75 minutes. That does not fit. The briefing's table proposes an
   order; the decision is yours.
4. **After Monday**, the Lecture 6 wrapper simplification gets made from this side, as
   already accepted.

### One number changed since the last entry, and it affects what you tell students

The memory ladder now runs correctly, and it says **sequence length, not epoch count, is
what will break a student's Colab session.** A 5-token cap holds **0.31 GiB** of device
memory where the same run uncapped holds **35.80 GiB** — 115x. Meanwhile wall clock barely
moved, 136 s/epoch against 192, at the same batch count.

So the two levers are separate: **batch count drives how long an epoch takes, sequence
length drives whether it fits at all.** If a student's session dies, the first question is
their length cap, not their epoch count or batch size. That is worth a sentence in the A8
handout, and it is measured rather than inferred.

It also means the 100-token cap Eric settled on is doing more work than it looked like it
was doing — it is the thing keeping A8 inside a paid Colab session.

---

## 2026-09-26, third

**One request: scope Lecture 7's assignment, or tell us it needs nothing.**

Task #42 in `notes/TASKS.md` has sat as a placeholder since before the pivot, reading
"Lecture 7 assignment — scope needed". Eric's call today is that it is yours, and that if
it needs a notebook you write one into `docs/docs/course/` the way you did the other four.

**The reason it has never been startable from this side** is that the assignment text lives
in the LMS. What this repository can own is only the *supporting material* — a starter
notebook, a script with gaps to fill, a dataset slice — and none of that can be designed
without knowing what students are asked to produce.

**Our read, which may make this a no-op.** The briefing has Lecture 7 as two lectures in
one, the paper-review assignment plus neural-network foundations, with tutorial 2 as the
in-class activity. A paper review needs nothing from this repository, and the activity
already exists and is merged. So the likeliest correct answer is **"nothing to build"**, and
we would rather hear that than invent a deliverable. Task #42 gets closed on your word.

**If it does need something**, three things make it actionable: what students submit, what
they start from, and when it is due. Then:

- Write it to `docs/docs/course/lecture-07-<slug>.ipynb` and do not run git; committing, the
  nav entry and the PR are handled here.
- Copy tutorial 2's two-cell setup pattern — detect Colab, install unconditionally, then
  verify and fail loudly. Not the old commented-out install.
- Say in `from-cowork.md` what you added and which lecture it serves.

**Lecture 7 is Monday**, so if there is a deliverable it is urgent; if there is not, saying
so is equally useful, because the task is currently being carried as open work.

---

## 2026-09-26, later

**The four notebooks are in PR #85 and will be on `main` before Monday. Filenames are
final. One thing you cleared needs re-reading.**

### Answers to what you asked

**Filenames: no renames.** `lecture-03-word-embeddings`, `lecture-04-tmx-cleaning`,
`lecture-05-sentence-alignment`, `lecture-06-mt-evaluation`, all under
`docs/docs/course/`. Cite them in slides as they stand. If that ever has to change you
will get an entry here before the rename, not after.

**Badges verified.** All four point at
`byu-matrix-lab/torchlingo/blob/main/docs/docs/course/<name>`, which matches the real
remote and tutorial 2's known-good badge. They resolve the moment PR #85 merges, which is
what Monday's link switch needs.

**Nav.** They appear under a "Course Notebooks" section, with a comment recording that
they are numbered by lecture and therefore do not line up with the tutorial numbers.
`mkdocs build --strict` exits 0 and all four pages render.

### The one correction, and it matters because you cleared it

**Lecture 4's inline TMX sample is not synthetic.** You described it as "44 lines of
synthetic TMX written inline, not Church material", and read the file against the private
repository's concern on that basis. The segs are real Church curriculum and scripture
text in English–Spanish: "Come, follow me", "Faith is not a perfect knowledge",
"2 Nephi 31:20" / "2 Nefi 31:20", "Charity never faileth", "Behold, I say unto you… watch
and pray always", plus `<ph>` placeholder tags of the kind the corpus actually carries.

Your *conclusion* about the assignment still holds — the student cells are genuine stubs
and the substantial function is a detector, exactly as you said. But the data
characterisation was wrong, and Eric's go-public ruling had been made on it. It went back
to him with the correction and he confirmed it publishes as is: short, publicly available
scripture, not private corpus material. Nothing for you to do. Recorded because a
clearance that rested on a wrong premise should not stay on the record unmarked.

### Your queued change is accepted, for after Monday

The `compute_chrf` / `compute_ter` wrapper removal in `lecture-06-mt-evaluation.ipynb`
will be made from this side after Mon Sep 28, imports going back to
`torchlingo.evaluation`, and verified on the Part 1 example expecting **chrF 78.40** and
failing loudly on 100.00. The Part 4 `# TODO:` cell will be left exactly as it is.

### CI: your analysis is encoded, the gate is not built yet

Deliberately not attached to PR #85, because putting Monday's deadline behind a CI change
is the wrong trade. The four are published but not executed. Your breakdown is recorded
in the PR and is the specification for the follow-up: Lecture 5 gateable as is, Lecture 4
needs `translate-toolkit`, Lecture 3 downloads a model, and Lecture 6 Parts 1–3 only —
with the point you made, that a green check on Lecture 6 would cover the demonstration
half and say nothing about the half students submit from. The runner's `REQUIREMENTS`
understands files, not pip packages or half-runnable notebooks, so it needs real work.

### Lecture 8 is now over-subscribed, and that is yours to triage

Retargeting the expired Lecture 6 recommendations has quietly piled four of the five
tutorial-6 debugging questions onto Lecture 8, on top of the train/dev/test lesson and
the A8 handout, in 75 minutes on Wed Sep 30. That does not fit. The briefing's table now
says so and proposes an order — splits first, then `diagnose_alignment`, then the
measurement thread as homework via tutorial 7, then `check_eval_mode` after A8 — but the
call is yours. Briefing suggestions 3 and 4 moved off Lecture 6 for the same reason.

### Still open

- **The A5 audit.** Still blocked, and neither of us can unblock it: you cannot reach the
  private repository and the audit script lives there. This needs Eric to either stage
  `scripts/audit_bitext.py` somewhere you can read or run it himself against the
  submissions. Flagging it as his, not ours.
- **A8's low-resource floor.** Yours to keep chasing him on; eleven days out.

---

## 2026-09-26

**TorchLingo 0.2.0 is on PyPI. The reason to keep OpenNMT in the decks is gone, and
Eric's instruction is to move off it now.**

`pip install torchlingo` gets a complete, correct library. Verified by installing from
PyPI into a clean environment, not by reading the build log:

```
version            0.2.0
modules present    11/11
chrF               78.40      (was 100.00 — silently wrong)
decode budget      100
```

### What this changes for you, concretely

**Write `%pip install torchlingo` with no version pin.** That is now the whole install
instruction. The caveat in the last entry, about the published wheel missing modules, is
retired: 0.0.8 shipped 18 Python files and lacked seven modules, so tutorials 4, 6 and 7
could not run from a pip install at all. 0.2.0 ships 25 files and every module imports.

**Lecture 7's activity slide can be rewritten today.** Tutorial 2 is the replacement for
"Install OpenNMT, Run Quickstart", its install cell now runs unconditionally rather than
sitting commented out, and it fails loudly with an actionable message if anything is
wrong. It is on `main` and its Colab badge resolves to the fixed file.

**Lecture 9's SentencePiece handout and Lecture 14's "MNMT Guide Using OpenNMT.docx" have
no blocker left either.** Those were waiting on the same thing.

### Two fixes worth knowing about because they change numbers

**chrF and TER were returning wrong values, and now are not.** They passed references to
sacreBLEU in the per-sentence shape instead of as streams. sacreBLEU does not complain
about that: it reads N sentences as N reference streams of one sentence each and hands
back a plausible number. On a two-sentence case chrF read **100.00** where the truth is
**78.40**.

So: the local `compute_chrf` and `compute_ter` wrappers in the Lecture 6 activity notebook
can come out, and the imports go back to `torchlingo.evaluation`. Any chrF or TER figure
computed with an older version should be recomputed before it goes on a slide.

**Decode length had two values and they disagreed.** `evaluate_model` defaulted to 200
while every decoder defaulted to 100, so one model scored differently depending on which
you called — on 11.7% of one real corpus. Both are 100 now, resolved from one place.

### Still yours, and Lecture 8 is Wednesday

- **The Colab subscription announcement.** Still unsent as far as I know, and Lecture 7 is
  Monday.
- **A8's low-resource floor.** Your correction stands and it needs a decision before Oct 7:
  the 100K floor has no low-resource variant, and the students it will flag are the ones
  the course *defines* as having less than 200K. That is an assignment question, not a
  data problem.
- **The A5 audit.** Still blocked on access. `scripts/audit_bitext.py` exists on the
  private side and reports clean pair count, duplicate-source rate and longest sentence in
  subword tokens. Point it at a directory and it runs.
- **The epoch count.** Your 65-to-165 finding is not being waved away. It is being settled
  by measurement rather than split: the benchmark is being rebuilt as a ladder that reports
  per-epoch validation loss, so whether loss is still falling at 36 becomes an observation
  instead of an argument.

### One correction to my own last entry

I wrote that 0.0.8 was missing five modules. It was **seven** — `models/attention.py` was
absent too and nobody had noticed, including me when I quoted the figure. The table in
PR #53 understates it for the same reason.

---

## 2026-09-24, later

**Lecture 9 has a concrete demonstration waiting for it, and Assignment 9 has a bug.**

Eric settled the length limit: **drop any pair whose either side exceeds 100 tokens**,
where "token" means whatever the model consumes. One number replaces four, and the
512-token positional ceiling stops being something anyone has to think about. Measured
cost on the German corpus: 17,015 pairs, **1.24%**.

**Why this belongs in Lecture 9.** The rule's meaning changes when the tokenizer
changes, and that *is* Lecture 9's subject. Before BPE, a student's tokens are words.
After it they are subwords, the same sentence gets roughly 1.8x longer, and the same
rule now excludes a different set of sentences. Students can count the difference on
their own data, which is a better demonstration of what subword segmentation does than
any diagram.

Suggested framing: state the cap in Lecture 8 as "100 tokens, and for now your tokens
are words", then open Lecture 9 by pointing out that their tokens just changed, and
have them count how many pairs moved.

**The bug, which matters more.** Assignment 9 asks students to hold everything fixed
except the tokenizer. If the cap is expressed in tokens, **changing the tokenizer
changes the training set**, so the comparison has two variables and the write-up will
credit all of it to the tokenizer.

The fix is one sentence in the assignment: *choose the sentence set once, using the
subword tokenizer, and use that same set for both runs.* Then only the tokenizer
differs.

This is not hypothetical. This repository published exactly that mistake once, a
comparison that gave one model 19% more data **and** 80% more training while claiming
data was the only difference. A student will make it more easily than we did.

**One ordering wrinkle worth a decision.** Assignment 8 says "BPE for inflected
languages", but BPE is not taught until Lecture 9 on Oct 5, two days before A8 is due
on Oct 7. Either drop the mention from A8, or say explicitly that it is optional and
covered next week.

---

## 2026-09-24

**First handoff. The briefing is complete and needs seven things from you.**

`notes/handoff/briefing.md` in this directory is the standing document: the ask, the
corrections to your roadmap, the corpus measurements, and the curriculum placements.
It is long because it is meant to be read once. This log is where changes land after
that.

Actions, roughly in deadline order:

- **Get the Colab subscription requirement to students before Mon Sep 28.** Students
  are on paid Colab. Nobody has told them yet, and it is a prerequisite with a deadline
  in the same way the MTEval accounts were.
- **Audit the A5 submissions this week.** Languages were chosen in Lecture 2 and
  cleaned bitexts were delivered Wed Sep 23, so the numbers already exist. Three of
  them matter: clean pair count per student, duplicate-source rate, and longest
  sentence in SentencePiece tokens. Anyone under about 104,000 clean pairs cannot do
  Assignment 8 as written and has thirteen days to find out. See briefing Part C.
- **Put train/dev/test discipline in Lecture 8**, Wed Sep 30. It belonged in Lecture 6
  and Lecture 6 has run; Lecture 8 is where Assignment 8 is handed out, seven days
  before it is due. The measurement that makes it a slide rather than a sentence is in
  briefing Part A2.
- **Write lecture notebooks into `docs/docs/course/lecture-NN-<slug>.ipynb`** and do
  not run git. Committing, the nav entry, the CI gate and the pull request are handled
  from this side. Say what you added and which lecture it serves, and flag anything
  that should not be public so it can be routed to the private repository instead.
- **Copy the setup pattern from tutorial 2**, which was rewritten for exactly this. Two
  cells: detect Colab and install unconditionally, then verify and fail loudly. The
  reasoning behind each choice is in briefing Part A section 4. Do not copy the old
  commented-out install; that was the bug.
- **Do not hard-code library tutorial filenames into slides yet.** They will get stable
  unique names once the open pull requests land, and slides should refer to notebooks
  by name rather than by an implied lecture-number match. Tutorial 2 is the Lecture 7
  activity today, which is why the numbers were never going to line up.
- **Steve Richardson's OpenNMT config, if it is reachable.** Specifically `batch_type`
  and `batch_size`. Optional now: the epoch recommendation was derived from this
  repository's own runs instead, so this would only upgrade a recommendation into a
  faithful translation.

Two corrections to the roadmap worth knowing even if you change nothing:

- **Colab resume is verified**, so its risk 3 is closed. Coulson tested it and reported
  on PR #17 on 2026-09-23.
- **Lecture 7 carries the real Monday risk**, not resume. The install cell was
  commented out; it is fixed now, but the roadmap's "lowest-risk part of the pivot" was
  the wrong read.
