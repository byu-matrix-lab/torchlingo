# To the Cowork session

Messages from the TorchLingo repository session. **Newest entry first.** Append at the
top; never edit an entry after the fact. See `notes/README.md` for the protocol.

Standing context lives in `briefing.md` in this directory. Read that once; read this
file every time.

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
