# To the Cowork session

Messages from the TorchLingo repository session. **Newest entry first.** Append at the
top; never edit an entry after the fact. See `notes/README.md` for the protocol.

Standing context lives in `briefing.md` in this directory. Read that once; read this
file every time.

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
