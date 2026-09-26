# From the Cowork session

Messages from the Cowork session that is rebuilding the CS 479 decks and writing the
in-class notebooks. **Newest entry first.** Append at the top; never edit an entry
after the fact. See `notes/README.md` for the protocol.

The protocol was set up on 2026-09-24, after the first exchange had already happened by
other means. (The line that used to sit here saying nothing had arrived yet is no longer
true; left noted rather than silently deleted.)

---

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
