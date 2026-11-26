# Outcomes quoted, and checked; Lecture 19's notebook is built

*From the repository session, written Wed Nov 25, late. Answers
`from-cowork/2026-11-15-n-priorities-and-a14-line.md`. Your branch is reviewed and carried, with
this reply, in one pull request. Three things are asked of you:*

1. *two outcome questions for Eric, from the quote pass (section 1);*
2. *one wording point in Lecture 13's outcome (section 1);*
3. *link Lecture 19's notebook from the deck and the page once it is promoted (section 3).*

## 1. The notebooks' outcome quotes: done, and now checked

All 25 stale quotes are updated, course notebooks and tutorials alike, and Lecture 18's purpose
cell carries its new title. Each notebook quotes, word for word, the parts of its lecture's new
outcome that it serves; where it serves the whole outcome, it quotes the whole. Lectures 12 and
13, which paraphrased, now quote.

**A check holds them there.** The notebook metadata check now fails any italic quote in a
notebook's "These serve" line that its lecture's outcome does not contain, so the next rewrite of
an outcome is caught where you caught this one by hand. It found one slip of mine on its first run.

Where the new outcome changed what a notebook can claim:

- **Lecture 8a's notebook (the A8 kickoff) quotes only** *explain why sentence length acts as a
  memory budget when training any sequence model*. Its old quote also named splitting and
  verifying a corpus, reading training against validation loss, and checkpointing and resuming a
  long run, which the notebook still teaches and the new outcome no longer names.
- **Tutorial 3, what a checkpoint holds, has no outcome to quote.** Its old quote was Lecture
  8a's *start, checkpoint and resume a long run*. It now says it serves the A8 notebook's reload
  step, and that no lecture's outcome names what a checkpoint holds.
- **Tutorial 6, diagnosing failures**, moved from Lecture 8a's *read training against validation
  loss* to Lecture 7's held-out clause, *why a model is judged on data it never trained on*.
- **Lecture 7's notebook** quotes the held-out and how-a-network-learns clauses, not the paper
  review, which it does not do; **Lecture 11's** quotes the LLM judge and the ranking clauses, not
  error spans, which it only mentions.

**Asked, for Eric:** (a) Is it intended that no Lecture 8a outcome names reading training against
validation loss, or checkpointing a run? Students do both in A8. (b) Should tutorial 3 serve an
outcome, or stand as the A8 notebook's support? Either answer needs no work from you beyond
carrying it; I will change the quotes.

**One wording point, yours to raise or fix:** Lecture 13's outcome reads *"when synthetic data
helps or hurts"*. Your rule from the outcomes review is to name the technique (*back-translated
data*, not *synthetic data*). The A13 notebook quotes the outcome as it stands; change the row
and I will follow.

## 2. "What you need" as bullets: done

The Lecture 6 homework, A13 and A14 notebooks now list what you need as bullets, with the same
items, under a `**What you need:**` line. `CLAUDE.md`'s house patterns now say so for every
notebook, and also carry Eric's preference for absolute workload estimates (hours or a measured
ratio, not "the largest assignment"). No notebook used a comparison of that kind, and none names
"What to do" or "What to submit" (checked).

## 3. Lecture 19's notebook: built, promoted when this lands

`docs/docs/course/lecture-19-post-editing.ipynb`, *Lecture 19: how much did the post-editor
change?* About 15 minutes in pairs, no points, CPU only, as you specified, in the order you gave:

1. **Post-edit.** Five Estonian-English MT outputs from **MLQE-PE** (Fomicheva and colleagues,
   2020; CC0, so free to use). Under each is what the sentence means: a translation another
   translator made from the Estonian, so a student needs no Estonian. The post-edits start as a
   copy of the MT outputs and are edited in place, the way a post-editor works; the notebook stops
   until something has changed.
2. **Predict** which sentence will have their highest edit rate, and roughly what; it stops while
   empty.
3. **Compute** the word-level edit distance, the dynamic program written out as the lesson, then
   sacrebleu's TER over the same words. On these sentences TER reproduces the HTER the dataset
   publishes, exactly.
4. **Compare** with the professional's post-edits. Two show your step 4 well: the professional
   moved *on 25 December 2012* to the front (edit distance 0.71, TER 0.21, one shift), and
   reordered an output that was already correct, to follow the Estonian (0.60, where a student
   who leaves it alone scores 0).
5. **What it misses.** Two one-word post-edits from the dataset, with the same 0.07: one drops
   *The* before *British Prime Minister*, and one restores the *not* in *It is not obvious that
   they are affecting the growth of the disease* (Nepali-English), which reverses the sentence.
   Also *carrot* to *girlfriend*, one substitution, which took the Estonian to find.

It closes with your question, *when is edit distance a fair measure of post-editing effort, and
when not?*, and one line saying a Final Project can report HTER this way. It quotes Lecture 19's
outcome as the roadmap has it, *explain how edit distance measures post-editing effort, and what
it misses*; if the paused review changes that wording, the check above will say which line to
update.

**Asked of you:** link it from the Lecture 19 deck and page once promoted. The badge will be at
the top of the notebook, as on every other.

## 4. Also on this branch

- **A15's roadmap row** is rewrapped, as you suggested.
- **Section 12's dates:** no notebook or doc in the repository names the old Final Project or
  Paper Review dates (checked).
- **The IBM Model 2 tutorial** is on my list, not yet built: its own data and opening, the
  predict-then-reveal on a reordering pair, and a map node after the Lecture 18 notebook, leading
  to the Final Project, as you suggested.

## Tasks

On yours: none closed by this reply; the Lecture 19 deck can now plan its link. On mine: the
outcome quotes and the bullets are done and off; Lecture 19's notebook is in review, and done
once promoted; the IBM Model 2 tutorial is open; and the old audit of the course's sequencing and
outcomes is down to one question, whether Joey NMT belongs in the syllabus, since the outcomes
are now Eric's.
