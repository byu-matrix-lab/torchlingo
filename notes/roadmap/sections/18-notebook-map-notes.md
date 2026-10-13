### Why lectures that have run are still listed

**Lectures 1 to 6 have already run.** They are listed anyway, for three reasons: a student
revisiting them needs to know what to open, assignments later in the course refer back to
them, and a Fall 2027 offering should not have to rediscover the mapping. Where a tutorial
*would have* improved a past lecture it is marked `reference` rather than silently dropped.

### What this table cannot say, and so is written here

The generated table lists notebooks that exist. Absence is the interesting part, and a
notebook that has not been written cannot declare anything:

- **Lecture 9 has no notebook and needs one.** That is #121, which carries the dates.
- **Lecture 14 has no notebook**, and the current handout is a Word document written for
  OpenNMT. That is #99, and #123 is the prior question of whether the code path even works.
- **Lecture 6 will gain a tutorial it cannot show yet.** Tutorial 2 was written for exactly
  this ground and is not merged (#88). Once it lands it will declare Lecture 6 itself, as
  reading *after* Lecture 8, since Lecture 6 has run.
- **Twelve of the twenty-three lectures pair with nothing** — 1, 2, 11 to 18, 20 and 21. Which
  is fine: not every lecture wants a notebook, and inventing one to fill a row would be the
  redundancy this document already warns about.

**Two notebooks declare no `leads_to`, deliberately**, and it is recorded here so nobody reads the
blank as an oversight and fills it in. Settled 2026-09-28, closing #145.

- **`04-attention-and-alignment`** explains attention and the Transformer. It gives a student no
  head start on A8's deliverable, which is a trained model on their own corpus — it makes the
  training make sense, which is a different thing.
- **`lecture-04-regex-refresher`** teaches regular expressions, which A4's cleaning is *written
  in*. A prerequisite skill is not a head start either.

The rule both follow is the one #139 established against tutorial 1's Lecture 9 claim: **a notebook
that teaches the prerequisite should not be recorded as covering the thing.** A visible blank is
better than a flattering map. Every other notebook now names an assignment.

One row appears twice on purpose. **Tutorial 4 serves Lectures 8a and 8b and will not be
split**, which reverses #140 **and an agreement with Cowork**.

That second half was stated wrongly here first, and the correction is the useful part. The claim
was that nothing sent to Cowork needed reversing, on the grounds that the 2026-09-27 hand-off
records this notebook as `["8a", "8b"]`, reading. True of that file, and irrelevant: the split was
Question 8 of the *previous* entry, and their reply agreed to it in terms — *"Split tutorial 4.
Parts 6 to 8 are architecture content and they belong to 8b."* Checking only the newest hand-off
missed an agreement that was two files back, in the archive.

Splitting it was argued from the heading list, which reads like two notebooks: Parts 1 to 6 are
LSTM attention, Parts 7 and 8 are the Transformer. Reading it settles the question the other
way. **Part 7 is titled "You have already seen the Transformer's mechanism"**, and its whole
move is that the attention the reader just ablated in Parts 1 to 6 *is* that mechanism. Split
the file and the 8b half opens by invoking an experiment its reader never ran.

The cost of not splitting is real but small: Parts 1 to 6 are nine of the ten code cells and
need nothing, so they would have been CI-executable on their own (#154), and the 8b half would
have been a one-code-cell notebook. Neither is worth the notebook's best transition.

Two incidental corrections, since #140's own description had them wrong: the seam is *after*
Part 6, not at it, and Part 6 — Bahdanau versus Luong — is Lecture 8a's material, not 8b's.
