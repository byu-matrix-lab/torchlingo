# Outcomes quoted, and checked; Lecture 19's notebook is built

*From the repository session, written Wed Nov 25, late; last updated Fri Nov 27. Answers
`from-cowork/2026-11-15-n-priorities-and-a14-line.md`. Your branch is reviewed and carried, with
this reply, in one pull request; a second, on Nov 27, added tutorial 10. Five things are asked
of you:*

1. *carry two outcome changes Eric made on Nov 26, to Lectures 8a and 13, wherever the course
   side shows the outcomes (section 1);*
2. *link Lecture 19's notebook from the deck and the page; it is promoted (section 3);*
3. *list tutorial 10, on IBM Model 2, as reading on the Lecture 18 page (section 4);*
4. *for the next offering, build one Lecture 8b slide that Eric approved, on why a Transformer is
   not recurrent (section 5);*
5. *for the next offering, build its follow-on in Lecture 12, on how generating text brings
   recurrence back, which Eric also approved (section 6).*

## 1. The notebooks' outcome quotes: done, and now checked

All 25 stale quotes are updated, course notebooks and tutorials alike, and Lecture 18's purpose
cell carries its new title. Each notebook quotes, word for word, the parts of its lecture's new
outcome that it serves; where it serves the whole outcome, it quotes the whole. Lectures 12 and
13, which paraphrased, now quote.

**A check holds them there.** The notebook metadata check now fails any italic quote in a
notebook's "These serve" line that its lecture's outcome does not contain, so the next rewrite of
an outcome is caught where you caught this one by hand. It found one slip of mine on its first run.

Where the new outcome changed what a notebook can claim:

- **Lecture 8a's new outcome, as you wrote it, named only the memory budget** among the things
  the A8 notebook teaches; reading training against validation loss, and checkpointing and
  resuming a long run, had dropped out, and with them tutorial 3's only outcome. Eric restored
  both (below).
- **Tutorial 6, diagnosing failures**, now also quotes Lecture 7's held-out clause, *why a model
  is judged on data it never trained on*.
- **Lecture 7's notebook** quotes the held-out and how-a-network-learns clauses, not the paper
  review, which it does not do; **Lecture 11's** quotes the LLM judge and the ranking clauses, not
  error spans, which it only mentions.

**Two outcome changes, Eric's, Nov 26.** I asked Eric about the two gaps above, and one wording
point, and he settled all three; the roadmap rows are changed in this pull request.

- **Lecture 8a gains two clauses**, since students do both in A8: *explain how to read training
  against validation loss to tell when a model stops improving; explain what a long run must save
  so that it can be resumed, or reloaded as the model you trained*. The A8 notebook now quotes
  all three of its clauses again.
- **Tutorial 3 serves an outcome again**, the second of those clauses, which Eric asked for: what
  a checkpoint leaves out (the vocabulary) is what a reload must restore. **Tutorial 6** quotes
  the first, beside Lecture 7's held-out clause.
- **Lecture 13's outcome names the technique**, per your rule: *when back-translated data helps or
  hurts*, not *synthetic data*. The A13 notebook quotes it.

**Asked of you:** carry both rows wherever the course side shows the outcomes (the decks'
outcome slides, the Learning Suite pages, the syllabus), word for word from the roadmap.

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

**Asked of you:** link it from the Lecture 19 deck and page. It is promoted (Nov 27), so the
badge at the top of the notebook opens it in Colab now.

## 4. Also on this branch

- **A15's roadmap row** is rewrapped, as you suggested.
- **Section 12's dates:** no notebook or doc in the repository names the old Final Project or
  Paper Review dates (checked).
- **Tutorial 10, *Does Word Position Help? IBM Model 2*, is built** (Nov 27), as Eric asked:
  `docs/docs/tutorials/10-ibm-model-2.ipynb`, reading, about 25 minutes, CPU only, no points. On
  fifteen English-French pairs, not Lecture 18's Spanish noun phrases, Model 2's learned diagonal
  settles *le jeune renard* (the young fox) and gets *la chouette grise* (the grey owl) wrong,
  confidently; students predict that first, and predict again before two more owl sentences fix
  Model 1 and not Model 2. It closes on how attention weighs position with the words instead,
  and on choosing a word aligner for a Final Project. The tutorial map has its node, leading to
  the Final Project; regenerate the map's picture from `notes/roadmap/generated/tutorial-map.mmd`.

  **Asked of you:** list it on the Lecture 18 page as reading for after class, and regenerate the
  tutorial map on the Assignments page.

## 5. One Lecture 8b slide, for the next offering: attention never reads back and down

**Asked of you, Eric approved it on Nov 27:** one slide in Lecture 8b, after slide 27,
*Replacing the Recurrence with Self-Attention*. It comes from Eric's research session on
recurrence, which put the question students ask: *why do we say a Transformer is not recurrent,
when attention at word n reads earlier words?* Slide 27 answers half of it, across the words:
one step between any two words, and parallel training. This slide adds the other half, up the
layers. Lecture 8b has run this term, so it is for next fall; nothing changes before then. The
slide's text, yours to fit to the deck:

**Title:** Attention Reads Back and Across, Never Back and Down

**The picture:** a grid, layers going up (input, layer 1, layer 2) and words going right (x1 to
x4). Every arrow goes up one layer, either straight up or up and to the right. None goes from
the top of one word down into the bottom of a later one.

**The points:**

- **An RNN has that downward arrow.** Step t's output is step t+1's input, so the chain of
  steps grows with the sentence: training waits for each step in turn, and the gradient runs
  back through all of them, which is where it vanishes or explodes.
- **A Transformer layer reads only the layer below.** At word n, attention reads the previous
  layer's vectors (in the decoder, at words up to n; in the encoder, at every word), and never
  a later layer's output at an earlier word.
- **So the depth is fixed.** Any path through the model climbs at most its number of layers,
  however long the sentence. In our encoder-decoder, cross-attention reads the encoder's top
  layer, so the longest path is the encoder's layers plus the decoder's: fixed all the same.
- **The trade:** a fixed number of steps in sequence, in exchange for training every word at
  once, and one step between any two words.

**A footnote, if there is room:** when the decoder translates, each word it writes becomes its
next input. Generating brings back one downward arrow, one word wide.

## 6. Its follow-on in Lecture 12, for the next offering: generating brings recurrence back

**Asked of you, Eric approved it on Nov 27:** one slide in Lecture 12, after slide 14,
*Translation as Continuation*, which is where students meet a decoder-only model continuing the
text. It completes section 5's slide: Lecture 8b says a Transformer's depth is fixed, and this
says what generating does to that. It also explains the reasoning mode that slides 17 and 18 ask
students to turn off.

**Title:** Continuing the Text Brings Recurrence Back, One Token Wide

**The points:**

- **Lecture 8b: no arrow runs back down.** Within one pass, every path climbs the layers once,
  so the depth is fixed.
- **Generating adds the arrow back.** The token the model writes at step n is its input at step
  n + 1. So a generating decoder is recurrent, but all that passes from one step to the next is
  that token: about 17 bits from a vocabulary of 128,000, against the thousands of numbers in
  each layer's vector at that position.
- **Chain of thought spends that arrow.** Each token a model writes before its answer is one
  more pass through all its layers, so a reasoning or thinking mode buys more steps in sequence,
  and costs the time of each. That is why slides 17 and 18 turn it off: the comparison is then
  of translation alone.

No pointer to outside research, per Eric's decision of Nov 17 on your course task for recurrence
against attention in the 8a to 8c decks; sections 5 and 6 are the framing and placement that
task was waiting on from this side.

## Tasks

On yours: two new course tasks, the Lecture 8b slide in section 5 and the Lecture 12 slide in
section 6, both for the next offering;
Lecture 19's deck and page can link its notebook, and Lecture 18's page can list tutorial 10.
On mine: the outcome quotes and the bullets are done and off; Lecture 19's notebook and
tutorial 10 are done and off; and the old audit of the course's sequencing and outcomes is down
to one question, whether Joey NMT belongs in the syllabus, since the outcomes are now Eric's.
