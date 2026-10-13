### Where the line is crossed

Four places, in order of how much they matter. Each ends with what is proposed; the decisions
are Eric's and sit on the task list.

**1. The A8 kickoff does the part the assignment says is the student's.** The A8 slide on AI
use says: *Yours: the splitting, the verification, the training decisions, and the judgment
about what is limiting your output.* The kickoff performs the cap, the dedupe, the split and
the verification in library calls, chooses every training decision, and provides the scoring
cell. The student edits two file names. A student can submit a BLEU score, six split files and
"a description of your process" copied from the notebook's own markdown without having
understood a step, and A9, A13 and A14 then re-run the same path. The executable path was a
deliberate choice (the run has to be going when class ends, and it was never tested on a
student corpus before Sep 30), so the fix is not to withdraw it. Proposed, in three parts:
(a) *this week, no notebook change*: the A8 write-up, whose Learning Suite text is being
rewritten anyway, replaces "a description of your process" with questions a copy cannot
answer: your ln V and your first logged loss, and what the gap between them means; how many
pairs the cap removed and how many the dedupe removed, and why the dedupe is on the source
side; which epoch's checkpoint you scored and why it is not the last one; which of the three
loss-curve shapes was yours. (b) *predict-before-you-run lines*, which the Lecture 4, 5 and 6
activities already use and 7, 8a and 9 do not: before Step 1, write down the fraction you
expect the cap to drop; before Step 6, predict ln V from your vocabulary size; before Step 7,
predict the first logged loss. One sentence of markdown each. (c) *for A9 and the next
offering*: Steps 1 and 2 (cap, dedupe) become TODO cells, with Step 4's contamination check as
the oracle that tells the student whether they got it right. A student who wrote sixteen
cleaning steps in A5 can write a length cap and a source-side dedupe; the library version stays
in a collapsed cell for anyone stuck. Steps 3 to 7 stay as given: `split_exact`, the
verification, the model and the training call are the library's job and the run must start in
class. The scoring cell keeps the decoding loop and leaves the `corpus_bleu` call to the
student, who wrote it in the A6 homework.

**2. A9 is a paste.** The Lecture 9 notebook ends by printing the three settings, verbatim,
and A9 is "paste them into Step 6 and run again". The outcome, that subwords change what the
model can say and that a one-variable comparison attributes the difference, can be satisfied
without reading a line. The printed settings should stay (the `sp_model_path` history argues
for not making students guess an API), but the understanding can be demanded around them.
Proposed: `unknown_rate` in Step 1 becomes a TODO (four lines; writing it is the lesson of
Step 1); the student chooses `VOCAB_SIZE` from 4K, 8K or 16K and says why in one line, with
8K as the default for anyone who does not want to choose; and the A9 write-up asks for a
prediction of the BLEU direction and rough size *before* the retrain, then the two `<unk>`
rates and the piece-per-word ratio, and a sentence reconciling the prediction with the result.

**3. The Lecture 7 toy model runs itself.** Part B's held-out phrase is fixed, and the second
held-out experiment ("Good night", where there is nothing to recombine from) is optional. A
student can produce the Part C printout without reading Part B. A7's two graded sentences are
the existing defence and they are the right one. Cheap improvement for the next offering (A7
is already in): the student chooses the held-out phrase, predicts in one line whether the
model will recombine it, and runs both the chosen phrase and "Good night", so that the report
contains one decision and one contrast rather than a fixed printout.

**4. A13 and A14 do not have notebooks yet, and the pattern is set.** If they are built as
kickoffs, a student will have run the same executable path four times by November. The rule
proposed for both: the scaffold supplies the pieces the student has already used
(`parallel_txt_to_dataframe`, `split_exact`, `check_contamination`, the training call,
`translate_batch`) and leaves the assembly to the student: reversing the corpus and training
the X-to-English model, back-translating the held-out target sentences, concatenating and
re-splitting without contaminating the A8 test set (A13); tagging directions and intermingling
them (A14). The verification cells stay, because they are what tell the student whether the
assembly is right.

Two things the in-class notebooks do well and the homework ones should borrow. *Predict before
you run* (Lecture 4's "predict the two numbers", Lecture 6's "predict the BLEU score before
you run the cell") costs nothing and makes auto-pilot visible to the student. *Report back*
questions carry the learning in the room, but outside the room the only lever is what is
graded, so the principle for every graded notebook-based assignment is: at least one number
the student had to choose, and at least one sentence that cannot be written without having read
the output. A7 meets it. A8 and A9 do not yet.
