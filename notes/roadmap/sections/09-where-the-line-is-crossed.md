### Where the line is crossed

**The rule for every assignment notebook** (Eric, Oct 13): the student writes the steps that are
the method, as short functions that raise `NotImplementedError` until written; each is checked on
a case whose answer is known and then by the library's own checks on the real data
(`check_contamination` where a split or a training set is built), so the check tells the student
when they have it right. The notebook supplies what is not the lesson: installs, loading,
training calls, decoding, scoring. Before each long run the student writes a prediction, and the
write-up asks them to reconcile it with the result. The test throughout: at least one thing the
student had to write or choose, and at least one sentence that cannot be written without having
read the output.

Where each assignment stands against it:

| assignment | the student writes | checked by |
|---|---|---|
| A7 | the held-out phrase, a prediction, the contrast | the report, which shows a missing prediction |
| A9 | `unknown_rate`; `VOCAB_SIZE` with a reason; a BLEU prediction | a known-answer case; Part B reloads the saved settings |
| A10 | `comet_inputs`; a prediction | known cases, including lists of different lengths |
| A11 | `comet_inputs` again; `pairwise_agreement`; a prediction | a known case; your files' alignment |
| A12 | `split_pool_and_test`, `build_prompt`; a prediction | known cases; `check_contamination` on the split |
| A13 | `reverse`, `combine`; a prediction | known cases; `check_contamination` on both held-out sets |
| A14 | to be built to the rule: tagging the directions and intermingling them | |

**The one place still open: the A8 kickoff does the part the assignment says is the student's.**
The A8 slide on AI use says *yours: the splitting, the verification, the training decisions, and
the judgment about what is limiting your output.* The kickoff performs the cap, the dedupe, the
split and the verification, and the student edits two file names. That was deliberate, since the
run has to be going when class ends, and A8 has run for this term. For the next offering, the
rule above applies: the cap and the source-side dedupe become the student's functions, with the
contamination check as the oracle; the split, the model and the training call stay as given;
and a prediction of ln V and the first logged loss goes in before training.
