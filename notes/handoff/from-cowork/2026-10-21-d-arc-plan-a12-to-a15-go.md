# Go on the assignment arc plan, A12 to A15, and notebook submission

*From the Cowork session, Wed Oct 21 2026, about 15:10. Eric in chat: "I support the arc-plan
decisions A12-A15. I support notebook submission." The plan, with each assignment's original
outcome quoted and kept: `Roadmap/Assignment Arc Plan_v2.md` in the course folder. Read it with
`from-cowork/2026-10-21-c-sampled-and-noised-decoding.md`, which feeds A13.*

## Order, by due date

1. **A12 (live, due Wed Oct 28).** Additions **optional this term**, required next year:
   chrF beside BLEU; three draws of the examples (three seeds), mean and spread; a "nearest
   examples" condition (the *k* pool sentences most similar to each test sentence, Moslem et
   al.), which needs a pool larger than 20; NLLB as an optional comparison row where it covers
   the language. Students who have finished are not asked to rerun. The required part does not
   change, except that the completed notebook is now submitted (Cowork is adding that to the page,
   with the rubric rebalanced toward the analysis at the same 15 points).
2. **A13 (due Mon Nov 2; Lecture 13 is Wed Oct 28): the TorchLingo rewrite, needed before
   Oct 28.**
   - Path A frame: an "if ..., then ..." hypothesis, a prediction before any score, the same A8/A9
     test set, the paired bootstrap, two sentences where it helped and two where it hurt, and rerun
     instructions.
   - Monolingual data named (low-resource students have none left): the Church data's other files,
     or an OPUS monolingual set, with a sentence on domain.
   - **NLLB allowed** for the back-translation where it covers the language (one training run
     fewer); a student who trains the reverse model keeps it for A14.
   - Back-translation decoding: greedy or beam now; sampling and noised beam as options when the
     decoding hand-off lands.
   - Scoring: both systems with greedy and with beam (k = 5).
   - Optional: filter back-translated pairs by multilingual embedding similarity (A3's idea).
   - Rater agreement: partners rank 25 of each other's X-to-English outputs (English, which every
     classmate reads) in MTEval and report agreement with A11's counting.
   - The notebook is submitted.
3. **A14 (due Wed Nov 4): rewrite.** A9's split in both directions (English to X on A9's test
   set, X to English on the same set reversed; A9's training set both ways, each pair tagged with
   its target); validation mixed. Compare the bidirectional model with the one-way models the
   student has (A9, and A13's reverse model or NLLB's direction if they used NLLB), one stated
   decoder for all, paired bootstrap. A write-up question on tagging and zero-shot. Target tags in
   TorchLingo (task #99). **Its due date is still Eric's open question** (a later date, a smaller
   run, or optional this term); Cowork will relay his answer.
4. **A15 (due Wed Nov 11): a notebook, if you agree it is wanted.** The student's own MT in the
   middle (their best of A8, A9 or A13, through TorchLingo), with open components: Whisper for
   speech recognition and MMS for speech synthesis where Azure does not cover the language (Azure
   stays allowed). It measures: word error rate of the speech recognition on the 10 sentences, and
   one sentence traced through the three stages, saying which stage the error came from. The
   notebook is submitted.

## Course side (Cowork), as each lands

The Learning Suite pages, outcomes and rubrics, `notes/assignments/*-directions.md`, and the deck
slides for Lectures 12 to 16. Please reply in `to-cowork/` with the step and file names as each
notebook reaches `main`.
