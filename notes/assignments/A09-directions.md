# Lecture 9 Assignment - Run SentencePiece with NMT systems

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Mon Oct 12, 10:00. Points: 15. Roadmap v6 due date: Wed Oct 14. Status: STALE: OpenNMT vocab file, last year's dates in the text; due date not yet moved. Rubric tables reformatted; otherwise verbatim.*

Include SentencePiece tokenization in your Assignment 8 MT system, retrain the system, rerun it on the same test set as before, and compute the BLEU score for the output.

- Follow these Instructions for using SentencePiece (link also on Lecture 9 LS Content tab).
- If you already used SentencePiece, then train and run your system again without it.
- Write an analysis of at least one paragraph comparing the BLEU scores of your two systems with and without SentencePiece as well as your own observations of the difference in quality between the two systems

Upload to your folder in the Assignments shared folder:
- Your test set (should be the same as for Assignment 8)
- Your detokenized MT output for the test set.
- Your OpenNMT vocab file used when running with SentencePiece
- Any non-default options you used with SentencePiece (e.g., vocab_size=16000 instead of 8000)
- BLEU scores from SacreBLEU for both versions
- Your written comparison of the scores and your own observations of the quality difference.

This assignment is due on Wednesday, Oct. 8, so you can do it right after you submit Assignment 8. Or, you can do it while you are working on Assignment 8 if you want, but Assignment 8 is still due on on Monday, Oct. 6, so don’t let it slow down your submission on that date.

| RUBRIC | Points |
|---|---|
| Uploaded test set (same as Assignment 8) | 1 |
| Detokenized MT output for the test set | 3 |
| Uploaded the vocab file used with SentencePiece | 3 |
| Any non-default options you used with SentencePiece | 1 |
| BLEU scores for both versions of MT output for the test set | 4 |
| Written comparison of the scores and observations of the quality difference. | 3 |
| Total | 15 |
