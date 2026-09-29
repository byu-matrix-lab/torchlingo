# Lecture 14 Assignment - Create & run a bidirectional MNMT model

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Wed Oct 28, 10:00. Points: 30. Roadmap v6 due date: Mon Nov 2. Status: STALE: OpenNMT system and the 'MNMT Guide Using OpenNMT.docx' handout (repository task #99); due date not yet moved. Rubric tables reformatted; otherwise verbatim.*

Create a simple bidirectional Multilingual NMT system containing only two languages: English and your language.

1. Use the OpenNMT system you created for Assignment 8 or 9.
2. Follow the instructions in the file found on the LS Lecture 14 Content tab (MNMT Guide Using OpenNMT.docx) to create and use the multilingual training data
3. You should also read the MNMT tutorial on the Content tab and use BPE/SentencePiece or another tokenizer if needed.
4. Your training data will be twice as large, since it will have both E->X and X->E segment pairs. Make sure the pairs for each direction are randomly intermingled with each other. Train for at least 20,000 steps (twice that many, or perhaps to convergence if you can).
5. Randomly create new test and validation sets (2000 each) from the multilingual data, ensuring that there are equal numbers of sentence pairs in both directions. Separate the test set into two sets of 1000 E->X pairs and 1000 X->E pairs. The validation set should have the different direction pairs randomly mixed together. Do not mix the test set pairs.

Upload to your shared subfolder on OneDrive:
- Each test set (E->X and X->E) and its corresponding MT output
- Your complete training and validation sets
- 5 random source/target sentence pairs from each test set, with the corresponding MT output and back translations of the output (using Google or some other system) for the E->X sentence pairs.
- BLEU scores from Sacrebleu and COMET scores for the MT output from each test set.
- A written description of your process, including how many iterations you trained for and the frequency of validation, what vocabulary processing you performed, your opinion of the output quality of each test set, issues encountered, if any, and time it took to: 1) prepare data, 2) train the system, 3) run the test sets

This assignment is due in one week.

Read/Review this paper on Lecture 14 Content tab in Learning Suite: Google 2020 - Complete MNMT

| RUBRIC | Points |
|---|---|
| Uploaded each test set (E->X and X->E) and its corresponding MT output | 10 |
| Uploaded your complete training and validation sets | 2 |
| Provided 5 random source/target sentence pairs from each test set (with corresponding MT output and back translations for E->X sentence pairs) | 5 |
| Provided BLEU and COMET scores for the MT output of each test set | 3 |
| Written description and analysis provided | 10 |
| Total | 30 |
