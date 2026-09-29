# Lecture 13 Assignment - Create & run NMT model with back-translated data

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Mon Oct 26, 11:59 pm. Points: 30. Roadmap v6 due date: Wed Oct 28. Status: STALE: OpenNMT configuration; due date not yet moved. Rubric tables reformatted; otherwise verbatim.*

Create another English-to-X system with back-translated data

1. Create an X-to-English system with the same data (train, validation, test – all reversed) you used for your English-to-X system, using the same OpenNMT configuration (as much as possible)
2. Take at least the same number of target sentences (e.g., 100K if you trained your previous system with 100K sentence pairs) from your cleaned data that you did not use to train your systems (assuming you did not use all of it) – do not take data from your current training, test, or validation sets! You could use data from other sources, but consider domain differences when you analyze the output.
3. Use the X-to-English system to back translate the sentences to English
4. Add the new English-X pairs to the English-X pairs you used to create your system for Assignment 8 (or 9 - whichever was best).
5. Train a new English-to-X system with this data (same configuration, test set, and validation set as before)
6. Compute and compare sacreBLEU and COMET scores (use the same test set as before) for your original English-to-X system and the new English-to-X system that contains the back-translated data

NOTE CAREFULLY: Upload to your folder in the Assignments shared folder: A written description of what you did, how much data you used originally and then how much you back translated and added, the sacreBLEU and COMET scores, any changes to the OpenNMT configuration you used for the X-to-English system, a thorough analysis of the results (Was the new system better or worse? Why?), 3 examples of sentences and their translations that were translated differently by the two systems (along with a Google back translation to English of the 3 target sentences), and the MT output of the new system for the whole test set (comparable to the MT output you uploaded for assignment 8/9).

Upload to your folder the following:
- Training set
- Validation set (should be the same as past assignment)
- Test set (should be the same as past assignment)
- MT Output (predictions from the model)

| RUBRIC: | Points |
|---|---|
| Uploaded training set, validation set (same), test set (same), and MT output | 15 |
| Describe what you did | 2 |
| Described amount of original data and added data | 1 |
| BLEU and COMET scores for the entire test set | 3 |
| Described any changes to OpenNMT configuration | 1 |
| Provided a thorough analysis of the results | 4 |
| Provided 3 examples of sentences and their translations (along with back translations) | 4 |
| Total | 30 |
