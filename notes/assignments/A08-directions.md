# Assignment 8: Create and run an NMT model with your cleaned data

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Wed Oct 7, 10:00. Points: 25. Roadmap v6 due date: Wed Oct 7. Status: STALE: Learning Suite still carries the OpenNMT text (20,000 steps, T4); the deck's A8 slides are current. Rubric tables reformatted; otherwise verbatim.*

## Current directions (the Lecture 8a deck, which is the authority)

*Copied from the Lecture 8a deck (slides "Assignment 8: What To Do", "What To Submit", "AI Use
on Assignment 8"), 2026-09-29. The deck is the authority; this file is a transcript.*

**Due Wednesday, October 7, at 10:00.** Start this week; it is the largest assignment in the
course.

## What to do

1. **Split your cleaned data into training, validation and test sets.** At least 100K training
   pairs, 2K validation, 2K test. If your cleaned data has less than that, use all of it and say
   so in your write-up.
2. **Deduplicate on the source side, split by source group, and verify no source appears
   twice.** This is the step that decides whether your BLEU means anything.
3. **Drop pairs where either side exceeds 100 tokens, and shuffle.** For now a token is a word.
   Subword tokenization arrives in Lecture 9.
4. **Train an English-to-X model with TorchLingo, about 35 epochs; keep going if validation
   loss is still falling.** The course model: d_model 512, 8 heads, 6 + 6 layers, feed-forward
   2048, about 56M parameters at an 8K vocabulary (more at a word vocabulary; the notebook prints
   your count). The best checkpoint by validation loss is kept, so running long costs time, not
   quality. Use an A100, L4 or G4 runtime, not the free T4.
5. **Checkpoint to Google Drive so a dropped Colab session resumes instead of restarting.** This
   run is long. Assume the session will drop at least once, because it will. Pass your validation
   set as `val_loader` with `save_dir`, so the best checkpoint is kept and not merely the last one.
6. **Translate your test set and score it with SacreBLEU.** Score all 2,000 sentences at once,
   the same rule as Assignment 6.

Assignments 9, 13 and 14 all rebuild this system. Whatever you get wrong here, you carry for a
month.

## What to submit

**To your subfolder in the shared folder**

- All three splits, source and target, six files in total.
- Your model's output on the test set.
- Ten random test pairs with the MT output and a back-translation of that output into English.
  Any MT system will do for the back-translation.

**To Learning Suite**

- Your SacreBLEU score over the whole test set.
- A description of your process: epochs trained, tokenization, decoding strategy, architecture
  settings, and how long it took.
- Your own opinion of the output quality, and what you think is limiting it.

**If you had to re-clean your data to get here:** upload the re-cleaned data and your updated
pipeline code as well, with new filenames. The cleaned corpus is a deliverable that three more
assignments depend on, so the copy we have should be the one you actually trained on.

## AI use on Assignment 8

Settled in Lecture 1: assignments are done individually; ask for helpful ideas, do not share
code; you may use AI to remind you how a library function works; do not use any kind of AI to
write your algorithm or your code.

- Fine: asking what a TorchLingo config field does, or how to read a traceback.
- Yours: the splitting, the verification, the training decisions, and the judgment about what
  is limiting your output.
- Watch for: a model will happily write you a split that shuffles rows. That is the bug this
  whole lecture is about.
- Either way: you should be able to explain every line you submit.

TorchLingo is new enough that a model asked about it will confidently invent an API. If the
answer does not match the documentation, the documentation is right.

## Numbers the kickoff notebook quotes from this page

| constant | value | from |
|---|---|---|
| training floor | 100,000 pairs | step 1 |
| validation / test | 2,000 / 2,000 | step 1 |
| length cap | 100 tokens either side | step 3 |
| epochs | about 35; continue while validation loss falls | step 4 |
| model | d_model 512, 8 heads, 6 + 6 layers, d_ff 2048 | step 4 |
| decoding | student's choice, stated in the write-up | submit |

## What Learning Suite says today (last year's text; to be replaced)

Create an English-to-Target NMT system using OpenNMT as follows: (NOTE: Due in one week!)

Using segment pairs from your *cleaned* data:
- 100,000 (at least) – training set
- 2,000 – test set (no overlap with training/validation)
- 2,000 – validation set (no overlap with training/test)
- Make sure you randomize the order of the segments in these three data sets and consider using the BPE option if your language is highly inflected to avoid a lot of <unk> tags in your output
- Training steps: 20000 (at least, more if you want to try to improve output quality)
- You can use a slower GPU (e.g., T4) to avoid using up your compute units
- Calculate BLEU score using sacreBLEU

NOTE: You must upload to your subfolder in the Assignments shared folder:
- Training, validation, and test sets (source and target for all three)
- MT output for the test set
- 10 random source/target sentence pairs from the test set, with the corresponding MT output and back-translations to English of the MT output. (You can use Google Translate, any other MT system, or human translate them.)
- BLEU score from sacreBLEU for the entire test set
- A written description of your process, including the OpenNMT configuration settings you used (# of steps, tokenization, architecture, etc., if applicable), your opinion of the output quality, issues encountered, if any, and time it took to: 1) prepare data, 2) train the system, 3) run the test set

>>>If you had to re-clean your data before building this system, upload the re-cleaned data, along with your updated, documented cleaning pipeline code, to your folder with updated filenames to not overwrite the existing files there

| RUBRIC: | Points |
|---|---|
| Uploaded training set, validation set, test set, and MT output | 10 |
| Uploaded 10 random pairs w/ MT output and backtranslations | 5 |
| BLEU score for the entire test set | 2 |
| Described OpenNMT configuration settings | 1 |
| Described opinion of output quality | 2 |
| Described issues encountered | 2 |
| Described time it took to: 1) prepare data, 2) train the system, 3) run the test set | 1 |
| Overall quality of NMT system & output (allowing for limited data & training) | 2 |
| Total | 25 |
