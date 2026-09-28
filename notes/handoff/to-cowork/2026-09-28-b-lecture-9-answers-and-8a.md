# Your two Lecture 9 questions, answered; and what changed in 8a

**Baton back to you, 2026-09-28. Live: last updated 18:00 MDT, and still being added to** as the
repository side works through its list. Answers `from-cowork/2026-09-28-b-lecture-9-activity.md`.
Section 5 has one decision still pending, marked as such. Ordered by what matters first.

## 1. Your question (a): no tokenizer hook, and one change to your scope

**`NMTDataset` needs neither pre-encoded text nor a `tokenizer=` hook.** It takes whatever
vocabulary it is given, and a `SentencePieceVocab` splits raw text into pieces itself. So the A9
path is 8a's Step 6 with two added arguments:

```python
train_loader, val_loader, src_vocab, tgt_vocab = create_dataloaders(
    OUT_DIR / "train.tsv", OUT_DIR / "val.tsv",
    use_sentencepiece=True, sp_model_path=OUT_DIR / "spm.model",   # the only new lines
    use_bucketing=True, num_workers=0, config=config,
)
```

`create_dataloaders` has had that option all along, and tutorial 5 works this way. Checked on the
pretrained model: "Thank you very much" becomes `▁Thank ▁you ▁very ▁much` inside the vocabulary,
with nothing pre-split outside it.

**The scope change: the activity should not re-encode the nine A8 files for A9 to train on.**
Pre-split text fed through the default word vocabulary makes the model output pieces, and BLEU
scored on pieces is not comparable with A8's word-level BLEU, which is the comparison A9 exists
to make. So:

- **A9 trains and scores on the same raw files as A8, never re-split.** Same files and same split
  is what makes the comparison fair; the tokenizer is the only thing that changes.
- **Re-encoding stays in the activity, but only to look at:** the `<unk>` counts before and
  after, and the ten-sentence round trip. Your other three deliverables stand as you wrote them.

## 2. Your question (b): the decode is already there

**`translate_batch` decodes through the target vocabulary before it returns**, so with a
`SentencePieceVocab` it hands back ordinary text. Checked: the pretrained model returns
"Muchas gracias." with no `▁` markers. So **8a's scoring cell works unchanged for A9**: once Step
6 returns SentencePiece vocabularies, scoring decodes through them, and `test.hyp` and the BLEU
are on real text.

Neither the library change nor the paste-in snippet is needed. The one way a student can still
get it wrong is by scoring pieces themselves, for instance BLEU on what `encode` returns, which
is worth one sentence on the A9 slide: **score the translations `translate_batch` returns, never
token ids or pieces.**

**So the whole A9 change is those two arguments.** That is also the strongest form of A9's
controlled comparison, and the slide can say so.

## 3. The first two cells of every notebook changed today

Eric asked how to make the notebooks' large code cells intuitive for a new student. The rule, now
in `CLAUDE.md`: **wrap the plumbing, keep the lesson inline.** TorchLingo 0.2.2 (on PyPI) added
`torchlingo.colab.setup`, and the notebooks moved onto it:

- **The install is four lines**: `%pip install` in Colab only, then `import torchlingo`, so a failed
  install stops in that cell. Pip prints "you may need to restart the kernel"; the cell's comment
  says to ignore it on a first install.
- **The second cell is `setup(...)`**, which prints the versions and device, checks for a GPU,
  mounts Drive, and downloads data. It replaces cells of twenty to fifty-seven lines.
- **Lecture 7's step A5 now mounts Drive with `setup(drive=True)`**, the same call 8a makes.

**If a deck shows a screenshot of any notebook's first cells, it is out of date.** That covers
Lecture 7, 8a, and tutorials 1, 2, 4 and 5. Everything below the setup cell is as it was: every
unedited cell re-executed to identical output.

## 4. 8a: faster, simpler, and running from the badge

- **Bucketed by length.** Training batches group sentences of similar length, removing most of the
  padding. Measured on the same model and data: **1.86x faster per batch** (Apple GPU, synthetic
  text; Colab will differ). A cell prints each student's own padding saving.
- **Plumbing into library calls**: `split_exact`, `create_dataloaders`, `padding_report`. The steps
  A8 teaches stay written out: the cap before the split, the source-side dedupe, the contamination
  check, training and scoring. 183 lines of code became 137, and the split files it writes are
  byte-identical to the previous version's.
- **Resuming after a disconnect was broken mid-epoch on 0.2.0**; fixed in 0.2.1, which 8a now
  requires.
- **Still owed: Eric's Colab run on a real A5 corpus** before Wednesday.

**Your deck's timings.** Coulson measured both A8 candidate models on Colab GPUs, and the
results are in `notes/reports/colab-memory.md`: peak memory, and milliseconds per batch at
worst-case lengths. **The Colab numbers should replace the Mac wall clocks the deck quotes.** With
bucketing, real batches are shorter than those worst cases, so they are upper bounds.

## 5. The model for A8: **decision pending**

Memory no longer argues against the larger 56.4M model, which the a8-benchmark report shows
scoring better at A8's floor and converging in fewer than half the epochs. **Students are expected
to have paid Colab** (Eric), where they choose their own GPU: an A100, L4 or G4 held both models in
every configuration measured, and only a T4 ran out of memory. **8a now says so, whichever model
A8 uses:** its first instruction is to choose an A100, L4 or G4, not a T4. If a slide shows the
runtime dialog, it should say the same. **Eric has not decided the model yet.** This section will
be updated in place when he does; until then, plan slides around the current 11.7M model.

## 6. Tutorial 3: keep its badge unlinked

Still true, and now with a second reason. Beyond its install, tutorial 3 loads the model *tutorial
2* saved, which a separate Colab runtime never has. It is Task #166, to be done with #162 in one
change; the Lecture 10 badge can return then, and we will say so.

## Which tasks moved since your 11:20 baton

`notes/TASKS.md` is reconciled in this same change.

| | |
|---|---|
| **#121** | your two questions answered (sections 1 and 2); one scope change recorded |
| **#107, #158** | closed: 8a buckets; `num_workers` stays default, callers pass what they need |
| **#118** | measured (reports/colab-memory.md); **decision pending** (section 5) |
| **#114, #86** | closed: tutorials 4 and 5 run from Colab; `evaluate_model` tested |
| **#169** | new: wrap the plumbing, keep the lesson inline; half done (section 3) |
| **#168** | new: `scripts/student_path.sh` runs a notebook as a Colab student would |
| **#166** | narrowed to tutorial 3, now with both of its failures |

**Questions for you:** none yet. More may be added below before the baton passes.
