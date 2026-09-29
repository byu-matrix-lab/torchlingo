# Your two Lecture 9 questions, answered; and what changed in 8a

**Baton back to you, 2026-09-28. Live: last updated 18:50 MDT, and still being added to** as the
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

**Correction, found while building the Lecture 9 notebook: A9 needs a third setting,
`max_decode_length`.** Decoding stops at 100 tokens unless told otherwise. That matched A8's
100-word cap, but in pieces it is far shorter: on the repository's real English-Spanish corpus,
the longest 100-word target is 171 pieces. Leave it at 100 and A9 cuts every long translation
off mid-sentence, and its BLEU drops for a reason that has nothing to do with subwords. The
Lecture 9 notebook measures each student's longest target and prints the value to use.

**So the whole A9 change is three settings in 8a's Step 6**, all printed for the student by the
Lecture 9 notebook: `use_sentencepiece=True` and `sp_model_path` in `create_dataloaders`, and
`max_decode_length` in `Config`. Same files, same split, same training and scoring cells. That is
also the strongest form of A9's controlled comparison, and the slide can say so.

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

## 5. The model for A8: **decided, the larger one**

**Eric's decision, 2026-09-28: A8 uses the 56.4M configuration**: `d_model` 512, 8 heads, 6 + 6
layers, feed-forward 2048, the original Transformer paper's architecture. The a8-benchmark report
shows it better at A8's 100,000-pair floor (17.79 against 15.95 BLEU) and converging in about 30
epochs where the 11.7M model needed 65. Memory no longer argues against it: students are expected
to have paid Colab, where they choose their GPU, and an A100, L4 or G4 held it in every
configuration measured. 8a's first instruction says to choose one of those, not a T4.

**Three things that change for your slides and the handout:**

- **The model slide**: 512 wide and 6 + 6 layers, not 256 and 3 + 3. "56.4 million parameters" is
  its size at an 8,000-piece vocabulary; at a word vocabulary the embeddings are larger (about 109M
  at the real corpus's 36,500 and 45,000 words), and the notebook prints each student's count.
- **Epochs: the handout's "60 to 70" becomes "about 35 epochs, and keep going if validation
  loss is still falling at the end"**, not a firm range. 8a uses 35. The evidence is one run: the
  56.4M model converged in about 30 epochs at 100,000 pairs, but with a subword vocabulary, where
  A8 uses words (about 109M parameters, convergence unmeasured), and an epoch count does not
  carry across corpus sizes. The benchmark itself recommends training until validation stops
  improving. Stopping too early is the risk; running long is safe, because the trainer keeps the
  best checkpoint by validation loss, so extra epochs cost time, not quality. **Eric owns the
  handout; this is flagged to him too.** The roadmap's "60 to 70" (lines 255, 461, 502, 546) is
  in your half and needs the same change.
- **Run time**: hours, on the GPU the student picks. The slide should not quote the old model's
  "just under two hours".

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

**Questions for you:**

1. **Please send copies of the Project Directions for every assignment**, into
   `notes/handoff/from-cowork/` or a path you name. The notebooks defer to "the handout" (8a's
   thresholds cell, its `EPOCHS` comment), and nobody on this side has ever seen it, so the
   notebooks cannot be checked against it. The first use: Section 5's epoch change needs the
   exact line in A8's directions that says "60 to 70".

More may be added below before the baton passes.
