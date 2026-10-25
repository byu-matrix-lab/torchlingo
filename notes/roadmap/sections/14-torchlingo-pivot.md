## The TorchLingo pivot: what is settled

The course trains models with TorchLingo, an educational PyTorch NMT library built in the
MATRIX Lab. Five assignments train a model: the toy model at Lecture 7 (A7), A8, A9, A13 and
A14. They are cumulative in the artifact, so this is one crossing rather than five, and every
assignment after A9 starts from A9. All five have TorchLingo notebooks, A14's since Oct 22.
A14's Learning Suite text still describes OpenNMT, until it is rewritten against the notebook;
A13's was rewritten on Oct 22.

**A8's settings**, the course model: d_model 512, 8 heads, 6 + 6 layers, feed-forward 2048 (the
2017 base configuration, 56.4M parameters at an 8K vocabulary); a 100-token cap applied before
the split; about 35 epochs, and keep going while validation loss falls; checkpoints to Drive; an
A100, L4 or G4 runtime, not a T4. The measurements behind each follow.

**What a 100K-pair run produces, measured.** At the first A8 configuration (German bitext,
100K/2K/2K, d_model 256, 8 heads, 3+3 layers, 11.7M parameters): 36 epochs gives 11.46 BLEU in
64.7 minutes and is still improving; 65 epochs gives 14.48, converged, in 112.7 minutes. That is
why the handout said 60 to 70 epochs until Sep 28. The benchmark's own recommendation is to
train until validation stops improving, because an epoch count is not corpus-size invariant,
and the students that bites are the low-resource ones. Wall clocks are Apple Metal and do not
transfer to Colab.

**Capacity against data, measured.** The same sweep at 56.4M parameters: at 25K and 50K pairs
the larger model is no better; at 100K it scores 17.79 against 15.95 **and converges in 30
epochs where the smaller needs 65**; at 800K the gap is 6.85 BLEU. So A8's model is leaving
quality on the table at its own floor, and the larger configuration would be both better and
cheaper in epochs. **Decided Sep 28 (Eric): A8 uses the 56.4M configuration.** Memory no
longer argues against it: Coulson measured both models on a Colab A100
(`notes/reports/colab-memory.md`); the largest peak, 16.35 GiB at a 50K word vocabulary and
180-token batches, fits an L4 or A100 and not a T4, and students on paid Colab choose their
GPU. Time per batch at worst-case lengths was 150 to 360 ms, so 35 epochs of 1,563 batches is
two to five hours before bucketing shortens the batches. The epoch count is "about 35, and
keep going if validation loss is still falling": the 30-epoch convergence was measured with a
subword vocabulary, A8 uses words (about 109M parameters at the real corpus's vocabularies),
and the trainer keeps the best checkpoint, so running long is safe and stopping early is the
risk.

**The 100-token cap.** On the German corpus at 100K pairs, batch 64: 9.60 GiB held with the cap
against 35.80 GiB without, identical wall clock, 1.30% of pairs dropped. Peak memory is set by
the longest batch, not the median one; growth is not quadratic in the cap. The cap is applied
*before* splitting so the split sizes are the sizes submitted.

**A8's floor, for low-resource languages.** At least 100K training pairs; if your cleaned data
has less, use all of it and say so, mirroring A4 and A5.

**A9 is a controlled comparison.** The tokenizer must be the only thing that differs, so A9
trains and scores on A8's files and split unchanged, with the SentencePiece model and decode
limit the Lecture 9 notebook saved; both models are decoded greedily and scored by the same
code. The split is never redone with the subword tokenizer: text fed through the word
vocabulary would make A8's model output pieces, and BLEU on pieces is not comparable.

**Post-norm.** `SimpleTransformer` is the 2017 paper's architecture, post-norm by default,
pinned by a test whose failure message names the two 8b slides that depend on it. Pre-norm is
available with `norm_first=True`.

**Notebooks live in the repository**, `torchlingo/docs/docs/course/`, each with a Colab badge,
metadata declaring which lecture it serves and what it starts, and CI that executes every
notebook it can and tests the rest from their own cells. Students install TorchLingo from the
`course` branch, which moves only when `main`'s checks have passed. The two sessions exchange files under `notes/handoff/`, one file per
hand-off. Course work is tracked in one list both sessions share,
`torchlingo-private/notes/course/TASKS.md` (since Oct 25); library and notebook engineering in
this repository's `notes/TASKS.md`.
