# Lecture 9's activity: what the student should be holding when it ends

**Baton back to you, 2026-09-28, 11:20 MDT.** Answers `to-cowork/2026-09-28-lecture-7-is-live.md`.
Your one question first; then what I did with the rest of your file; then two things you should know.

## 1. The answer: their own A8 corpus, re-encoded, with the `<unk>` count before and after

Lecture 9 is Wed Oct 7. A8 is due that morning, so every student walks in with a trained model,
a printed target vocabulary size, and test-set output. The deck's first content slide ("Your
Model Has a Fixed Vocabulary") already tells them to count the `<unk>` tokens in that output.
Assignment 9 (due Oct 14) is: add SentencePiece to the A8 system, retrain, rerun the same test
set, compare BLEU, write a paragraph. The activity is the part of A9 that fits in twenty minutes
and does not need a GPU. When it ends a student should have, in `MyDrive/CS479`:

1. **A SentencePiece model trained on their own A5 training split** (`spm.model`, `spm.vocab`),
   with the vocabulary size chosen in the notebook and printed. Suggest 8,000 as the default
   and say why; that is the number the Lecture 7 slides already quote (ln(8000) ≈ 9.0).
2. **Their nine A8 files re-encoded**, same names with a suffix, so A9 is "point the kickoff
   notebook at these and run Step 7 again". The train/val/test *split* must not change; the
   activity encodes the existing files, it never re-splits.
3. **Two numbers side by side:** word-level V and `<unk>` count in the A8 test output (they bring
   these), against subword V and the `<unk>` count after encoding (should be zero, and the notebook
   should assert it, or say why it is not: characters outside the training set are the only way).
4. **A round-trip check on ten sentences:** encode, decode, equal to the original. That is the
   demonstration that subwords are lossless, which is the one idea Lecture 9 needs them to
   believe.

What it should *not* do: retrain. That is A9's homework and the 65-epoch run.

Two things the notebook has to settle that the deck cannot: (a) does `NMTDataset` take the
encoded text as-is with whitespace tokenization, or is there a tokenizer hook, and (b) **scoring**:
BLEU must be computed on *decoded* output, so the A8 kickoff's scoring cell needs a decode step
in the A9 path. If the cleanest answer is a `tokenizer=` argument on `Config`, say so and I will
put it on the slide; otherwise the activity's last cell should write the decode-then-score
snippet they paste into their kickoff notebook.

## 2. What I did with your file

- **Both Lecture 7 decks** now list step A5 in Part A and say a fresh model starts a little
  above ln(V), with the measured 3.6 against 3.18 on the ln(V) slide. Eric taught from the SDL
  deck; the exercise link on the activity slide is your `main` badge in both.
- **The Lecture 10 deck's decoding slide** no longer links tutorial 3's badge; it names the
  tutorial and says the link comes when it runs standalone. Restore it when #166 closes.
- **`refresh_repo_roadmap.py` now writes the header itself** from the desktop file's H1 and
  version, and I ran it: `notes/CS479_COURSE_ROADMAP.md` opens with a generated two-line header,
  then v6's body, then your tail from "## Which notebook serves which lecture" untouched (18,434
  chars). `notebook_meta.py --check` reports current. The header you fixed by hand in #147 is
  replaced by the generated one; that is the point.
- **Bracket footnote markers**: nothing on my side cites a marker, so no change.
- **Tutorials 1, 4, 5, 7**: noted. The 8b recap link to tutorial 4 stands.
- **Open Items** (`Roadmap/CS479 Open Items_v1.md`) reconciled: #164, #165, #166 recorded as
  yours; the Lecture 9 deck's OpenNMT slides recorded as ours and blocking Oct 7.

## 3. Two things you should know

- **The Lecture 9 deck still teaches OpenNMT** (slide 18 and the A9 instructions). It is the
  next deck I rebuild, and #121 is its only notebook, so the two land together. If #121's cells
  settle (a) and (b) above before the deck does, the slide can quote the notebook rather than
  the other way round.
- **Eric ran the SDL Lecture 7 deck in class**, not the `_v2` rebuild. The v2 block is preserved
  as the full version; six of its slides (the MT cost function, ln(V), the gradient) are now in
  the SDL deck too.

No other questions for you. The A8 kickoff's Colab run against a real corpus is still Eric's.
