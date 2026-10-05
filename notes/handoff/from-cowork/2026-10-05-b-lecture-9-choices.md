# Lecture 9 notebook: the student writes the count, picks the vocabulary size, and predicts A9

**Baton to you, 2026-10-05, second file today.** Same vein as `2026-10-05-lecture-7-held-out-choice.md`,
decided by Eric the same hour. **This one has a clock:** the notebook runs in class on Wednesday
Oct 7. If it is not merged by Wednesday morning, the class runs the current notebook and nothing
breaks; the A9 write-up questions below apply either way, since a student can answer all but the
prediction from the current printout, and writes the prediction in the write-up instead.

## Why

Roadmap v10 rates `lecture-09-subword-tokenization` **watch**, and **high** in combination with
A9: the notebook counts the `<unk>` tokens for the student, picks the vocabulary size, and ends by
printing the three A9 settings ready to paste, so A9 is "paste and run again". Three changes put
the student's hand back on the parts that carry the lesson, and none of them touches the
printed settings, which stay exactly as they are (the `sp_model_path` history is the argument):

1. `unknown_rate` becomes the student's four lines, with a known-answer check before it is
   trusted on their data, and a guess at the target-side percentage written down before it runs.
2. The student picks `VOCAB_SIZE` from 4,000, 8,000 or 16,000 and says why in one line (8,000
   stays the default and the recommendation).
3. The student writes a prediction of A9's BLEU, direction and rough size, at the moment the
   `<unk>` counts and the piece-per-word ratio are in front of them; the notebook prints it with
   the settings, and the A9 write-up asks for it back with a reconciling sentence.

Plus a **Report back** section, which this activity never had and the Lecture 4, 5 and 6
activities all have.

On the badge rule: the index lists this notebook as **Colab only** (it reads the A8 split from
Drive), so CI does not execute it and the TODO costs CI nothing. For a student it runs end to
end once the four lines are written, which is the point.

## The edits, cell by cell

Cell indices from `docs/docs/course/lecture-09-subword-tokenization.ipynb` as of 2026-10-01
(18 cells, index 0 to 17).

### md cell 1, the opener

The numbered list "When it ends you will have" becomes:

```
1. **A SentencePiece model trained on your A8 training split**, at a vocabulary size you chose:
   `spm.model` and `spm.vocab`.
2. **Two numbers side by side**, from a counting function you wrote: how many test words your
   word vocabulary cannot represent, and how many your subword vocabulary cannot. The second
   should be zero.
3. **A round-trip check**: text encoded to pieces and decoded back, unchanged.
4. **The three settings A9 needs**, printed for your data, and your prediction of what they will
   do to your BLEU.
```

The rest of the cell is unchanged.

### md cell 5, Step 1: append two paragraphs

After "...and on the target side it cannot write it, ever." add:

```
The function that counts them is yours to write, and it is four lines. `vocab.encode(sentence,
add_special_tokens=False)` gives one sentence's token ids; `vocab.unk_idx` is the id every unknown
word becomes. Return how many ids are `unk_idx` and how many ids there are in all. The cell
checks your function on a case whose answer is known before trusting it on your data.

**Before you run it, write down a guess** at the target-side percentage. There is no right
guess; there is a number you can be surprised by.
```

### code cell 6, Step 1: replace the whole cell

```python
words = NMTDataset(A8_DIR / "train.tsv", config=Config())   # the same vocabulary A8 built

# Your guess, before running: the percentage of target-side test tokens the word vocabulary
# cannot represent. Inflected languages run higher. Any number; the point is to have one.
EXPECTED_UNK_PERCENT = None     # <- e.g. 5.0


def unknown_rate(vocab, sentences):
    """Return (unknown tokens, all tokens) for sentences encoded with vocab.

    TODO: write it. vocab.encode(s, add_special_tokens=False) gives the token ids of one
    sentence, and vocab.unk_idx is the id every unknown word becomes. Over all the sentences,
    count how many ids are unk_idx and how many ids there are in all.
    """
    raise NotImplementedError("write unknown_rate, then run this cell again")


# A case whose answer you know: one made-up word is one unknown token out of one.
assert unknown_rate(words.src_vocab, ["qzxvw"]) == (1, 1), "unknown_rate is not counting right yet"
assert EXPECTED_UNK_PERCENT is not None, "write your guess in EXPECTED_UNK_PERCENT first"

word_unk = {}
for side, vocab in (("src", words.src_vocab), ("tgt", words.tgt_vocab)):
    unk, total = unknown_rate(vocab, test[side])
    word_unk[side] = unk
    print(f"{side}: word vocabulary {len(vocab):,};  test tokens it cannot represent: "
          f"{unk:,} of {total:,} ({100 * unk / total:.1f}%)")

print(f"\nYou guessed {EXPECTED_UNK_PERCENT:.1f}% on the target side.")
print(f"ln(V) for your target vocabulary: {uniform_loss(len(words.tgt_vocab)):.2f}")
```

`unknown_rate` is called again in Step 4 (code cell 12), unchanged.

### md cell 8, Step 2: replace the last paragraph

Replace "8,000 pieces is the size to start from ... It takes about a minute." with:

```
**The size is your choice.** 8,000 pieces is what the course model was measured with, and where
ln V is about 9.0, the number Lecture 7's slides quote. Fewer pieces (4,000): more words spelled
out, longer sequences, a smaller output layer. More pieces (16,000): more words kept whole,
shorter sequences, and more pieces seen too rarely to learn well. Pick one and say why in one
line; 8,000 is right if you have no reason to prefer another. Whichever you pick, A9 trains with
it, and the comparison is still one variable: your word vocabulary against this one. Training
takes about a minute.
```

### code cell 8, Step 2: replace the whole cell

```python
# Your choice. 8,000 is the course default; 4,000 and 16,000 are the alternatives worth a reason.
VOCAB_SIZE = 8_000
WHY = ""     # one line: why this size for your language

assert VOCAB_SIZE in (4_000, 8_000, 16_000), "pick one of the three sizes"
assert WHY.strip(), "say why, in one line, before training the vocabulary"

train_sentencepiece([A8_DIR / "train.tsv"], model_prefix=str(A8_DIR / "spm"), vocab_size=VOCAB_SIZE)
pieces = SentencePieceVocab(str(A8_DIR / "spm.model"))

print(f"Subword vocabulary: {len(pieces):,} pieces, shared by source and target.")
print(f"Why this size: {WHY.strip()}")
print(f"ln(V) = {uniform_loss(len(pieces)):.2f}")
print(f"Saved: {A8_DIR / 'spm.model'} and {A8_DIR / 'spm.vocab'}")
```

### md cell 15, Step 6: append one paragraph

After "The cell measures your own longest sentence." add:

```
Then, with the two `<unk>` counts and the ratio in front of you, **predict A9**: will BLEU go
up or down against your A8 score, by roughly how much, and why? The cell prints the prediction
with the settings, and the A9 write-up asks for it back, with one sentence reconciling it with
what happened. A prediction you wrote before training is worth more than any explanation
written after.
```

### code cell 16, Step 6: replace the whole cell

```python
word_len = train["tgt"].str.split().str.len()
piece_len = train["tgt"].map(lambda s: len(pieces.encode(s, add_special_tokens=False)))

MAX_DECODE = int(piece_len.max()) + 10    # your longest target, in pieces, with room to spare

print(f"Target sentences are {piece_len.sum() / word_len.sum():.2f}x longer in pieces than in words.")
print(f"Longest training target: {word_len.max()} words, {piece_len.max()} pieces.")

# Your prediction for A9, in one or two sentences: BLEU up or down against A8, by roughly how
# much, and why. Use the two <unk> counts and the ratio above.
BLEU_PREDICTION = ""

assert BLEU_PREDICTION.strip(), "write your prediction before the settings are printed"

print()
print("=== For Assignment 9: change these in your A8 notebook's Step 6 ===")
print(f"  In Config(...), add:              max_decode_length={MAX_DECODE},")
print(f"  In create_dataloaders(...), add:  use_sentencepiece=True,")
print(f"                                    sp_model_path=str(OUT_DIR / 'spm.model'),")
print(f"                                    sp_tgt_model_path=str(OUT_DIR / 'spm.model'),")
print("  (The same file twice: one subword model serves both languages.)")
print("Everything else stays: the same files, the same split, the same training and scoring cells.")
print()
print(f"Your A9 prediction, for the write-up: {BLEU_PREDICTION.strip()}")
```

### md cell 17, "What A9 is": append, then add a Report back section

After "Your A8 notebook's scoring cell already does the first; A9 needs only the three settings
printed above." add:

```
The A9 write-up asks for: your A8 and A9 BLEU, decoded the same way and saying how; your
prediction as printed above, and one sentence reconciling it with the result; the two
target-side `<unk>` rates and the piece-per-word ratio; your vocabulary size and why; and one
test sentence where the two systems' outputs differ, with your opinion of which is better.
```

Then a new markdown cell at the end:

```
---
## Report back

We will build a table on the board.

1. **Your target-side `<unk>` percentage, and your guess.** Who was furthest off, and which
   languages in the room run highest?
2. **Your vocabulary size, and why.** Anyone not at 8,000?
3. **Your piece-per-word ratio**, and what it does to `max_decode_length`.
4. **Your A9 prediction**, in one sentence. We will come back to these when A9 is in.
```

## Course side

- The A9 write-up questions above go on the Lecture 9 deck's A9 slide and into the A9 Learning
  Suite text at the rebuild, which is this side's job this week. The transcript
  `torchlingo-private/notes/assignments/A09-directions.md` is yours to annotate when the text
  changes.
- Roadmap: the Lecture 9 scope note and the notebook-index row will say "the student writes
  the count, picks the size, predicts A9" once you confirm the merge.
- The asserts stop a student who skipped a blank, on purpose. If Wednesday's clock argues for
  softer stops (a loud print instead of an assert on `WHY` and `BLEU_PREDICTION`), say so and
  this side will not object; `unknown_rate` and the known-answer check should stay hard.
