# A8 and A9 now work for languages written without spaces; notebooks install from `course`

**Baton to you, 2026-10-05, late.** A second pass the same day, after
`2026-10-05-three-decisions-taken-in.md`. Nothing new came from your side since your Oct 5 files;
your roadmap v13 and its later edit (the 8b symbols slide) are committed. Two students on Asian
languages got empty A8 translations this afternoon, and most of today followed from that.
**Four requests for you, all Learning Suite or deck text, first; the rest is so your side matches
ours.**

## Requests

**1. The A8 handout: one paragraph for languages written without spaces.** The A8 notebook says
the handout governs, and the notebook now has an exception the handout does not mention. Proposed
text, yours to word:

> *If your language is written without spaces between words (Chinese, Japanese, Thai, Lao, Khmer,
> Burmese), set `NO_SPACES = True` in Step 0 of the A8 notebook. Your language is then split into
> characters instead of words; English stays in words. Your BLEU is spBLEU (sacreBLEU's
> `flores200` tokenizer), because ordinary BLEU finds words at spaces and scores such text near
> zero. Say so in your write-up.*

**2. The A9 instructions: three names, and one scoring rule.** Run unchanged, A9 reuses A8's
training-run name, its `best/` folder and its `test.hyp`, so it **overwrites the A8 model and
translations** the A9 write-up compares against. Found by running A8, Lecture 9 and A9 end to end.
The Lecture 9 notebook now prints, under the three settings:

- in Step 7: `experiment_name="assignment-9"` and `BEST_DIR = OUT_DIR / "best-a9"`;
- in the scoring cell: `OUT_DIR / "best"` becomes `OUT_DIR / "best-a9"`, and `"test.hyp"`
  becomes `"test-a9.hyp"`.

Please say the same in A9's text. And one rule: **score A8 and A9 the same way.** Decoding now
shows `<unk>` instead of dropping it (below), so an A8 score from before tonight and an A9 score
after it are not comparable; re-running A8's scoring cell on the saved A8 model fixes that in
minutes, with no retraining.

**3. A9 for the students with `NO_SPACES = True`:** they compare **characters against subword
pieces**, where everyone else compares words against pieces. Lecture 9 detects them from their
split, measures characters as the baseline, and tells them to leave `NO_SPACES = True` when they
paste A9's settings: the pieces replace the characters, and spBLEU stays, so their A8 and A9 are
comparable. If A9's text says "your word vocabulary", a parenthesis "(characters, if your language
has no spaces)" covers them.

**4. Any course notebook you write or edit that uses TorchLingo opens with the standard install
cell**, quoted in full in `briefing.md` (section 4, "Two conventions to copy, one to avoid"),
which installs from the `course` branch with a forced reinstall. Never a `torchlingo>=` PyPI pin, and never `@main`.

## What changed on this side

- **Notebooks install TorchLingo from the `course` branch on GitHub, not PyPI** (Eric). A notebook
  and the library code it needs now reach students together. `course` moves only to commits whose
  checks passed. Each notebook's setup line names the exact code, e.g.
  `TorchLingo 0.2.6 (course @ a46495b)`: ask for it with any student's bug report.
- **Decoding shows `<unk>`** instead of silently dropping it, and building a word vocabulary that
  would be mostly `<unk>` now warns, naming the no-spaces cause. A model that writes only `<unk>`
  used to print empty lines, which is what the two students saw.
- **The A8 notebook's `NO_SPACES` switch**: characters for the target language (`CharVocab`, now
  in the library), length capped in characters, spBLEU scoring. The default path is unchanged.
- **Releases 0.2.5 and 0.2.6 are on PyPI.** Students' own saved A8 copies, which install from
  PyPI, get all of the above in a new Colab session.
- **Tutorial 5 now shows the encoder's self-attention, head by head** (your offer from this
  morning): the first layer's four heads, their average, and a sharpness line per layer. On the
  pretrained model each head is sharp (0.90 to 1.00) and their average is not (0.40), the point
  your 8b slides make.
- **Two notes went to students tonight, from Eric**: one to the two students (set `NO_SPACES`,
  retrain), and one to the class (how to see their `<unk>`s without retraining).
- **Tutorial 8's rewrite is paused** until A8 is in shape (Eric). It still needs tutorial 3's
  checkpoint, so the Lecture 10 deck's decoding slide should keep naming the tutorial without its
  badge, as it does since Sep 28.

## Which tasks moved

`notes/TASKS.md` was reconciled in PR #220 (merged).

| | |
|---|---|
| **#192** | closed: empty A8 translations, fixed and released; students told |
| **#188** | closed: tutorial 5's self-attention section is merged |
| **#152** | closed: students run the A8 notebook in Colab successfully (Eric) |
| **#166** | paused (Eric): tutorial 8, with #162 and #92; no word ceiling for it (Eric) |
| **#167** | branch cleanup done except `myles_testing`, `coulson` and four unmerged local branches |

No questions for you this time; the four requests above are the ask.
