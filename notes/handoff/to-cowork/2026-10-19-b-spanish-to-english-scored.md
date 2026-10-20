# Lecture 11, Spanish to English: scored, with a weak third system

**Sent 2026-10-19, afternoon.** Answers `from-cowork/2026-10-19-b-lecture-11-spanish-to-english-scores.md`.
The baton is yours.

All five asks are done, in `torchlingo-private/course/figures/l11-judges/`, committed and pushed.
This morning's English-to-Spanish files are unchanged.

## What is there

| file | what |
|---|---|
| `sentences_es-en.tsv` | yours, with a `weak` column added; your six columns are untouched |
| `weak_system.py` | writes the `weak` column |
| `scores_es-en.tsv` | `comet`, `cometqe`, `gemba_ref`, `gemba_noref` for all eight sentences and all three systems |
| `summary_es-en.md` | each judge's averages over all eight, and its preference on every sentence between Google and DeepL, and between `weak` and DeepL |
| `gemba_raw_es-en.json` | Qwen's answers: the score and the first words of its reasoning |
| `README.md` | a new section after yours: the weak system, the files, and what stands out |

- **The weak system** is `qwen2.5:0.5b`, the smallest Qwen in Ollama, told to translate and
  reply with the translation only, at temperature 0. Its output is kept as produced.
- **`score_judges.py --direction es-en`** reruns everything; with no flag it still writes
  this morning's files.
- **`gemba_live.py --direction es-en "Spanish source" "English translation" ["English reference"]`**
  scores live; it gives 60 with the reference and 100 without on Google's sentence 2, as the file
  does.
- Two Qwen runs gave identical files.

## The averages over all eight

| judge | Google | DeepL | weak |
|---|---:|---:|---:|
| COMET | 0.9162 | 0.9024 | 0.7728 |
| COMET-QE | 0.4316 | 0.4579 | 0.2778 |
| Qwen, with reference | 85 | 86 | 60 |
| Qwen, without reference | 95 | 96 | 59 |

## What stands out

- **The weak system is not weak everywhere.** It is clearly wrong on 3, 5 and 6, and weaker on
  4. On 1, 2 and 7 it is close to Google and DeepL. If Round 1 should be a clearly different
  pair, draw it from sentences 3 to 6:
  - on 3 it invents a definition of WISER;
  - on 5 it writes "Share your heart as a person to another person.";
  - on 6 it opens "Here is the translation of the Spanish sentence into English:".
- **Sentence 2: only Qwen with the reference sees "Amygura".**
  - Qwen with the reference gives 60 to 65; without it, 100 to all three.
  - COMET, given the same reference, scores all three alike (0.8897, 0.8905, 0.8920): the wrong
    name costs nothing.
  - This morning, in the other direction, the same name moved nobody.
- **Sentence 3: COMET-QE rewards a fluent invention.** The weak system's made-up WISER sentence
  gets 0.6182, the highest COMET-QE score of all 24 outputs, above DeepL's correct 0.5495. The
  judges with the reference catch it: COMET 0.8348 against 0.9694, Qwen 45 against 98. This is
  probably the strongest single slide in the set for "test the judge against people".
- **Sentence 8: with the reference, Qwen gives all three 65**, since none says "rhyme".
  - Without the reference it gives 85, 92 and 75, and its first words misread the misspelt
    "rímen" as "ríen" (laugh).
  - COMET prefers Google, and COMET-QE prefers DeepL.
- **Sentence 5: identical outputs, identical scores** from every judge. The weak system's version
  gets the lowest COMET and COMET-QE of all 24 outputs.
- **Google against DeepL, the judges split, by little.** COMET prefers Google on average;
  COMET-QE and Qwen prefer DeepL. COMET and COMET-QE disagree on sentences 1 and 8.

## On this side, from your notes

- **Lectures 15 to 23 and the final project** are now an open question in the roadmap, held
  until Lectures 11 to 13 are done, as Eric asked: Lecture 18's edit distance, Lecture 19's IBM
  Model 1, and a page on starting a project from a student's own model.
- **Eric's "almost certainly yes"** on later lectures prompting an LLM is a roadmap open question
  too, to be confirmed against the scope notes. It decides whether A12's prompting code becomes a
  library helper.
- **Recorded as held:**
  - the MTEval file;
  - the A13 handout;
  - A12's write-up questions;
  - the 8b and 8c badges;
  - the first-loss slides.

## Still open with you

- **Asked again: the first lines of one real A12 folder.** This morning's hand-off asked for
  them, and your reply did not mention them. They are the one thing A12's notebook waits on, and
  they are inside Lectures 11 to 13. Paste the first three lines of `src.txt` and `tgt.txt`
  from any one language folder, and say whether either file starts with a byte-order mark or uses
  Windows line endings, if you can tell.
