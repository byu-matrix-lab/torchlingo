# Lecture 11 turns around: score the Spanish-to-English activity

**Sent 2026-10-19, afternoon.** Answers `to-cowork/2026-10-19-judges-scored-a12-a13-on-course.md`. Thank you for this morning's scores; they are good teaching results and stay in the folder.

## What changed, and why

Eric is recording Lecture 11 rather than teaching it in the room. He is not a Spanish speaker, and not every student is, so the judging activity turns around: Spanish into English, with the English original as the reference. Every student can then judge as a monolingual rater against a reference, as WMT's reference-based direct assessment does, and test COMET and Qwen against their own judgments. Lecture 11's message (Eric): *every judge of translation quality, an LLM included, is a metric, and it has to be tested against people.*

Scope (Eric): this session and yours stay on Lectures 11 to 13 and A11 to A13; anything later goes into the roadmap as notes. On whether later lectures prompt an LLM: "almost certainly yes", to be confirmed from the roadmap later.

## What needs you

1. **A weak third system, Spanish to English,** as column `weak` in `sentences_es-en.tsv`: something clearly worse than Google and DeepL, so that Round 1 is a clearly different pair. Your choice; two candidates are the smallest local model in Ollama prompted to translate (for example `qwen2.5:0.5b`), or a word-for-word gloss. Name what you used in the README. Tutorial 7's TorchLingo model only goes English to Spanish, so it cannot serve.
2. **`scores_es-en.tsv`:** the same four judges as this morning, now with English as the target, for all eight sentences and all three systems (`google`, `deepl`, `weak`):
   - `comet`: `Unbabel/wmt22-comet-da`, with the English reference;
   - `cometqe`: `Unbabel/wmt20-comet-qe-da`;
   - `gemba_ref` and `gemba_noref`: the same Qwen, the same GEMBA-DA prompt from Figure 1, with `Spanish` as the source language and `English` as the target.
3. **`summary_es-en.md`:** for every sentence, each judge's score per system and its preference between Google and DeepL, and between `weak` and DeepL; the system averages over all eight. Flag what stands out, especially sentence 2 (the source says "Amygura", the reference "Omidyar"), sentence 8 (neither output says "rhyme") and sentence 5 (Google and DeepL identical).
4. **Extend `score_judges.py`** (a direction flag is fine) so both directions rerun.
5. **`gemba_live.py`:** confirm it accepts Spanish to English, or add a flag for the languages.

Soon, not this minute: Eric records once the deck is rebuilt on these scores.

## Data

`torchlingo-private/course/figures/l11-judges/sentences_es-en.tsv`: columns `id test_row src ref google deepl`. The ids match this morning's file (same eight TED sentences); `test_row` is the row in tutorial 7's `test.tsv`. The README has a new section on what the turn-around changes.

## Your other asks

Held until Lectures 11 to 13 are done, in Eric's scope: the A6 MTEval results file (item 1), the A13 handout (item 3), and the carried items (A12's write-up questions on Learning Suite, the 8b and 8c badges, the first-loss slides). Item 2 (Lectures 14 to 23) becomes a roadmap note, per Eric.
