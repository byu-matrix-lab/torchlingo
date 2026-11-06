# Notebook code that belongs in TorchLingo (Task #310)

Reviewed: all 26 notebooks (17 course, 9 tutorials), every code cell, at `main` af79db2, 2026-11-05.
`course` is at 88e55bd. Nothing in the repository was changed. Cell numbers are indices into the
notebook's `cells` list, markdown included, the numbering `notes/TASKS.md` uses.

**Status, Nov 5.** F1 landed in PR #389, and F2's library half in PR #388, which found it wider
than stated: a column of numbers also became numbers. F3 to F7 and F9 became Tasks #314 to #318,
and the rest joined Task #294. F6 below is corrected. Findings F3 and onward were not each
re-verified when this was filed, so check one before building it. Eric, Nov 5: no students have
started A14, so the "not while A14 is open" cautions below no longer apply.

## Summary

- **Only three findings help a student this term:** F1 (Lecture 18, Dec 2), F2 (a library
  defect), and F4's small A14 piece if wanted. Everything else serves the next offering.
- **The cheapest, F1, needs no release.** `capture_cross_attention` is already on `course`. I ran
  it against Lecture 18's hand-written wrapper on the real checkpoint: largest difference 6e-8.
- **The most important finding is a defect, not a duplication.** `load_data` silently loses
  sentences (F2), and 13 hand-written `read_csv` calls in 9 notebooks avoid it. Students' A8, A9
  and A13 training goes through it.
- **Hand-written copies have already drifted.** The library's two chrF paths differ (F6, as
  corrected), and the COMET install pin differs across notebooks (F8).
- **A13 and A14's setup cells:** share them, next offering, as one function (F5). Not now.

| # | Finding | Library has it? | Who meets it | Size of change |
|---|---|---|---|---|
| F1 | Lecture 18 attention capture | Yes | **This term** (Dec 2) | Notebook only |
| F2 | `load_data` loses sentences; 13 workarounds | Defect | **This term** (library), notebooks next | Library, then notebooks |
| F3 | Reading and writing line files | No | Next offering | New functions |
| F4 | NLLB plumbing | Partly | Mostly next; A14 optional | Small library addition |
| F5 | A13 and A14 setup cells | Partly | Next offering | New function |
| F6 | Sentence chrF, three ways | Partly | Next offering | Small |
| F7 | COMET scoring, A10 and A11 | Partly; lacks what they need | Next offering | Library extension |
| F8 | COMET install, four notebooks | No | Next offering | Needs tooling change |
| F9 | BLEU tokenizer and no-spaces test | Mostly | Next offering | Small |
| F10 | Per-head capture, 8b and 8c | Partly | Next offering | Library extension |
| F11 | Model rebuilds and one-sentence decoding | Yes | Next offering; tutorial 7 anytime | Notebook only |
| F12 | Epochs read from a checkpoint | No | Next offering | Small |
| F13 | Hugging Face login | No | Next offering | Small |
| F14 | Hand-written JSON notes | Yes (`Notes`) | Tutorial 9 anytime; A11 next | Notebook only |
| F15 | Sample corpus filter | No | Next offering | Small |
| F16 | Tiny phrase model | Partly | Tutorials anytime | Judgement below |
| F17 | Tutorial odds and ends | Yes | Tutorials anytime | Notebook only |

"Tutorials anytime" means a tutorial edit can ship whenever; tutorials carry no points, so these
rank low (P3 by the project's own test).

---

## Findings, most valuable first

### F1. Lecture 18's attention capture repeats a library function

- **Where:** `lecture-18-ibm-model-1`, cell 18 (67 lines). The wrapper is 13 of them: `cross`,
  `seen`, `recording`, and the `try`/`finally`.
- **Does:** forces `need_weights=True` on the last decoder layer and records the averaged weights.
- **Library:** yes. `torchlingo.models.transformer_simple.capture_cross_attention` does exactly
  this, on every layer, heads averaged, and is on `course`.
- **Verified:** on `data/pretrained/model.pt` with "The red car." / "El coche rojo.", the library's
  last layer matches the hand wrapper to 6e-8. Replacement:

  ```python
  with capture_cross_attention(transformer.transformer.decoder) as captured, torch.no_grad():
      transformer(source, target[:, :-1])
  weights = captured[-1][0]
  ```

  `attention_for_sequence` also matches, but its contract is a *generated* sequence and this
  Spanish is given, so `capture_cross_attention` is the honest name.
- **Why plumbing:** the task names attention capture; the cell is already marked Setup. The lesson
  is comparing the two alignment tables, and EM (cells 8 to 16) stays.
- **The rest of the cell:** `words_of` (15 lines) and the piece-to-word merge (4 lines) could
  become `merge_pieces(weights, src_pieces, tgt_pieces)` in `visualization.py`, but they serve one
  notebook, so they can stay. I would leave them.
- **This term: yes.** Lecture 18 is Dec 2. A notebook edit alone; no promotion of library code,
  though the notebook itself still reaches students through `course`.
- `tests/test_lecture_18.py` has no match for `recording`, `attention_by_word` or `words_of` by
  grep, so it should need no change; confirm by running it.

### F2. `load_data` silently loses sentences, and 13 cells work around it

- **Where:** `pd.read_csv(..., sep="\t", dtype=str, keep_default_na=False)` in 13 places across 9
  notebooks: A9 cells 4, 12, 22; A11 cell 20; A12 cell 7; A13 cell 4; A14 cell 4; 8b cell 8;
  tutorial 5 cell 5; tutorial 7 cells 5, 13; tutorial 9 cell 6 (twice). Each is 1 line, or a
  2-line `read_tsv` in A13 and A14.
- **Library:** `preprocessing.load_data` exists, but calls `pd.read_csv(path, sep="\t")`.
- **Verified:** a TSV with the row `NA<tab>null` loads as NaN on both sides, and `NMTDataset` then
  returned 2 sentences of 3, with no warning. pandas treats `NA`, `null`, `None`, `nan`, `N/A` and
  others as missing. `NMTDataset` (`dataset.py:101`), `train_sentencepiece` and the multilingual
  helpers all go through `load_data`, so **the notebooks protect their own reads while the
  library's training path does not.**
- **Proposed:** make `load_data` read TSV and CSV with `dtype=str, keep_default_na=False`. Then
  replace the 13 sites with `load_data(path)`.
- **Check before changing:** `dataset.py:114` does `fillna("")` and later code filters empties, so
  a blank cell should still end up as `""`. Confirm that, and that nothing relied on numeric dtypes
  (the `kind` column in `data/example.tsv` is text).
- **Effect is small but silent:** only a whole cell equal to one of those strings is lost. It is
  still the wrong way round for a library that exists to teach careful data handling.
- **This term: the library fix, yes,** for anyone training after promotion. The notebook
  replacements: next offering, except tutorials.

### F3. Reading and writing one-sentence-per-line files

Extends Task #294. **Correction:** `def read_lines` exists in only two notebooks (A11 cell 7, A12
cell 7), not five; but the logic is written out in far more, in four different spellings:

| Spelling | Where | Risk |
|---|---|---|
| `read_text("utf-8-sig").rstrip("\n").split("\n")` | A11 c7, A12 c7, A13 c6 | Correct |
| `.read_text("utf-8").split("\n")[:n]` | A12 c25; A13 c16, c26 (2); A14 c17, c23 | Correct; needs n |
| `text[:-1].split("\n")` (`read_hyp`) | A11 c20 | Correct; 3 lines |
| `.read_text().splitlines()` | A9 c16; A13 c14, c16; lecture 4, 5 cells | **Splits inside a line on U+2028, U+0085 and others** |

- `parallel_txt_to_dataframe` (`base.py:68`) reads plain `utf-8`, so a byte-order mark, which A11
  and A12 take care to drop, survives in A8's first sentence.
- Writing, `write_text("\n".join(x) + "\n")`: A8 c18 (2), A12 c19, A13 c6, c20 (2), tutorial 1 c6.
- `inference.py:1309` has the same read inline, so the library would use its own function.
- **Proposed:** `read_lines(path, n=None)` and `write_lines(path, lines)` in
  `preprocessing/base.py`, the first dropping a byte-order mark and splitting on `"\n"` only; have
  `parallel_txt_to_dataframe` use it.
- **Plumbing:** file loading, named in the rule. A11's and A12's docstrings explain the encoding
  trivia a student gains nothing from.
- **This term: no.** A9, A11, A12 and A13 have run or close by Nov 6. Lectures 4 and 5 stay as
  they are: they install no TorchLingo.

### F4. NLLB plumbing

Extends Task #291. **Corrections to that row:** the A12 cell is now **21**, not 20 (20 is
markdown). Tutorial 9's cell 14 is markdown too, and tutorial 9 no longer resumes decodes by hand,
so that half of the task appears to be gone.

| Block | Where | Lines | Library |
|---|---|---|---|
| Load model and decode greedily | A12 c21 | 29 | `load_nllb` does it |
| Set `src_lang`, build `NLLBTranslator` | A14 c5, c7, c20; T9 c6 | 2 each, 4 sites | No |
| The code list from `additional_special_tokens` | A14 c5; T9 c6 | 1 each | No |
| `from_pretrained(..., float32).to(device)` | A14 c5; T9 c6 | 1 to 2 each | Differs from `load_nllb` |

- **Why `load_nllb` does not cover A14 and tutorial 9:** it loads float16 on CUDA and is fixed to
  one direction. Both notebooks need float32 for training and many directions on one in-memory
  model. So these two need an addition, not a swap.
- **Proposed in `torchlingo_harness/nllb.py`:** `nllb_codes(tokenizer) -> set[str]`, and
  `NLLBTranslator.between(model, tokenizer, src_code, tgt_code, device)`, which sets `src_lang` on
  each call. The `tokenizer.src_lang = ...` line is the fragile one: forgetting it translates from
  the previous direction without complaint.
- **A12 caveat:** `load_nllb` raises on an unknown code and A12 keeps its `NLLB_CODES` dict for
  Spanish and Telugu; keep that dict, as it decides whether there is an NLLB row at all.
- **This term:** A12 no. A14: optional. It is open until Nov 13, its cells work, and a student
  gains nothing from a change now. I would not touch it. Tutorial 9: anytime.

### F5. A13's and A14's setup cells: share them, next offering

Compared line by line, A13 cell 4 (124 lines) and A14 cell 4 (62 lines) share about 30 lines:

| Shared | A13 | A14 | Identical? |
|---|---|---|---|
| `course_dir(... needs=[...])` | 6 | 6 | Except one file |
| `settings`, `NO_SPACES`, `TOKENIZE` | 4 | 3 | Yes |
| `Config(d_model=512, ...)` | 10 | 2 (reformatted) | Same values |
| `SentencePieceVocab` | 2 | 1 | Yes |
| `Notes` and its comment | 4 | 4 | Yes |
| `read_tsv` | 2 | 2 | Yes |
| Read the three splits and print | 2 | 2 | Yes |
| `load_best` / `scratch_translate` | 4 / 7 | Same `load_model` call | Yes |

The A9 `Config` is written five times: A8 c20, A9 c24, A9 c27 (`config_a8`), A13 c4, A14 c4.

- **Proposed:** `torchlingo.colab.load_a9_run(env, needs=())`, returning a small object with
  `dir`, `settings`, `no_spaces`, `tokenize`, `vocab`, `config`, `train`, `val`, `test`, and
  `load_best(best_dir)`.
- **Why plumbing:** A13 and A14 are about back-translation and multilingual tags. Neither teaches
  anything by reloading A9. "Same configuration as A9" is a fact a function states better than
  ten repeated lines, and it cannot drift.
- **Keep inline:** A8's own `Config` (cell 20) and A9's first (cell 24). A student should see the
  model's size once. A13's `train`, `back_translate` and A14's `nllb_translate` each serve one
  notebook. `train` in particular is worth a read: it is A9's call, and says so.
- **Against doing it now:** `tests/test_lecture_14.py` keys on `SETUP_CELL = "def
  scratch_translate("`, and A14 is live. Nothing ahead this term uses A9's files in a notebook:
  A15 is dropped (Task #271) and Lecture 18 uses the pretrained model.
- **This term: no.** Do it as one change with F2, F3 and F9, since those shrink the same cells.

### F6. Sentence-level chrF, three ways, and two are not the library's chrF

- **Where:** A11 c16 and c20: `CHRF(word_order=2)` with `sentence_score`. A13 c26:
  `sacrebleu.sentence_chrf`. Lecture 7 c19: `sacrebleu.corpus_chrf`, two uses. Homework for
  Lecture 6, c16: `compute_chrf([h], [r])`.
- **The drift:** `compute_chrf` uses `word_order=2`, and sacreBLEU's `sentence_chrf` and
  `corpus_chrf` default to `0` (checked from their signatures). On one test pair the two gave
  55.11 and 54.85. Tutorial 2 cell 14 teaches exactly this trap. A11's comment shows the author
  knew.
- **Correction, Nov 5.** This finding first said A13's "helped most" ranking differs from the table
  above it. It does not: `paired_bootstrap` uses `CHRF()`, so `word_order=0`, the same as
  `sentence_chrf`. The real inconsistency is inside the library: `compute_chrf` is chrF++ and
  `paired_bootstrap` plain chrF, so A14 prints both as "chrF". Word bigrams also depend on spaces:
  on one Chinese sentence plain chrF gave 37.54 and chrF++ 32.18. Task #314 proposes 0 everywhere.
- **Library:** `compute_chrf` works per sentence today (the Lecture 6 homework does this), so A13
  and Lecture 7 need only a swap.
- **Proposed:** `sentence_chrf(hypotheses, references) -> list[float]` in `evaluation.py`, sharing
  `compute_chrf`'s settings.
- **This term: no.** Both have run; A13 closes tomorrow.

### F7. COMET scoring in A10 and A11

Extends Task #169, which lists this as "still by hand". **The new point: `comet_score` cannot
replace these blocks as it stands.**

- **Where:** A10 c11 (20 lines), c14 (9). A11 c16 (about 16 of 42) and c20 (about 16 of 88), which
  are near-identical to each other.
- **Library gaps:** `comet_score` requires `references`, so it cannot run COMETKiwi, and it does
  not turn COMET's misleading `KeyError` on an unaccepted license into a usable message. Those two
  are what the blocks mostly consist of.
- **Keep the student's function in the path:** these blocks score the output of `comet_inputs`,
  which the student wrote (A10 c7, A11 c9). A library call taking three lists would orphan it.
- **Proposed:** `comet_score_records(records, model=...) -> CometScores` in `evaluation.py`, taking
  `comet_inputs(...)`'s output, with the license message; `comet_score` calls it.
- **Also:** A13 c24 still shells out to `comet-score` and has the student read the output; A14
  uses `comet_scores_in_new_process`, which A13 could too.
- **This term: no.** All have run.

### F8. The COMET install, four copies, and the pin has drifted

- **Where:** A10 c3, A11 c3, A13 c23, A14 c22; 7 to 9 lines each, plus 3 lines of comment.
- **Drift:** A13 pins `jsonargparse==4.46.0`; A10, A11 and A14 pin `3.13.1`. This looks harmless:
  A13 alone calls the `comet-score` command, which is why it moved (commit a70524b). The others use
  the Python API. Still, the next COMET break must be fixed four times.
- **Proposed:** `torchlingo.colab.install_comet()` running pip through `sys.executable`.
- **The obstacle:** `scripts/student_path.sh` finds install cells by the text `"pip install"` in
  the cell, and `--colab --install-only` is the guard that caught October's numpy break. A hidden
  install would be **skipped by that check**, which has already earned its place. So the tool must
  change first, and A10 installs no TorchLingo at all, so it would gain a dependency.
- **Lighter alternative:** leave the cells and add a test that all four pip lines agree, allowing
  A13's stated difference. That removes the drift risk at almost no cost. I prefer this.
- **This term: no.**

### F9. BLEU tokenizer and the no-spaces test

- `TOKENIZE = "flores200" if NO_SPACES else "13a"`: A8 c32, A9 c27, A11 c20, A13 c4, A14 c4.
  A9 c16 has it inline.
- `which = "spBLEU (flores200)" if NO_SPACES else ...`: A9 c29, A11 c22, A13 c26, A14 c25.
- The test "averages under 3 words a line": A8 c8, A9 c4, A11 c20, the same threshold each time.
- `sacrebleu.corpus_bleu(hyp, [refs], tokenize=...)`: A9 c16, c27 (2), A11 c20. **`compute_bleu`
  already takes `tokenize`.**
- **Proposed:** `bleu_tokenizer(no_spaces) -> str` and `BLEU_NAMES` in `evaluation.py`;
  `written_without_spaces(sentences) -> bool` in `preprocessing/base.py`.
- **Keep:** A8 c32's `corpus_bleu` with `[references]`. Its comment teaches the reference-shape
  trap at the point students meet it.
- **This term: no.**

### F10. Per-head capture: 8b and 8c

Extends Task #294 and refines it: 8c already uses the library, and only its input hooks are by hand.

- **Where:** 8b c10 (21 lines, per-head weights plus decoder state X and memory M). 8c c9 (23
  lines, `capture_self_attention` plus pre-hooks for each layer's input).
- **Library gap:** `capture_cross_attention` has no `average_heads` argument, though
  `capture_self_attention` does, and neither records inputs.
- **Proposed:** add `average_heads=True` to `capture_cross_attention`, and a `with_inputs=False`
  option to both, yielding inputs alongside. Then 8b and 8c each lose about 15 lines.
- **Why plumbing:** both cells are marked Setup. The lesson is the student's `attention` and
  `split_heads` matching the model, which stay.
- **This term: no.** Both have run.

### F11. Model rebuilds and one-sentence decoding the library already does

The cheapest fixes after F1, but all on work already past.

| Where | Lines | Replace with |
|---|---|---|
| A8 c32 (`load_checkpoint`, `load_state_dict`) | 2 | `load_model` (Task #169) |
| A9 c27, rebuild of A8's model | 10 | `load_model(..., config=config_a8)` (Task #169) |
| Tutorial 7 c3, rebuild | 7 | `load_model`; it also still needs the raw dict for its print, so read that too (Task #294) |
| Tutorial 7 c5 `translate` | 8 | `translate_batch(model, [s], vocab, vocab, ...)` |
| Tutorial 7 c13, last loop | 3 | `translate_batch` |
| Tutorial 8 c22 `library_greedy`, `library_beam` | 14 | `translate_batch`, if it reaches the same decoders; check first, since this cell's purpose is to compare with those decoders |

**Caveat on A8 c32:** the rebuild there depends on the vocabularies from cell 20, and its comment
tells students to rerun the earlier cells. `load_model` would remove that step; the A8 test needs
updating. Task #169 already says these have run. **This term: no**, tutorial 7 anytime.

### F12. Epochs read from a checkpoint

- **Where:** A9 c22 and A13 c4, 3 lines each inside a 10-line block: `torch.load(...)["state"]`,
  then `len(state["train_losses"])`. The only places a course notebook reaches into the file format.
- **Proposed:** `TrainingCheckpointer.epochs_done()`, in `training_checkpoint.py`.
- **This term: no.**

### F13. Hugging Face login

- **Where:** A10 c13 (11 lines), A11 c15 (13 lines). Same code, different error text.
- **Proposed:** `torchlingo.colab.hf_login(hint=...)`.
- A10 installs no TorchLingo, so it would have to, or keep its own. Modest value.
- **This term: no.**

### F14. Hand-written JSON where `Notes` exists

- **Where:** tutorial 9 c8, c11, c13 (4 json calls, re-saving the same dict); A11 c16, c18, c20,
  c22; A9 c18 and c22 for `a9-settings.json`.
- **Library:** `torchlingo.colab.Notes`, used by A13 and A14.
- **Keep:** A9's settings file is a contract with A11, A13 and A14; those read it directly, so
  moving it would touch all of them for little.
- **This term:** tutorial 9 anytime. A11 next offering.

### F15. The sample corpus filter

- **Where:** A12 c7 (5 lines), tutorial 9 c6 (5 lines), tutorial 7 c13 (2 lines): fetch
  `data/example.tsv`, keep `kind == "transcript"`, filter by words, dedupe.
- **Proposed:** `torchlingo.colab.sample_pairs(min_words=, max_words=)`.
- Note A12's branch also numbers lines for "the sentence before", which must stay in A12.
- **This term: no.**

### F16. The tiny phrase model (Task #294)

My judgement, for that row:

- Three notebooks build it: Lecture 7 c7 and c9, tutorial 3 c6, tutorial 8 c4 (44 lines). All use
  the same `Config` (`d_model=64, n_heads=4, ...`).
- **Lecture 7 keeps its code.** Building and training is that notebook's purpose, and it is a
  student's first model.
- **Tutorial 3's training is plumbing for its point** (what a checkpoint does not hold), and tutorial
  8's cell is already marked Setup. A `train_tiny_model(pairs, repeats=, epochs=, device=)` in
  `training.py`, or in a teaching module, would serve both. But a function this specific is awkward
  in a general library, and it saves about 15 lines per tutorial. **Leave it unless a third
  tutorial needs it.** The one sure saving: `write_tsv`, `write_corpus` and T8's `to_csv` can be
  `save_data`, which already makes parent folders.

### F17. Tutorial odds and ends

- **Tutorial 2 c21 imports a private name**, `_as_reference_streams`. Make it public, or show
  the behaviour through `compute_chrf`.
- **Tutorial 4 c8, padding arithmetic** (the `pad` helper, 3 lines): `collate_fn` does it. Named in
  the rule, but the synthetic ids make the swap a little wider than it looks.
- **Tutorial 6, four `train_model` calls** (Task #294): confirmed at c9, c13, c16, c22, 8 to 9
  lines each. A local `fit(model, frame, epochs, lr)` in its Setup cell; a notebook helper, not
  a library one. c16 needs to keep its `lr=0.0` visible, and it is the bug shown, so it can pass it.
- **Tutorial 6 c7 `translate`, padding by hand** (3 lines): `translate_batch` would need
  `Vocab.encode` to take `add_special_tokens`. One keyword, and worth doing.
- **`to_csv(sep="\t", index=False)` in 9 notebooks:** `save_data` exists. Replace in passing only.

---

## Looks like plumbing, but should stay

- **The install cell, 21 notebooks, 9 lines.** The largest repetition in the tree, and it cannot
  be in the library: it is what fetches the library. CLAUDE.md fixes its form.
- **The prediction and hypothesis asserts.** A house pattern meant to be seen; a helper would hide
  the stop that makes students commit.
- **A13 c21 and A14 c18, the `REAL` check.** Similar 10-line cells, but the student's reading of
  p is the exercise.
- **`comet_inputs`, `split_heads`, `attention`, `sharpness`, `unknown_rate`,
  `pairwise_agreement`, `bleu_from_parts`, `reverse`, `combine`, `tag`, `both_directions`,
  `split_pool_and_test`, `build_prompt`.** Exercises. Some have library twins (`split_exact`);
  that is the point of writing them.
- **A8 c10, c12, c16.** Length cap, dedupe, contamination check, named lessons. Note c16 already
  calls the library's check.
- **Tutorial 8's `greedy_decode` and beam search; Lecture 18's `em_round`.** Named lessons.
- **Lecture 5 c8, c12, c16, c18.** NLTK's Gale-Church and the hand `length_correlation`, even
  though `gale_church_align` and `length_correlation` exist. The notebook installs no TorchLingo;
  it teaches the algorithm and the parameters. The library's `length_correlation` takes a frame
  anyway. Lecture 5's slipped-by-one-row comparison is the lesson.
- **Lecture 4 TMX cells, including `inspect` (c16, 23 lines) and the hand parser (c21).** The
  first is a Setup helper for one notebook; the second shows what a package does for you. It
  uses no TorchLingo.
- **A11 c7 `read_mteval` (19 lines).** One notebook, one file format, and its error messages speak
  to that assignment.
- **A12 c7's `load_pairs`.** Eric's call, Oct 19. Only its `read_lines` moves, under F3.
- **A12's LLM code (c14, c15).** Task #169 waits for a second notebook that prompts an LLM. None
  exists; Lectures 15 to 19 add none. The `<think>` check is a rule of A12's, not a general one.
- **A12 c12 `trigrams`, `nearest_examples`.** Choosing examples is what A12 measures.
- **A12 c27's and A8 c30's plots.** A12's serves one notebook. A shared `plot_losses` for A8 and
  Lecture 7 would save about 5 lines each, and "your first loss curve" is itself a lesson.
- **A8 c18's three-way write, A9 c29's and A13 c26's report blocks.** The output differs with each
  assignment; generalising them would need more arguments than the lines it saves.
- **`sum(p.numel() ...)`, 6 notebooks.** One line, and it says what it does.
- **Tutorial 6's `Vocab`.** CLAUDE.md's own example of plumbing kept in a Setup cell.
- **8b c8 and tutorial 5 c5 look like a copy.** They load and decode the same sentence, about 15
  shared lines, with each notebook then doing different work. It is a tutorial and an assignment
  sharing material, which the rules address by giving them separate purposes, not a function. Each
  line is already a library call.

---

## If this becomes work

Order: F1 now. F2 next, as a small library PR with a test using a row of `NA` and `null`, then
promoted. Then, after term, one pass for the next offering: F3, F9 and F5 together, since all three
shrink the same setup cells, then F4, F6, F7 and the pin test from F8.

Every cell move has to keep the tests that drive notebooks by cell text; 17 test modules
use `tests/notebook_cells.py`, whose substitutions must match the cells they change.
