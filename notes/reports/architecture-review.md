# An architectural review of TorchLingo, pedagogy first (Task #243)

Reviewed: `src/torchlingo/` (10 modules and 3 subpackages, 14,860 lines), `src/torchlingo_harness/`
where course notebooks call it, all 26 notebooks (17 course, 9 tutorials, every code cell), and
the docs under `docs/docs/`. Commit `2ea9bf2` on `main`, 2026-11-13. Nothing in the repository
was changed. Cell numbers are indices into a notebook's `cells` list, markdown included, as in
`notes/NOTEBOOK_LIBRARY_REVIEW.md`. Findings are numbered R1 to R13, so they cannot be confused
with assignments (A8, A9 and so on).

**The test applied to every finding:** does it make the library easier for a CS 479 student to
read, use and learn from? Speed, generality and third-party use come second, and a change that
buys them with readability counts against itself. Every claim below was checked against the
code; where a behaviour is claimed, it was run. How the counts were made is at the end.

**Not repeated here:** the long-cell survey (`NOTEBOOK_LIBRARY_REVIEW.md`, Task #310). Its F1 to
F9 are done; F10 to F17 sit in Task #294. Where a finding below touches one, it says so.

Each finding gives what is true now, why a student meets it, a recommendation, and a cost:
**S** (an afternoon, one PR), **M** (a day or two, tests and docs move with it), **L** (a week or
more, or touches most notebooks). "Who meets it" uses the P1 to P4 test in `notes/TASKS.md`.

## Summary

- **Two findings are wrong output, not style.** `compute_bleu` silently overrides an explicit
  `tokenize=` (R7), and 23 `Config` fields do nothing when set, six of them knobs a student
  would expect to work, such as `grad_clip` and `beam_size` (R8).
- **The two files a student is most likely to open are the hardest to read.** `config.py` is
  3,448 lines that state each setting three times (R9); `train_model` is one 582-line function
  with its validation loop written twice (R3).
- **The API reference has no page for training or decoding,** the modules 9 and 11 notebooks
  import, and none of the eight most-used classes and functions has a docstring example (R13).
- **The decoder layout rule is broken in one place, and the fast path drops the LSTM** (R4). No
  notebook meets the defect today.
- **Task #272 should close** (R5): it serves speed only, and no assignment waits on it.
- **Of 125 public names, 70 appear in a notebook.** Most of the other 55 are reached through those
  70. Eighteen are dead weight a student can find in the reference and nowhere else (R12).

---

## 1. Module boundaries and the data flow

The modules follow the pipeline a student meets: `preprocessing` (files and DataFrames),
`data_processing` (vocabularies, tensors, loaders), `models`, `training`, `inference`,
`evaluation`. A8's notebook imports from 11 of the 13 modules and packages, in roughly that
order, and each module's docstring opens with one sentence that states its job. That part holds
up; see "Checked and found fine". Three things blur it.

### R1. Colab plumbing lives in two modules

- **Now.** `training_checkpoint.py` defines `is_colab` (:66), `mount_drive` (:83),
  `DRIVE_MOUNT_POINT` and `default_checkpoint_dir` (:146), and `colab.py:40` imports them from
  there. The usage example at the top of `training_checkpoint.py` (:18-25) still shows
  `mount_drive()` and `default_checkpoint_dir(...)`. No notebook calls either: all 21 that install
  TorchLingo call `setup()`, which mounts Drive when asked.
- **Why it matters.** A student who opens `training_checkpoint` to learn how a run resumes meets
  Drive mounting first, and an example that is not how their own notebook does it.
- **Recommendation.** Move the three functions and the constant to `colab.py`, re-export them from
  `training_checkpoint` for a term, and change the module example to `setup(drive=True)` followed
  by `TrainingCheckpointer`.
- **Cost: S.** Two imports and a docstring; `tests/test_training_checkpoint.py` patches these names.

### R2. `preprocessing` and `data_processing` never say which is which

- **Now.** A9's notebook imports `load_data` from `preprocessing` and `NMTDataset` from
  `data_processing`. Neither package docstring states the split. `data_processing/__init__.py:3-4`
  says it "mirrors the top-level `data_processing` module in the repo", which does not exist.
  `torchlingo/__init__.py:3-5` names four modules of thirteen. The API reference opens with
  "TorchLingo is organized into four main modules" (`docs/docs/reference/index.md:7`).
- **Why it matters.** The two names read as synonyms. The split itself is sound (text on one
  side, tensors on the other); nothing says so.
- **Recommendation.** One sentence in each package docstring and in the reference index: files and
  DataFrames in `preprocessing`, vocabularies and tensors in `data_processing`. Do not rename:
  that touches 17 notebooks and every test that drives one.
- **Cost: S.**

### R3. The first training loop a student reads is 582 lines

- **Now.** `train_model` runs from `training.py:199` to `:783`. The lesson, forward, loss,
  backward and step, is about 25 lines at :451-490, among mixed precision, accumulation, resume,
  three schedulers, early stopping, TensorBoard, and three ways of saving (R6).
  - The validation loop is written twice (:528-540 mid-epoch, :676-686 at the end of an epoch).
  - The dictionary saved as the best model is built four times with three different key sets
    (:552-563, :593-602, :721-734, :738-748).
  - Ten `getattr(cfg, "...", fallback)` calls (:310, :316, :331, :514, :519, :528, :582, :590,
    :619, :752) read fields every `Config` has, some with fallbacks that differ from the real
    default (`getattr(cfg, "patience", 0)`; the default is 10).
  - The `patience` counter advances both at each mid-epoch validation (:581-583) and at each
    epoch's end (:750-753), while `PATIENCE`'s documentation in `config.py` counts validation
    intervals only.
- **Why it matters.** Eric's rule is that the machinery runs in the library and the notebook names
  it. A student who follows a notebook's "`train_model` runs the loop" into the source is the
  reader this file most needs to serve, and the core is hard to find.
  `docs/docs/concepts/training.md` teaches the loop with its own code, not this function's.
- **Recommendation.** Extract `_validate(model, loader, loss_fn, device)` and `_model_state(...)`,
  so the epoch loop reads top to bottom; replace the ten `getattr` calls with plain attribute
  reads; count patience in one place and say which. Keep the epoch and batch loops inline: they
  are the lesson. Point the docstring at the lines of the five core steps, or give a ten-line
  sketch of them.
- **Cost: M.** A pure refactor; `tests/test_training_inference.py` and the checkpoint tests cover it.

---

## 2. Two ways to do one thing

| Pair | Reason written where a student finds it? | Finding |
|---|---|---|
| `load_checkpoint` / `load_model` | Yes, `checkpoint.py:1-15` | Fine; the remaining by-hand uses are F11 (Task #169), and tutorial 3 uses `load_checkpoint` on purpose |
| `checkpoint` / `training_checkpoint` | Yes, `training_checkpoint.py:7-11` | Fine, but see R6 |
| `translate_batch` / `translate_to_file` | Yes, `inference.py:1105-1113` | Fine as a pair; see R4 |
| reference decoders / `inference_fast` | Yes, both module docstrings and `concepts/decoding.md` | R4, R5 |
| `Config` default / explicit parameter | Partly | R8 to R10 |
| `compute_bleu(tokenization=)` / `bleu_tokenizer()` | No | **R7** |
| `split_data` / `split_exact` | Partly | R12 |
| `get_model` / `checkpoint.MODEL_CLASSES` | No | R12 |

### R4. The decoder layout rule, a duplicated `translate_batch`, and an LSTM the fast path drops

- **Now.**
  - **The import cycle.** `inference.py:1160` imports `inference_fast.translate_batch` inside
    `translate_to_file`; `inference_fast.py:50` imports `inference` at module level and again at
    :252. The cycle holds only because of the function-level import. This is the break the brief
    names (PR #239, Oct 12).
  - **The copy.** `inference_fast.translate_batch` (:200-337, 139 lines) repeats
    `inference.translate_batch` (:908-1048, 141 lines) and differs only in which
    `beam_search_decode` it calls.
  - **A defect.** The fast beam search accepts only a model with `encode` and `decode`
    (`inference_fast.py:127`). So `translate_to_file(..., decode_strategy="beam")` on a
    `SimpleSeq2SeqLSTM` raises `ValueError`, while the reference `translate_batch` decodes the same
    model. Run at `2ea9bf2`: reference beam returns a result; fast beam and `translate_to_file`
    with beam raise; `translate_to_file` with greedy works. `translate_to_file`'s docstring says
    "Seq2seq model (Transformer or LSTM)" (:1117) and that its output is identical to
    `translate_batch`'s (:1112-1113). Neither `tests/test_decoding_equivalence.py` nor
    `tests/test_translate_to_file.py` has an LSTM fixture (no match for "LSTM" in either).
- **Why it matters.** "Same output, faster" is the promise that lets a student move from the
  readable decoder to the fast one without thinking, and it is false for one of the two
  architectures. Nobody meets it today: tutorial 4 is the only LSTM notebook, and it decodes no file.
- **Recommendation.**
  1. In `inference_fast.beam_search_decode`, hand a model without `encode` and `decode` to the
     reference implementation (one `if`, saying why), and add an LSTM fixture to the
     equivalence suite.
  2. Keep `translate_to_file` in `inference`: four notebooks import it from there, and it is the
     call a student makes. Write the actual rule into `CLAUDE.md`'s "Two decoders": the reference
     module imports the fast one in exactly one place, the bulk file decode, because that is the
     job the fast path exists for. A documented exception reads better than moving the call
     students know.
  3. Leave the 139-line copy. Collapsing it would send a student reading the reference
     `translate_batch` through an indirection, and the equivalence suite already guards against
     drift. Revisit only if the two drift in practice.
- **Cost: S.**

### R5. Task #272 against the Sep 22 descoping: close Task #272

- **Now.** `CLAUDE.md` records that batching across sentences was descoped (Eric, Sep 22).
  Task #272 proposes it, with a measurement: 372 ms a sentence for beam, so about 10 hours to
  beam-decode 100,000 sentences. A13 back-translates by sampling (Eric, Oct 22), so no assignment
  waits on beam. The fast module's docstring already says it pulls one lever of two
  (`inference_fast.py:18-21`), and `inference.py:23-26` warns that a flattened
  `(batch * beam, vocab)` top-k breaks the tie-breaking rule. That is exactly the bookkeeping
  Task #272 would add.
- **Why it matters.** Nothing a student does gets faster. What it adds is the hardest code in the
  library to keep token-identical, in a module students are told they may skip.
- **Recommendation.** Close Task #272 and keep the descoping. Reopen only if a Final Project needs
  beam over more than about 10,000 sentences; until then CTranslate2 is the answer
  `related-work.md` already gives.
- **Cost: S** (a decision).

### R6. `train_model` saves three ways, and one is silent

- **Now.** `save_dir=` writes `model_best.pt` there; `checkpointer=` writes `latest.pt` and
  `best.pt`; and with a `val_loader` but no `save_dir`, `train_model` writes `cfg.checkpoint_path`,
  by default `./checkpoints/model_best.pt` (`config.py:1074-1076`; `training.py:576-579` and
  :736-749, with `cfg.last_checkpoint_path` at :610-613). It prints nothing when it does, and
  `TrainResult.best_checkpoint` stays `None`, because only the `save_dir` branches set
  `best_path`. Tutorial 4 (c10) and tutorial 6 (c9, c13, c16, and c22) take this path: five runs,
  each overwriting the same file in the working directory.
- **Why it matters.** `CLAUDE.md` teaches two mechanisms, `save_dir` for the artifact and
  `checkpointer` for resuming. The third writes a file the student did not ask for, while the
  result says no file was written.
- **Recommendation.** Save the best model only when `save_dir` is given; delete `checkpoint_path`
  and `last_checkpoint_path` from `Config`.
- **Cost: S.**

### R7. Two ways to choose a BLEU tokenizer, and the explicit one loses

- **Now.** `compute_bleu(predictions, references, tokenize="13a", tokenization="auto")`
  (`evaluation.py:199-205`) has two tokenizer arguments. With the default `tokenization="auto"`,
  it scans the first ten references for CJK, Thai, Burmese, Khmer, or Lao characters
  (`_needs_char_tokenization`, :109-128) and, if it finds one, splits every sentence into
  characters and **replaces `tokenize` with `"none"`** (:268-282). Run at `2ea9bf2`:
  - `compute_bleu(zh_hyp, zh_ref, tokenize="flores200")` returns a score whose signature reads
    `tok:none|tokenization:char`. The flores200 the caller asked for is gone.
  - English with one Chinese name in it, `"I met 王 yesterday..."`, is scored by characters
    throughout.

  The notebooks decide the same question another way: `written_without_spaces`
  (`preprocessing/base.py:59`, under three words a line on average) and `bleu_tokenizer`
  (`evaluation.py:51`). A8 c32, A9 c16 and c27, and A11 c20 use those with
  `sacrebleu.corpus_bleu` directly, bypassing `compute_bleu`. A11 c16 and A12 c19 call
  `compute_bleu(mt, references)` with its defaults. So for a student whose language is written
  without spaces, A11 reports char-BLEU in c16 and spBLEU in c20, and A12's BLEU is not the BLEU
  A9 reported.
- **Why it matters.** It breaks `CLAUDE.md`'s first rule for parameters, that the explicit one
  wins, in the module that teaches students to report a signature. The two detectors disagree on
  mixed text. And a Final Project that calls `compute_bleu` on its own language gets a different
  metric from its own A9 baseline, with nothing but the signature to say so.
- **Recommendation.** One detector and one choice. Drop `tokenization="auto"`, or make the
  default `None`, meaning "use `tokenize`"; have callers pass `tokenize=bleu_tokenizer(no_spaces)`;
  keep `"char"` as an explicit option for tutorial 2, which teaches it. Then the
  `sacrebleu.corpus_bleu` calls in A8, A9, and A11 can become `compute_bleu` (F9's remaining half).
- **Cost: S to M.** Small in the library. `tests/test_bleu_signature.py` and tutorial 2's committed
  outputs change. The notebooks follow next offering.
- **Who meets it.** Final Project students this term (P2); A11 and A12 next offering.

---

## 3. The `Config` pattern

Of 36 functions that take `config=`, most follow the rule: the parameter defaults to `None`,
then falls back to `cfg.<field>`. `SimpleTransformer` (`transformer_simple.py:229-270`) and
`NMTDataset` (`dataset.py:20-60`, which even documents the fallback order) are clean examples.
The problems are fields that do nothing, a file that cannot be read, and one fact kept in two
places.

### R8. Twenty-three `Config` fields change nothing, six of them knobs a student would reach for

- **Now.** No code outside `config.py` reads these 23 fields (searched by name, `getattr` reads
  included):
  - **Shadowed by a parameter with its own default (6).** `grad_clip` (default 1.0;
    `train_model(gradient_clip=None)` at `training.py:207` never falls back), `beam_size`
    (the decoders default to 5), `length_penalty` (the decoders take `alpha=0.6`),
    `use_bucketing`, `bucket_boundaries`, and `use_sentencepiece` (`create_dataloaders`,
    `batching.py:272-276`).
  - **Documented as behaviours that do not exist (5).** `tie_embeddings`, `use_compile`,
    `use_packed_projection`, `use_scaled_dot_product_attention`, and `use_greedy`. The comment
    for `USE_COMPILE` in `config.py` promises a "20-50% speedup"; the one for `GRAD_CLIP` says
    "Applied via torch.nn.utils.clip_grad_norm_".
  - **Paths for pipelines nobody runs (12).** `back_trans_src`, `back_trans_tgt`,
    `combined_train_src`, `combined_train_tgt`, `reverse_model_checkpoint`,
    `sentencepiece_model`, `sentencepiece_vocab`, `sentencepiece_src_vocab`,
    `sentencepiece_tgt_vocab`, `output_dir`, `output_translations`, and `output_scores`.

  Of the same family: `train_model` ignores `cfg.device` (`training.py:77-83`), while
  `create_dataloaders` reads it (`batching.py:387`). Counting `gradient_clip`, `alpha`, and the
  literal `beam_size=5` and `batch_size=32` defaults on the decoders and `evaluate_model`, 12
  parameters in the 36 config-taking functions never fall back to the `Config` field they
  shadow. (`seed=0` on the decoders is a different seed from `Config.seed`, and is fine.)
- **Why it matters.** `Config(grad_clip=0.5)`, `Config(beam_size=10)`, or
  `Config(tie_embeddings=True)` is accepted, validated, and ignored, with no error. Nine of the 15
  `train_model` calls in the notebooks pass `gradient_clip=1.0` by hand, which is the workaround
  showing. The rule "explicit parameters win" holds; what a student cannot predict is which
  `Config` fields are wired at all.
- **Recommendation.** Make the rule "`Config` holds the model's shape and the training
  hyperparameters; a decoding choice is a parameter of the call", and delete rather than wire.
  Delete all 23 fields (the 12 paths go with R12's pipelines). Deleting is safe in a way wiring is
  not: `Config.__init__` takes fixed keywords, so a stale `Config(beam_size=10)` raises
  `TypeError` instead of silently doing nothing, whereas wiring `grad_clip=1.0` as a default
  would change the results of the six notebook calls that pass none. Then have `train_model` take
  `device` from `cfg.device` when it is set.
- **Cost: M.** `tests/test_config.py` and the reference page change; no notebook sets any of the 23.

### R9. `config.py` is 3,448 lines, says each setting three times, and describes a mechanism it does not have

- **Now.**
  - **Three statements of each setting.** 98 module constants with comment blocks (:29-785); an
    `__init__` with 101 literal defaults (:816-931); and 100 property and setter pairs, each with
    its own docstring (about 2,090 lines, :1330-3414).
  - **A promise the constants do not keep.** The constants do not feed the defaults: setting
    `config.BATCH_SIZE = 8` leaves `get_default_config().batch_size` at 64 (run at `2ea9bf2`).
    That contradicts `__init__`'s docstring ("defaults from the module-level constants above are
    used", :935-936) and `get_default_config`'s ("seeded from the current module-level values",
    :3421-3422), and the module docstring (:18-20) invites readers to use them. The 96 constants
    that name a field all agree with `__init__` today, but `tests/test_config.py` compares only
    `batch_size` (:114, :257).
  - **Not a dataclass.** `CLAUDE.md` calls `Config` a dataclass; it is not one, and a comment at
    :1316 still refers to "the dataclass machinery".
  - **The notebooks use little of it.** Fourteen fields are set by any notebook: the six of the
    model's shape, plus `batch_size`, `learning_rate`, `num_workers`, `label_smoothing`,
    `max_decode_length`, and the three special-token ids.
- **Why it matters.** `config.py` is where a student goes to learn what `d_model` or
  `label_smoothing` does. They meet 3,448 lines, the same sentence three times, and a
  module-constant interface that silently does nothing.
- **Recommendation.** After R8, make `Config` a `@dataclass` with one field line and one comment
  per setting. Run the validators from `__post_init__`, and from `__setattr__` so that
  `cfg.d_model = ...` still checks `n_heads`. Delete the module constants, or derive them from the
  dataclass defaults for one term. About 75 fields at two or three lines each puts the file near
  300 lines. Fix `CLAUDE.md`'s description in the same PR.
- **Cost: M.** The public names stay; `tests/test_config.py` is rewritten; the reference page
  renders from the dataclass.

### R10. Special-token ids live in `Config` and in the vocabulary, and nothing checks they agree

- **Now.** The decoders start from `cfg.sos_idx` and stop at `cfg.eos_idx`
  (`inference.py:388`, :853, :859, :895). `translate_batch` pads the source with `cfg.pad_idx`
  (:990) but strips the output with the *vocabulary's* ids, through
  `getattr(tgt_vocab, "pad_idx", cfg.pad_idx)` (:982-984). That `getattr` fallback is dead, since
  `BaseVocab.__init__` always sets all four ids (`vocab.py:115-118`). Tutorial 6 c7 has to write
  `Config(..., pad_idx=0, sos_idx=1, eos_idx=2)` to match its own `Vocab`, which is the workaround
  showing.
- **Why it matters.** A vocabulary whose ids differ from the `Config`'s decodes to `max_len` every
  time and returns plausible-looking text, with no error. The library's own vocabularies and
  `train_sentencepiece` take their ids from the `Config`, so this bites only a student who brings
  a vocabulary, which is the Final Project's case.
- **Recommendation.** In `translate_batch` and `translate_to_file`, raise a `ValueError` naming
  both values when `src_vocab` or `tgt_vocab` disagrees with `cfg` on an id. Raising beats quietly
  preferring one: the model was trained with one of them, and only the student knows which.
- **Cost: S.**

---

## 4. Interfaces detected with `hasattr`

### R11. Five model contracts, written in five places

- **Now.** The vocabularies have an explicit contract: `BaseVocab` is an ABC with six abstract
  methods (`vocab.py:71`, :123-193). The models have none. Decoding code tests them five ways:

  | Where | Transformer | LSTM |
  |---|---|---|
  | `_check_decodable`, `inference.py:327-334` | `encode`, `decode` | `src_embed`, `encoder`, `decoder`, `output` |
  | `_decode_steps`, `inference.py:385`, :401-407 | `encode`, `decode` | `encode_source`, `decode_step`; else the four modules, plus `tgt_embed` by `getattr` |
  | reference beam, `inference.py:806-807` | `encode`, `decode` | `encode_source`, `decode_prefix` |
  | fast beam, `inference_fast.py:127` | `encode`, `decode` | none (R4) |
  | `attention_for_sequence`, `inference.py:296-305` | `forward(..., return_attention=True)`, by catching `TypeError` | the same |

  The four-module path serves only a test double, `DummyLSTMModel`
  (`tests/test_training_inference.py:52`). `SimpleSeq2SeqLSTM` always takes the
  `encode_source` and `decode_step` path (`lstm_simple.py:186`, :304, :337). `CLAUDE.md`'s
  "Interfaces" paragraph still describes the four-module contract, the one only the test double
  uses.
- **Why it matters.** To learn what a model must provide to be decoded, a student has to read five
  checks that disagree, and the error message names a different contract depending on which
  decoder raised it. For vocabularies, one class answers the question. Duck typing also hides
  that an LSTM's decoder step and a Transformer's `decode` are the same idea behind different
  signatures, which is the point `beam_search_decode`'s docstring makes (:731-734).
- **Recommendation.** Write the two contracts once, as two `typing.Protocol` classes beside the
  models (`EncoderDecoder`: `encode`, `decode`; `RecurrentEncoderDecoder`: `encode_source`,
  `decode_step`, `decode_prefix`), each method with a one-line docstring. Use one helper,
  `_model_kind(model)`, in all five places. Delete the four-module path and give the test double
  `encode_source` and `decode_step`. Protocols rather than an ABC, because a student's own model
  need not inherit anything, and an `isinstance` check against a `runtime_checkable` Protocol
  still reads as one line. Update `CLAUDE.md`'s Interfaces paragraph.
- **Cost: M.** Inference code and two test doubles; the decoding-equivalence suite guards it.

---

## 5. What the notebooks actually use

Of 125 public top-level names (120 in `torchlingo`, 5 in `torchlingo_harness`), **70 appear in at
least one notebook's code**. Of the other 55, most are reached through those 70: return types
(`LoadedModel`, `CometScores`, `BeamStep`, `PaddingReport`, and others), helpers that
`train_model`, `setup`, or the decoders call, the LSTM's attention modules, and `sample_decode`
and `add_noise` behind `decode_strategy=`. These are fine.

### R12. Eighteen public names nothing on a student's path reaches

**Dead: no notebook, no docs walk-through, no library caller (5).** The safe cuts.

| Name | Where | Note |
|---|---|---|
| `get_model` | `models/__init__.py:15` | A second registry; `checkpoint.MODEL_CLASSES` (:43-46) is the one used |
| `create_masks` | `transformer_simple.py:551` | Tests only |
| `get_config_dict` | `config.py:3430` | Returns the module constants, not a config |
| `print_config` | `config.py:3442` | Prints the default, not yours |
| `save_translations` | `evaluation.py:654` | Tests only |

**The multilingual pipeline (7 functions, 549 lines).** `add_language_tags` and
`preprocess_multilingual` (`multilingual.py`); `add_language_tags_multi`,
`create_multilingual_dataset`, `save_multilingual_splits`, `get_language_tags_from_vocab`, and
`ensure_language_tags_in_vocab` (`multilingual_helpers.py`).
- No notebook calls them. They keep 11 `Config` path and tag fields alive (`back_trans_*`,
  `combined_train_*`, `reverse_model_checkpoint`, `multi_*`, `test_en_x_file`, `test_x_en_file`,
  and `lang_tag_*`).
- They have an API reference page and two pages outside the nav: `MULTILINGUAL_QUICKSTART.md`,
  and `MULTILINGUAL_ANALYSIS.md`, which imports four names that do not exist (`LanguageRegistry`,
  `MultilingualNMTDataset`, `MultilingualVocab`, and `translate_multilingual`).
- `preprocess_multilingual` is a library answer to A14's `tag` and `both_directions` exercise, by a
  different mechanism (a `<2X>` token rather than A14's code columns), so a student who finds it
  learns the wrong one.

**The config-file pipelines (3).** `preprocess_base` (`base.py:342`), `preprocess_sentencepiece`,
and `apply_sentencepiece` (`sentencepiece.py:238`, :149). They read and write the paths named in
`Config`. No notebook works that way, and they are what keeps `split_data` (by ratio) beside
`split_exact` (by exact size), which A8 uses.

**`evaluate_model`** (`evaluation.py:468`). No notebook calls it, though `concepts/decoding.md:8`
and :21 present it as the way to score. It decodes through the reference path (:538), the slow
path the reference itself warns about, and it scores BLEU through R7's `tokenization="auto"`.

**`MeCabVocab` and `JiebaVocab`** (`vocab.py:835-1346`, about 510 of the file's 1,346 lines). No
notebook or roadmap entry uses them; the course handles languages without spaces with
`CharVocab` and SentencePiece. A checkpoint cannot carry them (`checkpoint.py:96-98`), so
`load_model` needs them passed back by hand. The reference index lists both under "Most Used
Classes".

- **Why it matters.** The API reference is part of what a student reads, and a fifth of it
  describes code no assignment uses. The multilingual functions also hand over an exercise's
  answer in the wrong form.
- **Recommendation.** Cut the 5 dead names, the multilingual pipeline with its 11 fields and three
  pages, and the three config-file pipelines with `split_data` and their path fields. Move MeCab
  and Jieba out of "Most Used" now, and decide separately whether to keep them (a Final Project in
  Chinese or Japanese is the case for keeping them). Make `evaluate_model` the one scoring call,
  using `bleu_tokenizer` and the fast decoder (after R7), or cut it and fix
  `concepts/decoding.md`. Note each cut in `CHANGELOG.md`, since a PyPI user may call these.
- **Cost: M** in total, as two or three PRs by group. No notebook changes.

**Plumbing every notebook repeats.** The long-cell survey found it, and five pull requests (PR
#388, PR #389, PR #391, PR #392, and PR #394) moved most of it. What remains is in Task #294 and Task #291. I found no new
candidate.

---

## 6. Docstrings and docs as the student's first reading

### R13. No reference page for training or decoding, and no examples on the most-used names

- **Now.**
  - **No reference page for training or decoding.** The API reference renders no page for
    `torchlingo.training`, `torchlingo.inference`, `torchlingo.inference_fast`, or
    `torchlingo_harness` (every `:::` directive under `docs/docs/reference/` checked).
    `concepts/decoding.md:638` lists "API reference for `torchlingo.inference` and
    `torchlingo.inference_fast`" as plain text, because there is nothing to link to. The
    preprocessing page renders five functions and omits `read_lines`, `write_lines`, and
    `written_without_spaces`, which five, three, and two notebooks call.
  - **No examples on the names notebooks call most.** 79 of the 125 public names have an
    Examples section, and 71 have a line that runs. Among those without: `train_model`
    (9 notebooks), `SimpleTransformer` (8), `NMTDataset` (6), `SentencePieceVocab` (6),
    `collate_fn` (6), `translate_batch` (3), the reference `beam_search_decode`, `SimpleVocab`,
    `SimpleSeq2SeqLSTM`, `load_checkpoint`, `TrainResult`, and the four names notebooks import
    from `torchlingo_harness.nllb` (`NLLBTranslator`, `nllb_codes`, `load_nllb`, and `fine_tune`).
  - **Names in the docs that do not exist.** `train_test_split` (`reference/preprocessing/index.md`
    :52, :65, :75, :105, :115, :127; `reference/preprocessing/base.md:116`, :118),
    `set_default_config` (`reference/preprocessing/multilingual.md:184`, :193), and `SimpleLSTM`
    (`concepts/models.md:11`). Nothing executes the 278 Python blocks in `docs/docs/**/*.md`.
  - **Stale package docstrings**, as in R2. Two task numbers sit in library comments
    (`training.py:79`, `inference.py:1053`) that a student cannot look up.
- **Why it matters.** `CLAUDE.md` makes "documentation that executes" goal 3 and asks for runnable
  Examples on public functions. The functions where a student most needs one, those every
  assignment calls, are the ones without. A student who follows the reference to
  `train_test_split` gets an `ImportError`.
- **Recommendation.**
  1. Add `reference/training.md`, `reference/inference.md` (both decoder modules), and
     `reference/harness.md`; add the three line-file functions to `base.md`.
  2. Write a runnable Example for each of the 15 names listed, on the tiny `SimpleVocab` and
     `SimpleTransformer` pattern that `translate_to_file`'s docstring already uses
     (`inference.py:1146-1157`).
  3. Fix the three wrong names.
  4. Later, and separately: a check that every `from torchlingo... import` in `docs/` resolves.
     That check found the wrong names above in seconds; how it works is under "How this was
     checked".
- **Cost: S** for parts 1, 3, and 4; **M** for part 2.

---

## The short list: what I would act on first

Ordered by who meets it, then by cost.

| Rank | Finding | Cost | Who meets it |
|---|---|---|---|
| 1 | R7: one BLEU tokenizer choice; an explicit `tokenize` always wins | S to M | Final Project students this term (P2) |
| 2 | R8: delete the 23 `Config` fields that do nothing | M | Any student who sets one (P2) |
| 3 | R13 parts 1 and 3: reference pages for training and decoding; fix three wrong names | S | Students following the docs (P2) |
| 4 | R10: raise when vocabulary and `Config` ids disagree | S | Final Project students who bring a vocabulary (P2) |
| 5 | R4 part 1: an LSTM through the fast beam, and an LSTM equivalence fixture | S | Nobody today; guards a promise (P3) |
| 6 | R6: save only to `save_dir` | S | Tutorials 4 and 6 (P3) |
| 7 | R5: close Task #272 | S | Nobody; removes a task (P3) |
| 8 | R9: `Config` as a dataclass, after R8 | M | Anyone reading `config.py` (P3, or P4 for the next offering) |
| 9 | R3: `train_model` that reads top to bottom | M | Anyone reading the loop (P3) |
| 10 | R13 part 2: Examples on the 15 most-used names | M | Students reading docstrings (P3) |
| 11 | R11: two Protocols for the models | M | Students reading the decoders (P3) |
| 12 | R12: cut the unused API, by group | M | Readers of the reference (P3) |
| 13 | R1, R2: Colab plumbing in one module; one sentence per package | S | Few (P3) |

R8 comes before R9, since the dataclass is smaller once the dead fields are gone. R7 comes before
the F9 notebook pass, since that pass wants `compute_bleu` to be the one BLEU call.

---

## Checked and found fine

- **Pipeline order and module jobs.** Each module's docstring opens with its job in one sentence,
  and A8's imports run in pipeline order across 11 modules and packages. The exceptions are R1
  and R2.
- **The vocabulary contract.** `BaseVocab` is an explicit ABC (`vocab.py:71`) with a runnable
  example in its docstring.
- **Model and training checkpoints.** The difference is explained in both module docstrings
  (`checkpoint.py:1-15`, `training_checkpoint.py:7-11`). `load_model` rebuilds the model and its
  vocabularies from the file alone, and declines a vocabulary it cannot carry rather than guessing
  (`checkpoint.py:96-98`).
- **The reference decoders are readable.** The beam search loop is about 50 lines
  (`inference.py:853-906`) with tracing alongside; `_canonical_topk` and `_rank_key` carry the
  tie-breaking rule documented at the top of the module; and the reference warns once on a large
  input, naming the fast path (:73-98).
- **The decoding contract.** `tests/test_decoding_equivalence.py` enforces it for Transformers,
  and `concepts/decoding.md` explains both decoders and when to use which.
- **One source for generation length.** Every decoder and `evaluate_model` read
  `max_decode_length` from `Config`, with a comment at each site saying why.
- **Decoding strategy names are validated** (`_check_strategy`, `inference.py:337`), not silently
  defaulted.
- **`SimpleTransformer` and `NMTDataset`** follow the `Config` pattern exactly and record their
  resolved settings for checkpoints.
- **The harness boundary holds.** Nothing in `torchlingo` imports `torchlingo_harness`, and
  `import torchlingo` does not load matplotlib (`__init__` does not import `visualization`).
- **`SimpleTransformer` wraps `nn.Transformer`, so its attention is not in the library's source.**
  That is the right split: 8b and 8c have students write attention themselves and compare it with
  the model through `capture_cross_attention` and `capture_self_attention`.
- **`setup()` prints `version_label()`** in every notebook that installs TorchLingo (21 of 21).
- **`diagnostics` returns one `CheckResult` shape throughout**, and the notebooks call it where the
  lesson is: five use `check_contamination`, four use `uniform_loss`.
- **chrF is consistent.** One setting serves the whole library (`CHRF_WORD_ORDER = 0`,
  `evaluation.py:47`), so the long-cell survey's F6 drift is closed.
- **`num_workers` and bucketing defaults** match `CLAUDE.md`'s standing decisions, and every course
  notebook that builds a loader passes `num_workers`.
- **Explanatory comments.** The library's 13 "this used to..." comments mostly explain a failure
  mode a student can learn from; only the two task numbers in R13 are noise.

## How this was checked

- **Public names**: every top-level `def` and `class` not starting with `_`, found with `ast`, in
  `src/torchlingo` and `src/torchlingo_harness`. "Used by a notebook" means the name appears as a
  word in a notebook's code cells. That overcounts slightly (Lecture 5 defines its own
  `length_correlation`) and never undercounts.
- **Unread `Config` fields**: each `Config.__init__` parameter searched as `.name`, as
  `getattr(cfg, "name"`, and as `"name"`, in every source file other than `config.py`.
- **Shadowed parameters**: every function taking `config` with a parameter named like a field (or
  `gradient_clip`, or `alpha`) whose default is not `None`.
- **Docs imports**: every `from torchlingo... import` in a fenced Python block in
  `docs/docs/**/*.md`, resolved against the package.
- **Behaviour claims** (R4, R7, and R9) were run against `src/` at `2ea9bf2` with the repository's
  `.venv`. The scripts are in the session scratchpad, not the repository.
