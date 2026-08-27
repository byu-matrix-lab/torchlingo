# TorchLingo vs. OpenNMT — Feature & Fitness Comparison for Student MT Projects

*Assessed 2026-08-22 against TorchLingo v0.0.8 (`main` @ `64b30de`) and OpenNMT-py 3.x.*

## Executive summary

TorchLingo and OpenNMT are not really competitors. **OpenNMT is a production research
toolkit you configure; TorchLingo is a teaching codebase you read.** For a semester-length
MT course, that difference matters more than any feature checklist.

That calculus is further shifted by maintenance status: **OpenNMT-py is officially in
maintenance mode**, and the OpenNMT team now directs new development to
[Eole](https://github.com/eole-nlp/eole). Recommending OpenNMT-py to students starting
a project in 2026 means pointing them at a frozen codebase.

**Recommendation in one line:** use TorchLingo to *understand* NMT, and Eole (or Fairseq2 /
HuggingFace) when a project needs to *compete* on BLEU or ship.

## Feature comparison

| | TorchLingo (v0.0.8) | OpenNMT-py 3.x |
|---|---|---|
| Size | ~8.5k LOC, of which `config.py` is 3.3k lines *mostly docstrings* — real implementation is ~4–5k | ~30k+ LOC across a plugin/registry architecture |
| Models | Transformer enc-dec (`SimpleTransformer`), LSTM seq2seq | Transformer variants, RNN/CNN, LM, encoder pretraining, adapters/LoRA |
| Interface | Python API you import and call | YAML config + `onmt_train` / `onmt_translate` CLI |
| Tokenization | SentencePiece (BPE + unigram) | SentencePiece, BPE, pyonmttok, on-the-fly transforms pipeline |
| Training | AMP, gradient accumulation, grad clipping, Noam + cosine-warmup schedulers, label smoothing, TensorBoard, bucketed batching | All of the above **plus** multi-GPU/DDP, multi-node, token-based batching, checkpoint averaging, resumable sharded data |
| Decoding | Greedy + beam search w/ Wu et al. length normalization | Beam, sampling, top-k/top-p, n-best, coverage penalty, replace-unk, ensembles |
| Evaluation | sacreBLEU built in — BLEU, chrF, TER (`evaluation.py`) | Relies on external sacreBLEU scripts |
| Multilingual | Language-tag multilingual + back-translation helpers | Via data transforms; more manual |
| Deployment | None | CTranslate2 export, quantized fast inference, REST server |
| Tests | 425 unit tests, all passing locally (21 skipped) | Large suite, but CI now frozen |
| Maintenance | Active but young — 34 commits, Jan–Jul 2026 | **Maintenance mode**; no new PyPI release in ~12 months |
| License | AGPL-3.0-or-later | MIT |

## Where TorchLingo wins: readability

TorchLingo wins on the thing students actually struggle with — reading the code.

- A student can open `src/torchlingo/models/transformer_simple.py` (328 lines) and see the
  entire encoder-decoder in one sitting.
- `src/torchlingo/inference.py` implements beam search in ~85 readable lines with the
  length-normalization formula sitting right there in the function body.
- `src/torchlingo/config.py` is effectively a documented hyperparameter glossary — 3,322
  lines, ~400 docstrings, one per constant.

In OpenNMT-py, tracing what happens to a single batch means following it through a
transforms pipeline, a model-builder registry, and a trainer abstraction. Students
routinely spend a week on plumbing before understanding a single attention head.
OpenNMT's batched beam search is excellent engineering and poor pedagogy.

Supporting material is aimed squarely at learners: three Jupyter tutorials
(data/vocab → tiny model → beam search), concept pages on the data pipeline and
vocabulary, and Google-style docstrings with runnable examples throughout. The
Cebuano→Mandarin example under `examples/` suggests low-resource coursework is an
explicit design target.

## Where TorchLingo walls you

Verified against the source, not the README:

- **Beam search is batch-size-1 only** (`inference.py` raises on `src.size(0) != 1`) and
  re-encodes the full prefix at every step — no incremental decoding or KV cache. Decoding
  a 3k-sentence test set is genuinely slow.
- **No distributed training.** `config.py:663` states outright that multi-GPU "requires
  custom DataParallel setup" — i.e. it is not implemented. Single device only.
- **The LSTM has no attention mechanism.** `models/lstm_simple.py:117` notes attention
  masks are "currently unused." This blocks the classic Bahdanau/Luong attention lesson
  and the with-vs-without-attention ablation on the LSTM path.
- **No checkpoint averaging, no ensembling, no export path** (no CTranslate2, ONNX,
  TorchScript, or quantization).
- **Beam search is Transformer-only** — it requires `encode()`/`decode()` methods.

## Eole — the OpenNMT successor

[Eole](https://github.com/eole-nlp/eole) ("Extensible Open Language Modeling Toolkit") is
a spin-off of OpenNMT-py by the same team, and is where active development now happens.
It is a full refactor rather than a rename: encoders, decoders, adapters, model classes,
the trainer, distributed training, and inference were all reworked, and configuration
moved to Pydantic-based schemas.

**Scope shift.** Eole is no longer an NMT-first toolkit. It targets language modeling
broadly — encoder-only, decoder-only, and encoder-decoder transformers — with substantial
LLM and multimodal support. NMT remains supported, but it is now one use case among many.

**Notable capabilities**

- Converters for HuggingFace models: Llama-3.x, Mistral-3.1, Gemma-3, Qwen3.5 (incl.
  vision), Phi-2/3, Whisper, and translation-specific models such as Hunyuan-MT-7B
- 8-bit and 4-bit quantization with LoRA adapters; finetuning 7B–13B models on a single
  24GB GPU
- Tensor parallelism for models exceeding single-GPU memory
- `torch.compile`-compliant inference reaching vLLM-comparable speeds; Flash Attention
  with in-place KV cache; custom CUDA kernels for RMSNorm, RoPE, and activations
- Pure-BF16 training via Kahan summation
- COMET and MetricX scoring built into the training loop — a real upgrade over BLEU-only
  evaluation for MT coursework
- GGUF conversion and AutoRound int4 quantization; streaming chatbot serving
- Dynamic data transforms applied at load time (inherited from OpenNMT-py)

**Practicalities**

- Version 0.6.0, actively developed
- **MIT licensed** — notably more permissive than TorchLingo's AGPL-3.0, which matters if
  student work might be released or commercialized
- Requires Python ≥ 3.11 and PyTorch ≥ 2.10, < 2.13 — a tighter, more modern dependency
  window than TorchLingo's Python ≥ 3.10 / PyTorch ≥ 2.0
- Docker images available (torch 2.11.0)

**Caveat for teaching.** Eole is *further* from student-readable than OpenNMT-py was. The
LLM machinery, quantization paths, CUDA kernels, and tensor parallelism make it a
formidable research platform and a poor first codebase. Its COMET/MetricX integration and
modern MT model support are strong arguments for capstone and thesis work, not for week 3
of an intro course.

## Recommendation by project type

| Project type | Recommendation |
|---|---|
| Intro NMT coursework — implement attention, ablate positional encodings, examine BPE and morphology | **TorchLingo.** Designed for exactly this. |
| Low-resource / back-translation experiments at class scale | **TorchLingo.** Built-in multilingual tagging and back-translation helpers. |
| Capstone or thesis targeting competitive BLEU on a real benchmark | **Eole**, Fairseq2, or HuggingFace `transformers` + Marian/NLLB finetuning. Do **not** start on OpenNMT-py. |
| Anything needing deployment or inference speed | **Eole** (torch.compile / Flash Attention / quantization) or CTranslate2. |

A pragmatic hybrid many courses land on: TorchLingo for weeks 1–8 (build and understand),
then a stronger toolkit for the final project.

## Contribution opportunities

The three gaps identified above are well-scoped student contribution projects in their own
right, and the repo has clean test scaffolding (425 tests) to support them:

1. **Batched beam search with incremental decoding / KV cache** — the highest-impact fix.
2. **Attention for the LSTM decoder** — restores a core pedagogical exercise.
3. **Multi-GPU training** via DDP.

`CONTRIBUTING.md` invites PRs.

## Sources

- [OpenNMT-py is now in maintenance mode — Use Eole instead](https://forum.opennmt.net/t/opennmt-py-is-now-in-maintenance-mode-use-eole-instead/5792)
- [OpenNMT-py releases](https://github.com/OpenNMT/OpenNMT-py/releases)
- [OpenNMT-py on PyPI](https://pypi.org/project/OpenNMT-py/)
- [eole-nlp/eole on GitHub](https://github.com/eole-nlp/eole)
- [Eole documentation](https://eole-nlp.github.io/eole/docs/)
- TorchLingo source at `main` @ `64b30de` (direct code inspection)
