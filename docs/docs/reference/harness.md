# NLLB Harness

`torchlingo_harness.nllb` puts Meta's NLLB-200, one model that translates between any two of
200 languages, behind the same decoding strategies TorchLingo's own models use.

## Why it is separate

TorchLingo is the library the course learns from, about the models students build. Some
assignments also call someone else's model, as a comparison or a shortcut: A12 sets NLLB beside
an LLM, A13 may back-translate with it, and A14 fine-tunes it on a student's pairs. That code
ships with TorchLingo but lives outside the `torchlingo` package, so `import torchlingo` never
loads it and nothing in the library depends on it. It needs Hugging Face `transformers`.

## Load a translator and call it

```python
from torchlingo_harness.nllb import load_nllb

nllb = load_nllb("tel_Telu", "eng_Latn", device=device)
nllb(["...", "..."])                                        # greedy
nllb(sentences, decode_strategy="beam", beam_size=5)        # as translate_batch decodes
```

Languages are named by their FLORES-200 codes, such as `eng_Latn` and `tel_Telu`. A code NLLB
does not know raises a `ValueError` pointing to the list. The translator takes the same
`decode_strategy` values as [`translate_batch`](inference.md), so a notebook can switch between
a student's own model and NLLB without changing how it decodes.

## Both directions from one model

A tokenizer holds one source language, so two translators sharing one would silently change
each other's direction. `NLLBTranslator.between` builds a translator for one direction of a
model already in memory, and sets its source language each time it is called:

```python
from torchlingo_harness.nllb import NLLBTranslator

to_english = NLLBTranslator.between(model, tokenizer, code, "eng_Latn", device)
from_english = NLLBTranslator.between(model, tokenizer, "eng_Latn", code, device)
```

`nllb_codes(tokenizer)` returns every code a tokenizer knows. `fine_tune` is the training loop
A14 uses, resumable through a [`TrainingCheckpointer`](training_checkpoint.md).

## API

::: torchlingo_harness.nllb
    options:
      show_source: true
      members:
        - load_nllb
        - NLLBTranslator
        - nllb_codes
        - fine_tune
        - coded_batch
        - NLLB_MODEL
        - FLORES_200_LIST
