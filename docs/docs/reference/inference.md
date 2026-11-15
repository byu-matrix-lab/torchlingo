# Decoding

Turning a trained model's scores into translations. Two modules hold the same decoders:

| module | what it is for |
| --- | --- |
| `torchlingo.inference` | the **reference** decoders, written to be read line by line; they define the correct output |
| `torchlingo.inference_fast` | the same decoders, faster on large inputs; token-identical output, checked by `tests/test_decoding_equivalence.py` |

For how greedy decoding and beam search work and why the fast version is faster, read
[Decoding](../concepts/decoding.md) first. This page is the API.

## Translate a few sentences

```python
from torchlingo.inference import translate_batch

translate_batch(model, ["Good morning.", "Thank you."], src_vocab, tgt_vocab)
translate_batch(model, sentences, src_vocab, tgt_vocab, decode_strategy="beam", beam_size=5)
```

`decode_strategy` is one of `DECODE_STRATEGIES`:

| strategy | does | for |
| --- | --- | --- |
| `"greedy"` (default) | takes the most likely token at each step | quick checks |
| `"beam"` | keeps the `beam_size` best partial translations | scoring a system; usually better |
| `"sample"` | draws each token at random, shaped by `temperature` and `top_k` | making back-translations |
| `"beam+noise"` | beam search, then corrupts each translation with `add_noise` | making back-translations |

The two random strategies take a `seed`, and each sentence gets its own random stream, so a
sentence comes out the same whatever batch it is decoded in.

## Translate a test set or a corpus to a file

```python
from torchlingo.inference import translate_to_file

translate_to_file(model, sentences, src_vocab, tgt_vocab, OUT_DIR / "test.hyp",
                  decode_strategy="beam")
```

It writes one translation per line, a chunk at a time, and a run that stops partway resumes
where it stopped when called again with the same arguments. It decodes through
`inference_fast.translate_batch`, so the file holds exactly what the reference
`translate_batch` would return, sooner.

## The vocabularies and the config must agree

The decoders start each translation with `config.sos_idx` and stop at `config.eos_idx`, while
the vocabulary reads the output with its own ids. `translate_batch` and `translate_to_file`
raise a `ValueError` naming each id that differs. Vocabularies the library builds take their
ids from the config, so this meets only a vocabulary built some other way.

## When to use the fast module

Use the reference to read, teach or debug, and for small inputs. Switch to the fast module
when beam decoding thousands of sentences: the reference warns, once per process, when an input
is large enough for that to matter. The two functions share their names, so switching is an
import:

```python
from torchlingo.inference_fast import translate_batch
```

## API: `torchlingo.inference`

::: torchlingo.inference
    options:
      show_source: true
      members:
        - translate_batch
        - translate_to_file
        - greedy_decode
        - beam_search_decode
        - sample_decode
        - add_noise
        - translate_lines_to_file
        - DECODE_STRATEGIES
        - NOISE_FILLER

## API: `torchlingo.inference_fast`

::: torchlingo.inference_fast
    options:
      show_source: true
      members:
        - translate_batch
        - beam_search_decode
