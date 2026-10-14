# Model Checkpoints

A finished model in one file, with everything needed to use it again: the weights,
the settings the model was built with, and its vocabularies.

## Why the vocabulary goes in the file

A model's weights are numbers indexed by token id. Which word or piece each id
means lives in the vocabulary, not the weights, so a model reloaded with a
different vocabulary of the same size loads without error and translates into
fluent nonsense. Keeping the vocabulary in the same file as the weights is what
makes that mistake impossible.

## Train, then reload in a new session

`train_model` writes such a file whenever it saves a model: with
`save_dir=OUT_DIR / "best"`, that is `OUT_DIR / "best" / "model_best.pt"`. It
takes the vocabularies from `train_loader.dataset` (an `NMTDataset`, which is what
`create_dataloaders` builds), or from `src_vocab=` and `tgt_vocab=` if you pass
them.

A day later, in a new session, one call rebuilds the model and both vocabularies:

```python
from torchlingo.checkpoint import load_model
from torchlingo.inference import translate_batch

model, src_vocab, tgt_vocab = load_model(OUT_DIR / "best" / "model_best.pt")
translate_batch(model, ["Good morning."], src_vocab, tgt_vocab)
```

Nothing is re-declared: no `Config`, no model sizes, no rebuilding the vocabulary
from the training file. The model comes back on the GPU if there is one, in
`eval()` mode.

## What the file holds

| key | what |
| --- | --- |
| `model_state_dict` | the weights |
| `model_class` | `"SimpleTransformer"` or `"SimpleSeq2SeqLSTM"` |
| `model_config` | every setting the model was built with, so `model_class(**model_config)` builds the same architecture |
| `src_vocab`, `tgt_vocab` | the vocabularies as plain data; a SentencePiece model as its bytes. A vocabulary shared by both sides is stored once |
| `epoch`, `val_loss`, `train_losses`, ... | the training state at the save, from `train_model` |

`SimpleVocab`, `CharVocab` and `SentencePieceVocab` are recorded. Any other kind
(`MeCabVocab`, `JiebaVocab`) is left out, and you pass it to `load_model` yourself:
`load_model(path, src_vocab=..., tgt_vocab=...)`.

## Files saved before this

Model files written by earlier versions hold the weights but not the model's
settings. They load exactly as they always did:

```python
from torchlingo.checkpoint import load_checkpoint

model = SimpleTransformer(len(src_vocab), len(tgt_vocab), config=config)
model.load_state_dict(load_checkpoint(path)["model_state_dict"])
```

`load_model` refuses such a file with that instruction rather than guessing the
settings.

## API

::: torchlingo.checkpoint
    options:
      show_source: true
      members:
        - load_model
        - LoadedModel
        - save_checkpoint
        - load_checkpoint
        - describe_model
