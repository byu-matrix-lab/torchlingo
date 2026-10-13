## What exists

### Tutorials

| | Runs on | Depends on |
|---|---|---|
| 1. Data and Vocabulary | `data/sample_train.tsv` | nothing |
| 2. Train a Tiny Model | `data/example.tsv` | 1 |
| 3. Inference and Beam Search | tutorial 3's checkpoint | **2, at runtime** |
| 4. Attention and Alignment | synthetic reversal task | 2 |
| 5. Translating Unseen Sentences | `data/pretrained/` | 1-3 conceptually |

### Concept pages

`what-is-nmt.md`, `data-pipeline.md`, `vocabulary.md`, `models.md`, `training.md`,
`decoding.md`, `when-it-fails.md`.

### Measured artifacts

Generated into `docs/docs/_generated/`, each from a script, none hand-typed:
`decode_bench` (#12), `decoding_sweep` (#40), `alignment_diagnosis` (#46),
`realign_report` (#29).
