## Redundancy

Not necessarily wrong, but currently undeliberate.

**Beam search appears four times**: explained in `concepts/decoding.md`, reimplemented
from scratch in tutorial 8, visualized via `format_beam_search` (#39), and measured in
#40. The reimplementation is defensible — writing it yourself is the lesson — and
tutorial 8 explicitly reconciles its version against the library's. Worth deciding
deliberately rather than by accumulation.

**Attention appears three times**: `concepts/models.md`, tutorial 4, and
`reference/visualization.md`.

*Corrected 2026-09-18, and this is a gap rather than redundancy.* All three cover the
**LSTM**. `SimpleTransformer` has no attention-returning path at all, so a student who
follows the tutorials onto the Transformer — which is what tutorials 3 and 7 train, and
what the pretrained checkpoint is — cannot inspect attention on the model they are
actually using. Tutorial 4 teaches the concept honestly on an LSTM and a synthetic task;
nothing carries it across. See #34, whose scope was recorded too small for the same
reason.

**The corpus repair story now appears twice**: `concepts/data-pipeline.md` (#46, #29) and
`scripts/realign_corpus.py`'s docstring. These are aimed at different readers and the
overlap is probably correct.
