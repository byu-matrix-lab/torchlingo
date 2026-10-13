## Gaps

What the material still does not teach, ordered by how much it costs a student. Three gaps
listed here before are closed: tutorial 6 teaches diagnosis by breaking a model on purpose and
watching each check fire; tutorial 2 teaches where BLEU misleads; tutorial 5 reads a
Transformer's attention.

**Training dynamics.** `concepts/training.md` covers the loop, optimizers, schedulers,
clipping and early stopping. It does not cover what a student does when training goes wrong:
batch size against learning rate, or what overfitting looks like on a small corpus.

**How much data is enough.** Still open in general. The material has a measured answer to a
better question, *which lever are you actually pulling?*: at this scale, training longer
helped and more data did not, and the first attempt to answer it was wrong in a way only a
control run caught. The numbers are in `notes/reports/training-budget.md`. The episode is worth
teaching directly.

**Inference cost in practice.** `concepts/decoding.md` covers beam search's cost and why a
GPU at batch size 1 is mostly overhead. Nothing covers model size against latency, which is
what a student meets when a Colab session is slow.
