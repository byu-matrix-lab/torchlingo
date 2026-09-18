# TorchLingo — Curriculum Audit

Opened 2026-09-17. An instructor's working document, not student-facing: an honest audit
has to name gaps and redundancy, and that is not writing to hand a class.

The material here was built task by task. Each piece was justified on its own merits and
none of it was ever checked against a curriculum. This is that check.

Student-facing sequencing lives in `docs/docs/tutorials/index.md`. That page is a table
of contents; this one is the audit behind it.

---

## What exists

### Tutorials

| | Runs on | Depends on |
|---|---|---|
| 1. Data and Vocabulary | `data/sample_train.tsv` | nothing |
| 2. Train a Tiny Model | `data/example.tsv` | 1 |
| 3. Inference and Beam Search | tutorial 2's checkpoint | **2, at runtime** |
| 4. Attention and Alignment | synthetic reversal task | 2 |
| 5. Translating Unseen Sentences | `data/pretrained/` | 1-3 conceptually |

### Concept pages

`what-is-nmt.md`, `data-pipeline.md`, `vocabulary.md`, `models.md`, `training.md`,
`decoding.md`.

### Measured artifacts

Generated into `docs/docs/_generated/`, each from a script, none hand-typed:
`decode_bench` (#12), `decoding_sweep` (#40), `alignment_diagnosis` (#46),
`realign_report` (#29).

---

## Sequencing, including the parts nobody wrote down

Two dependencies are load-bearing and were invisible until this audit.

**Tutorial 3 cannot run without tutorial 2.** It loads the checkpoint tutorial 2 saves.
`scripts/execute_notebooks.py` encodes this — it runs the notebooks in one shared working
directory, in filename order, and lists tutorial 3 as needing the corpus even though it
never names it. A student who opens tutorial 3 in Colab on its own gets a file-not-found
error and no explanation.

**Tutorial 3's model is too small to teach what tutorial 3 is about.** Its beam-size
sweep prints five identical rows because the model is decisive. This is why #40's
measurement had to be done on tutorial 5's checkpoint instead, and why #50 exists. The
sequencing implication is real: *the lesson about beam search requires a model that is
wrong often enough to be uncertain, and the tutorial that teaches beam search does not
have one.*

**Tutorial 4 is independent.** It trains a synthetic reversal task with known ground
truth, which is what lets it measure alignment accuracy rather than assert it. It could
be moved without breaking anything.

---

## Learning outcomes

Stated as what a student can do afterwards. **These are proposed, not confirmed** — they
are reverse-engineered from the material, and they need checking against the course this
feeds.

**Tutorial 1.** Load a parallel corpus; explain why a vocabulary needs `<pad>`, `<sos>`,
`<eos>` and `<unk>`; predict what happens to an out-of-vocabulary word at inference.

**Tutorial 2.** Train a Transformer end to end; read a loss curve well enough to tell
"still learning" from "converged"; recognize that a loss near `ln(vocab_size)` means the
model has learned nothing.

**Tutorial 3.** Implement greedy and beam search from scratch; state what beam search
buys and what it costs; explain why two implementations of the same algorithm can
disagree on ties.

**Tutorial 4.** Explain the encoder-decoder bottleneck as a concrete discarded variable;
run an ablation; judge whether attention learned the *right* alignment rather than merely
a confident one; connect cross-attention to self-attention.

**Tutorial 5.** Distinguish a held-out set that is genuinely held out from one that
leaks; interpret a BLEU score; compare decoding strategies on a model whose answers
actually differ.

**`concepts/decoding.md`.** Recognize beam search as best-first search with a fixed-width
frontier (#41); read a table with error bars and tell a real difference from sampling
noise (#40).

**`concepts/data-pipeline.md`.** Check whether a parallel corpus is actually parallel
(#46); repair one that has drifted, and verify the repair rather than trusting it (#29).

---

## Gaps

Ordered by how much they would cost a student.

**Why a model fails.** Everything teaches how the machinery works when it works. Nothing
teaches diagnosis: loss not falling, output collapsing to a single token, translations
fluent but unrelated to the source, a model that scores well and translates badly. This
is what a student actually hits, and the library's own history is full of examples —
tutorial 2 once shipped producing empty translations for every phrase.

**Evaluation beyond BLEU.** BLEU is introduced and used. Its failure modes are not: it
rewards length-matching, is unusable on single sentences, and is not comparable across
tokenizations. #40 measured a BLEU difference smaller than its own noise and the docs now
say so, which is the only place this idea appears.

**Training dynamics.** `concepts/training.md` covers the loop, optimizers, schedulers and
clipping. It does not cover what a student does when training goes wrong: batch size
against learning rate, when to stop, what overfitting looks like on a small corpus.

**Data quantity.** #29 added 18% more data and nobody has checked whether it helps (#49).
"How much data do I need" is among the first questions a student asks and the material is
silent.

**Inference cost in practice.** `decoding.md` covers this well for beam search
specifically. Nothing covers model size against latency, or CPU against GPU, which is
what a student meets when their Colab session is slow.

---

## Redundancy

Not necessarily wrong, but currently undeliberate.

**Beam search appears four times**: explained in `concepts/decoding.md`, reimplemented
from scratch in tutorial 3, visualized via `format_beam_search` (#39), and measured in
#40. The reimplementation is defensible — writing it yourself is the lesson — and
tutorial 3 explicitly reconciles its version against the library's. Worth deciding
deliberately rather than by accumulation.

**Attention appears three times**: `concepts/models.md`, tutorial 4, and
`reference/visualization.md`.

**The corpus repair story now appears twice**: `concepts/data-pipeline.md` (#46, #29) and
`scripts/realign_corpus.py`'s docstring. These are aimed at different readers and the
overlap is probably correct.

---

## Open questions for the instructor

1. Are the outcomes above the right ones? They are inferred from the material, which
   means they describe what exists rather than what the course needs.
2. Should tutorial 3 keep its from-scratch implementations, or call the library and spend
   the space on diagnosis instead?
3. Where does the lecture 7 assignment (#42) attach?
4. Is "why models fail" in scope for this library, or is it lecture material?
