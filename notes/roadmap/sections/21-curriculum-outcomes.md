## Learning outcomes

Stated as what a student can do afterwards. **These are proposed, not confirmed** — they
are reverse-engineered from the material, and they need checking against the course this
feeds.

**Tutorial 1.** Load a parallel corpus; explain why a vocabulary needs `<pad>`, `<sos>`,
`<eos>` and `<unk>`; predict what happens to an out-of-vocabulary word at inference.

**Tutorial 3.** Train a Transformer end to end; read a loss curve well enough to tell
"still learning" from "converged"; recognize that a loss near `ln(vocab_size)` means the
model has learned nothing.

**Tutorial 8.** Implement greedy and beam search from scratch; state what beam search
buys and what it costs; explain why two implementations of the same algorithm can
disagree on ties.

**Tutorial 4.** Explain the encoder-decoder bottleneck as a concrete discarded variable;
run an ablation; judge whether attention learned the *right* alignment rather than merely
a confident one; connect cross-attention to self-attention.

**Tutorial 7.** Distinguish a held-out set that is genuinely held out from one that
leaks; interpret a BLEU score; compare decoding strategies on a model whose answers
actually differ.

**`concepts/decoding.md`.** Recognize beam search as best-first search with a fixed-width
frontier (#41); read a table with error bars and tell a real difference from sampling
noise (#40).

**`concepts/data-pipeline.md`.** Check whether a parallel corpus is actually parallel
(#46); repair one that has drifted, and verify the repair rather than trusting it (#29).
