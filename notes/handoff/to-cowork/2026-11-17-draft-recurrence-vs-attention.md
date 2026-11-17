# Draft: recurrence against attention, for the 8a to 8c decks

*From the repository session. A draft, last updated Tue Nov 17: it goes out with the next baton
pass to Cowork, folded into that pass's main file.*

## What we need from you

1. Update the 8a, 8b and 8c decks in place with the framing below, for students reading back and
   for the next offering (Eric, Nov 17). Nothing this term waits on it.

## The framing

Eric, Nov 17, after a student-style question: *"Why do we say a Transformer is not recurrent?
Don't the attention heads at position n refer back to earlier positions?"* The decks motivate
attention without saying what recurrence is given up for it.

**The answer: attention looks back and across, never back and down.** At position n, layer l, a
head reads keys and values built from layer l−1's outputs at positions ≤ n, never from any layer
≥ l at an earlier position. On a grid of layers × positions, every edge goes up one layer, and
some also move right:

    layer 2   ·    ·    ·    ·
              ↑ ↗  ↑ ↗  ↑ ↗  ↑
    layer 1   ·    ·    ·    ·
              ↑ ↗  ↑ ↗  ↑ ↗  ↑
    input     x1   x2   x3   x4

No edge runs from the top of one position to the bottom of a later one, so the longest chain of
sequential steps is L layers, however long the sentence.

- **8a, the RNN:** it feeds the top of step t into the bottom of step t+1, so its serial depth
  grows with the sentence. That edge is why step t+1 waits for step t in training, and why
  gradients cross T steps and vanish or explode.
- **8b, the Transformer:** it drops that edge. The grid has a fixed depth, so every position is
  computed at once in training, and any two positions are one attention hop apart, not T steps.
  The trade is serial depth for parallelism and short paths: the real motivation for attention,
  and it says what is given up. One diagram, the grid above, carries it.
- **8c, where recurrence comes back:** generating restores exactly one top-to-bottom edge, since
  the output at position n becomes the input at n+1 as a single sampled token. In NMT that edge
  is the decoder's alone; the encoder has none. The KV cache does not change this: nothing
  computed later flows back down to an earlier layer. It fits 8c's outcome, "say how order gets
  into a model with no recurrence".
- **Optional, one line:** this is why writing out intermediate tokens (chain of thought) helps an
  LLM: each token buys another L layers of serial computation.
- **Caveats, if a student asks:** linear attention can be rewritten as an RNN over positions, but
  its depth is still bounded by the layers; looped or universal Transformers repeat layers within
  a position, which is recurrence in depth, a different route.

Eric, Nov 17: keep it to the course's own material. No pointer to outside research on widening
that edge.
