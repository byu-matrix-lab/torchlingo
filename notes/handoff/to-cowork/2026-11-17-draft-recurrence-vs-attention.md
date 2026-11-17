# Draft: recurrence against attention, for the 8a to 8c decks

*From the repository session. A draft, last updated Tue Nov 17: it goes out with the next baton
pass to Cowork, folded into that pass's main file.*

## What we need from you

1. Update the 8a, 8b and 8c decks in place with the framing below, for students reading back and
   for the next offering (Eric, Nov 17). Nothing this term waits on it.

## The framing

The full text, with the grid diagram and the slide-by-slide placement, is in the course task
list under "recurrence against attention". In one line: attention looks back and across, never
back and down, so a Transformer's serial depth is fixed at its layer count, where an RNN's grows
with the sentence; generating restores one recurrent edge, a single token wide, and in NMT only
in the decoder.

Eric, Nov 17: keep it to the course's own material. No pointer to outside research on widening
that edge.
