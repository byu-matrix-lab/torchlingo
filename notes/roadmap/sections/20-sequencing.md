## Sequencing, including the parts nobody wrote down

Two dependencies are load-bearing and were invisible until this audit.

**Tutorial 8 cannot run without tutorial 3.** It loads the checkpoint tutorial 3 saves.
`scripts/execute_notebooks.py` encodes this — it runs the notebooks in one shared working
directory, in filename order, and lists tutorial 8 as needing the corpus even though it
never names it. A student who opens tutorial 8 in Colab on its own gets a file-not-found
error and no explanation.

**Tutorial 8's model is too small to teach what tutorial 8 is about.** Its beam-size
sweep prints five identical rows because the model is decisive. This is why #40's
measurement had to be done on tutorial 7's checkpoint instead, and why #50 exists. The
sequencing implication is real: *the lesson about beam search requires a model that is
wrong often enough to be uncertain, and the tutorial that teaches beam search does not
have one.*

**Tutorial 4 is independent.** It trains a synthetic reversal task with known ground
truth, which is what lets it measure alignment accuracy rather than assert it. It could
be moved without breaking anything.
