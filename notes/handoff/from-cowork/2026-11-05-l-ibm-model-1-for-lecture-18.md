# L: Yes to IBM Model 1 for Lecture 18

From the Cowork session, Thu 2026-11-05. Answers `to-cowork/2026-11-04-lecture-15-landed.md`. Baton to the repository session.

## 1. Task #253: yes (Eric, Nov 5)

Eric: *"we can do an IBM model 1 notebook."* It is yours to build. Everything the course side knows is below, so nothing should need another pass before Lecture 18.

- **Where it runs:** in class, about 10 minutes, run together; **no points** and no quiz. So it is a course notebook, `docs/docs/course/lecture-18-ibm-model-1.ipynb` (or your preferred slug), not an assignment notebook: no research questions block.
- **The outcome it serves**, from the roadmap's Lecture 18 row: *"Explain how word alignments relate to attention in NMT."* Quote it in the opening cell.
- **Stands alone:** it must not assume tutorial 4 was read. Say in two sentences what attention alignment is at first use; link tutorial 4 for depth only. It must not look like tutorial 4 either (that one trains three small models on a synthetic task): different data, different opening.
- **The corpus:** about a dozen English-Spanish pairs, chosen so EM has something to resolve: shared words across pairs ("the house", "the red house", "a house"), and at least one adjective-noun swap ("the red car" / "el coche rojo"; Lecture 16's wait-k slide uses this very pair, so students will recognize it). Spanish is the language most of the class reads. Keep the pairs visible in a cell, as a fixture, not hidden in a helper.
- **One prediction, before the first iteration**, in its own cell that stops while empty, per the house pattern. Suggested: "After one round of EM, which Spanish word will `house` be most likely to translate to, and why can it not be sure yet?" After a few iterations, the table shows it settling.
- **What students do with their hands:** if it fits in 10 minutes, then one small edit worth making: add a pair of their own and rerun, and see what moves. Your call on whether that fits.
- **Beside attention:** show the same sentence pair's cross-attention from a pretrained model (the English-to-Spanish model tutorial 5 already loads would do), next to Model 1's alignment, and ask the one question the lecture is about: where do they agree, where do they differ, and why does attention not need EM?
- **CPU is enough;** no GPU, so no disconnect reminder needed unless the pretrained model calls for one.
- **When:** Lecture 18 is Wed Dec 2. If the notebook is on `course` by Mon Nov 23, then the deck links it with a Colab badge when Cowork builds Lecture 18. If anything above does not fit 10 minutes, then cut the hands-on edit first, and keep the prediction and the comparison with attention.

## 2. Nothing else asked

- The CLAUDE.md rule is noted: from now on the branch alone, no copies in your main checkout.
- The hidden Lecture 15 slide is fixed (`_F2026_v24`, "four designs"); no repository change.

## Tasks

On the course list, Task #253 moved from "waiting on Eric" to yours: build the Lecture 18 notebook above. Nothing else moves.
