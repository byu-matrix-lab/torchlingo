## Gaps

Ordered by how much they would cost a student.

**Why a model fails.** ~~Everything teaches how the machinery works when it works.
Nothing teaches diagnosis.~~ **First pass written: `concepts/when-it-fails.md`.** A
diagnostic order — data, then whether the model is learning at all, then whether it is
learning the wrong thing, then whether the measurement is lying, then the environment —
with each step's check drawn from a real incident in this repository's history rather
than invented.

The table at its head is the part worth keeping: six symptoms, the obvious explanation
for each, and what it actually turned out to be. Every "looked like" column entry is a
reasonable first guess and every one is wrong.

Still missing, and harder: a *runnable* version. The alignment lesson in #46 works
because `shuffle_target_side` lets a student break a corpus and watch the check fire.
The equivalent here would be deliberately breaking a model — freezing a parameter,
detaching a graph, training one epoch — and watching each diagnostic catch it. That is
a tutorial, not a page.

**Evaluation beyond BLEU.** BLEU is introduced and used. Its failure modes are not: it
rewards length-matching, is unusable on single sentences, and is not comparable across
tokenizations. #40 measured a BLEU difference smaller than its own noise and the docs now
say so, which is the only place this idea appears.

**Training dynamics.** `concepts/training.md` covers the loop, optimizers, schedulers and
clipping. It does not cover what a student does when training goes wrong: batch size
against learning rate, when to stop, what overfitting looks like on a small corpus.

**Data quantity.** ~~#29 added 18% more data and nobody has checked whether it helps.~~
**Checked, and the answer is no** — at this scale. With training budget held fixed, +20%
data is worth +0.29 ± 0.22 BLEU, an interval crossing zero, while going from 20 to 36
epochs on the same data is worth +2.05. See the training-budget section in `TASKS.md`.

"How much data do I need" is still a gap, but the material now has a real answer to a
better question: *how do you tell which lever you are actually pulling?* The episode is
worth teaching directly, because the first attempt to answer it got the wrong answer with
error bars and a paired bootstrap attached, and only a control run caught it.

**Inference cost in practice.** `decoding.md` covers this well for beam search
specifically. Nothing covers model size against latency, or CPU against GPU, which is
what a student meets when their Colab session is slow.

**Attention on the architecture students actually use.** Covered for the LSTM in three
places, and not at all for the Transformer, which cannot return its attention weights.
See the correction under Redundancy below, and #34.
