# Related work, and why TorchLingo exists

TorchLingo is not the first library to put a readable NMT implementation in front of
students, and pretending otherwise would be a poor start for a project about intellectual
honesty. This page says what else exists, what we reuse instead of rebuilding, and where
we think the remaining gap is.

If you are choosing a tool rather than taking a course, the short version is that
**[Joey NMT](https://github.com/joeynmt/joeynmt) is the closest thing to this and is
excellent**. Read that section first.

## Toolkits with overlapping goals

### Joey NMT

[Joey NMT](https://aclanthology.org/D19-3019/) (Kreutzer, Bastings & Riezler, EMNLP 2019)
is a minimalist PyTorch NMT toolkit written explicitly for novices. It is actively
maintained, and it covers most of what a course needs: RNN and Transformer encoders and
decoders, several attention variants, beam search with length penalty, attention
visualization, learning-curve plots, word/BPE/character tokenization, and multilingual
training with language tags.

It is also evaluated as a teaching tool rather than only asserted to be one. The paper
reports a user study in which novices, after working through the Joey NMT tutorial,
performed nearly as well as experts on a subsequent code quiz. That is a higher standard
of evidence than most educational software offers, including this project.

The difference is one of shape rather than quality. Joey NMT is a **toolkit you
configure**: a YAML file describes the experiment and `joeynmt train` runs it. That is
the right design for getting a student to a working translation system quickly, and for
letting them read a clean implementation afterwards.

TorchLingo is a **library you call**, arranged around a sequence of lessons. The
difference matters most when something goes wrong, which is the case we have optimized
for — see [When it fails](concepts/when-it-fails.md) and Tutorial 6.

If your goal is a working NMT system with minimal ceremony, use Joey NMT.

#### What running it actually looked like

We ran their quickstart rather than relying on the README. Joey NMT 2.3.0, the shipped
`configs/transformer_reverse.yaml` toy task, on a CPU laptop:

**It does what it says.** Five epochs in 3m52s, ending at **93.62 BLEU** on the test
split. The training log is better instrumented than ours — per-step batch accuracy,
tokens/sec, and current learning rate — and it prints the full sacreBLEU signature
(`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`) alongside every score.
That last habit is worth copying: it makes a BLEU number reproducible by someone who
was not there.

**Three things had to be fixed before it would start.** The shipped toy config sets
`use_cuda: True` and `fp16: True`, so it fails immediately on a machine without a GPU.
More seriously, `joeynmt/builders.py` passes `verbose=False` to
`torch.optim.lr_scheduler.ReduceLROnPlateau`, which PyTorch has since removed — so on
current PyTorch the shipped toy config raises `TypeError` before the first step. Since
`scheduling: "plateau"` is what that config uses, the quickstart does not run
out of the box.

None of this says the project is unmaintained; it says a tutorial that is not executed
automatically will eventually stop working, whoever wrote it. That is the exact failure
[our notebook gate](https://github.com/byu-matrix-lab/torchlingo/blob/main/scripts/execute_notebooks.py)
exists to prevent, and finding it here is the strongest argument we have for keeping that
gate — and for the fact that it currently
[skips most of our own tutorials in CI](concepts/when-it-fails.md).

### OpenNMT-py and Eole

[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) is the long-running research toolkit;
[Eole](https://github.com/eole-nlp/eole) is its modernized successor. Both are
substantially more capable than TorchLingo and substantially harder to read. They are the
right next step for a student who has outgrown a teaching library, and the wrong place to
meet beam search for the first time.

### fairseq, Marian, CTranslate2, Hugging Face

Production and research infrastructure. [Marian](https://marian-nmt.github.io/) and
[CTranslate2](https://github.com/OpenNMT/CTranslate2) in particular are what you should
reach for when inference speed actually matters; neither is trying to be readable, and
both are far faster than anything in this repository will ever be.

This is a deliberate boundary. TorchLingo's decoders are written to be followed line by
line, and we do not intend to compete on throughput.

## Expositions and courses

These are not libraries, and several are better than any library at explaining a single
idea. Recommended alongside this project rather than against it:

| Resource | What it is best at |
| -------- | ------------------ |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | Reading the original paper and its implementation side by side |
| [Lena Voita's NLP Course](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) | Seq2seq and attention, with unusually good visual intuition |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | The single clearest picture of what attention computes |
| [Dive into Deep Learning](https://d2l.ai/) | A full textbook with runnable code, MT included |

## What we reuse rather than rebuild

A teaching library is still a library, and reimplementing a solved problem in order to
teach it is only justified when the implementation *is* the lesson. Where it is not, we
take the dependency:

| We use | Instead of | Why |
| ------ | ---------- | --- |
| [sacreBLEU](https://github.com/mjpost/sacrebleu) | our own BLEU | BLEU is notoriously sensitive to tokenization; a non-standard implementation produces numbers nobody can compare to anything |
| [SentencePiece](https://github.com/google/sentencepiece) | our own subword model | The algorithm is not the lesson; the *effect* of subword segmentation is, and that shows up regardless of who implemented it |
| `torch.nn.Transformer` | our own attention stack | The architecture is the lesson, and PyTorch's implementation is the one students will meet everywhere else. Where this costs us something — it hides cross-attention weights — we document [why and work around it](reference/visualization.md) |

The rule we try to follow: **implement it when reading the implementation teaches the
concept; depend on it when only the behaviour matters.**

That rule cuts against us in places. Length-based sentence alignment is a good example:
[NLTK ships Gale-Church](https://www.nltk.org/api/nltk.translate.gale_church.html)
already, so writing our own is justified only for the lesson — and if the goal is
actually to recover data, embedding-based aligners such as Vecalign and Bertalign
[measurably outperform](https://www.nature.com/articles/s41598-023-47479-w) length-based
methods and should be used instead.

## Where we think the gap is

Everything above teaches the path where things work. We could not find an equivalent for
any of the following, which is where this project puts its effort:

**Diagnosis as a subject.** A model that does not work fails silently: training runs, the
loss curve looks like a loss curve, translations appear, and every number is wrong.
Tutorial 6 breaks a working model five different ways and makes the student watch a
specific check fire for each. The checks ship as `torchlingo.diagnostics` so they can be
used on real work, not only read.

**Empirical discipline, taught through MT rather than asserted.** Controlled comparison,
test-set contamination, `ln(V)` as the reference for "learned nothing", significance
testing on BLEU. These are research methods; MT is the vehicle. Most of them entered the
curriculum because we got them wrong first — the enlarged corpus "worth +2.33 BLEU" that
was mostly 80% more training time is
[documented as a failure](concepts/when-it-fails.md), not quietly fixed.

**A corpus with a known history.** `data/example.tsv` shipped misaligned. Rather than
repair it silently we kept the repair as a lesson, and `shuffle_target_side` exists so
any corpus can be broken the same way on demand. A teaching corpus that has actually been
broken is more useful than a clean one.

**Documentation that executes.** Every docstring example runs under
`pytest --doctest-modules`, and every tutorial notebook is executed in CI. A teaching
library whose examples have rotted is worse than none, because a student cannot tell
"the tutorial is broken" from "I did it wrong."

## Honest caveats

- Joey NMT is the only project here we have actually run. Everything said about the
  others is from documentation and published papers, and claims about a project's
  *behaviour* deserve hands-on verification before they inform a syllabus.
- The Joey NMT findings above are from one toy task on one CPU machine. They are enough
  to say the quickstart needs fixing on current PyTorch; they are not a judgement of the
  toolkit's quality, which on the evidence of that run is high.
- The gap described above is a gap in what we *found*. Absence of evidence is weak
  evidence, particularly across course materials that are not indexed as software.
- Joey NMT's user study is real evidence about teaching effectiveness. TorchLingo has no
  equivalent, and until it does, "better for learning" is a design intention rather than
  a finding.
