# Post-norm, the split is in, and one thing I am not promising

**2026-09-27.** Answering your baton. First hand-off under the new one-file-per-hand-off
layout — see `notes/handoff/README.md`; the old append-only logs are in `archive/` whole.

## 1. Pre-norm or post-norm: **post-norm.** Both your 8b slides stand

`SimpleTransformer` constructs `nn.Transformer` without passing `norm_first`, so PyTorch's
default of `False` applies. Confirmed on the built module rather than read off the
constructor:

```python
m = SimpleTransformer(...)
m.transformer.encoder.layers[0].norm_first   # False
```

That is post-norm — sublayer, then residual add, then LayerNorm — the ordering of the 2017
paper. So the encoder-block anatomy slide is right, and the residual-stream slide is right
that the channel gets rescaled at every block.

Worth saying how it was settled, because you will hit this class of question again: the call
site does not mention the argument, so no amount of reading `transformer_simple.py` answers
it. I built the model and read the attribute. If a slide ever depends on a PyTorch default,
that is the cheap way to be sure.

## 2. Your check is green. `8a` and `8b` parse, and three notebooks moved

**Done and in PR #127.** Lecture identifiers are **strings** now, because `8a` has no
integer form — that was the real cost of the scheme, and it is paid.

A normalizer means nothing else had to change: the integer `9` a notebook already held, your
schedule's `9`, and a filename's `09` all land on one key.

Your three moves are applied, each validated before it was written:

| notebook | now serves | role |
|---|---|---|
| `06-diagnosing-failures` | `[9]` | reading |
| `03-inference-and-beamsearch` | `[10]` | reading |
| `04-attention-and-alignment` | `["8a", "8b"]` | reading |

**The first was not optional, and this is the part worth your attention.** Lecture 8 stopped
existing, so tutorial 6 was pointing at nothing — the check said so by name. Had you
renumbered 9 onward instead, tutorial 6 would have kept claiming "8", the row would still
have existed, and **nothing anywhere would have complained** while it pointed at the wrong
lecture. The suffix scheme is the one the tooling can protect. Thank you for choosing it.

Two side effects you should know about rather than discover: tutorials 3 and 4 lost their
`note` fields. Both described the old pairings, so they are correctly gone, but that was a
consequence of restamping and not a judgement I made.

## 3. Your three notebooks are in, checked rather than trusted

`lecture-04-regex-refresher`, `lecture-10-comet-install`, `lecture-12-llm-context`. I
re-verified independently of your scan: zero outputs, zero execution counts, no tokens or
keys, no INSTRUCTOR copies, badges present, metadata sound. All three are in the mkdocs nav
now — without that the docs build carries pages nothing links to.

**Your roles read correctly to me**, so I have not changed them. `regex-refresher` as
`reference` at Lecture 4 is right given Learning Suite posts it on the Lecture 3 tab too.

**One gap this opened**, and it is now Task #154: the notebook gate runs `docs/docs/tutorials`
only. `docs/docs/course/` now holds seven notebooks and **none of them is executed by
anything.** That matters more for yours than for mine, because yours are opened in the room
on a clock. Several need Drive, a download or a HF token so they cannot all run in CI, but the
self-contained ones can, and the `needs` field already says which are which.

## 4. The A8 kickoff notebook: **I am not promising it for Wednesday**

You asked to be told rather than discover it late, so: **plan the deck without a pointer to
it.** If it exists by Tuesday I will say so and you can add the slide back cheaply; a slide
pointing at nothing is the outcome you said you wanted to avoid, and I will not risk it.

The reason is not the ten steps, which are well specified — it is steps 2 and 7. They mount
Google Drive and write six files back to it, and **I cannot execute that path here.** Every
other notebook in this repository is verified by running it. This one would ship to eighteen
students, in a room, on a twenty-minute clock, having never been run end to end by anyone.

What would change my answer, in order of how much it buys:

1. **Eric runs it once in Colab against a real A5 corpus** before Wednesday. That is the
   whole gap; with it I would ship confidently.
2. A Drive-free variant for the room — same ten steps against a corpus fetched by URL — with
   the Drive path as the take-home. Weaker pedagogically, since it stops being *their* data.

**Your structural argument is right and I am not arguing with it.** Lectures 4, 5 and 6 each
had an activity that started the assignment and the largest assignment in the course has
none. That is worth fixing. It is worth fixing with a notebook that has been run.

## 5. Two answers to things you flagged

**#118 is still Coulson's and still unmeasured.** I have nothing to add except agreeing with
your framing: it now blocks a live decision, not a documentation claim. Raising the epoch
count on the 11.7M model is the more expensive path to the worse result *if* the 56.4M
configuration fits, and nobody knows whether it fits.

**A8's "What To Do" overlapping the kickoff notebook** is Task #155 and I have not acted on
it either — you were right that it is Eric's, because it changes an assignment students are
about to start.

## 6. Where your list and mine agree

Your items 1 to 6 map onto Tasks #138 (done), #152, #121, the three moves (done), your three
notebooks (done), and #153 with #118. **I have not found anything in your "waiting on the
repository session" list that is wrong in either direction.**

New on my side since your baton: **#154** the course notebook gate, **#155** the A8 handout
overlap, and **#153** which this entry closes.

## 7. Housekeeping

- `notes/legacy-f2025/` is committed. Your README recording what last year's notebook got
  wrong is the most useful thing in it — it turned #121 from a guess into a scope.
- One correction to my own record: I dated yesterday's hand-off `2026-09-27` when it was the
  26th. Fixed, including one past heading, which is the only time I have edited an entry
  after the fact.
- The roadmap's 13,151 characters are indeed back, and the generated notebook map regenerated
  cleanly on top of your schedule changes.
