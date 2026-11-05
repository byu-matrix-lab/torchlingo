# Lecture 18's notebook is built, and in review

*From the repository session, Thu Nov 5. Answers
`from-cowork/2026-11-05-l-ibm-model-1-for-lecture-18.md`. Nothing is asked of you. This file
stays live until the notebook is on `course`, and says so here when it is.*

## 1. The notebook: `docs/docs/course/lecture-18-ibm-model-1.ipynb`

Built in one pass, to your brief: in class, about ten minutes, run together, no points, CPU only,
no research questions block. Its opening cell quotes the outcome *"Explain how word alignments
relate to attention in NMT"*. It stands alone: attention is explained in a few sentences where
it first appears, and tutorial 4 is linked for depth only. It looks nothing like tutorial 4: no
training, and a corpus of noun phrases instead of a synthetic task.

1. **The corpus**, visible in a cell: thirteen pairs, *the house* / *la casa* through *a big
   book* / *un libro grande*, including *the red car* / *el coche rojo*.
2. **EM**, written out in about fifteen lines, since it is the lesson.
3. **Your prediction, word for word**, in a cell that stops while empty: after one round, which
   Spanish word will `house` most likely translate to, and why can Model 1 not be sure yet? One
   round gives `casa` first and `la` close behind; ten rounds settle every noun, and leave `the`
   and `red` split by gender, which the notebook explains.
4. **Model 1's alignment of *the red car* beside a Transformer's attention** on the same pair,
   then your question: where they agree, where they differ, and why attention needs no EM. A
   cell asks students to write one row of each before the answer.
5. **The hands-on edit fits**: an optional last step adds *the boy* / *el niño* and reruns, and
   `el` moves toward `the`. If time is short, it is the step to skip, as you said.

What the comparison shows, for the deck if you want it:

```
Model 1, after ten rounds of EM:          The Transformer's cross-attention:
        the   red   car                           the   red   car
el       66     0    34                   el       94     3     3
coche     0     0   100                   coche     5    45    50
rojo      0    99     1                   rojo      8    46    46
```

Both put `el` on `the`, and Model 1 keeps a third of it for `car`, since in the corpus `el` never
appears without `car` or `book`. On `coche` and `rojo` it is the other way round: Model 1 is sure
of each, while attention spreads both across *red car*. The notebook's reading: attention is
trained to help predict the next word, not to align, so a row of attention is where the model
found it useful to look; and it needs no EM because its weights are a differentiable function
trained by the translation loss, with no hidden alignment to guess.

## 2. Two changes from your brief

- **The attention model is Lecture 8b's small English-Spanish Transformer.** Tutorial 5 loads no
  pretrained model, so there was none to reuse there. Lecture 8b's runs on a CPU, downloads in
  seconds, and students have seen it. If the deck names the model, say it is "the model from
  Lecture 8b".
- **Two feminine nouns, not one.** With `casa` the only noun after `la`, counting can never
  separate them, and `house` stalls with a third of its weight on `la`. Adding *door* / *puerta*
  gives `the` a way to claim `la`, and `house` then settles on `casa`.

## 3. When it reaches `course`

It is in a pull request now. Every check passes, and a run as a student takes it (a fresh
install from `course`, the model downloaded, every cell). Once it is reviewed and merged, it goes
to `course`, and this file will say so. That is well before Nov 23. The badge for the deck is the
one in the notebook's first cell:

`https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/course/lecture-18-ibm-model-1.ipynb`

The roadmap gains the notebook's entry and a Nov 5 line in the decisions log. The open question
on Lecture 18 now says its notebook is settled. Scoping the rest of the lecture is still yours
and Eric's.

## 4. Your branch

Your hand-off went in through a pull request from your branch, with this reply, and
`cowork/2026-11-05-l-ibm-model-1` is deleted. Nothing was left in the main checkout this time.

## Tasks

On this side, the Lecture 18 notebook has a row of its own, marked in review, due on `course` by
Nov 23. Nothing else moves. The single Colab L4 run of the A14 notebook stays open, with Coulson.
