# Lecture 18's notebook is on `course`

*From the repository session, Thu Nov 5. Answers
`from-cowork/2026-11-05-l-ibm-model-1-for-lecture-18.md`. The baton goes back to Cowork with
this file. Nothing is asked of you; the deck can link the notebook when you build Lecture 18.*

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

## 3. It is on `course`

Merged and promoted on Nov 5, ahead of your Nov 23 date, and the student-path run on `course`
passed. That run skips a notebook that stops for a prediction, so this one was also run once by
hand as a student would, with the prediction filled in: a fresh install from `course`, the model
downloaded, every cell. The badge for the deck is the one in the notebook's first cell:

`https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/course/lecture-18-ibm-model-1.ipynb`

The roadmap gains the notebook's entry and a Nov 5 line in the decisions log. The open question
on Lecture 18 now says its notebook is settled. Scoping the rest of the lecture is still yours
and Eric's.

## 4. Your branch

Your hand-off went in through a pull request from your branch, with this reply, and
`cowork/2026-11-05-l-ibm-model-1` is deleted. Nothing was left in the main checkout this time.

## 5. Two COMET failures, both fixed on `course` (added Nov 5, evening)

Students finishing A13 reported Step 6 failing on Discord, and checking A14 turned up a different
failure in its Step 7. Both are fixed and promoted. Neither needs retraining, since the
translations are on Drive.

- **A13, Step 6.** `comet-score` stopped with an `intermixed` error on Python 3.13, which Colab now
  runs. The install cell pins a newer `jsonargparse`. A student already in their own copy adds one
  cell after the COMET install, `%pip install --quiet "jsonargparse==4.46.0"`, and reruns the
  scoring cell. Eric has that line for Discord.
- **A14, Step 7.** Scoring failed for anyone running the notebook in order, with a message saying
  COMET was not installed. It now scores in a separate process, and runs in a new session after
  the setup cells. No students have started A14 (Eric, Nov 5), so everyone gets the fix from the
  badge.
- **The red `huggingface-hub` / `gradio` / `diffusers` list** is harmless in both, and both
  notebooks now say so. If Learning Suite or a deck describes these steps, it may want the same
  sentence.

Neither fix has run on real Colab with a GPU yet; they were tested against Colab's published
package versions. The single Colab L4 run of the A14 notebook, with Coulson, will be the first.

## Tasks

On this side, the Lecture 18 notebook is done and its row is gone; on your list it can close too,
leaving the deck's link to it. A review of the notebooks for code that belongs in the library
produced five tasks, under way on this side. One fix reaches students: a sentence reading `NA`,
`null`, or `None` was silently dropped from training, and now is not. The single Colab L4 run of
the A14 notebook stays open, with Coulson.
