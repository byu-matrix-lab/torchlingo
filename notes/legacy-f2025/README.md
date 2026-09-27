# Fall 2025 notebooks

Brought in from the course working folder on 2026-09-27, at Eric's instruction: use what
exists rather than reinvent it.

## What was found

Eight `.ipynb` files in `Fall 2025/`, which collapse to **four distinct notebooks**. The
regex activity existed in three identical copies and the LLM assignment in two.

All four were scanned before anything moved: no access tokens, no API keys, no student names,
no email addresses, no personal filesystem paths, in source or in outputs. The COMET notebook
reads its HuggingFace token from Colab Secrets, which is the right pattern and worth keeping.

Outputs were stripped from every file. Two of them carried training logs, 23 KB in one and
**675 KB** in the other.

## Where they went

| Fall 2025 file | now | lecture | role |
|---|---|---|---|
| `CS 479 Regex Activity.ipynb` | `docs/docs/course/lecture-04-regex-refresher.ipynb` | 4 | reference |
| `COMET_Install_and_Authenticate.ipynb` | `docs/docs/course/lecture-10-comet-install.ipynb` | 10 | homework |
| `LLM_Assignment.ipynb` | `docs/docs/course/lecture-12-llm-context.ipynb` | 12 | homework |
| `OpenNMT_and_Sentencepiece.ipynb` | **stayed here** | 9 | superseded |

The first three are framework-independent: nothing in them touched OpenNMT, so they carry
forward unchanged apart from the badge, the metadata block and the stripped outputs.

## Why the fourth one stayed here

`OpenNMT_and_Sentencepiece.ipynb` is Lecture 9's predecessor and is being replaced by the new
subword notebook (#121). It is kept as reference for whoever writes that, not as student
material, because putting it in `docs/docs/course/` would publish a notebook whose fourth cell
reads:

```
!pip install OpenNMT-py
!pip install "numpy<2.0" # This fixes an error caused by OpenNMT not being maintained
```

Nineteen cells, of which three are SentencePiece and the rest is OpenNMT plumbing. Two things
it does that the replacement must not:

- **It fits the tokenizer on OpenNMT's `toy-ende` corpus, not on the student's own data.**
  Assignment 9 is a controlled comparison on their corpus, so the tokenizer has to be fit on
  their training split. Fitting on the whole corpus leaks test material into the vocabulary,
  quietly, because nothing crashes.
- **It says nothing about the length cap.** Changing the tokenizer changes which pairs a
  token-based cap excludes, which breaks the comparison. The sentence set has to be chosen
  once, with the subword tokenizer, and reused for both runs.

## One thing this fixes beyond tidiness

Lecture 4's deck linked the regex notebook **twice**, both times to a personal Google Drive
URL. Lecture 12's deck links its assignment notebook the same way. Those links die with the
account that owns them, which is the same fragility as `grader.exe`. Now that the notebooks are
here, those links can become Colab badges off `main`. **The decks have not been changed yet**;
that is a separate edit, and the Drive copies must stay alive while students are working in
them.
