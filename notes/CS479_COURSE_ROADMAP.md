# CS 479 Fall 2026 course roadmap

Course-side reference, maintained by the Cowork session. **v3, 2026-09-26.** Replaces the
version you read on 2026-09-25 in place; git history has the previous one.

What changed since v2, in one list, so you do not have to diff it:

- **Lectures 7 and 8 are rebuilt** and their scope notes are real rather than descriptions
  of last year's decks. Lecture 7 is Monday.
- **The data-splitting lesson moved from Lecture 8 to Lecture 7**, which is closer to the
  placement briefing Part A2 actually wanted. Lecture 8 keeps a recap.
- **The pivot section is rewritten around your measurements.** Three risks v2 listed are
  closed: the library ships complete at 0.2.0, Colab resume is verified, and the units
  question is settled at 30 to 36 epochs. The length-cap table and the two-levers framing
  are in, with the Metal caveat attached.
- **New: what the pivot settled** — the low-resource floor wording for A8, the Assignment 9
  control bug, and the BPE ordering wrinkle.
- **New: the grader has no source and no home.** Traced today; details below. Eric has
  written to its author.
- **New: a section recording the handoff protocol** and where the notebooks now live, so the
  arrangement is documented somewhere other than in this log.

Everything below the line is the course map itself, verbatim from
`CS479 Fall 2026 Roadmap_v3.md` in Eric's course folder.

---


Introduction to Machine Translation, BYU. Monday and Wednesday, 11:00 to 12:15.

**What this is.** A forward map of the semester: what each lecture covers, what its
assignment asks for, and what tooling that assignment depends on. Built from the Fall 2026
Learning Suite course structure, the six rebuilt F2026 decks, and the Fall 2025 decks for
everything not yet rebuilt.

**How to read the status column.** *F2026* means the deck has been rebuilt for this
semester. *F2025* means the lecture will run from last year's deck unless it is reworked
first, so its scope note describes what that deck does today.

**Dates.** Confirmed against the Learning Suite Schedule tab on Sep 25.

**This version.** v3, Sep 26. Lectures 7 and 8 are now rebuilt, the pivot's open
engineering questions have started returning measurements rather than estimates, and the
repository session and this one now exchange written handoffs. What changed is collected
under “What the pivot has settled” below; the lecture scope notes for 9 onward are unchanged.

---

## Semester at a glance

| # | Date | Lecture | Assignment due that day | Deck |
|---|---|---|---|---|
| 1 | Wed Sep 2 | Course Overview and History of MT | | F2026 |
| 2 | Wed Sep 9 | Translation Challenges for MT | | F2026 |
| 3 | Mon Sep 14 | Introduction to Word Embeddings | | F2026 |
| 4 | Wed Sep 16 | Data Preparation for MT Training | | F2026 |
| 5 | Mon Sep 21 | Data Preparation for MT Training, Part 2 | **A4** initial cleaning steps | F2026 |
| 6 | Wed Sep 23 | Human and Automatic MT Evaluation | **A5** complete cleaning pipeline | F2026 |
| 7 | Mon Sep 28 | Research Paper Reviews; Intro to Neural Networks | **A6** human vs. automatic evaluation | **F2026** |
| 8 | Wed Sep 30 | Neural MT Overview and Architectures | | **F2026** |
| 9 | Mon Oct 5 | Morphology and Terminology in NMT | | F2025 |
| 10 | Wed Oct 7 | Overview of MT Quality Estimation | **A8** create and run an NMT model | F2025 |
| 11 | Mon Oct 12 | Neural Quality Estimation and Evaluation | **A9** SentencePiece · **A10** install COMET | F2025 |
| 12 | Wed Oct 14 | Using LLMs for MT; Expanding Context Awareness | **A11** run COMET on the A6 sentences | F2025 |
| 13 | Mon Oct 19 | Strategies for NMT of Low-Resource Languages | **A12** context and LRL translation | F2025 |
| 14 | Wed Oct 21 | Multilingual NMT and "Zero-shot" NMT | | F2025 |
| 15 | Mon Oct 26 | Overview of Speech-to-Speech MT | **A13** back-translated data | F2025 |
| 16 | Wed Oct 28 | Automatic Dubbing and Interpretation | **A14** bidirectional MNMT | F2025 |
| 17 | Mon Nov 2 | Multimodal NMT | | F2025 |
| — | Wed Nov 4 | Project proposal outline reviews with instructor | **A16** SLT pipeline | |
| — | Mon Nov 9 | Project proposal outline reviews with instructor | | |
| — | Wed Nov 11 | Presentations of final project proposals | **Final project proposal** | |
| — | Mon Nov 16 | Finish proposal presentations; report on TAUS conference | | |
| 18 | Wed Nov 18 | HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation | | F2025 |
| 19 | Mon Nov 23 | Word and Sentence Alignment | | F2025 |
| — | Wed Nov 25 | **No class**, Thanksgiving Eve | | |
| 20 | Mon Nov 30 | What can we learn from previous MT methods: RBMT, KBMT, EBMT, PBSMT, SBSMT | | F2025 |
| — | Wed Dec 2 | Project checkpoint reviews with instructor | | |
| 21 | Mon Dec 7 | Writing Research Articles; MATRIX Lab research | | F2025 |
| 22, 23 | Wed Dec 9 | MT Applications and Jobs; Opportunities for Further MT Research | **Paper review assignment** | F2025 |
| — | Thu Dec 10 | Last day of class | Final project **presentations, submission, write-up** | |
| — | Wed Dec 16 | Final exam, 11:00 to 14:00 | | |

An assignment appears on the row of the day it is **due**, which is how the Learning Suite
schedule lists it. Lecture quizzes run through Lecture 20 only. There is no Lecture 7
assignment; that week is for reading and preparing the paper review. Lecture 15 carries a
reading assignment for the quiz but nothing to submit. Lecture 17, 18, 19 and 20 carry no
assignments, because the final project has taken over by then.

---

## The through-line

The course is one long build. A student who keeps up finishes the semester holding a
cleaned bilingual corpus, a trained NMT system, and a stack of measurements of it. Each
assignment consumes the previous artifact:

1. **Lectures 4 and 5** produce the corpus. Church translation-memory data, extracted from
   TMX, put through a 16-step cleaning pipeline the student writes, delivered as two
   sentence-aligned text files. Medium- and high-resource languages: at least 200K pairs.
2. **Lecture 6** measures *other people's* systems on that corpus. Two commercial MT
   systems, human ranking in MTEval, then SacreBLEU and chrF.
3. **Lectures 7 through 9** build the student's own system. Toy model first, then a real
   English-to-X model on 100K pairs, then the same model with SentencePiece so the two can
   be compared.
4. **Lectures 10 and 11** measure it properly, with COMET and COMET-QE, and compare those
   against the Lecture 6 numbers.
5. **Lectures 12 through 14** improve it: in-context prompting of an LLM, back-translation,
   and a bidirectional multilingual model.
6. **Lectures 15 and 16** step outside text into speech.
7. **Lecture 17 onward** is context, history and the final project.

The pivot is Lecture 7. Everything before it is about data and measurement; everything
after it assumes the student can train a model.

---

## Lecture scope notes

### 1. Course Overview and History of MT (F2026, 46 slides)
Weaver's 1947 letter, IBM-Georgetown, ALPAC, the evolution to NMT, hype versus reality,
MT at BYU back to Eldon Lytle in 1976. Also the course's own rules: quizzes, assignments,
AI use, grades, the late policy and the new Early Policy.

### 2. Translation Challenges for MT (F2026, 46 slides)
Why FAHQT is the wrong question. Lexical and structural ambiguity, false friends,
long-distance dependencies, pronominal reference, word order, topicalization, world
knowledge, morphology. Students pick their language here, which sets up everything after.
In-class activity on difficult sentences.

### 3. Introduction to Word Embeddings (F2026, 51 slides)
The distributional principle, syntagmatic versus paradigmatic relations, word2vec both
ways, LSA, PMI, GloVe, contextualized embeddings, and multilingual embeddings in a shared
space. Colab activity testing whether the shared space really lines up across languages.
This is the slide deck the Lecture 4 heat map comes from.

### 4. Data Preparation for MT Training (F2026, 41 slides)
Translation memories, the CAT loop, TMX anatomy, ISO language codes, the 16 cleaning
steps, the GILT best-practices document, Python TMX packages and TMX editors.
**Assignment:** extract segment pairs from TMX, fix everything that breaks alignment
first, combine sources, write at least three of the cleaning steps, prepare 200K pairs.

### 5. Data Preparation, Part 2 (F2026, 29 slides)
The full pipeline, plus Gale-Church sentence alignment brought forward from Fall 2025's
Lecture 19: length correlation, the six link types, dynamic programming, the cost model,
and where length-based alignment runs out. Colab activity estimates Gale and Church's
parameters on the student's own language.
**Assignment:** the complete pipeline, all 16 steps, over the student's Church data.

### 6. Automatic and Human MT Evaluation (F2026, 34 slides)
Human evaluation (ranking, adequacy and fluency, MQM), MTEval, then BLEU from the inside:
n-gram precision, the brevity penalty, the geometric mean, a worked example, why raw BLEU
is not comparable, SacreBLEU, and chrF. Colab activity computes BLEU and chrF on the
student's own data and shows BLEU returning zero on ten short sentences.
**Assignment (due Mon Sep 28):** 500+ sentences through two MT systems, human ranking of
50 in MTEval *before* seeing any score, SacreBLEU and chrF over the whole sample, and a
written analysis of where the metrics and the ranking diverge. Extra credit for writing
your own BLEU script, explicitly without AI.

### 7. Research Paper Reviews; Introduction to Neural Networks (F2026, 38 slides)
Rebuilt Sep 25. Two halves still, plus the pivot. The paper-review assignment is split into
"What To Do" and "What Your Presentation Must Cover"; the neural-network foundations from
the AMTA 2018 tutorial carry over untouched, with the instructor's own "describe ReLU"
placeholder replaced by a real activation-functions slide.

New for F2026: an Objectives slide; an Assignment 6 debrief built around where a metric and
a human ranking disagreed; **the data-splitting lesson, moved up from Lecture 8**; "A Change
of Framework", explaining why the materials now say TorchLingo; a rewritten Colab slide
carrying the paid-plan requirement; and "What Is That Loss Number?", giving `ln(V)` as the
reference point for a student's first training loss and warning that label smoothing puts a
floor under it.
**In-class activity:** install TorchLingo and train a toy model, via tutorial 2's Colab
badge. **Assignment:** none. The week is for choosing and reading a paper.

### 8. NMT Overview and Architectures (F2026, 38 slides)
Rebuilt Sep 26. The twenty architecture slides, encoder-decoder through multi-head
attention, carry over untouched. The Fall 2025 debrief on why student models came out bad
is kept and restyled, because it is still the sharpest illustration in the deck.

New for F2026: Objectives; a TorchLingo install debrief; a short recap of the splitting
lesson now taught in Lecture 7, carrying the argument that a contaminated split propagates
through Assignments 9, 13 and 14; "Sentence Length Is a Memory Budget", built on measured
figures rather than estimates; and the assignment split into What To Do, What To Submit and
an AI-use slide. Two slides duplicated from Lecture 7, the paper reminder and the papers
table, were cut.
**Assignment (due Wed Oct 7):** train an English-to-X model on your own cleaned data. At
least 100K training pairs, 2K validation, 2K test, or all of it if you have less; source-side
deduplication and a verified split; 100-token cap; 30 to 36 epochs; checkpointing to Drive;
SacreBLEU over the whole test set.

### 9. Handling Morphology and Terminology in NMT (F2025, 26 slides)
Morphological preprocessing, byte-pair encoding, SentencePiece, and approaches to
injecting terminology into NMT.
**In-class activity:** a notebook installing SentencePiece and wiring it into OpenNMT.
**Assignment:** add SentencePiece to the Assignment 8 system, retrain, rerun on the same
test set, and write a comparison of the two BLEU scores and the quality difference. If the
student already used SentencePiece, they run it again without.

### 10. MT Quality Estimation Overview (F2025, 31 slides)
Evaluation versus estimation, uses of QE, traditional QE training data and features, the
WMT QE shared task metric, QUETCH as the first neural QE model, then COMET and COMET-QE,
whether references are really needed, and how to read a COMET score.
**Assignment:** read the COMET repository, run the provided Colab notebook end to end with
a HuggingFace token, and read two papers for the quiz.

### 11. Neural Quality Estimation and Evaluation Toolkits (F2025, 19 slides)
COMET and COMET-QE in more depth, HTER distributions, partial-input baselines, lexical
artifacts, xCOMET, and QE with LLMs.
**Assignment:** install COMET; run the default model and COMET-QE-DA on the Lecture 6
outputs; compare COMET against the BLEU, chrF and human ranking numbers already collected;
one-page summary. Extra credit for xCOMET-XL.
*Note this assignment reaches back to Assignment 6 for its data. A student who skipped
Lecture 6 cannot do it.*

### 12. Using LLMs for MT and Expanding Context Awareness (F2025, 32 slides)
Encoder-only, decoder-only and large language models, MT with decoder-only models, uses of
LLMs for MT, document-level and context-aware MT, dropped and ambiguous pronouns,
contrastive test sets.
**Assignment:** in a provided notebook, pick a non-MT-specific HuggingFace model, load a
low-resource dataset (Efik, Palauan, Pohnpeian, Yapese, Kosraean or Telugu), translate a
test set with 0, 5, 10 and 20 in-context examples, and chart BLEU against context size.

### 13. Introduction to Low-Resource NMT Strategies (F2025, 30 slides)
What counts as low-resource, mitigation with and without LLMs, LRL challenges beyond data
volume, evaluation metrics and datasets for LRLs, data sources, and data augmentation.
**Assignment:** back-translation. Train an X-to-English system on the reversed data, back
translate at least as many held-out target sentences as the original training set, add the
synthetic pairs, retrain English-to-X, and compare SacreBLEU and COMET against the
original system.

### 14. Multilingual NMT and Zero-shot NMT (F2025, 23 slides)
Bilingual versus multilingual, how zero-shot works, tagging approaches, NLLB, complete
MNMT, and why the Church data suits cMNMT.
**Assignment:** a bidirectional two-language multilingual system, English and X, built
from the Assignment 8/9 system, with the two directions randomly intermingled and separate
test sets per direction. BLEU and COMET for both directions.

### 15. Overview of Speech-to-Speech MT (F2025, 33 slides)
Spoken language translation, early systems, cascade versus end-to-end architectures,
Skype Translator, Whisper, and speech data for projects.
**Assignment:** reading only, two papers, for the quiz.

### 16. Automatic Dubbing and Interpretation (F2025, 34 slides)
Neural voices, LINGUA/ToAll, automatic video dubbing, Wav2Lip, HeyGen. Also the final
project timeline: proposals, presentations, write-up, submission.
**Assignment:** build a three-component speech-to-speech pipeline (ASR, MT, TTS) on a free
Azure account, explicitly *not* using the Speech Translation API. Ten spoken sentences
recorded in and out, plus a video of one.

### 17. Multimodal NMT (F2025, 42 slides)
Whether visual context helps, video-guided MT, the VaTeX and MAD datasets, architecture,
ablations, and project examples. No assignment; proposals are due around here.

### 18. HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation (F2025, 40 slides)
The BYU interactive translation system, CAT tools, post-editing, normalized edit distance,
why post-editing helps some translators and not others, adaptive MT and Lilt, and whether
LLMs can do adaptive MT. In-class activity.

### 19. Word and Sentence Alignment (F2025, 39 slides)
IBM Models for word alignment worked through in detail, Awesome Align, then sentence
alignment. *The sentence-alignment half of this deck has already been moved forward into
the F2026 Lecture 5, so this lecture needs rescoping before it runs.* Also carries final
project tips.

### 20. Overview of Previous MT Paradigms (F2025, 44 slides)
The Vauquois triangle, direct/transfer/interlingua RBMT, lexical functional grammar, KANT,
EBMT, and statistical MT including phrase-based SMT. The history lecture, placed late so
students can see what NMT replaced.

### 21 to 23 (F2025)
Writing research articles with Overleaf and LaTeX, current MATRIX Lab research, MT
applications and jobs, and opportunities for further research. Final project presentations
close the semester.

---

## Tooling the semester depends on

| Tool | First needed | Used for |
|---|---|---|
| Google Colab | Lecture 3 | every in-class activity and most assignments |
| Python TMX libraries | Lecture 4 | extracting segment pairs |
| TMX editors (Olifant, Heartsome) | Lecture 4 | inspection only, never in the pipeline |
| grader.exe | Lecture 4 | checking cleaned output. **Binaries only, no source, no repo.** See below |
| MTEval (mteval.matrix.byu.edu) | Lecture 6 | human ranking; students self-register |
| SacreBLEU, chrF | Lecture 6 | automatic scoring, and again in 8, 9, 13, 14 |
| **TorchLingo** | **Lecture 7** | **every model the students train: 7, 8, 9, 13, 14.** 0.2.0 is on PyPI; `pip install torchlingo`, no pin |
| SentencePiece | Lecture 9 | subword tokenization |
| HuggingFace account | Lecture 10 | COMET model downloads; LLMs in Lecture 12 |
| COMET / COMET-QE / xCOMET | Lecture 10 | neural evaluation, and again in 13 |
| Azure free tier | Lecture 16 | ASR, MT and TTS for the speech pipeline |

---

## The OpenNMT to TorchLingo pivot

**OpenNMT-py is in maintenance mode and the course is moving to TorchLingo.** This is the
largest single change to the second half of the semester, and it lands at Lecture 7.

Five assignments train a model, and all five are written against OpenNMT-py today:
Lectures 7, 8, 9, 13 and 14. They are consecutive on the calendar and cumulative in the
artifact, so this is one crossing rather than five. Assignment 9 retrains Assignment 8's
system, 13 reuses it, and 14 is built from "the system you created for Assignment 8/9".

### What has to change, and when

Dates below are the day the material is first *used*, not the day the assignment is due.
The gap between them is the slack available.

| Needed in class | Assignment due | What |
|---|---|---|
| Mon Sep 28 | — | Lecture 7's in-class activity: install and a toy model, on TorchLingo |
| Wed Sep 30 | **Wed Oct 7** | Lecture 8's assignment: framework, configuration vocabulary, what to submit |
| Mon Oct 5 | **Mon Oct 12** | Lecture 9: the SentencePiece instructions are OpenNMT-specific |
| Mon Oct 19 | **Mon Oct 26** | Lecture 13: back-translation workflow |
| Wed Oct 21 | **Wed Oct 28** | Lecture 14: the "MNMT Guide Using OpenNMT.docx" handout needs replacing |

The schedule is kinder than it first looks. Assignment 8, the heavy one, is not due until
**October 7**, nine days after the first TorchLingo contact in class and a week after the
lecture. That is the buffer the pivot has to work in.

### Where TorchLingo stands today

Rewritten Sep 26. The estimates in v2 have been replaced by measurements, and two of the
three risks named there are closed.

**Closed: the library ships complete.** TorchLingo **0.2.0** is on PyPI, verified by
installing from PyPI into a clean environment. 0.0.8 had shipped 18 files and was missing
seven modules, so several tutorials could not run from a pip install at all. The install
instruction is now a plain `pip install torchlingo` with no version pin and no git URL.

**Closed: Colab resume works.** Verified in Colab, not only in tests: a run mounts Drive,
writes its checkpoints there, and continues rather than restarting from epoch zero. v2
listed this as the highest risk on the strength of a task note saying the code had been
written twice and run zero times. That note was out of date when I quoted it.

**Closed: the units question.** Epochs, not steps. At 100K pairs and batch 64 an epoch is
about 1,560 batches, so "20,000 steps" is roughly 12.8 epochs in TorchLingo and roughly 65
to 165 in the OpenNMT configuration Fall 2025 students actually ran. The number does not
survive the move, which is the argument for dropping it. The recommendation is **30 to 36
epochs**, from measured convergence on this library.

**Still open: what a 100K run actually produces.** No BLEU figure and no wall-clock figure
exist for a run at the assignment's scale. A 36-epoch attempt consumed all application
memory and took a machine down, which is what produced the length-cap work below. Until a
run completes, Assignment 8 asks for 30 to 36 epochs without telling students what quality
to expect.

**Still open: what a paid Colab session provides.** The memory figures below are Apple Metal
unified-memory numbers on a 64 GiB machine. They are not CUDA numbers, and no memory figure
belongs in the assignment text until the Colab ceiling is known.

### What the pivot has settled

**A 100-token length cap, and why.** Drop any pair where either side exceeds 100 tokens.
Measured on the German corpus, 100K pairs, batch 64, 3 layers, 8 heads:

| | cap 100 | no cap |
|---|---|---|
| device memory held | 9.60 GiB | 35.80 GiB |
| seconds per epoch | 192.0 | 192.0 |
| pairs truncated | 1.30% | 0% |

73% of the memory cost removed for 1.30% of the data, at identical wall clock, because peak
memory is set by the longest batch rather than the median one. Across the measured range
memory moves 28x while time moves 1.39x, which gives the line students need: **batch count
sets how long an epoch takes; the length cap sets whether it fits at all.** Growth is not
quadratic in the cap, because embedding and feed-forward activations are linear in length
and dominate until sequences get long.

**Assignment 8's floor, for low-resource languages.** The 100K floor now mirrors the wording
of Assignments 4 and 5: at least 100K training pairs, and if your cleaned data has less than
that, use all of it and say so in the write-up. Low-resource students are defined by the
course as having under 200K available before cleaning, so the original floor had no variant
for exactly the students most likely to miss it.

**A bug in Assignment 9, fixed in the wording.** That assignment is a controlled comparison
of the tokenizer, so the tokenizer must be the only thing that differs. Expressing the
length cap in tokens breaks that, because changing the tokenizer changes which pairs the cap
excludes. The fix: choose the sentence set once, using the subword tokenizer, and use that
same set for both runs.

**An ordering wrinkle still to resolve.** Assignment 8 says "BPE for inflected languages",
but BPE is not taught until Lecture 9 on Oct 5, two days before A8 is due. Either drop the
mention or mark it optional and covered next week.

### The two sessions now hand off in writing

The repository session and this one cannot message each other. They exchange files in the
TorchLingo working tree instead, under `notes/handoff/`: `briefing.md` for standing context,
`to-cowork.md` for messages in, `from-cowork.md` for messages out. Course notebooks live in
that repository now and are changed by the repository session, so changes to them are
requested through that file rather than made here.

**Notebooks, as of Sep 26.** Four student notebooks are public in
`torchlingo/docs/docs/course/`, named `lecture-NN-<slug>.ipynb`, each with an Open-in-Colab
badge off `main`. The two instructor copies, which carry worked answers, are in
`torchlingo-private/course/`. The Lecture 6 deck links the public badge rather than a Drive
copy.

### The grader has no source and no home

`grader.exe`, which Lectures 4 and 5 both send students to, is four compiled binaries and an
`Instructions.md` in a PhD student's personal OneDrive. Sizes from 27 MB to 301 MB indicate
PyInstaller bundles. There is no source code, no repository in `byu-matrix-lab` or on the
author's GitHub account, no license and no version. The Windows and M-series builds date
from September 2023.

Two problems follow. The course depends on an artifact that disappears when that OneDrive
account does. And students are told to download an unsigned 301 MB executable and, on macOS,
to override Gatekeeper to run it.

The path forward is to ask the author for the source, put it in `byu-matrix-lab` with a
license, and ship it as a script or a pip install. If the source is gone, the checks are all
documented in the Lecture 4 deck and TorchLingo's `diagnostics` module is the natural home.

## What is already modernized, and what is not

Lectures 1 through 8 are rebuilt for F2026: an Objectives slide on each, assignment slides
split into "What To Do" and "What To Submit", AI-use guidance tied to the department's
levels, Colab activities ending in a Report Back, and Fall 2026 submission links. Lecture 5
gained Gale-Church. Lecture 6 gained "Why Evaluate?", editable formulas in place of bitmap
images, and an MTSurvey demo. Lectures 7 and 8 carry the pivot.

Lectures 9 through 23 are still last year's decks. In the order they arrive:

1. **Lecture 9, Mon Oct 5.** Its SentencePiece handout is OpenNMT-specific. It also owns the
   demonstration the length cap sets up: before subwording a student's tokens are words,
   after it the same rule excludes a different set of sentences, and they can count the
   difference on their own data.
2. **Lecture 13, Mon Oct 19, and Lecture 14, Wed Oct 21.** Back-translation and multilingual
   tagging, both written against OpenNMT. Lecture 14's handout is a Word document,
   "MNMT Guide Using OpenNMT.docx", that needs replacing outright.
3. **Lecture 19, Mon Nov 23.** Overlaps Lecture 5 now that sentence alignment moved forward.
   Its word-alignment half stands on its own; the rest needs pruning.
4. **Lectures 10, 11, 12, 15 to 18, 20 to 23.** No framework dependency, so they run as they
   are until rebuilt for style.

Two prerequisites still need to reach students: the paid Colab plan, which the syllabus
already requires and which a reminder has been drafted for, and MTEval accounts, which
students self-register for.

## Open questions

- **What does a 100K-pair run produce, and how long does it take?** Assignment 8 is due Oct
  7 and this is still unmeasured. Everything about how the assignment states expectations
  depends on it.
- **What memory does a paid Colab session actually provide?** Being measured separately. No
  memory figure goes in the assignment text until it exists.
- **Does the grader survive?** Depends on whether its source still exists.
- **The A5 audit.** Clean pair count, duplicate-source rate and longest sentence per student,
  from the Assignment 5 submissions. It tells you how many students cannot reach Assignment
  8's floor and by how much. Still not run; it needs the submissions staged somewhere
  readable.
- **Lecture 19 needs rescoping** around the material now in Lecture 5.
- **Lecture 15's assignment is reading only.** That free week is the obvious place to absorb
  slippage from the pivot, and Lectures 13 and 14 are heavy.
