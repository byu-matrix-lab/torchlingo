# CS 479 Fall 2026 course roadmap, and what it asks of TorchLingo

Handed over 2026-09-24, corrected 2026-09-25, from the Cowork session where Eric's CS 479
decks are being rebuilt.

**The correction matters, so it goes first.** An earlier version of this file assumed the
course would stay on OpenNMT-py for Fall 2026 and that TorchLingo was a target for a later
offering. That was wrong. OpenNMT-py is in maintenance mode, and **Eric is pivoting the
course to TorchLingo this semester**. Lecture 7 is Monday, September 28. Plan against that.

Part 1 is the course map, verbatim from `CS479 Fall 2026 Roadmap_v2.md` in Eric's course
folder, with all dates confirmed against the Learning Suite Schedule tab. Part 2 is what it
asks of this repository.

---

# Part 1: the course


Introduction to Machine Translation, BYU. Monday and Wednesday, 11:00 to 12:15.

**What this is.** A forward map of the semester: what each lecture covers, what its
assignment asks for, and what tooling that assignment depends on. Built from the Fall 2026
Learning Suite course structure, the six rebuilt F2026 decks, and the Fall 2025 decks for
everything not yet rebuilt.

**How to read the status column.** *F2026* means the deck has been rebuilt for this
semester. *F2025* means the lecture will run from last year's deck unless it is reworked
first, so its scope note describes what that deck does today.

**Dates.** Confirmed against the Learning Suite Schedule tab on Sep 25.

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
| 7 | Mon Sep 28 | Research Paper Reviews; Intro to Neural Networks | **A6** human vs. automatic evaluation | F2025 |
| 8 | Wed Sep 30 | Neural MT Overview and Architectures | | F2025 |
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

### 7. Research Paper Reviews; Introduction to Neural Networks (F2025, 32 slides)
Two halves. First, the paper-review assignment: 20 papers, each student picks one and
gives a 5-minute presentation later in the semester. Second, neural network foundations
from the AMTA 2018 tutorial: hidden layers, activation functions, cost functions, gradient
descent.
**In-class activity:** install OpenNMT-py and run the Quickstart toy English-German model,
on Colab with a GPU.
*This is the first lecture where the course depends on a training framework.*

### 8. NMT Overview and Architectures (F2025, 36 slides)
Encoder-decoder, the fixed-representation bottleneck, degradation with sentence length,
RNN with attention, self-attention, softmax, the Transformer, multi-head attention, pros
and cons of NMT. Includes a frank debrief of what went wrong for students in the install
activity, including how badly dirty training data shows up in output.
**Assignment:** train an English-to-X system on at least 100K cleaned pairs, with 2K test
and 2K validation held out and no overlap, at least 20,000 training steps, BPE if the
language is inflected. Deliver the three data splits, MT output on the test set, ten
sample pairs with back-translations, a SacreBLEU score, and a write-up of the
configuration. Re-cleaned data gets re-uploaded.
*This is the heaviest assignment in the course.*

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
| grader.exe | Lecture 4 | checking cleaned output |
| MTEval (mteval.matrix.byu.edu) | Lecture 6 | human ranking; students self-register |
| SacreBLEU, chrF | Lecture 6 | automatic scoring, and again in 8, 9, 13, 14 |
| **TorchLingo** | **Lecture 7** | **every model the students train: 7, 8, 9, 13, 14** |
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

### Where TorchLingo stands today, against those dates

Assessed from the repository on Sep 25.

**Lecture 7 looks fine.** Six tutorials already carry Open-in-Colab badges, and tutorial 2,
"Train a Tiny Model", is a direct replacement for the OpenNMT Quickstart activity. This is
the lowest-risk part of the pivot, and it is the part that happens first.

**Assignment 8 is the risk, and it is specific.** Three things:

1. *Scale is unproven.* The largest controlled training run recorded in the repository is
   64,311 pairs for 36 epochs, reaching BLEU 7.32. The assignment asks for at least
   100,000 pairs. No run at that scale has been done, so neither the wall-clock time nor
   the achievable quality is known.
2. *Units do not match.* The assignment specifies "20,000 training steps." TorchLingo is
   configured in epochs. Students will set this wrong unless the assignment is restated in
   the library's own terms.
3. *Colab resume has never been run in Colab.* The repository's own task list records that
   the Drive-backed checkpointing code has been written twice and executed zero times. A
   100K-pair run will outlive a free Colab session. If resume silently restarts from epoch
   zero, students lose the assignment, and they lose it late.

Item 3 is the one to fix first, and it is cheap to test: start a run, interrupt the
runtime, re-run the cell, and check that it prints a resume line rather than starting over.

**A useful precedent.** The Fall 2025 Lecture 10 deck opens with "For those who obtained
reasonably intelligible output from your OpenNMT systems," which suggests some students did
not clear that bar on the old stack either. Worth knowing before setting expectations for
the new one.

---

## What is already modernized, and what is not

Lectures 1 through 6 have been rebuilt for F2026: an Objectives slide on each, assignment
slides split into "What To Do" and "What To Submit", AI-use guidance tied to the
department's levels, Colab activities ending in a Report Back, and Fall 2026 submission
links. Lecture 5 gained Gale-Church. Lecture 6 gained a "Why Evaluate?" slide, editable
formulas in place of bitmap images, and an MTSurvey demo.

Lectures 7 onward are still last year's decks. The largest gaps, in the order they arrive:

1. **Every OpenNMT reference in Lectures 7, 8, 9, 13 and 14** has to become TorchLingo,
   starting Monday. See the section above.
2. **Lecture 7's install activity** is the course's first real infrastructure hurdle, and
   the Lecture 8 debrief shows it went badly for several students last year, on a stack
   that had been in use for years. A new framework raises that risk, not lowers it.
3. **Lecture 19 overlaps Lecture 5** now that sentence alignment moved forward. Its word
   alignment half stands on its own; the rest needs pruning.
4. **Assignment due dates** after Lecture 6 still need confirming against the Schedule tab.
5. **The Lecture 6 assignment's MTEval dependency** means students need accounts before
   Sep 28. Self-registration works, but nobody has told them yet.

## Open questions

- Can TorchLingo train a usable English-to-X model on 100K pairs inside a student's Colab
  budget? Nothing in the repository answers this yet, and Assignment 8 is due Sep 30.
- Does Assignment 8 keep the 100K-pair floor and the 20,000-step figure, or get restated
  for the new stack? A number carried over from a framework nobody is using is worse than
  no number.
- Lecture 19 needs rescoping around the material now in Lecture 5.
- Lecture 15's "assignment" is reading only. Whether that stays a free week is worth a look
  given how heavy Lectures 13 and 14 are, and it is now the obvious place to absorb slippage
  from the pivot.

---

# Part 2: what this asks of TorchLingo

## The shape of it

Five assignments in this course train a model: Lectures 7, 8, 9, 13 and 14. All five are
written against OpenNMT-py today. They are consecutive on the calendar and cumulative in
the artifact, so this is one crossing rather than five. Assignment 9 retrains Assignment
8's system, 13 reuses it, and 14 is built from "the system you created for Assignment 8/9".

TorchLingo is now the course's training stack. That converts this repository from a
teaching library with a hypothetical audience into the thing eighteen students have to get
working, on their own data, on a deadline.

## The dates, and the slack

| Needed in class | Assignment due | What the course needs |
|---|---|---|
| **Mon Sep 28** | — | install, and a toy model, in twenty minutes of class |
| **Wed Sep 30** | **Wed Oct 7** | a real English-to-X model, 100K pairs, scored with SacreBLEU |
| **Mon Oct 5** | **Mon Oct 12** | SentencePiece on and off, everything else held fixed |
| **Mon Oct 19** | **Mon Oct 26** | back-translation: reverse the direction, generate, retrain |
| **Wed Oct 21** | **Wed Oct 28** | bidirectional two-language multilingual, with target-language tagging |

The first date is three days out. The one that decides whether the pivot worked is
**October 7**, which is nine days out.

## What is already in good shape

**Lecture 7 is close to solved.** Six tutorials carry Open-in-Colab badges, and tutorial 2,
"Train a Tiny Model", maps directly onto the OpenNMT Quickstart activity it replaces. The
work here is framing, not capability: the activity has to survive twenty minutes on
eighteen student laptops, and it has to fail loudly rather than quietly.

**Tutorial 6 is unexpectedly well aimed.** It checks corpus alignment with
`diagnose_alignment` and makes the point that a misaligned corpus still loads, still
batches, and still shows a falling loss. That is the most expensive mistake available in
this course: Lectures 4 and 5 are entirely about producing an aligned corpus, and Lecture
8 is where a bad one finally surfaces, three weeks later. Getting that check in front of
students before they start Assignment 8 is high value for low effort.

**The failure catalogue already exists.** The Fall 2025 Lecture 8 deck has a debrief slide
listing exactly why student models produced bad output: 10K training pairs instead of 100K,
dirty data with entities and mismatched segments, and 1K training iterations instead of
20K. Those are the three failures the `diagnostics` module and the "when it fails" page
are built around. A page that names them in the course's own terms would land.

## The three risks on Assignment 8

This is the assignment that decides the pivot. Its requirements, from the Fall 2025 slide:
at least 100,000 training pairs, 2,000 test and 2,000 validation with no overlap, shuffled,
at least 20,000 training steps, BPE for inflected languages, inference over the 2,000-line
test set, a SacreBLEU score, and a write-up of the configuration.

**1. Nothing here has been run at 100K pairs.** The largest controlled run in
`notes/TASKS.md` is 64,311 pairs for 36 epochs, reaching BLEU 7.32. The course asks for
100K and expects "reasonably intelligible output." Unknown: how long that takes on a free
Colab T4, whether it fits in a student's compute budget, and what BLEU is actually
reachable. One end-to-end run would answer all three, and the answer is needed before the
assignment text can state honest expectations.

**2. The assignment counts in steps; TorchLingo counts in epochs.** "20,000 training steps"
does not translate without knowing the batch size and corpus size. Either the assignment
gets restated in epochs, or the library reports both. Carrying over a number from a
framework nobody is using is worse than having no number.

**3. Colab resume has been written twice and run zero times.** Task #38 says this plainly:
`training_checkpoint.py` has `is_colab()`, `mount_drive()` and a Drive-backed default
directory, none of which has ever executed in Colab, and CI cannot cover it. A 100K-pair
run will outlive a free Colab session. If resume restarts from epoch zero, students lose
the assignment late, after they have spent the compute.

Of the three, this is the one to do first. It is also the cheapest: start a run, interrupt
the runtime, re-run the cell, confirm it prints a resume line and trains only the remaining
epochs. Item 3 of that task is exactly the check, and it is still open.

## What the course does not need from this library

Worth stating, because it bounds the work:

- **Evaluation beyond BLEU and chrF.** The course uses SacreBLEU directly and COMET
  directly. TorchLingo's evaluation module has to be *correct*, because students read it
  and compare systems with it, but it does not need to become a metrics suite. The chrF and
  TER transpose bug fixed in PR #58 is exactly the correctness that matters: a student
  comparing two systems on a silently wrong number learns the wrong lesson and cannot tell.
- **LLM prompting** (Lecture 12), which runs on HuggingFace models directly.
- **Quality estimation** (Lectures 10 and 11), which runs on COMET.
- **Speech** (Lectures 15 and 16), which runs on Azure components.

## Suggested order

Straight down the calendar, because each capability is only useful once the previous one
works:

1. **Now.** Verify Colab resume actually resumes. Task #38, item 3.
2. **Before Sep 28.** A single Colab link for the Lecture 7 activity that runs top to
   bottom on a free GPU with no local install.
3. **Before Oct 7.** One end-to-end 100K-pair run, to get a real wall-clock number and a
   real BLEU. Everything about Assignment 8's wording depends on those two numbers.
4. **Before Oct 12.** Tokenizer on versus off as a controlled comparison, with the rest
   held fixed. Same concern as Task #59.
5. **Before Oct 26.** Back-translation as a documented workflow, with reverse-direction
   training as a configuration change rather than a second project.
6. **Before Oct 28.** Multilingual tagging tutorial, replacing the OpenNMT Word document.

Items 1 and 3 are the whole question. If a student cannot train a usable English-to-X model
on their own cleaned data in a Colab session, and cannot recover when the session drops,
nothing downstream matters. If they can, the rest is documentation.

## Caveats

Part 1 is assembled from the Fall 2026 Learning Suite schedule, the six rebuilt F2026
decks, and the Fall 2025 decks for Lectures 7 onward. All dates are confirmed. Lectures 7
through 23 have not been rebuilt for F2026, so their content may change before they run,
and the assignment requirements quoted above are the Fall 2025 versions, which the pivot
will itself change.
