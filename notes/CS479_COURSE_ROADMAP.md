# CS 479 Fall 2026 course roadmap

Course-side reference, maintained by the Cowork session. **v5, 2026-09-27.**

Changes from v4:

- **The TBC markers are gone. Every date in the table is now decided.** The eleventh session
  the 8a/8b split needed comes from **Mon Nov 16**, which Lecture 17 now occupies.
- **Assignment 8 stays Wed Oct 7.** v4 had it sliding to Oct 12 with its lecture. Eric
  decided against the slip, so A8 now appears on **Lecture 9's row** rather than Lecture 10's.
  The "Assignment due that day" column is about due dates, not ownership: A8 is still
  Lecture 8's assignment. If your `leads_to` validation pairs assignments to the lecture whose
  row they sit on, this row change is the one to look at.
- **Assignments 9 through 14 and 16 each gained two to five days.** Exact new due dates are in
  the table and itemized under "The Lecture 8 split, and what it moved".
- **The TAUS conference report is dropped from Lecture 17.** That was a 2025 event, not a 2026
  one, so it does not recur.
- **Lecture 8 is now two decks on disk, 8a (26 slides) and 8b (25 slides).** The single
  Lecture 8 deck is superseded. 8b gained three slides on Sep 27: query/key/value, positional
  encoding, and the anatomy of an encoder block. None of them existed in the F2025 deck.
- **Assignment 8's epoch count is raised from 30-to-36 to 60-to-70**, on your a8-benchmark
  report. See the handoff entry: the report's own recommendation was a step budget, and
  Eric chose the epoch count; the tradeoff is recorded rather than papered over.
- **8a and 8b each get their own quiz.** One more quiz in the semester than last year.
- **Learning Suite is now the stale copy.** These dates are decisions made course-side; the TA
  (Coulson) is updating Learning Suite to match. Treat this file as the authority meanwhile.

---

Introduction to Machine Translation, BYU. Monday and Wednesday, 11:00 to 12:15.

**What this is.** A forward map of the semester: what each lecture covers, what its
assignment asks for, and what tooling that assignment depends on. Built from the Fall 2026
Learning Suite course structure, the six rebuilt F2026 decks, and the Fall 2025 decks for
everything not yet rebuilt.

**How to read the status column.** *F2026* means the deck has been rebuilt for this
semester. *F2025* means the lecture will run from last year's deck unless it is reworked
first, so its scope note describes what that deck does today.

**Dates.** Lectures 1 through 8a were confirmed against the Learning Suite Schedule tab on
Sep 25. Everything from Lecture 8b onward is **our decision, made here**, not yet reflected in
Learning Suite; Coulson is updating Learning Suite to match. Until he has, **this file is the
authority** and Learning Suite is the stale copy.

**This version.** v5, Sep 27. Dates are firm: the Lecture 8 split is scheduled and the shift it forces is resolved. Revised later the same day on two corrections: **Assignment 8 holds at Wed Oct 7** rather than sliding with its lecture, and the **TAUS conference report is dropped**, because that was a 2025 event. Revised again the same day: **Lecture 8 is built as two decks**, the A8 epoch count is raised to **60 to 70** on the benchmark result, and each half gets its own quiz. v4, Sep 27. Lecture 8 is split into **8a** and **8b**, and assignments 1 to 3 are added to the schedule table, which had omitted them. Dates from Lecture 9 onward shift by one session and are marked TBC until confirmed against Learning Suite. v3, Sep 26. Lectures 7 and 8 are now rebuilt, the pivot's open
engineering questions have started returning measurements rather than estimates, and the
repository session and this one now exchange written handoffs. What changed is collected
under “What the pivot has settled” below; the lecture scope notes for 9 onward are unchanged.

---

## Semester at a glance

| # | Date | Lecture | Assignment due that day | Deck |
|---|---|---|---|---|
| 1 | Wed Sep 2 | Course Overview and History of MT | **A1** read the syllabus; MT history | F2026 |
| 2 | Wed Sep 9 | Translation Challenges for MT | **A2** rank translation challenges for your language | F2026 |
| 3 | Mon Sep 14 | Introduction to Word Embeddings | **A3** multilingual embedding space | F2026 |
| 4 | Wed Sep 16 | Data Preparation for MT Training | | F2026 |
| 5 | Mon Sep 21 | Data Preparation for MT Training, Part 2 | **A4** initial cleaning steps | F2026 |
| 6 | Wed Sep 23 | Human and Automatic MT Evaluation | **A5** complete cleaning pipeline | F2026 |
| 7 | Mon Sep 28 | Research Paper Reviews; Intro to Neural Networks | **A6** human vs. automatic evaluation | **F2026** |
| 8a | Wed Sep 30 | Neural MT: Encoder-Decoder, and Why Attention Was Invented | | **F2026** |
| 8b | Mon Oct 5 | Neural MT: The Transformer | | **F2026** |
| 9 | Wed Oct 7 | Morphology and Terminology in NMT | **A8** create and run an NMT model | F2025 |
| 10 | Mon Oct 12 | Overview of MT Quality Estimation | | F2025 |
| 11 | Wed Oct 14 | Neural Quality Estimation and Evaluation | **A9** SentencePiece · **A10** install COMET | F2025 |
| 12 | Mon Oct 19 | Using LLMs for MT; Expanding Context Awareness | **A11** run COMET on the A6 sentences | F2025 |
| 13 | Wed Oct 21 | Strategies for NMT of Low-Resource Languages | **A12** context and LRL translation | F2025 |
| 14 | Mon Oct 26 | Multilingual NMT and "Zero-shot" NMT | | F2025 |
| 15 | Wed Oct 28 | Overview of Speech-to-Speech MT | **A13** back-translated data | F2025 |
| 16 | Mon Nov 2 | Automatic Dubbing and Interpretation | **A14** bidirectional MNMT | F2025 |
| — | Wed Nov 4 | Project proposal outline reviews with instructor | **A16** SLT pipeline | |
| — | Mon Nov 9 | Project proposal outline reviews with instructor | | |
| — | Wed Nov 11 | Presentations of final project proposals | **Final project proposal** | |
| 17 | Mon Nov 16 | Multimodal NMT | | F2025 |
| 18 | Wed Nov 18 | HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation | | F2025 |
| 19 | Mon Nov 23 | Word and Sentence Alignment | | F2025 |
| — | Wed Nov 25 | **No class**, Thanksgiving Eve | | |
| 20 | Mon Nov 30 | What can we learn from previous MT methods: RBMT, KBMT, EBMT, PBSMT, SBSMT | | F2025 |
| — | Wed Dec 2 | Project checkpoint reviews with instructor | | |
| 21 | Mon Dec 7 | Writing Research Articles; MATRIX Lab research | | F2025 |
| 22, 23 | Wed Dec 9 | MT Applications and Jobs; Opportunities for Further MT Research | **Paper review assignment** | F2025 |
| — | Thu Dec 10 | Last day of class | Final project **presentations, submission, write-up** | |
| — | Wed Dec 16 | Final exam, 11:00 to 14:00 | | |

### The Lecture 8 split, and what it moved

Lecture 8 is two sessions: **8a** on Wed Sep 30 and **8b** on Mon Oct 5. Lecture *numbers*
do not shift, so every notebook's declared `serves_lectures` stays valid; only dates move.

Eleven lecture sessions were needed between Sep 30 and early November where ten existed. The
extra one comes from **Mon Nov 16**, whose second item in Fall 2025 was a report on the TAUS
conference. That was a 2025 event and does not recur, so the slot was already free.
**Lecture 17 moves there.**

What that buys and what it costs:

- **Lectures 9 through 16 each shift one session later.** Their assignments move with them,
  which is where the breathing room comes from.
- **A8 does not move. It stays Wed Oct 7**, now falling on Lecture 9's day rather than
  Lecture 10's. It is the heaviest assignment of the semester and the one whose expected
  output nobody can state yet, but the fix for that is a completed 100K run, not five more
  days. Holding it also keeps Assignment 9, which retrains A8's system, from being pushed
  into the same week.
- **Everything after A8 gains two to five days.** A9 and A10 move Oct 12 to **Oct 14**; A11
  Oct 14 to **Oct 19**; A12 Oct 19 to **Oct 21**; A13 Oct 26 to **Oct 28**; A14 Oct 28 to
  **Nov 2**; A16 Nov 2 to **Nov 4**. The two longest gains land on A11 and A14, the COMET
  comparison and the bidirectional multilingual system, which are the two that depend on the
  most prior machinery working.
- **Both proposal review sessions survive**, Nov 4 and Nov 9, as does the presentation
  session on Nov 11.
- **Nothing from Nov 18 onward moves.** Lectures 18 through 23, the checkpoint reviews, the
  final presentations and the exam are all untouched.
- **Two costs.** Lecture 17 is now two weeks after Lecture 16 and sits after the proposal
  presentations, which is the least damaging place for a gap since Multimodal NMT is the
  most self-contained lecture in the run. And there is no longer a catch-up session if the
  proposal presentations overrun Nov 11.

An assignment appears on the row of the day it is **due**, which is how the Learning Suite
schedule lists it. Lecture quizzes run through Lecture 20 only, and **8a and 8b each get their own**, so the semester carries one more quiz than last year. There is no Lecture 7
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
a human ranking disagreed; **the data-splitting lesson, moved up from Lecture 8**; "The
Library You Will Use: TorchLingo" (rewritten Sep 27; it was "A Change of Framework" and
explained OpenNMT's retirement, which per Eric students do not need. No F2026 deck mentions
OpenNMT now except a stale repository link in Lecture 1's resources list); a rewritten Colab slide
carrying the paid-plan requirement; and "What Is That Loss Number?", giving `ln(V)` as the
reference point for a student's first training loss and warning that label smoothing puts a
floor under it.
**In-class activity:** install TorchLingo and train a toy model, via tutorial 2's Colab
badge. **Assignment:** none. The week is for choosing and reading a paper.

### 8a. Encoder-Decoder, and Why Attention Was Invented (F2026, 27 slides)
Split out of the single Lecture 8 deck on Sep 27. **The principle of the seam: 8a is what you
need in order to do the assignment, 8b is what the model actually is.** A8 is handed out here,
so nobody waits for the Transformer to start work.

Carries the opening block and the Quiz 7 review, then Objectives, the TorchLingo install
debrief, the Fall 2025 debrief on why student models came out bad, the encoder-decoder
sequence with its two animations, the fixed-representation bottleneck, degradation with
sentence length, and attention as the fix through "RNN with Attention". Then the practical
half: the splitting recap carrying the argument that a contaminated split propagates through
Assignments 9, 13 and 14; "Sentence Length Is a Memory Budget", built on measured figures; and
the assignment as What To Do, What To Submit and an AI-use slide.
A slide added Sep 27, **"Reading Your Loss Curve"**, closes two gaps found when the loss
explanation was audited. Label smoothing had been promised in Lecture 7 ("we will come back to
this in Lecture 8") and appeared in neither 8a nor 8b; the claim itself checks out, since
`label_smoothing` defaults to 0.1 and is passed into the loss. And validation loss was a
required deliverable that no deck explained: students build 2,000 validation pairs and nothing
said what for, while overfitting appeared nowhere in the course. The slide gives three curve
shapes and the action the handout was not reaching for, namely passing `val_loader` and
`save_dir` so the best checkpoint by validation loss is kept rather than the last one.
**Assignment (due Wed Oct 7):** train an English-to-X model on your own cleaned data. At
least 100K training pairs, 2K validation, 2K test, or all of it if you have less; source-side
deduplication and a verified split; 100-token cap; **60 to 70 epochs**; checkpointing to Drive;
SacreBLEU over the whole test set.

### 8b. Neural MT: The Transformer (F2026, 26 slides)
Split out on Sep 27. Opens with its own quiz and a **placeholder for the discussion question**,
which still needs writing, then Objectives and a "Where We Left Off" recap that restates the
bottleneck and the fix in three cards. The architecture run carries over untouched:
self-attention, softmax, multi-head attention, the full Transformer, and pros and cons of NMT.

New for F2026: "Welcome to the Birthplace of the Transformer", a light full-bleed slide,
Eric's image of the architecture glowing inside a NICU incubator with the real newborns in
the row behind it, dating the architecture to 12 June 2017; "What Capacity Buys, and
When", the measured 11.7M-against-56M crossover with its one-seed and greedy-decoding caveats
stated on the slide; and an Assignment 8 reminder, since A8 is due two days after this session.

Three slides added Sep 27 to close gaps the F2025 deck had left, after a content-weight
comparison across Lectures 6 to 9 found 8b the thinnest session of the five while carrying
the hardest material. **"Attention, Mechanically: Query, Key, Value"** gives the mechanism
the vocabulary it was missing, with the scaled dot-product formula set as editable text.
**"No Recurrence. So How Does It Know the Order?"** answers the question the architecture
slide provokes and never addressed: positional encoding, and why every later system still
injects position somewhere. **"What Is Actually Inside One Encoder Block"** names the
residual connection, layer norm and feed-forward sublayer, notes that the feed-forward layer
holds most of the parameters, which sets up the capacity slide, and closes by identifying
the decoder's extra sublayer as the cross-attention taught in 8a.

A fourth slide, **"The Residual Stream"**, added Sep 27 at Eric's direction: teach the later
reading rather than pretend it is 2017. The 2017 paper presents the residual connection as a
skip borrowed from ResNets so the gradient survives depth, which is true and the least
interesting thing about it. The slide gives the mechanistic-interpretability reading instead,
from Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021: one channel of
width `d_model` running the whole depth, which every attention head and feed-forward block
reads from and adds back to, with nothing overwriting anything. The mathematics is identical;
what changed is what we think it is for. The slide states the provenance and one caveat, that
the picture is cleanest in pre-norm models where the stream is untouched, while the original
post-norm design the students are training normalises after the addition and so rescales the
channel at every block.
The Koehn video and book references close it.
**Assignment:** none of its own. A8 was handed out in 8a.

### 9. Handling Morphology and Terminology in NMT (F2025 with corrections, 26 slides)
`Lectures 9 ..._F2026_v1.pptx` carries two F2026 changes and is otherwise last year's deck.
**The stale assignment dates are corrected**: A8 was showing Mon Oct 6 and A9 Wed Oct 8, in
slide titles and in body text, and now read Wed Oct 7 and Wed Oct 14. **And the decoding
constraint is stated on the A9 slide**, where the assignment is set: decode both runs the same
way, say how you decoded, Lecture 10 covers why. The teaching itself is at Lecture 10 (below).

**Its tokenization notebook does not exist yet** and is the one thing Lecture 9 still needs that
is about Lecture 9's own subject. Last year's predecessor,
`Fall 2025/Jupiter notebooks/OpenNMT_and_Sentencepiece.ipynb`, is nineteen cells of which three
are SentencePiece and the rest is OpenNMT plumbing, including a `numpy<2.0` pin with a comment
blaming OpenNMT's lack of maintenance. Two things it did that the replacement must not: it fit
the tokenizer on a toy corpus rather than the student's own training split, which for a
controlled comparison leaks test material into the vocabulary, and it said nothing about the
length cap, which is the recorded A9 bug.

**That file is not a rebuild.** The SentencePiece handout is still OpenNMT-specific.


Morphological preprocessing, byte-pair encoding, SentencePiece, and approaches to
injecting terminology into NMT.
**In-class activity:** a notebook installing SentencePiece and wiring it into OpenNMT.
**Assignment:** add SentencePiece to the Assignment 8 system, retrain, rerun on the same
test set, and write a comparison of the two BLEU scores and the quality difference. If the
student already used SentencePiece, they run it again without.

### 10. MT Quality Estimation Overview (F2025 plus a new block, 34 slides)
**Beam search lives here, decided Sep 27.** Until then decoding was taught nowhere in the course:
a search of every F2026 and F2025 deck returned no occurrence of "beam", so students trained a
model, decoded greedily, reported a BLEU number and were never told greedy was a choice. Three
slides in `Lectures 10 ..._F2026_v1.pptx`, inserted after the assignment block and before the QE
content: the distribution and the greedy default, beam search with a worked example where a
locally worse first token wins on total log probability, and the rule, which closes back to
Lecture 6 and forward into the lecture's own subject, since a quality number is only as good as
what was held fixed to produce it. Tutorial 3 is the reading, moved here from Lecture 22.

It sits here rather than at Lecture 9 because Lecture 9 was already carrying its own subject plus
an A8 debrief, diagnostics reading and a subword activity. **The cost: A9 is due Wed Oct 14 and
this lecture is Mon Oct 12**, so the teaching arrives two days before the assignment it protects,
which is why the rule itself is also stated on Lecture 9's A9 slide.


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
ablations, and project examples. No assignment; by the time it runs the proposals are in.

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
| Wed Oct 7 | **Wed Oct 14** | Lecture 9: the SentencePiece instructions are OpenNMT-specific |
| Wed Oct 21 | **Wed Oct 28** | Lecture 13: back-translation workflow |
| Mon Oct 26 | **Mon Nov 2** | Lecture 14: the "MNMT Guide Using OpenNMT.docx" handout needs replacing |

The schedule is tighter than it first looks. Assignment 8 is due **October 7**, nine days
after the first TorchLingo contact in class, a week after Lecture 8a and only two days after
Lecture 8b. That is the whole buffer the pivot has to work in, and it is the reason the
remaining open question below matters as much as it does.

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
survive the move, which is the argument for dropping it. The recommendation was 30 to 36
epochs; it is now **60 to 70**, for the reason in the next item.

**Closed, and it changed the assignment: what a 100K run actually produces.** The run exists.
At A8's own configuration, German bitext, 100K/2K/2K, 11.7M parameters: **36 epochs gives 11.46
BLEU in 64.7 minutes and is still improving when it stops; 65 epochs gives 14.48 and converges,
in 112.7 minutes.** So the original 30-to-36 recommendation stopped the model mid-climb and
cost about 3 BLEU. **Eric's call, Sep 27: raise the handout to 60 to 70 epochs.** The report
argues for a step budget instead, about 100,000 optimizer steps, on the grounds that it
survives a student changing corpus size where an epoch count does not; that is the known
weakness of the number now in the handout, and the students it bites are the low-resource ones.
Wall clocks are Apple Metal on a 64 GiB machine and do not transfer to Colab.

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
but BPE is not taught until Lecture 9 on Oct 7, the same day A8 is due. Either drop the
mention or mark it optional and covered next week.

### The two sessions now hand off in writing

The repository session and this one cannot message each other. They exchange files in the
TorchLingo working tree instead, under `notes/handoff/`: `briefing.md` for standing context,
`to-cowork/` for messages out, `from-cowork/` for messages in. Course notebooks live in
that repository now and are changed by the repository session, so changes to them are
requested through that file rather than made here.

**Notebook coverage, as of Sep 27.** Checked against the notebooks themselves rather than
their declared metadata. **Lecture 7** is covered: tutorial 2 is its activity, and the rebuilt
neural-network block now uses that notebook's own corpus as its running example. **Lecture 8a and 8b** are
covered by tutorial 4, reassigned there from Lecture 19 on Sep 27 (see below). **An A8 kickoff notebook is commissioned**, requested Sep 27 for
use in 8a on Sep 30, closing the one gap that had no owner: Lectures 4, 5 and 6 each had an
in-class activity that began the assignment and A8, the largest in the course, had none. It
takes a student from their A5 corpus to a training run that is already going when they leave
the room: dedupe, group-aware split, a loud `check_contamination`, the length cap reported on
their own data, then the model built and training started with checkpointing to Drive. It does
not clean anything, because cleaning was A4 and A5 and is graded.
Three Fall 2025 notebooks were brought into the repository on Sep 27 rather than rewritten:
the **regex refresher** (Lecture 4, reference), the **COMET install** (Lecture 10, homework) and
the **LLM in-context assignment scaffold** (Lecture 12, homework). All three are
framework-independent and none of them ever touched OpenNMT. The fourth, last year's
OpenNMT-and-SentencePiece notebook, is kept as reference for the Lecture 9 rebuild rather than
published. **Lecture 9** went from nothing to two: tutorial 6, reassigned from Lecture 8 on Sep 27, and the
subword notebook when it exists, which is planned to close Lecture 9 and start Assignment 9
together. Tutorial 1's vocabulary section is word-level only and was an over-claim as coverage
for Lecture 9; the repository session said so rather than leaving the map flattering.

**Tutorial 6 moved from Lecture 8 to Lecture 9, Sep 27.** It is a diagnostics notebook, and a
student cannot diagnose a model they have not built: at 8a they have trained only the toy model.
Two things are worth holding onto about this pairing. It is **by sequence, not by topic** —
nothing in it concerns morphology, BPE or SentencePiece. And **A8 is due Wed Oct 7 at 10:00 while
Lecture 9 runs at 11:00 the same day**, so as Lecture 9's reading it arrives an hour after the
assignment it would most have helped. Its value there is for Assignments 9, 13 and 14, which all
rebuild the A8 system. To reach students while it still matters, 8b's Assignment 8 reminder slide
names it directly, and 8b runs two days before A8 is due. **When Lecture 9's deck is rebuilt it
should open with an A8 debrief**, the way Lecture 7 opens with an A6 debrief; that is what
anchors this notebook to the session rather than leaving it a calendar accident.

**Tutorial 4 moved from Lecture 19 to 8a and 8b, Sep 27.** Lecture 19 is statistical word
alignment, the IBM models; tutorial 4 is neural attention alignment. Parts 1 through 6 support
8a, from the bottleneck through the with-and-without-attention ablation to whether the model
learned the *right* alignment, and Part 7 is titled "You have already seen the Transformer's
mechanism", which is where 8b starts. 8b's recap slide links it. **Lecture 19 now has no
notebook**, which is acceptable only because it already needed rescoping and does not run until
Nov 23.

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

Lectures 1 through 8b are rebuilt for F2026, nine decks now that Lecture 8 is two: an Objectives slide on each, assignment slides
split into "What To Do" and "What To Submit", AI-use guidance tied to the department's
levels, Colab activities ending in a Report Back, and Fall 2026 submission links. Lecture 5
gained Gale-Church. Lecture 6 gained "Why Evaluate?", editable formulas in place of bitmap
images, and an MTSurvey demo. Lectures 7, 8a and 8b carry the pivot.

Lectures 9 through 23 are still last year's decks. In the order they arrive:

0. **Lecture 7's neural-network block, slides 12 to 32.** Not a framework problem: 21 bitmap
   slides from the AMTA 2018 tutorial, in SDL's template, teaching neurons, activation
   functions, forward pass, cost and gradient descent through a tic-tac-toe running example.
   Framework-agnostic, so nothing in it is wrong, but it cannot be edited, it carries another
   organization's branding inside a dark deck, and its running example is not translation. The
   rebuild worth doing swaps tic-tac-toe for the toy model students train twenty minutes later
   in the same session. Roughly 12 to 14 slides in our format instead of 21. Largest single
   rebuild left in the first half; needs the diagrams redrawn as vector shapes.
1. **Lecture 9, Wed Oct 7.** Its SentencePiece handout is OpenNMT-specific. It also owns the
   demonstration the length cap sets up: before subwording a student's tokens are words,
   after it the same rule excludes a different set of sentences, and they can count the
   difference on their own data.
2. **Lecture 13, Wed Oct 21, and Lecture 14, Mon Oct 26.** Back-translation and multilingual
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

- **Does a 60-to-70 epoch instruction hold up for a student with 30K pairs?** The epoch count
  is calibrated on a 100K run. An epoch count is not corpus-size invariant, which is the
  argument the benchmark report makes for a step budget. Worth revisiting once the A5 audit
  says how many students are well below 100K.
- **What memory does a paid Colab session actually provide?** Being measured separately. No
  memory figure goes in the assignment text until it exists.
- **Does the grader survive?** Depends on whether its source still exists.
- **The A5 audit.** Clean pair count, duplicate-source rate and longest sentence per student,
  from the Assignment 5 submissions. It tells you how many students cannot reach Assignment
  8's floor and by how much. Still not run; it needs the submissions staged somewhere
  readable.
- **Is Lecture 9 now carrying too much?** It went from no notebooks to three in one afternoon,
  and it has gained the decoding block on top of its own subject. If it proves too full, the
  decoding block is the part that moves cleanly to Lecture 10, whose subject is what a quality
  number means and whose assignment is light.
- **Lecture 19 needs rescoping** around the material now in Lecture 5, and now has no notebook
  either, since tutorial 4 moved to 8a and 8b.
- **Lecture 15's assignment is reading only.** That free week is the obvious place to absorb
  slippage from the pivot, and Lectures 13 and 14 are heavy.

---

## Which notebook serves which lecture

Two notebook families now exist and their
numbering schemes do not line up, deliberately:

| family | numbered by | who writes it |
|---|---|---|
| `docs/docs/tutorials/` | library topic | this repository |
| `docs/docs/course/` | lecture | Cowork writes content, this repository commits it |

So nothing about which artifact belongs in which lecture is derivable from a filename. It
used to be written down only here, by hand. It is now **declared by each notebook** in its
own `torchlingo` metadata and generated from them, because a hand-kept table next to the
artifacts it describes is the redundancy this document warns about everywhere else — and it
had already drifted twice.

Dates, topics and assignment deadlines are **in the schedule above** rather than repeated
here. Lecture numbers are the join.

The table below is a build artifact. **Do not edit it**: change the notebook's metadata and
run `python scripts/notebook_meta.py --write`. CI fails if the two disagree.

<!-- BEGIN generated:notebook-map -->

| # | Lecture | Course notebook | TorchLingo tutorial |
|---|---|---|---|
| 1 | Course Overview and History of MT | — | — |
| 2 | Translation Challenges for MT | — | — |
| 3 | Introduction to Word Embeddings | `lecture-03-word-embeddings` (activity) | — |
| 4 | Data Preparation for MT Training | `lecture-04-regex-refresher` (reference) ², `lecture-04-tmx-cleaning` (activity) | `01-data-and-vocab` (reference) ¹ |
| 5 | Data Preparation for MT Training, Part 2 | `lecture-05-sentence-alignment` (activity) | — |
| 6 | Human and Automatic MT Evaluation | `lecture-06-mt-evaluation` (activity) | — |
| 7 | Research Paper Reviews; Intro to Neural Networks | — | `02-train-tiny-model` (reading) |
| 8a | Neural MT: Encoder-Decoder, and Why Attention Was Invented | — | `04-attention-and-alignment` (reading) |
| 8b | Neural MT: The Transformer | — | `04-attention-and-alignment` (reading) |
| 9 | Morphology and Terminology in NMT | — | `01-data-and-vocab` (reference) ¹, `06-diagnosing-failures` (reading) |
| 10 | Overview of MT Quality Estimation | `lecture-10-comet-install` (homework) ³ | `03-inference-and-beamsearch` (reading), `05-real-translations` (reading) |
| 11 | Neural Quality Estimation and Evaluation | — | — |
| 12 | Using LLMs for MT; Expanding Context Awareness | `lecture-12-llm-context` (homework) ⁴ | — |
| 13 | Strategies for NMT of Low-Resource Languages | — | — |
| 14 | Multilingual NMT and "Zero-shot" NMT | — | — |
| 15 | Overview of Speech-to-Speech MT | — | — |
| 16 | Automatic Dubbing and Interpretation | — | — |
| 17 | Multimodal NMT | — | — |
| 18 | HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation | — | — |
| 19 | Word and Sentence Alignment | — | — |
| 20 | What can we learn from previous MT methods: RBMT, KBMT, EBMT, PBSMT, SBSMT | — | — |
| 21 | Writing Research Articles; MATRIX Lab research | — | — |
| 22, 23 | MT Applications and Jobs; Opportunities for Further MT Research | — | — |

Notes on the rows that are not simple:

- ¹ **`01-data-and-vocab`** — covers loading and cleaning a parallel corpus, which is
  Lecture 4's subject from the library side, and its vocabulary half belongs to Lecture 9.
  Lecture 4 has already run this year, so the pairing is retrospective there and genuine
  for a future offering.
- ² **`lecture-04-regex-refresher`** — Fall 2025 material, brought into the repository on
  2026-09-27. A fifteen-minute tour of Python regular expressions, which is what the
  sixteen cleaning steps are written in. Lecture 4's deck links it twice and Learning
  Suite posts it on the Lecture 3 tab as well, so it is offered ahead of Lecture 4 rather
  than used inside it.
- ³ **`lecture-10-comet-install`** — Fall 2025 material, brought into the repository on
  2026-09-26. Installs unbabel-comet, scores a worked example, and sets up the HuggingFace
  token through Colab Secrets, which the reference-free models require. It is Lecture 10's
  assignment and the setup for Lecture 11's. Framework-independent: nothing in it touched
  OpenNMT.
- ⁴ **`lecture-12-llm-context`** — Fall 2025 material, brought into the repository on
  2026-09-26. The Assignment 12 scaffold: pick a non-MT HuggingFace model, translate a
  low-resource test set with growing numbers of in-context examples, chart BLEU against
  context size. Every code cell is a TODO, so it contains no answers.
  Framework-independent.

<!-- END generated:notebook-map -->

### Why lectures that have run are still listed

**Lectures 1 to 6 have already run.** They are listed anyway, for three reasons: a student
revisiting them needs to know what to open, assignments later in the course refer back to
them, and a Fall 2027 offering should not have to rediscover the mapping. Where a tutorial
*would have* improved a past lecture it is marked `reference` rather than silently dropped.

### What this table cannot say, and so is written here

The generated table lists notebooks that exist. Absence is the interesting part, and a
notebook that has not been written cannot declare anything:

- **Lecture 9 has no notebook and needs one.** That is #121, which carries the dates.
- **Lecture 14 has no notebook**, and the current handout is a Word document written for
  OpenNMT. That is #99, and #123 is the prior question of whether the code path even works.
- **Lecture 6 will gain a tutorial it cannot show yet.** Tutorial 7 was written for exactly
  this ground and is not merged (#88). Once it lands it will declare Lecture 6 itself, as
  reading *after* Lecture 8, since Lecture 6 has run.
- **Twelve of the twenty-three lectures pair with nothing** — 1, 2, 11 to 18, 20 and 21. Which
  is fine: not every lecture wants a notebook, and inventing one to fill a row would be the
  redundancy this document already warns about.

One row appears twice on purpose. **Tutorial 4 serves Lectures 8a and 8b and will not be
split**, which reverses #140 and the recommendation sent to Cowork on 2026-09-27.

Splitting it was argued from the heading list, which reads like two notebooks: Parts 1 to 6 are
LSTM attention, Parts 7 and 8 are the Transformer. Reading it settles the question the other
way. **Part 7 is titled "You have already seen the Transformer's mechanism"**, and its whole
move is that the attention the reader just ablated in Parts 1 to 6 *is* that mechanism. Split
the file and the 8b half opens by invoking an experiment its reader never ran.

The cost of not splitting is real but small: Parts 1 to 6 are nine of the ten code cells and
need nothing, so they would have been CI-executable on their own (#154), and the 8b half would
have been a one-code-cell notebook. Neither is worth the notebook's best transition.

Two incidental corrections, since #140's own description had them wrong: the seam is *after*
Part 6, not at it, and Part 6 — Bahdanau versus Luong — is Lecture 8a's material, not 8b's.

## What exists

### Tutorials

| | Runs on | Depends on |
|---|---|---|
| 1. Data and Vocabulary | `data/sample_train.tsv` | nothing |
| 2. Train a Tiny Model | `data/example.tsv` | 1 |
| 3. Inference and Beam Search | tutorial 2's checkpoint | **2, at runtime** |
| 4. Attention and Alignment | synthetic reversal task | 2 |
| 5. Translating Unseen Sentences | `data/pretrained/` | 1-3 conceptually |

### Concept pages

`what-is-nmt.md`, `data-pipeline.md`, `vocabulary.md`, `models.md`, `training.md`,
`decoding.md`, `when-it-fails.md`.

### Measured artifacts

Generated into `docs/docs/_generated/`, each from a script, none hand-typed:
`decode_bench` (#12), `decoding_sweep` (#40), `alignment_diagnosis` (#46),
`realign_report` (#29).

---

## Sequencing, including the parts nobody wrote down

Two dependencies are load-bearing and were invisible until this audit.

**Tutorial 3 cannot run without tutorial 2.** It loads the checkpoint tutorial 2 saves.
`scripts/execute_notebooks.py` encodes this — it runs the notebooks in one shared working
directory, in filename order, and lists tutorial 3 as needing the corpus even though it
never names it. A student who opens tutorial 3 in Colab on its own gets a file-not-found
error and no explanation.

**Tutorial 3's model is too small to teach what tutorial 3 is about.** Its beam-size
sweep prints five identical rows because the model is decisive. This is why #40's
measurement had to be done on tutorial 5's checkpoint instead, and why #50 exists. The
sequencing implication is real: *the lesson about beam search requires a model that is
wrong often enough to be uncertain, and the tutorial that teaches beam search does not
have one.*

**Tutorial 4 is independent.** It trains a synthetic reversal task with known ground
truth, which is what lets it measure alignment accuracy rather than assert it. It could
be moved without breaking anything.

---

## Learning outcomes

Stated as what a student can do afterwards. **These are proposed, not confirmed** — they
are reverse-engineered from the material, and they need checking against the course this
feeds.

**Tutorial 1.** Load a parallel corpus; explain why a vocabulary needs `<pad>`, `<sos>`,
`<eos>` and `<unk>`; predict what happens to an out-of-vocabulary word at inference.

**Tutorial 2.** Train a Transformer end to end; read a loss curve well enough to tell
"still learning" from "converged"; recognize that a loss near `ln(vocab_size)` means the
model has learned nothing.

**Tutorial 3.** Implement greedy and beam search from scratch; state what beam search
buys and what it costs; explain why two implementations of the same algorithm can
disagree on ties.

**Tutorial 4.** Explain the encoder-decoder bottleneck as a concrete discarded variable;
run an ablation; judge whether attention learned the *right* alignment rather than merely
a confident one; connect cross-attention to self-attention.

**Tutorial 5.** Distinguish a held-out set that is genuinely held out from one that
leaks; interpret a BLEU score; compare decoding strategies on a model whose answers
actually differ.

**`concepts/decoding.md`.** Recognize beam search as best-first search with a fixed-width
frontier (#41); read a table with error bars and tell a real difference from sampling
noise (#40).

**`concepts/data-pipeline.md`.** Check whether a parallel corpus is actually parallel
(#46); repair one that has drifted, and verify the repair rather than trusting it (#29).

---

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

---

## Redundancy

Not necessarily wrong, but currently undeliberate.

**Beam search appears four times**: explained in `concepts/decoding.md`, reimplemented
from scratch in tutorial 3, visualized via `format_beam_search` (#39), and measured in
#40. The reimplementation is defensible — writing it yourself is the lesson — and
tutorial 3 explicitly reconciles its version against the library's. Worth deciding
deliberately rather than by accumulation.

**Attention appears three times**: `concepts/models.md`, tutorial 4, and
`reference/visualization.md`.

*Corrected 2026-09-18, and this is a gap rather than redundancy.* All three cover the
**LSTM**. `SimpleTransformer` has no attention-returning path at all, so a student who
follows the tutorials onto the Transformer — which is what tutorials 2 and 5 train, and
what the pretrained checkpoint is — cannot inspect attention on the model they are
actually using. Tutorial 4 teaches the concept honestly on an LSTM and a synthetic task;
nothing carries it across. See #34, whose scope was recorded too small for the same
reason.

**The corpus repair story now appears twice**: `concepts/data-pipeline.md` (#46, #29) and
`scripts/realign_corpus.py`'s docstring. These are aimed at different readers and the
overlap is probably correct.

---

## Open questions for the instructor

1. Are the outcomes above the right ones? They are inferred from the material, which
   means they describe what exists rather than what the course needs.
2. Should tutorial 3 keep its from-scratch implementations, or call the library and spend
   the space on diagnosis instead?
3. ~~Where does the lecture 7 assignment attach?~~ **Answered 2026-09-26: it needs nothing.**
   Lecture 7 is a paper review, its activity is tutorial 2, and Task #42 is closed.
4. Is "why models fail" in scope for this library, or is it lecture material?
