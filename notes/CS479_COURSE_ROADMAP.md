# CS 479 Fall 2026: Course Roadmap

*Repository copy of `CS479 Fall 2026 Roadmap_v7.md` (course-side roadmap v7), refreshed 2026-09-29. Everything above "## Which notebook serves which lecture" is regenerated from the desktop file by the course-side session; edit it there. Everything from that heading down is written in this repository and is preserved by the refresh.*

---

Introduction to Machine Translation, BYU. Monday and Wednesday, 11:00 to 12:15.

**What this is.** The forward map of the semester: what each lecture covers, which notebooks
serve it and what each one is for, what assignment it sets and when that is due. One document,
current as of its date. What changed and why is in the Decisions log at the end, not inline.

**v7, Sep 29.** A coherence pass over v6 after Lecture 7 ran and the A8 decisions landed. No
schedule change. What is new since v6 is in the log: A8's model and epoch count, A9's shape
(same files, three settings), the Lecture 9 notebook, LLM-as-judge at Lecture 11, the paper
sign-up sheet, and the assignment directions transcribed into the repository. Stale claims
that v6 still carried (the subword notebook "does not exist", "no pin" on TorchLingo, Colab
memory "unmeasured") are corrected in place.

**What is authoritative.** Learning Suite and the decks are the source of truth for what
students see; this file is the distilled reference the two working sessions share. Where the
two disagree, this file says so and names the stale one.

**Deck status, three states.** *F2026* is a rebuilt deck. *F2026 corrected* is last year's deck
with dated fixes applied and a note saying which; it is not a rebuild. *F2025* is last year's
deck as it stands.

**Dates.** Lectures 1 through 8a were confirmed against Learning Suite on Sep 25. Everything
after is decided here. Coulson has `Learning Suite Schedule Changes for Coulson_v3.md`; as of
Sep 29 Learning Suite still shows the pre-shift dates from Lecture 9 onward and the
pre-shift due dates from A9 onward, so for those **this file is current and Learning Suite is
the stale copy**. Assignment descriptions are transcribed in the repository at
`notes/assignments/`, with a status line per assignment saying which Learning Suite texts
are still last year's (A8, A9, A13, A14).

---

## Semester at a glance

| # | Date | Lecture | Assignment due that day | Deck |
|---|---|---|---|---|
| 1 | Wed Sep 2 | Course Overview and History of MT | **A1** read the syllabus; MT history | F2026 |
| 2 | Wed Sep 9 | Translation Challenges for MT | **A2** rank translation challenges for your language | F2026 |
| 3 | Mon Sep 14 | Introduction to Word Embeddings | **A3** multilingual embedding space | F2026 |
| 4 | Wed Sep 16 | Data Preparation for MT Training | | F2026 |
| 5 | Mon Sep 21 | Data Preparation for MT Training, Part 2 | **A4** initial cleaning steps | F2026 |
| 6 | Wed Sep 23 | Automatic and Human MT Evaluation | **A5** complete cleaning pipeline | F2026 |
| 7 | Mon Sep 28 | Research Paper Reviews; Intro to Neural Networks | **A6** human vs. automatic evaluation | F2026 |
| 8a | Wed Sep 30 | Neural MT: Encoder-Decoder, and Why Attention Was Invented | **A7** toy-model report | F2026 |
| 8b | Mon Oct 5 | Neural MT: The Transformer | | F2026 |
| 9 | Wed Oct 7 | Handling Morphology and Terminology in NMT | **A8** create and run an NMT model | F2026 corrected |
| 10 | Mon Oct 12 | Overview of MT Quality Estimation | | F2026 corrected |
| 11 | Wed Oct 14 | Neural Quality Estimation and Evaluation | **A9** SentencePiece · **A10** install COMET | F2025 |
| 12 | Mon Oct 19 | Using LLMs for MT; Expanding Context Awareness | **A11** run COMET on the A6 sentences | F2025 |
| 13 | Wed Oct 21 | Strategies for NMT of Low-Resource Languages | | F2025 |
| 14 | Mon Oct 26 | Multilingual NMT and "Zero-shot" NMT | **A12** context and LRL translation | F2025 |
| 15 | Wed Oct 28 | Overview of Speech-to-Speech MT | **A13** back-translated data | F2025 |
| 16 | Mon Nov 2 | Automatic Dubbing and Interpretation | **A14** bidirectional MNMT | F2025 |
| — | Wed Nov 4 | Project proposal outline reviews with instructor | | |
| — | Mon Nov 9 | Project proposal outline reviews with instructor | **A16** SLT pipeline | |
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

An assignment appears on the row of the day it is **due**, which is how Learning Suite lists
it, at 10:00 unless said otherwise (Learning Suite has A7 at 11:59 pm; the deck says 10:00,
and the two should agree). A7 is new this year: a short report from Lecture 7's toy-model
exercise. There is no A15: Lecture 15 assigns two papers for the quiz and nothing to submit.
Lectures 17 to 20 carry no assignments; the final project has taken over. Lecture quizzes run
through Lecture 20, and 8a and 8b each have their own, one more than last year.

**Paper reviews** run alongside: 23 papers on the sign-up sheet (Google Sheet "CS 479 Fall
2026", tab "Reading Report"), presented Oct 5 through Nov 30 at one or two per session, each
placed next to the lecture its topic belongs with, and the sheet's LECTURE TOPIC ALIGNMENT
column says which. Dec 7 and Dec 9 are open slots if a student has to move.

---

## How the second half was scheduled

Lecture 8 is two sessions, 8a on Wed Sep 30 and 8b on Mon Oct 5. That is one more session
than the calendar had between Sep 30 and the proposal week. The extra one is Mon Nov 16, which
last year held a report on the TAUS conference, a 2025 event that does not recur; Lecture 17
moved there. Lectures 9 to 16 each shifted one session later. Lecture *numbers* did not change,
so every notebook's declared lecture stays valid.

**A8 did not move.** It stays Wed Oct 7, which is now Lecture 9's day. Its problem was never
calendar; it was that nobody had measured what a 100K-pair run produces or what a Colab GPU
holds, and both are measured now (see the pivot section).

**Assignment windows, measured from the lecture that sets each one.** This is the measure that
matters and the one v5 got wrong: it counted gains from the *old due date*, and the Monday /
Wednesday parity flip meant one session later is sometimes five days and sometimes two.

| assignment | set at | due | window |
|---|---|---|---|
| A7 | 7, Mon Sep 28 | Wed Sep 30 | 2, deliberately: three numbers and two sentences from an exercise started in class |
| A8 | 8a, Wed Sep 30 | Wed Oct 7 | 7 days |
| A9 | 9, Wed Oct 7 | Wed Oct 14 | 7 |
| A10 | 10, Mon Oct 12 | Wed Oct 14 | 2, deliberately: an install, and A11 needs it done |
| A11 | 11, Wed Oct 14 | Mon Oct 19 | 5 |
| A12 | 12, Mon Oct 19 | **Mon Oct 26** | 7, moved from Oct 21 (was 2) |
| A13 | 13, Wed Oct 21 | Wed Oct 28 | 7 |
| A14 | 14, Mon Oct 26 | Mon Nov 2 | 7 |
| A16 | 16, Mon Nov 2 | **Mon Nov 9** | 7, moved from Nov 4 (was 2) |

Two costs of the split, accepted knowingly: Lecture 17 sits two weeks after Lecture 16 and
after the proposal presentations, which is the least damaging place for a gap since Multimodal
NMT is the most self-contained lecture in the run; and there is no catch-up session if the
proposal presentations overrun Nov 11.

---

## The through-line

The course is one long build. A student who keeps up finishes holding a cleaned bilingual
corpus, a trained NMT system, and a stack of measurements of it. Each assignment consumes the
previous artifact.

1. **Lectures 4 and 5** produce the corpus: Church translation-memory data, extracted from TMX,
   through a 16-step cleaning pipeline the student writes, delivered as two sentence-aligned
   text files. Medium- and high-resource languages: at least 200K pairs.
2. **Lecture 6** measures *other people's* systems on it: two commercial MT systems, human
   ranking in MTEval, then SacreBLEU and chrF.
3. **Lectures 7 through 9** build the student's own system: a toy model in class, then a real
   English-to-X model on 100K pairs, then the same model with SentencePiece so the two can be
   compared. Decoding as a choice is taught at Lecture 10, and the rule that comparisons must
   hold it constant is stated where A9 is set.
4. **Lectures 10 and 11** measure it properly, with COMET and COMET-QE, against the Lecture 6
   numbers.
5. **Lectures 12 through 14** go beyond it: an LLM prompted with in-context examples (A12, a
   different model, not the student's), then two ways to improve the student's own system,
   back-translation (A13) and a bidirectional multilingual model (A14).
6. **Lectures 15 and 16** step outside text into speech.
7. **Lecture 17 onward** is context, history and the final project.

The pivot is Lecture 7. Before it the course is about data and measurement; after it, every
assignment assumes the student can train a model.

---

## Lecture scope notes

Each note ends with the notebooks that serve the session, named, with what each is for, and the
assignment it sets. A notebook's *role* is how it is used: an **activity** is run in the
session, **reading** is assigned alongside it, **homework** is the assignment's own scaffold,
**reference** is offered rather than urged.

### 1. Course Overview and History of MT (F2026, 46 slides)
Weaver's 1947 letter, IBM-Georgetown, ALPAC, the evolution to NMT, hype versus reality, MT at
BYU back to Eldon Lytle in 1976. The course's own rules: quizzes, assignments, AI use, grades,
the late policy and the new Early Policy.
**Sets:** A1, read the syllabus and the MT history.

### 2. Translation Challenges for MT (F2026, 46 slides)
Why FAHQT is the wrong question. Lexical and structural ambiguity, false friends, long-distance
dependencies, pronominal reference, word order, topicalization, world knowledge, morphology.
Students pick their language here, which sets up everything after. In-class work on difficult
sentences.
**Sets:** A2, rank the translation challenges for your language.

### 3. Introduction to Word Embeddings (F2026, 51 slides)
The distributional principle, syntagmatic versus paradigmatic relations, word2vec both ways,
LSA, PMI, GloVe, contextualized embeddings, multilingual embeddings in a shared space.
**Notebooks:** `lecture-03-word-embeddings` (activity): tests whether the shared multilingual
space really lines up across the student's languages; it produces the heat map A3 asks for.
**Sets:** A3, the multilingual embedding space.

### 4. Data Preparation for MT Training (F2026, 41 slides)
Translation memories, the CAT loop, TMX anatomy, ISO language codes, the 16 cleaning steps, the
GILT best-practices document, Python TMX packages and TMX editors.
**Notebooks:** `lecture-04-tmx-cleaning` (activity): a ten-unit TMX with planted problems,
Translate Toolkit beside a plain parser, and a checker that names the cleaning step behind each
finding; starts A4. `lecture-04-regex-refresher` (reference): a fifteen-minute tour of the
regular expressions the cleaning steps are written in; posted on the Lecture 3 tab too, so it
is offered ahead of this session. `01-data-and-vocab` (reference): the library's own view of
loading a parallel corpus; Lecture 4 had already run when it was paired, so the pairing is for
a future offering.
**Sets:** A4, extract segment pairs from TMX, fix what breaks alignment, combine sources, write
at least three cleaning steps, prepare 200K pairs.

### 5. Data Preparation, Part 2 (F2026, 31 slides)
The full pipeline, plus Gale-Church sentence alignment brought forward from last year's
Lecture 19: length correlation, the six link types, dynamic programming, the cost model, and
where length-based alignment runs out.
**Notebooks:** `lecture-05-sentence-alignment` (activity): estimates Gale and Church's
parameters on the student's own language; starts A5.
**Sets:** A5, the complete 16-step pipeline over the student's Church data. Two sentence-aligned
files; this is the corpus every later assignment consumes.

### 6. Automatic and Human MT Evaluation (F2026, 34 slides)
Human evaluation (ranking, adequacy and fluency, MQM), MTEval, then BLEU from the inside:
n-gram precision, the brevity penalty, the geometric mean, a worked example, why raw BLEU is
not comparable, SacreBLEU, chrF. The heaviest deck in the course: no image-only slides, a
derivation, an activity and the largest handout of the first half.
**Notebooks:** `lecture-06-mt-evaluation` (activity): computes BLEU and chrF on the student's
own data and shows BLEU returning zero on ten short sentences; runs in CI.
`lecture-06-mt-evaluation-homework` (homework): the former Part 4, split out Sep 27 because it
uploads the student's A5 corpus and is the assignment's scaffold rather than the session's.
`07-evaluating-translations` (reading): the library's out-of-class treatment of evaluation,
leading to A6 and A8; the in-class notebook keeps the teaching.
**Sets:** A6 (due Mon Sep 28): 500+ sentences through two MT systems, human ranking of 50 in
MTEval before seeing any score, SacreBLEU and chrF over the whole sample, a written analysis of
where metrics and ranking diverge. Extra credit for writing your own BLEU, explicitly without AI.

### 7. Research Paper Reviews; Introduction to Neural Networks (F2026, 43 slides)
An Assignment 6 debrief built around where a metric and a human ranking disagreed; the
data-splitting lesson, taught here first because A8 is where it costs; the paper-review
assignment as "What To Do" and "What Your Presentation Must Cover"; the neural-network
foundations; "The Library You Will Use: TorchLingo"; the paid-Colab requirement; and "What Is
That Loss Number?", which gives ln(V) as the reference point for a first training loss.

The neural-network block, as taught Sep 28 and kept. The AMTA 2018 tutorial's tic-tac-toe
sequence (Munteanu, SDL) teaches neurons, activations, forward pass, cost and gradient descent
on a problem small enough to hold in the head; Eric's judgment after teaching it is that the
intuition it builds is worth keeping and then transferring. The transfer is six course-format
slides inserted where the tutorial reaches the same idea: after its MSE cost slide, "How the
Words Become Numbers", "What Comes Out: A Distribution Over the Vocabulary", "Translation Is
Predicting the Next Token", "Step 2: Compute the Cost" (cross-entropy) and "And That Is Where
ln(V) Comes From"; after its gradient slide, "What a Gradient Is". One correction from the
repository's measurement is on both ln(V) slides: a fresh model starts a little *above* ln(V)
(3.6 against 3.18 on the toy model), not at or below it. The full rebuild that replaced the
tutorial outright (`_v2`, 41 slides: the activation function elaborated, backpropagation one
weight at a time, the spine slide) is in `Archive/` with its comparison PDF and change notes;
it is the source if the block is ever rebuilt again.
**Notebooks:** `lecture-07-toy-model` (activity; on `main` since Sep 28, PR #140, and the
deck's Part A links its Colab badge): a three-part exercise adapted from tutorial 2.
Part A in class: install, train the twelve-phrase toy model, read the loss curve against ln(V),
which the notebook prints beside the vocabulary size and parameter count the slides point at;
then Part A's last step, added by the repository: mount Google Drive, create `MyDrive/CS479`,
and put the two Assignment 5 files there before 8a, so that 8a's Drive mount is not the
student's first. Part B also carries an optional resume demonstration: a training run faked to
drop mid-way and resumed from its checkpoint, the first time a student sees the checkpointer
that A8 depends on for hours.
Part B outside class: hold "The dog sleeps" out, retrain on eleven, and score seen against
unseen; the seen phrases come back 11/11 and the unseen one comes back *El perro corre*, the
verb it saw next to *perro*. BLEU is 0.0 on both, because three-word sentences have no
4-grams, which is Lecture 6's lesson arriving on cue; the notebook reports chrF and exact match
and says why. Part C: the report. Executed end to end on Sep 28, thirty seconds on a CPU, and
by the repository's CI on every pull request since.
`02-train-tiny-model` (reading): the library's own tutorial this was adapted from, for anyone
who wants the plain version.
**Sets:** A7 (due Wed Sep 30, 10:00): the notebook's printed report plus two sentences, on what
the seen/unseen gap says about what the model learned and what they will do differently on A8.
Graded for completion and engagement, not for the numbers.

### 8a. Encoder-Decoder, and Why Attention Was Invented (F2026, 29 slides)
The seam with 8b: *8a is what you need in order to do the assignment; 8b is what the model
actually is.* The toy-model debrief; the encoder-decoder sequence with its two animations; the
fixed-representation bottleneck; degradation with sentence length; the attention build-up;
**"Attention Is Learned Alignment"**, one text slide stating what attention computes, that its
weights form an alignment matrix nobody wrote a model for, and that tutorial 4 measures whether
it is the right one, tying Lecture 5's length-based alignment to 8b's self-attention; then
"RNN with Attention". The practical half opens with why a trained model comes out bad, moved
there on Sep 28 from before the architecture so that it introduces the data lessons rather than
interrupting the architecture: the splitting recap, "Sentence
Length Is a Memory Budget" on measured figures, "Reading Your Loss Curve" (three curve shapes,
the validation set's purpose, label smoothing's floor), "In Class: Start Assignment 8", and the
assignment as What To Do, What To Submit and AI use.
**Notebooks:** `lecture-08a-a8-kickoff` (activity): from the student's A5 corpus to a training
run already going when class ends; cap, dedupe, seeded exact split, a `check_contamination`
that raises, nine files to Drive, the course model, ln(V) beside the first logged loss, training
with `val_loader`, `save_dir` and a Drive checkpointer; batches bucketed by length (1.86x
faster per batch, measured); the first two cells are the four-line install and
`torchlingo.colab.setup(...)`, which prints versions and device, checks the GPU, mounts Drive.
The handout (the A8 slides) is the authority; the notebook quotes its thresholds from one cell.
Coulson's real-corpus Colab run is the remaining end-to-end test (repository task #152).
`04-attention-and-alignment` (reading, Parts 1 to 6): trains the same model with and without
attention and checks whether the attention it learned points at the right words; assigned in
the five-day gap before 8b.
**Sets:** A8 (due Wed Oct 7): an English-to-X model on your own cleaned data; at least 100K
training pairs, 2K validation, 2K test, or all of it if you have less; source-side dedupe and a
verified split; 100-token cap; the course model (d_model 512, 8 heads, 6 + 6 layers,
feed-forward 2048, the 2017 base configuration, 56.4M parameters at an 8K vocabulary); **about
35 epochs, and keep going if validation loss is still falling at the end** (Eric, Sep 28: not
a firm range; the trainer keeps the best checkpoint by validation loss, so running long costs
time, not quality); an A100, L4 or G4 runtime, not a T4; checkpoint to Drive; SacreBLEU over
the whole test set; state your decoding strategy.

### 8b. Neural MT: The Transformer (F2026, 27 slides)
Its own quiz and a **discussion-question placeholder that still needs Eric's question**. A
"Where We Left Off" recap; "Welcome to the Birthplace of the Transformer"; self-attention and
its diagrams; softmax, now a recap of Lecture 7 and placed before it is used; "Attention,
Mechanically: Query, Key, Value" with the scaled dot-product formula as editable text; the
Transformer replaces recurrence; "No Recurrence. So How Does It
Know the Order?" (positional encoding); multi-head attention; "What Is Actually Inside One
Encoder Block"; "The Residual Stream", the 2021 reading of the same diagram, with the pre-norm
caveat and the note that TorchLingo defaults to the paper's post-norm and offers both; **"The
Decoder Block: Where Translation Happens"**, masked self-attention over the target so far (the
mask is teacher forcing mechanically), cross-attention over the encoder (8a's attention,
the only place the two sentences meet), feed-forward, then the softmax over V that produces
Lecture 7's p(y_t | y_<t, x); the full architecture; pros and cons; "What Capacity Buys, and When", the measured 11.7M-against-56M
crossover with its caveats on the slide; an A8 reminder pointing at tutorial 6, with the
Colab clock (two to five hours for 35 epochs on an A100 before bucketing, and not a T4); the
Koehn references.
**Notebooks:** `04-attention-and-alignment` (reading, Part 7, "You have already seen the
Transformer's mechanism"): where this session starts. Kept as one notebook with 8a's parts on
purpose; the transition is the best thing in it.
**Sets:** nothing of its own. A8 is due two days later.

### 9. Handling Morphology and Terminology in NMT (F2026 corrected, 27 slides)
Morphological preprocessing, byte-pair encoding, SentencePiece, and approaches to injecting
terminology. A8 is submitted at 10:00 the morning this runs.
**Corrections applied:** the assignment dates (A8 was showing Mon Oct 6, A9 Wed Oct 8; now Wed
Oct 7 and Wed Oct 14, in titles and body); the decoding rule stated on the A9 slide, where the
assignment is set; and **"Your Model Has a Fixed Vocabulary"**, a bridge slide before Morphology
that gives the lecture its reason: the output is a distribution over a fixed V, every unseen
form is `<unk>`, inflected languages make that worse, subwords are the fix, and the fix changes
which pairs the length cap excludes. **Still last year's, and blocking Oct 7:** the
SentencePiece handout slide (OpenNMT-specific) and the in-class activity slide, which points at
an OpenNMT notebook; both become the subword notebook below at the rebuild, and the A9 slide
becomes the three settings.
**Notebooks:** `lecture-09-subword-tokenization` (activity; merged Sep 28, PR #174): closes
this lecture and starts A9. On a CPU, from the student's A8 split in Drive: counts the test
tokens the word vocabulary cannot represent; trains SentencePiece at 8,000 pieces on the
training split only; shows the pieces; counts `<unk>` again under subwords (zero unless a
character never appears in training); round-trips ten sentences; and prints the three A9
settings for that student's data. Retrains nothing. `06-diagnosing-failures` (reading): five
questions to ask of a model that is not working, each with a planted bug; paired here by
sequence, not topic, because this is the first session at which a student has a trained model
to diagnose. `01-data-and-vocab` (reference): its vocabulary half, word-level only, stopping at
`<unk>` for "Hello universe", the motivating example for A9 rather than coverage of it. Last
year's OpenNMT notebook is in the repository's `notes/legacy-f2025/` as scope.
**When rebuilt:** open with an A8 debrief, the way Lecture 7 opens with an A6 debrief; quote
the notebook's printed settings rather than restating them.
**Sets:** A9 (due Wed Oct 14): the same files and the same split as A8, never re-split, and
three settings changed in the kickoff's Step 6: `use_sentencepiece=True`, `sp_model_path`, and
`max_decode_length` raised to what the notebook measures (pieces run about 1.8x longer than
words; left at 100 it cuts long translations off and the BLEU drop has nothing to do with
subwords). Retrain, rerun the same test set, compare the two BLEU scores and the quality.
Score what `translate_batch` returns, which is already decoded text, never pieces or ids;
decode both runs the same way and say how.

### 10. Overview of MT Quality Estimation (F2026 corrected, 34 slides)
Evaluation versus estimation, uses of QE, traditional QE training data and features, the WMT QE
shared-task metric, QUETCH, COMET and COMET-QE, whether references are needed, how to read a
COMET score.
**Corrections applied:** the A9 reminder's date, and **the decoding block**, three slides after
the assignment reminders: the greedy default nobody chose, beam search with a worked example in
which a locally worse first token wins on total log probability, and the rule that comparisons
must pin decoding, closing back to Lecture 6 and forward into what a quality number can claim.
Decoding was taught nowhere in the course before this.
**Notebooks:** `lecture-10-comet-install` (homework): installs COMET, scores a worked example,
sets up the HuggingFace token through Colab Secrets; A10's own scaffold. `03-inference-and-
beamsearch` (reading): greedy and beam search side by side, checked against the library, both
scored with SacreBLEU; **it cannot yet run from its Colab badge** (it loads the model tutorial
2 saved, which a fresh runtime never has; repository task #166), so the deck names it without
linking it. `05-real-translations` (reading): a trained model's output on real sentences; it
now points at A11 rather than A8, and would help A8 more if read at 8b; where it sits is
Eric's call (Open questions).
**Sets:** A10 (due Wed Oct 14): install COMET and run the notebook end to end with a token; read
two papers for the quiz.

### 11. Neural Quality Estimation and Evaluation Toolkits (F2025, 19 slides)
COMET and COMET-QE in depth, HTER distributions, partial-input baselines, lexical artifacts,
xCOMET, and **LLM-as-judge** (decided Sep 28): this is the one lecture where the course treats
prompting an LLM to score translations as a method in its own right. The F2025 deck carries two
slides on it; the combined May 2025 "Lectures 10, 11" deck carries five (the Google 2023
score-prediction prompt, GEMBA, a GEMBA-DA example, GEMBA against the other QE metrics, and the
system-level vs segment-level caveat), and the rebuild should use the five. Student reviews
set it up: the GEMBA paper ("LLMs are SOTA Evaluators of Translation Quality") and "To Ship or
Not to Ship" on Oct 12, COMET itself on Oct 14; the 2025 data-contamination study comes earlier,
on Oct 5.
**Notebooks:** none of its own; A10's install notebook is the prerequisite.
**Sets:** A11 (due Mon Oct 19): run the default COMET model and COMET-QE-DA on the Lecture 6
outputs; compare against the BLEU, chrF and human-ranking numbers already collected; one page.
Extra credit for xCOMET-XL. *Reaches back to A6 for its data.*

### 12. Using LLMs for MT and Expanding Context Awareness (F2025, 32 slides)
Encoder-only, decoder-only and large language models; MT with decoder-only models; document-
level and context-aware MT; dropped and ambiguous pronouns; contrastive test sets.
**Notebooks:** `lecture-12-llm-context` (homework): A12's scaffold, every code cell a TODO.
**Sets:** A12 (due **Mon Oct 26**): pick a non-MT HuggingFace model, load a low-resource dataset
for **Efik, Kiribati, Palauan, Pohnpeian, Yapese, Kosrean or Kamba**, translate a test set with
0, 5, 10 and 20 in-context examples, chart BLEU against context size. *The deck's own slide
still lists last year's languages (Telugu rather than Kiribati and Kamba); fix it when the deck
is rebuilt.*

### 13. Introduction to Low-Resource NMT Strategies (F2025, 30 slides)
What counts as low-resource, mitigation with and without LLMs, challenges beyond data volume,
evaluation for LRLs, data sources, data augmentation. **Written against OpenNMT**; the
back-translation workflow needs rewriting for TorchLingo before Oct 21. Candidate home for the
NLLB spotlight (Open questions).
**Notebooks:** none yet.
**Sets:** A13 (due Wed Oct 28): back-translation. Train X-to-English on the reversed data, back-
translate at least as many held-out target sentences as the training set, add the synthetic
pairs, retrain English-to-X, compare SacreBLEU and COMET against the original.

### 14. Multilingual NMT and Zero-shot NMT (F2025, 23 slides)
Bilingual versus multilingual, how zero-shot works, tagging, NLLB, complete MNMT, why the
Church data suits cMNMT. **Its handout is "MNMT Guide Using OpenNMT.docx"** and needs replacing
outright before Oct 26. The other candidate home for the NLLB spotlight.
**Notebooks:** none yet.
**Sets:** A14 (due Mon Nov 2): a bidirectional two-language system, English and X, built from
the A8/A9 system, directions randomly intermingled, separate test sets per direction, BLEU and
COMET both ways.

### 15. Overview of Speech-to-Speech MT (F2025, 33 slides)
Spoken language translation, early systems, cascade versus end-to-end, Skype Translator,
Whisper, speech data for projects.
**Sets:** reading only, two papers for the quiz. The free week in the run.

### 16. Automatic Dubbing and Interpretation (F2025, 34 slides)
Neural voices, LINGUA/ToAll, automatic video dubbing, Wav2Lip, HeyGen. The final project
timeline.
**Sets:** A16 (due **Mon Nov 9**): a three-component speech-to-speech pipeline, ASR, MT and TTS,
on a free Azure account, explicitly not the Speech Translation API. Ten spoken sentences in and
out, plus a video of one.

### 17. Multimodal NMT (F2025, 42 slides)
Whether visual context helps, video-guided MT, VaTeX and MAD, architecture, ablations, project
examples. Runs after the proposal presentations; no assignment.

### 18. HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation (F2025, 40 slides)
The BYU interactive translation system, CAT tools, post-editing, normalized edit distance, why
post-editing helps some translators and not others, adaptive MT and Lilt, whether LLMs can do
adaptive MT. In-class activity.

### 19. Word and Sentence Alignment (F2025, 39 slides)
IBM Models worked through in detail, Awesome Align, then sentence alignment. **Needs rescoping:**
the sentence-alignment half moved forward into Lecture 5, and its notebook (tutorial 4) moved
to 8a/8b, correctly, since that notebook is about neural attention alignment and this lecture is
about statistical word alignment. Its word-alignment half stands on its own.

### 20. Overview of Previous MT Paradigms (F2025, 44 slides)
The Vauquois triangle, RBMT, LFG, KANT, EBMT, phrase-based SMT. The history lecture, placed late
so students can see what NMT replaced.

### 21 to 23 (F2025)
Writing research articles with Overleaf and LaTeX, current MATRIX Lab research, MT applications
and jobs, opportunities for further research. Final project presentations close the semester.

---

## Notebook index

Every notebook in the repository, what it is for, and where it lands. Purposes are in the
notebooks' own metadata; the repository's generated map is the machine-checked version of this
table, and the two are kept in agreement by hand at every hand-off.

| notebook | lecture | role | starts | purpose | CI |
|---|---|---|---|---|---|
| `lecture-03-word-embeddings` | 3 | activity | A3 | does the shared multilingual space line up across your languages | needs download |
| `lecture-04-tmx-cleaning` | 4 | activity | A4 | planted TMX problems, two parsers, a checker that names the cleaning step | yes |
| `lecture-04-regex-refresher` | 4 | reference | | fifteen-minute regex tour; a worksheet with blanks | blanks |
| `lecture-05-sentence-alignment` | 5 | activity | A5 | estimate Gale-Church parameters on your language | yes |
| `lecture-06-mt-evaluation` | 6 | activity | A6 | BLEU and chrF on your data; BLEU returns zero on ten short sentences | yes |
| `lecture-06-mt-evaluation-homework` | 6 | homework | A6 | the assignment's scaffold over your uploaded A5 corpus | Colab only |
| `07-evaluating-translations` | 6 | reading | A6, A8 | the library's out-of-class treatment of evaluation | yes |
| `lecture-07-toy-model` | 7 | activity | A7 | the three-part toy-model exercise; hold one phrase out and score seen against unseen; optional resume demo | yes |
| `02-train-tiny-model` | 7 | reading | A8 | the library's own tiny-model tutorial, which the exercise above was adapted from | yes |
| `lecture-08a-a8-kickoff` | 8a | activity | A8 | from your A5 corpus to a running, checkpointed, bucketed training job | Colab only |
| `04-attention-and-alignment` | 8a, 8b | reading | | with and without attention; did it learn the right alignment; Part 7 is the Transformer | yes |
| `lecture-09-subword-tokenization` | 9 | activity | A9 | `<unk>` before and after SentencePiece, a round trip, and the three A9 settings printed | Colab only |
| `06-diagnosing-failures` | 9 | reading | A8 | five questions to ask of a model that is not working | yes |
| `01-data-and-vocab` | 4, 9 | reference | A5 | the library's view of corpus loading; word-level vocab, stops at `<unk>` | yes |
| `lecture-10-comet-install` | 10 | homework | A10 | install COMET, score an example, set up the HF token | needs token |
| `03-inference-and-beamsearch` | 10 | reading | A9 | greedy against beam, checked against the library, both scored; badge not yet runnable (#166) | yes |
| `05-real-translations` | 10 | reading | A11 | a trained model on real sentences; placement open | yes |
| `lecture-12-llm-context` | 12 | homework | A12 | in-context translation with 0 to 20 examples; all TODOs | pip |

Every notebook now opens with the same two cells: a four-line install and
`torchlingo.colab.setup(...)`, which prints versions and device, checks the GPU, mounts Drive
and downloads data. Notebook text carries no weekdays or dates ("before Lecture 8a", never
"Wednesday"), so the notebooks outlive one term's calendar.

**Nothing serves 11, 13, 14, 15, 16, 17, 18, 19, 20.** Lectures 13 and 14 need notebooks
because their assignments train models and are still written against OpenNMT; the rest are
lectures without code.

---

## Tooling the semester depends on

| Tool | First needed | Used for |
|---|---|---|
| Google Colab, paid plan | Lecture 3 | every in-class activity and most assignments; the paid plan is required from 8a, where A8 needs an A100, L4 or G4 (a T4 does not hold the course model at long lengths) |
| Python TMX libraries | Lecture 4 | extracting segment pairs |
| TMX editors (Olifant, Heartsome) | Lecture 4 | inspection only, never in the pipeline |
| grader.exe | Lecture 4 | checking cleaned output. Binaries only, no source, no repository; see Open questions |
| MTEval (mteval.matrix.byu.edu) | Lecture 6 | human ranking; students self-register |
| SacreBLEU, chrF | Lecture 6 | automatic scoring, and again in 8, 9, 13, 14 |
| TorchLingo | Lecture 7 | every model the students train: 7, 8, 9, 13, 14. `pip install torchlingo>=0.2.1` (0.2.3 current): 0.2.1 fixed mid-epoch resume, 0.2.2 added `colab.setup`, 0.2.3 quietened SentencePiece training |
| SentencePiece | Lecture 9 | subword tokenization |
| HuggingFace account | Lecture 10 | COMET model downloads; LLMs in Lecture 12 |
| COMET / COMET-QE / xCOMET | Lecture 10 | neural evaluation, and again in 13 and 14 |
| Azure free tier | Lecture 16 | ASR, MT and TTS for the speech pipeline |

---

## The TorchLingo pivot: what is settled

The course trains models with TorchLingo, an educational PyTorch NMT library built in the
MATRIX Lab. Five assignments train a model: A8, A9, A13, A14, and the toy model at Lecture 7.
They are cumulative in the artifact, so this is one crossing rather than five. **A7, A8 and A9
and their notebooks are on TorchLingo. A13 and A14 are still written against OpenNMT, in
Learning Suite and on the decks, and are due for rewriting by Oct 21 and Oct 26.**

**What a 100K-pair run produces, measured.** At the first A8 configuration (German bitext,
100K/2K/2K, d_model 256, 8 heads, 3+3 layers, 11.7M parameters): 36 epochs gives 11.46 BLEU in
64.7 minutes and is still improving; 65 epochs gives 14.48, converged, in 112.7 minutes. That is
why the handout said 60 to 70 epochs until Sep 28. The benchmark's own recommendation is to
train until validation stops improving, because an epoch count is not corpus-size invariant,
and the students that bites are the low-resource ones. Wall clocks are Apple Metal and do not
transfer to Colab.

**Capacity against data, measured.** The same sweep at 56.4M parameters: at 25K and 50K pairs
the larger model is no better; at 100K it scores 17.79 against 15.95 **and converges in 30
epochs where the smaller needs 65**; at 800K the gap is 6.85 BLEU. So A8's model is leaving
quality on the table at its own floor, and the larger configuration would be both better and
cheaper in epochs. **Decided Sep 28 (Eric): A8 uses the 56.4M configuration.** Memory no
longer argues against it: Coulson measured both models on a Colab A100
(`notes/reports/colab-memory.md`); the largest peak, 16.35 GiB at a 50K word vocabulary and
180-token batches, fits an L4 or A100 and not a T4, and students on paid Colab choose their
GPU. Time per batch at worst-case lengths was 150 to 360 ms, so 35 epochs of 1,563 batches is
two to five hours before bucketing shortens the batches. The epoch count is "about 35, and
keep going if validation loss is still falling": the 30-epoch convergence was measured with a
subword vocabulary, A8 uses words (about 109M parameters at the real corpus's vocabularies),
and the trainer keeps the best checkpoint, so running long is safe and stopping early is the
risk.

**The 100-token cap.** On the German corpus at 100K pairs, batch 64: 9.60 GiB held with the cap
against 35.80 GiB without, identical wall clock, 1.30% of pairs dropped. Peak memory is set by
the longest batch, not the median one; growth is not quadratic in the cap. The cap is applied
*before* splitting so the split sizes are the sizes submitted.

**A8's floor, for low-resource languages.** At least 100K training pairs; if your cleaned data
has less, use all of it and say so, mirroring A4 and A5.

**A9 is a controlled comparison.** The tokenizer must be the only thing that differs, so A9
trains and scores on A8's raw files and split unchanged; the three settings above are the whole
change, and decoding is held constant and stated. (v6 said the sentence set should be chosen
once with the subword tokenizer; that was withdrawn on Sep 28 because re-split text fed through
the word vocabulary makes the model output pieces, and BLEU on pieces is not comparable with
A8's.)

**Post-norm.** `SimpleTransformer` is the 2017 paper's architecture, post-norm by default,
pinned by a test whose failure message names the two 8b slides that depend on it. Pre-norm is
available with `norm_first=True`.

**Notebooks live in the repository**, `torchlingo/docs/docs/course/`, each with a Colab badge
off `main`, metadata declaring which lecture it serves and what it starts, and CI that executes
every notebook it can. The two sessions exchange files under `notes/handoff/`, one file per
hand-off; course-side items are tracked in `CS479-TASKS.md`, repository items in the
repository's `notes/TASKS.md`.

---

## Open questions

- **Does "about 35 epochs" hold for a student with 30K pairs?** The 30-epoch convergence was
  measured at 100K pairs with a subword vocabulary. Revisit once the A5 audit says how many
  students are well below 100K; the audit itself still needs the submissions staged somewhere
  readable. The instruction's escape hatch is "keep going while validation loss falls".
- **Has Coulson's real-corpus Colab run of the A8 kickoff happened?** Repository task #152. It
  is the only end-to-end test before 8a; CI cannot mount Drive or draw a GPU.
- **Does the grader survive?** `grader.exe` is four unsigned PyInstaller binaries in a PhD
  student's OneDrive with no source, no repository, no license. Ammon has been asked for the
  source. Fallback: the checks are documented in Lecture 4 and belong in `torchlingo.diagnostics`.
- **An NLLB spotlight lecture** (Eric, Sep 27): what makes NLLB special beyond its data, namely
  the sparsely gated mixture-of-experts architecture with its regularisation against low-resource
  pairs overfitting, and the deliberate training curriculum, which has no coverage in the course.
  Home is Lecture 13 or 14. The repository session will verify the paper's specifics before any
  number reaches a slide.
- **Tutorial 5's placement.** Read at Lecture 10 and now pointing at A11; it would help A8
  more if read at 8b. Eric's call, and a deck change if it moves.
- **Learning Suite's assignment texts.** A8 (tomorrow), A9, A13 and A14 still carry last year's
  OpenNMT directions; the current ones are in `notes/assignments/`. A7's due time differs
  between Learning Suite and the deck.
- **Lecture 9's load.** Its own subject plus an A8 debrief, diagnostics reading and a subword
  activity. The decoding block already moved out for this reason. Watch it.
- **Lecture 6 is overloaded** and nothing has been done about it.
- **Lecture 19 needs rescoping** around the material now in Lecture 5, and has no notebook.
- **Deck links to personal Google Drive URLs.** Lecture 4 links the regex notebook twice and its
  own activity once; Lecture 12 links its assignment notebook. All die with the account. Now
  fixable as badges, but the Drive copies must stay alive while students are in them.

---

## Decisions log

Newest first. Each is a decision that changed the schedule, an assignment, or a notebook's
place, with who made it.

- **Sep 29.** Lecture 7 taught from the SDL tic-tac-toe block with six transfer slides; the
  full rebuild archived (Eric). Every assignment description transcribed from Learning Suite
  into the repository's `notes/assignments/` (Eric asked; course-side). The repository's copy of
  this file is now regenerated header and all by the course-side refresh script.
- **Sep 28, evening.** A8's model is the 56.4M configuration, on Coulson's Colab memory
  measurement (Eric). Epochs: "about 35, and keep going if validation loss is still falling",
  not a range (Eric). A9 keeps A8's raw files and split and changes three settings
  (`use_sentencepiece`, `sp_model_path`, `max_decode_length`); the Lecture 9 notebook prints
  them (repository). Notebooks carry no weekdays or dates (Eric). Tutorial 5 points at A11, not
  A8 (repository; placement still Eric's). Tutorial 2 stays `reading` with a note (repository).
- **Sep 28, afternoon.** LLM-as-judge is Lecture 11's, using the five GEMBA slides at the
  rebuild (Eric). The paper sign-up sheet gains URL and LECTURE TOPIC ALIGNMENT columns, three
  papers (the contamination study, COMET, Koehn 2003) to reach 23, dates in lecture order with
  the first on Oct 5 (Eric). The Lecture 7 to 9 run given its spine (Eric, all agreed): the
  next-token sentence at Lecture 7, attention as learned alignment at 8a, the decoder block at
  8b, the vocabulary problem at 9; "why a trained model comes out bad" moved to open 8a's
  practical half; softmax placed before it is used in 8b.
- **Sep 28, midday.** Lecture 7's activity becomes a three-part exercise with a turn-in, A7, due
  before 8a (Eric). That makes it a course notebook, so tutorial 2 returns to `reading` and the
  "one exception" of the same morning is withdrawn: the rule that tutorials are out-of-class
  holds with no exceptions.
- **Sep 28, morning.** A12 due Mon Oct 26 and A16 due Mon Nov 9, moved from Oct 21 and Nov 4,
  after an audit found the schedule shift had left them two-day windows (Eric). Tutorial 2 is
  Lecture 7's in-class activity, the one exception to the rule that tutorials are out-of-class
  (Eric; withdrawn by midday). A12's language list is the notebook's, not the old deck's (Eric).
  The A8 kickoff notebook written, course-side, with the repository session's eleven findings
  built in. Post-norm confirmed; `norm_first` added as an option (repository).
- **Sep 27.** Lecture 8 split into 8a and 8b; Lecture 17 takes Nov 16; Lectures 9 to 16 shift
  one session; A8 holds at Oct 7 (Eric). TAUS report dropped, a 2025 event (Eric). A8 raised
  from 30-36 to 60-70 epochs on the benchmark (Eric). One quiz each for 8a and 8b (Eric).
  Tutorial 4 moved from Lecture 19 to 8a/8b, and stays one notebook (Eric; repository).
  Tutorial 6 moved from Lecture 8 to 9 (Eric). Beam search taught at Lecture 10, tutorial 3
  moved there from 22 (Eric). OpenNMT removed from all F2026 decks (Eric). The residual-stream
  reading added to 8b (Eric). Three Fall 2025 notebooks brought into the repository. The A8
  handout stays complete and is the authority over its kickoff notebook (Eric, via repository
  task #155).
- **Sep 26.** Lectures 7 and 8 rebuilt. TorchLingo 0.2.0 on PyPI; Colab resume verified; the
  units question settled as epochs.
- **Sep 25.** Dates for Lectures 1 to 8 confirmed against Learning Suite.
- **Sep 21 to 23.** Lectures 5 and 6 rebuilt; Gale-Church moved forward into Lecture 5.

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
| 4 | Data Preparation for MT Training | `lecture-04-regex-refresher` (reference) [5], `lecture-04-tmx-cleaning` (activity) | `01-data-and-vocab` (reference) [1] |
| 5 | Data Preparation for MT Training, Part 2 | `lecture-05-sentence-alignment` (activity) | — |
| 6 | Automatic and Human MT Evaluation | `lecture-06-mt-evaluation-homework` (homework) [6], `lecture-06-mt-evaluation` (activity) [7] | `07-evaluating-translations` (reading) [4] |
| 7 | Research Paper Reviews; Intro to Neural Networks | `lecture-07-toy-model` (activity) [8] | `02-train-tiny-model` (reading) [2] |
| 8a | Neural MT: Encoder-Decoder, and Why Attention Was Invented | `lecture-08a-a8-kickoff` (activity) [9] | `04-attention-and-alignment` (reading) |
| 8b | Neural MT: The Transformer | — | `04-attention-and-alignment` (reading) |
| 9 | Handling Morphology and Terminology in NMT | `lecture-09-subword-tokenization` (activity) [10] | `01-data-and-vocab` (reference) [1], `06-diagnosing-failures` (reading) |
| 10 | Overview of MT Quality Estimation | `lecture-10-comet-install` (homework) [11] | `03-inference-and-beamsearch` (reading), `05-real-translations` (reading) [3] |
| 11 | Neural Quality Estimation and Evaluation | — | — |
| 12 | Using LLMs for MT; Expanding Context Awareness | `lecture-12-llm-context` (homework) [12] | — |
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

- [1] **`01-data-and-vocab`** — covers loading and cleaning a parallel corpus, which is
  Lecture 4's subject from the library side, and its vocabulary half belongs to Lecture 9.
  Lecture 4 has already run this year, so the pairing is retrospective there and genuine
  for a future offering.
- [2] **`02-train-tiny-model`** — The library's own tutorial. Lecture 7's in-class
  exercise, lecture-07-toy-model, was adapted from it.
- [3] **`05-real-translations`** — Read at Lecture 10, so it no longer claims A8, which is
  due at Lecture 9. Its substance is evaluating and decoding a real model -- held-out
  BLEU, greedy against beam search -- which is what A11's metric work builds on. It would
  help A8 more if read at 8b; where it sits is Eric's call.
- [4] **`07-evaluating-translations`** — The out-of-class treatment of evaluation.
  lecture-06-mt-evaluation is the in-class activity and owns the teaching; this one makes
  the student choose between two systems (Cowork's Q7 call).
- [5] **`lecture-04-regex-refresher`** — Fall 2025 material, brought into the repository
  on 2026-09-27. A fifteen-minute tour of Python regular expressions, which is what the
  sixteen cleaning steps are written in. Lecture 4's deck links it twice and Learning
  Suite posts it on the Lecture 3 tab as well, so it is offered ahead of Lecture 4 rather
  than used inside it.
- [6] **`lecture-06-mt-evaluation-homework`** — Part 4 of the original Lecture 6 notebook,
  split out on 2026-09-27. Uploads the student's own Assignment 5 corpus through the Colab
  file picker, so it cannot run outside Colab.
- [7] **`lecture-06-mt-evaluation`** — Parts 1 to 3 of the original Lecture 6 notebook.
  Part 4 moved to lecture-06-mt-evaluation-homework on 2026-09-27, because `role` holds
  one value and this file was an activity with homework inside it. Needs nothing and no
  Colab runtime, so it executes in CI.
- [8] **`lecture-07-toy-model`** — Lecture 7's in-class activity, written 2026-09-28 as a
  three-part exercise: Part A in class (install, train the toy model, read the loss curve
  against ln(V)), Part B outside class (hold one phrase out, retrain, score seen against
  unseen with SacreBLEU), Part C a short report handed in as A7 before Lecture 8a. Adapted
  from tutorial 2, which stays the library's own tutorial. Trains a 64-dimensional model
  in seconds, so it runs in CI.
- [9] **`lecture-08a-a8-kickoff`** — The in-class start of Assignment 8, written
  2026-09-28 for Lecture 8a. Takes a student from their Assignment 5 corpus to a training
  run that is already going when class ends. The handout is the authority; this is the
  executable path through it, and every threshold is quoted from the handout in one cell.
  Cannot run in CI: it mounts Drive and needs a GPU.
- [10] **`lecture-09-subword-tokenization`** — Lecture 9's in-class activity, written
  2026-09-28 to the scope Cowork set: SentencePiece fit on the student's A8 training
  split, word-level against subword unknown counts on their test set, a round trip, and
  the three settings A9 needs (use_sentencepiece, sp_model_path, and a max_decode_length
  measured on their data). Retrains nothing. Reads the A8 split from Drive, so it cannot
  run in CI.
- [11] **`lecture-10-comet-install`** — Fall 2025 material, brought into the repository on
  2026-09-26. Installs unbabel-comet, scores a worked example, and sets up the HuggingFace
  token through Colab Secrets, which the reference-free models require. It is Lecture 10's
  assignment and the setup for Lecture 11's. Framework-independent: nothing in it touched
  OpenNMT.
- [12] **`lecture-12-llm-context`** — Fall 2025 material, brought into the repository on
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

**Two notebooks declare no `leads_to`, deliberately**, and it is recorded here so nobody reads the
blank as an oversight and fills it in. Settled 2026-09-28, closing #145.

- **`04-attention-and-alignment`** explains attention and the Transformer. It gives a student no
  head start on A8's deliverable, which is a trained model on their own corpus — it makes the
  training make sense, which is a different thing.
- **`lecture-04-regex-refresher`** teaches regular expressions, which A4's cleaning is *written
  in*. A prerequisite skill is not a head start either.

The rule both follow is the one #139 established against tutorial 1's Lecture 9 claim: **a notebook
that teaches the prerequisite should not be recorded as covering the thing.** A visible blank is
better than a flattering map. Every other notebook now names an assignment.

One row appears twice on purpose. **Tutorial 4 serves Lectures 8a and 8b and will not be
split**, which reverses #140 **and an agreement with Cowork**.

That second half was stated wrongly here first, and the correction is the useful part. The claim
was that nothing sent to Cowork needed reversing, on the grounds that the 2026-09-27 hand-off
records this notebook as `["8a", "8b"]`, reading. True of that file, and irrelevant: the split was
Question 8 of the *previous* entry, and their reply agreed to it in terms — *"Split tutorial 4.
Parts 6 to 8 are architecture content and they belong to 8b."* Checking only the newest hand-off
missed an agreement that was two files back, in the archive.

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
