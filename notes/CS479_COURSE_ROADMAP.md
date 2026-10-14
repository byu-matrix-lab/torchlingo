# CS 479 Fall 2026: Course Roadmap

*Generated from `notes/roadmap/` by `scripts/render_roadmap.py`: edit there, never here.*

---

Introduction to Machine Translation, BYU. Monday and Wednesday, 11:00 to 12:15.

**What this is.** The forward map of the semester: what each lecture covers, which notebooks
serve it and what each one is for, what assignment it sets and when that is due. It describes
the course as it is now. What changed, when and on whose decision is in the Decisions log at the
end, and only there.

**What is authoritative.** Learning Suite and the decks are the source of truth for what
students see; this file is the distilled reference the two working sessions share. Where the
two disagree, this file says so and names the stale one.

**Deck status, three states.** *F2026* is a rebuilt deck. *F2026 corrected* is last year's deck
with dated fixes applied and a note saying which; it is not a rebuild. *F2025* is last year's
deck as it stands.

**Dates and assignment texts.** The lecture rows, assignment due dates and quiz names match
Learning Suite. The assignment descriptions are transcribed in the private repository at
`torchlingo-private/notes/assignments/` (they never reach the public tree), each with a status
line. Two Learning Suite texts are still last year's, written for OpenNMT: A13 and A14. Each is
rewritten against its notebook.

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
| 8c | Wed Oct 7 | Finishing the Transformer, the A8 Debrief, and the Subword Activity | **A8** create and run an NMT model | F2026 |
| 9 | Mon Oct 12 | Handling Morphology and Terminology in NMT | | F2026 |
| 10 | Wed Oct 14 | Overview of MT Quality Estimation | **A9** SentencePiece | F2026 corrected |
| 11 | Mon Oct 19 | Neural Quality Estimation and Evaluation | **A10** install COMET | F2025 |
| 12 | Wed Oct 21 | Using LLMs for MT; Expanding Context Awareness | | F2025 |
| 13 | Mon Oct 26 | Strategies for NMT of Low-Resource Languages | **A11** run COMET on the A6 sentences | F2025 |
| 14 | Wed Oct 28 | Multilingual NMT and "Zero-shot" NMT | **A12** context and LRL translation | F2025 |
| 15 | Mon Nov 2 | Overview of Speech-to-Speech MT | **A13** back-translated data | F2025 |
| 16 | Wed Nov 4 | Automatic Dubbing and Interpretation | **A14** bidirectional MNMT | F2025 |
| — | Mon Nov 9 | Proposal week: one day of outline reviews with the professor, two days of proposal presentations (Eric and Coulson, Sep 30); which day is which is being set in Learning Suite | | |
| — | Wed Nov 11 | (proposal week) | **A15** SLT pipeline | |
| — | Mon Nov 16 | (proposal week) | **Final project proposal** (date may move with the presentation days) | |
| 17 | Wed Nov 18 | Multimodal NMT | | F2025 |
| 18 | Mon Nov 23 | HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation | | F2025 |
| — | Wed Nov 25 | **No class**, Thanksgiving Eve | | |
| 19, 20 | Mon Nov 30 | Word Alignment, and What We Can Learn from Previous MT Methods: RBMT, KBMT, EBMT, PBSMT, SBSMT | | F2025, merged at rebuild |
| — | Wed Dec 2 | Project checkpoint reviews with instructor | | |
| 21 | Mon Dec 7 | Writing Research Articles; MATRIX Lab research | | F2025 |
| 22, 23 | Wed Dec 9 | MT Applications and Jobs; Opportunities for Further MT Research | **Paper review assignment** | F2025 |
| — | Thu Dec 10 | Last day of class | Final project **presentations, submission, write-up** | |
| — | Wed Dec 16 | Final exam, 11:00 to 14:00 | | |

An assignment appears on the row of the day it is **due**, which is how Learning Suite lists
it, at 10:00 unless said otherwise (Learning Suite has A7 at 11:59 pm; the deck says 10:00,
and the two should agree). A7 is new this year: a short report from Lecture 7's toy-model
exercise. The SLT pipeline is A15, set at Lecture 15; Lecture 16 sets nothing, and the
numbering runs 1 to 15 without a gap.
Lectures 17 to 20 carry no assignments; the final project has taken over. Lecture quizzes run
through Lecture 20, and 8a and 8b each have their own, one more than last year.

**Paper reviews** run alongside: 24 papers on the sign-up sheet (Google Sheet "CS 479 Fall
2026", tab "Reading Report"), presented Oct 5 through Nov 30 at one or two per session, each
placed next to the lecture its topic belongs with, and the sheet's LECTURE TOPIC ALIGNMENT
column says which. From Oct 12 on, that column is one lecture off from the schedule (the QE
papers of Oct 12 land on Lecture 9, COMET on Lecture 10); the unassigned Och and Ney slot on
Nov 23 belongs on Nov 30, the merged alignment session. Dec 7 and Dec 9 are open slots if a
student has to move.

---

## How the second half was scheduled

Lecture 8 is three sessions: 8a on Wed Sep 30, 8b on Mon Oct 5 and 8c on Wed Oct 7. The first
extra session (Sep 27) was Mon Nov 16, which last year held a report on the TAUS conference, a
2025 event that does not recur; the second (Oct 7, when 8b's unreached slides and the A8 debrief
filled the Oct 7 session) comes from merging Lectures 19 and 20 on Mon Nov 30: Lecture 19 had
already lost its sentence-alignment half to Lecture 5, and its word-alignment half, the IBM
models, is the natural opening of the previous-paradigms lecture. Lectures 9 to 18 each sit two
sessions later than last year's calendar. Lecture *numbers* did not change, so every notebook's
declared lecture stays valid; the Lecture 9 notebook's Part A ran in 8c.

**A8 did not move.** It stays Wed Oct 7, which is now Lecture 8c's day. Its problem was never
calendar; it was that nobody had measured what a 100K-pair run produces or what a Colab GPU
holds, and both are measured now (see the pivot section).

**Assignment windows, measured from the lecture that sets each one**, not from an old due date:
the Monday / Wednesday parity flip means one session later is sometimes five days and sometimes
two.

| assignment | set at | due | window |
|---|---|---|---|
| A7 | 7, Mon Sep 28 | Wed Sep 30 | 2, deliberately: three numbers and two sentences from an exercise started in class |
| A8 | 8a, Wed Sep 30 | Wed Oct 7 | 7 days |
| A9 | 8c, Wed Oct 7 | **Wed Oct 14** | 7; Eric, Oct 7: one more session, since the subword slides are taught Monday. Part A ran in class on Oct 7, Part B is live |
| A10 | 10, Wed Oct 14 | Mon Oct 19 | 5: an install, and A11 needs it done |
| A11 | 11, Mon Oct 19 | **Mon Oct 26** | 7; one session would give 2, too short for running COMET on A6's data |
| A12 | 12, Wed Oct 21 | Wed Oct 28 | 7 |
| A13 | 13, Mon Oct 26 | Mon Nov 2 | 7 |
| A14 | 14, Wed Oct 28 | Wed Nov 4 | 7 |
| A15 | 15, Mon Nov 2 | Wed Nov 11 | 9; the pipeline belongs to the speech-to-speech lecture, and the Azure account needs the extra days |

Costs accepted knowingly: Lecture 17 sits two weeks after Lecture 16 and after the proposal
presentations, the least damaging place for a gap since Multimodal NMT is the most
self-contained lecture in the run; there is no catch-up session if the proposal presentations
overrun Nov 16; and the merged Nov 30 session has to be built from two 40-slide decks.

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

## Learning outcomes, and who does the work

The notebooks have become good enough to run themselves, and that is the risk: a notebook that
does every step for the student can be completed on auto-pilot, and then the assignment it
starts is completed the same way. This section states what each lecture and notebook is for,
in terms of what the student can do afterwards, and then says for each notebook who does the
work: what the student must write, decide or predict, and what runs for them. The test
throughout is the one the audit set for the notebooks' text: *the job of each notebook is to
teach.* A notebook teaches when the student has to do something the output depends on.

### Lecture outcomes

What a student who did the work can do afterwards, and what shows it. Two or three per
lecture; the scope notes below carry the content, this carries the point.

| # | Lecture | A student who did the work can | Shown by |
|---|---|---|---|
| 1 | Course overview, history | Place Weaver, Georgetown, ALPAC, SMT and NMT in order and say what each promised and delivered; state the course's rules on quizzes, late work and AI. | A1, Quiz 1 |
| 2 | Translation challenges | Name the kinds of difficulty (lexical and structural ambiguity, word order, morphology, reference, world knowledge) and rank them for their own language with real examples. | A2 |
| 3 | Word embeddings | Explain the distributional idea behind word2vec, GloVe and contextual embeddings; test whether a multilingual space lines up for their language pair and read a similarity matrix row by row. | A3 heat map and analysis |
| 4 | Data preparation | Read a TMX file through `repr`, find a planted problem with code rather than by eye, write a cleaning step as one function, and say which of the 16 steps fixes which problem. | A4: extraction plus three steps |
| 5 | Data preparation, part 2 | Run a full cleaning pipeline to a sentence-aligned corpus; explain Gale-Church (length correlation, link types, the cost model, c and s²) and where length-based alignment fails. | A5, the corpus everything else consumes |
| 6 | Evaluation | Rank translations by hand before seeing a score; compute BLEU from its parts and explain a zero; use chrF; say where a metric and a human disagree and what the metric was rewarding. | A6 |
| 7 | Paper reviews; neural networks | Review a paper to the course template; explain neuron, activation, forward pass, loss and gradient; read a loss curve against ln V; tell memorisation from translation with a held-out item. | A7 report; paper reviews |
| 8a | Encoder-decoder and attention | Describe the encoder-decoder and its bottleneck; explain attention as a learned soft alignment and read an alignment matrix; split and verify a corpus and say why the dedupe is source-side; read training against validation loss; start, checkpoint and resume a long run. | A8 |
| 8b | The Transformer | Describe self-attention (query, key, value, scaled dot product, softmax) and compute one step by hand; say why there are three attentions and what each is for; read a tensor shape. | Quiz 8b |
| 9 | Morphology, subwords, terminology | Describe multi-head attention, the encoder block and residual stream, and the decoder's masked self-attention and cross-attention; say how order gets into a model with no recurrence; place their own A8 BLEU on the measured capacity curve and say what capacity buys and when; explain the fixed-vocabulary problem; measure `<unk>` before and after SentencePiece; retrain with one variable changed and attribute the BLEU difference to it. | A9 comparison |
| 10 | Quality estimation | Distinguish evaluation from estimation; say what COMET and COMET-QE score and what the number means; explain greedy against beam search and why a comparison must pin decoding. | A10; Quiz 10 |
| 11 | Neural QE and toolkits | Run COMET and COMET-QE on their own A6 data; reconcile them with BLEU, chrF and their human ranking; state what LLM-as-judge scores and its system- against segment-level caveat. | A11 |
| 12 | LLMs for MT; context | Prompt a decoder-only model to translate with in-context examples; measure what context buys and where it stops; name the document-level problems (pronouns, consistency). | A12 chart and analysis |
| 13 | Low-resource strategies | Define low-resource beyond pair counts; execute back-translation end to end and measure whether it helped. | A13 |
| 14 | Multilingual and zero-shot | Explain tagging and why zero-shot works at all; build a bidirectional system from their A9 one and score both directions. | A14 |
| 15 | Speech-to-speech MT | Contrast cascade and end-to-end systems and say what each loses; build an ASR to MT to TTS pipeline and say where the errors compound. | A15 |
| 16 | Dubbing and interpretation | Say what dubbing adds to translation (timing, voice, lip sync) and where current systems fail. | Quiz 16 |
| 17 | Multimodal NMT | Say when visual context helps translation and how to test the claim. | Quiz 17 |
| 18 | HAMT and MAHT | Measure post-editing with edit distance; explain why post-editing helps some translators and not others; describe adaptive MT. | In-class activity; Quiz 18 |
| 19 | Word alignment | Work IBM Model 1 by hand on a toy pair; explain EM; relate statistical alignment to the attention alignment of 8a. | Quiz 19 |
| 20 | Previous paradigms | Place RBMT, EBMT and PBSMT on the Vauquois triangle and say what NMT replaced and what it kept. | Quiz 20 |
| 21 to 23 | Writing, research, careers | Structure a research article; present a project. | Final project |

### Notebook outcomes, and who does the work

For every notebook: the lesson, what the student must do by hand, what runs for them, and a
rating. **Low** means the result depends on something the student wrote, chose or predicted.
**Watch** means the notebook runs itself and the learning rests on the questions around it.
**High** means a student can produce the assignment's deliverable without understanding the
steps.

| notebook | the lesson | the student must | the notebook does | rating |
|---|---|---|---|---|
| `lecture-03-word-embeddings` | a shared multilingual space can be tested, not assumed | pick five words, read the matrix row by row, answer four questions; A3 adds the heat-map code, which is deliberately not given | embed and print the matrix | low |
| `lecture-04-regex-refresher` | read the patterns the cleaning steps are written in | fill seven blanks | check each against an expected output | low |
| `lecture-04-tmx-cleaning` | find data problems with code, through `repr` | predict two counts, write the line-break regex, write two cleaners, defend a keep-or-drop call | parse, write files, run the checker | low. The model for the course |
| `lecture-05-sentence-alignment` | length can align sentences and cannot see a deletion | change the priors and watch it fail, upload their A4 files, report c | aligner, scoring, the c and s² formulas | low |
| `lecture-06-mt-evaluation` | what BLEU and chrF count | predict before each of three runs, three talk-it-over answers | the scoring | low |
| `lecture-06-mt-evaluation-homework` | the metrics on their own data | upload, sample, translate ten by hand, rank before re-reading the scores | three scoring calls | low; A6's substance is MTEval and the analysis |
| `02-evaluating-translations` (reading) | which metric to believe when they disagree | decide before scoring; four "Your turn" exercises | the examples | low |
| `lecture-07-toy-model` | a model can score perfectly on what it memorised and know nothing | choose the held-out phrase, predict, run Parts A and B, report back in class, write A7's two sentences; a contrast phrase runs as well | corpus, model, training, both scorings, the resume demonstration, the report printout | low |
| `03-train-tiny-model` (reading) | a saved model is weights without a vocabulary, so the vocabulary must be rebuilt exactly | read; run the three rebuilds | everything | low, as reading |
| `lecture-08a-a8-kickoff` | from cleaned corpus to a running, checkpointed model | type two file names; raise `EPOCHS` if validation loss is still falling | cap, dedupe, split, verification, nine files, the configuration, the training call, the resume, the scoring | **high** |
| `04-attention-and-alignment` (reading) | attention is measurable alignment | read Parts 1 to 5 and 7; run them | the ablation and the alignment accuracy | low, as reading |
| `lecture-08b-cross-attention` | attention can find the right words and the decoder still write the wrong ones | write `attention`, predict where three output pieces will look, say which prediction missed | the model, the capture of each head's inputs and weights, the map | low |
| `lecture-08c-attention-heads` | each head is sharp and the average is not; deeper layers spread out | write `split_heads` and `sharpness`, predict before the measurement, say which part of the prediction was wrong | the model, the attention step, the capture, the maps | low |
| `lecture-09-subword-tokenization` | a fixed vocabulary cannot say unseen words; subwords can | write the `<unk>` counter, pick the vocabulary size with a reason, predict A9's BLEU, report back, and reconcile the prediction in the A9 write-up | the SentencePiece training, the round trip, saving the settings to Drive, and in Part B the retraining and both scorings | low |
| `06-diagnosing-failures` (reading) | a procedure for a model that is not working | read once, then look symptoms up | five planted bugs and their checks | low; it is a lookup table |
| `01-data-and-vocab` (reference) | the library's view of a corpus | read | everything | low, as reference |
| `lecture-10-comet-install` | a learned metric, with and without a reference, and the records it scores | write comet_inputs (checked on known cases), predict which system each model prefers, set up the token | the install, both models' scoring, the summary A10 grades | low |
| `lecture-11-comet-on-a6` | whether metrics agree with each other and with a human, sentence by sentence | reuse comet_inputs, write pairwise_agreement (checked on a known case), predict, and explain the agreement in the summary | loading, all four metrics on every sentence, the report | low |
| `08-inference-and-beamsearch` (reading) | decoding is part of the system being measured | read | two decoders written to be read | low, as reading |
| `07-real-translations` (reading) | what an undertrained model looks like on unseen text | read; one cell trains a small model | everything | low, as reading |
| `lecture-12-llm-context` | in-context examples and what they buy | write split_pool_and_test and build_prompt (the method, each checked for sizes, reproducibility and contamination, and for the prompt's required contents, its wording their own); predict the best context size; reconcile the prediction with the curve | the LLM, the loop, BLEU, the chart and the upload files | low |
| `lecture-13-back-translation` | synthetic pairs whose target side is real, and whether they help | find monolingual text new to their split; write reverse and combine (the method, checked against known answers and the contamination check); predict before retraining; add Google back-translations to three examples and explain the result | both trainings, the back-translation, BLEU, COMET and the upload files | low |

### Where the line is crossed

**The rule for every assignment notebook** (Eric, Oct 13): the student writes the steps that are
the method, as short functions that raise `NotImplementedError` until written; each is checked on
a case whose answer is known and then by the library's own checks on the real data
(`check_contamination` where a split or a training set is built), so the check tells the student
when they have it right. The notebook supplies what is not the lesson: installs, loading,
training calls, decoding, scoring. Before each long run the student writes a prediction, and the
write-up asks them to reconcile it with the result. The test throughout: at least one thing the
student had to write or choose, and at least one sentence that cannot be written without having
read the output.

Where each assignment stands against it:

| assignment | the student writes | checked by |
|---|---|---|
| A7 | the held-out phrase, a prediction, the contrast | the report, which shows a missing prediction |
| A9 | `unknown_rate`; `VOCAB_SIZE` with a reason; a BLEU prediction | a known-answer case; Part B reloads the saved settings |
| A10 | `comet_inputs`; a prediction | known cases, including lists of different lengths |
| A11 | `comet_inputs` again; `pairwise_agreement`; a prediction | a known case; your files' alignment |
| A12 | `split_pool_and_test`, `build_prompt`; a prediction | known cases; `check_contamination` on the split |
| A13 | `reverse`, `combine`; a prediction | known cases; `check_contamination` on both held-out sets |
| A14 | to be built to the rule: tagging the directions and intermingling them | |

**The one place still open: the A8 kickoff does the part the assignment says is the student's.**
The A8 slide on AI use says *yours: the splitting, the verification, the training decisions, and
the judgment about what is limiting your output.* The kickoff performs the cap, the dedupe, the
split and the verification, and the student edits two file names. That was deliberate, since the
run has to be going when class ends, and A8 has run for this term. For the next offering, the
rule above applies: the cap and the source-side dedupe become the student's functions, with the
contamination check as the oracle; the split, the model and the training call stay as given;
and a prediction of ln V and the first logged loss goes in before training.

---

## Lecture scope notes

What each session covers, the notebooks that serve it and what it sets. The decks hold the
slides; this holds the scope. A notebook's *role* is how it is used: an **activity** is run in
the session, **reading** is assigned alongside it, **homework** is the assignment's own
notebook, **reference** is offered rather than urged.

### 1. Course Overview and History of MT (F2026, 46 slides)
Weaver's 1947 letter, IBM-Georgetown, ALPAC, the evolution to NMT, hype versus reality, MT at
BYU back to Eldon Lytle in 1976. The course's rules: quizzes, assignments, AI use, grades, the
late policy and the Early Policy.
**Sets:** A1, read the syllabus and the MT history.

### 2. Translation Challenges for MT (F2026, 46 slides)
Why FAHQT is the wrong question. Ambiguity, false friends, long-distance dependencies, reference,
word order, world knowledge, morphology. Students pick their language here.
**Sets:** A2, rank the translation challenges for your language.

### 3. Introduction to Word Embeddings (F2026, 51 slides)
The distributional principle, word2vec, LSA, PMI, GloVe, contextual and multilingual embeddings.
**Notebooks:** `lecture-03-word-embeddings` (activity): does the shared multilingual space line
up across the student's languages; it produces the heat map A3 asks for.
**Sets:** A3, the multilingual embedding space.

### 4. Data Preparation for MT Training (F2026, 41 slides)
Translation memories, TMX, language codes, the 16 cleaning steps, TMX tools.
**Notebooks:** `lecture-04-tmx-cleaning` (activity): planted TMX problems, two parsers, and a
checker that names the cleaning step behind each finding; starts A4. `lecture-04-regex-refresher`
(reference): the regular expressions the cleaning steps are written in. `01-data-and-vocab`
(reference): the library's view of loading a parallel corpus.
**Sets:** A4, extract segment pairs, fix what breaks alignment, write at least three cleaning
steps, prepare 200K pairs.

### 5. Data Preparation, Part 2 (F2026, 31 slides)
The full pipeline, and Gale-Church sentence alignment: length correlation, link types, the cost
model, and where length-based alignment fails.
**Notebooks:** `lecture-05-sentence-alignment` (activity): estimates Gale and Church's parameters
on the student's language; starts A5.
**Sets:** A5, the complete pipeline over the student's Church data: two sentence-aligned files,
the corpus every later assignment consumes.

### 6. Automatic and Human MT Evaluation (F2026, 34 slides)
Human evaluation (ranking, adequacy and fluency, MQM), MTEval, BLEU from the inside, SacreBLEU,
chrF. The heaviest deck of the first half.
**Notebooks:** `lecture-06-mt-evaluation` (activity): BLEU and chrF, and BLEU's zero on short
sentences. `lecture-06-mt-evaluation-homework` (homework): the metrics on the student's own A5
data. `02-evaluating-translations` (reading): which metric to believe when they disagree.
**Sets:** A6: 500+ sentences through two MT systems, a human ranking of 50 in MTEval before any
score, SacreBLEU and chrF, and an analysis of where metrics and ranking diverge.

### 7. Research Paper Reviews; Introduction to Neural Networks (F2026, 43 slides)
An A6 debrief; data splitting, taught here because A8 is where it costs; the paper-review
assignment; neural-network foundations (the AMTA tic-tac-toe sequence with six transfer slides
to translation); TorchLingo; ln(V) as the reference point for a first training loss.
**Notebooks:** `lecture-07-toy-model` (activity): train a toy model and read its loss curve in
class; outside class, hold out a phrase of the student's choice, predict, and run a contrast;
watch a run resume from its checkpoint. `03-train-tiny-model` (reading): what a checkpoint holds.
**Sets:** A7, the notebook's report plus two sentences on what the seen/unseen gap says.

### 8a. Encoder-Decoder, and Why Attention Was Invented (F2026, 30 slides)
*8a is what you need in order to do the assignment; 8b is what the model actually is.* The
encoder-decoder and its bottleneck; attention as learned alignment, with real heatmaps; then why
a trained model comes out bad: splitting, sentence length as a memory budget, reading a loss
curve; A8 started in class.
**Notebooks:** `lecture-08a-a8-kickoff` (activity): from the student's A5 corpus to a
checkpointed training run going when class ends. `04-attention-and-alignment` (reading, Parts 1
to 7; Part 7 is where 8b starts).
**Sets:** A8: an English-to-X model on the student's data, at least 100K training pairs (all of
it if fewer), 2K validation, 2K test; the course model and settings are in "The TorchLingo
pivot" below; SacreBLEU over the whole test set; the decoding strategy stated.

### 8b. Neural MT: The Transformer (F2026 `_v4`, 24 slides shown)
Tensor shapes; the three attentions and what each is for; softmax; query, key and value, with
one attention step by hand; self-attention on a real model's maps; replacing recurrence. The
multi-head and block slides were not reached, and are taught in 8c and 9.
**Notebooks:** `lecture-08b-cross-attention` (activity): the student writes the attention step,
shown to be the trained model's in every head, then predicts and reads cross-attention against a
wrong translation. `04-attention-and-alignment` Part 7 (reading); `05-transformer-attention`
(reference): this notebook and 8c's in one worked example, without the exercises.
**Sets:** nothing; A8 is due two days later.

### 8c. Finishing the Transformer, the A8 Debrief, and the Subword Activity (F2026, 22 slides)
The by-hand solution; multi-head attention, the whole Transformer, the encoder block, the
residual stream and the decoder block; pros and cons of NMT; the A8 debrief around the capacity
table; Part A of the Lecture 9 notebook, in class.
**Notebooks:** `lecture-08c-attention-heads` (activity): the student splits the projections into
heads, shown to be the model's in every encoder layer, and measures how averaging and depth blur
attention. `lecture-09-subword-tokenization` Part A (activity); `06-diagnosing-failures`
(reading): five questions to ask of a model that is not working.
**Sets:** A9.

### 9. Handling Morphology and Terminology in NMT (F2026 `_v4`, 21 slides shown)
Deployment; how order gets into a model with no recurrence; the fixed vocabulary; morphology and
why vocabularies explode; BPE and SentencePiece on real sentences; characters; what changes in
the model's shapes; terminology and getting terms into a neural system.
**Notebooks:** `lecture-09-subword-tokenization` (activity, and A9 itself). Part A, on a CPU:
the student writes the `<unk>` counter, trains SentencePiece on the training split, chooses the
vocabulary size with a reason, and predicts A9's BLEU; the settings go to Drive. Part B, on a GPU:
retrains with the pieces under A9's own run names and scores A8 and A9 with the same code.
`01-data-and-vocab` (reference): its vocabulary half stops at `<unk>`, where this lecture starts.
**Sets:** A9 (set at 8c): the same split as A8, the tokenizer the only change, both decoded and
scored the same way; the write-up reconciles the prediction with the result.

### 10. Overview of MT Quality Estimation (F2026 corrected, 34 slides)
Evaluation against estimation, QE features and shared tasks, COMET and COMET-QE and how to read a
score; and the decoding block: greedy, beam search with a worked example, and why a comparison
must pin decoding.
**Notebooks:** `lecture-10-comet-install` (homework, A10): the student writes `comet_inputs`,
predicts, and scores two systems with and without a reference. `08-inference-and-beamsearch`
(reading): greedy and beam search written out, and why a comparison holds decoding fixed.
`07-real-translations` (reading): a trained model on unseen text, greedy against beam.
**Sets:** A10, install COMET and run the notebook end to end; two papers for the quiz.

### 11. Neural Quality Estimation and Evaluation Toolkits (F2025, 19 slides)
COMET and COMET-QE in depth, HTER, partial-input baselines, xCOMET, and LLM-as-judge (the
rebuild uses the five GEMBA slides of the combined May 2025 deck).
**Notebooks:** `lecture-11-comet-on-a6` (homework, A11): COMET and COMET-QE-DA on the student's
A6 sentences beside BLEU and chrF; the student writes `pairwise_agreement`, which counts how
often two metrics prefer the same system, sentence by sentence.
**Sets:** A11: the default COMET model and COMET-QE-DA on the A6 outputs, compared with the BLEU,
chrF and human-ranking numbers already collected; one page. Extra credit for xCOMET.

### 12. Using LLMs for MT and Expanding Context Awareness (F2025, 32 slides)
Encoder-only, decoder-only and large language models; MT with decoder-only models;
document-level and context-aware MT; dropped and ambiguous pronouns; contrastive test sets.
**Notebooks:** `lecture-12-llm-context` (homework, A12): the student writes the context/test
split and the prompt; an open chat model translates at 0, 5, 10 and 20 examples.
**Sets:** A12: a non-MT model from the Hugging Face Hub, one of Efik, Palauan, Pohnpeian, Yapese,
Kosraean or Kamba (the Learning Suite list), BLEU against context size, and an analysis. The
deck's own slide still lists last year's languages; fix it at the rebuild.

### 13. Introduction to Low-Resource NMT Strategies (F2025, 30 slides)
What counts as low-resource, mitigation with and without LLMs, evaluation for low-resource
languages, data sources, data augmentation. The deck's back-translation workflow is written for
OpenNMT; the notebook replaces it. Candidate home for the NLLB spotlight (Open questions).
**Notebooks:** `lecture-13-back-translation` (homework, A13): from the A9 files, the student
writes the reversal and the combination; two trainings and the back-translation run for them.
**Sets:** A13: back-translation from A9, compared with A9 on BLEU and COMET.

### 14. Multilingual NMT and Zero-shot NMT (F2025, 23 slides)
Bilingual against multilingual, how zero-shot works, tagging, NLLB, complete MNMT, why the
Church data suits it. Its handout, "MNMT Guide Using OpenNMT.docx", needs replacing outright.
**Notebooks:** none yet; A14's is built from A9 to the same rule.
**Sets:** A14: a bidirectional English and X system from A9, directions intermingled, separate
test sets per direction, BLEU and COMET both ways.

### 15. Overview of Speech-to-Speech MT (F2025, 33 slides)
Spoken language translation, cascade against end-to-end, Whisper, speech data for projects.
**Sets:** A15: an ASR, MT and TTS pipeline on a free Azure account, not the Speech Translation
API; ten spoken sentences in and out, and a video of one; two papers for the quiz.

### 16. Automatic Dubbing and Interpretation (F2025, 34 slides)
Neural voices, automatic video dubbing, Wav2Lip, HeyGen; the final-project timeline.
**Sets:** nothing.

### 17. Multimodal NMT (F2025, 42 slides)
Whether visual context helps, video-guided MT, VaTeX and MAD, ablations, project examples.

### 18. HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation (F2025, 40 slides)
Interactive translation, CAT tools, post-editing and normalized edit distance, adaptive MT and
whether LLMs can do it. In-class activity.

### 19 and 20, merged. Word Alignment, and Previous MT Paradigms (F2025, 39 and 44 slides; Mon Nov 30)
The IBM models worked through, then the Vauquois triangle, RBMT, EBMT and phrase-based SMT,
which word alignment was built for. One deck of about 30 slides at the rebuild; the Och and Ney
reading report belongs here.

### 21 to 23 (F2025)
Writing research articles, MATRIX Lab research, MT applications and jobs, further research.
Final project presentations close the semester.

---

## Notebook index

Every notebook in the repository, what it is for, and where it lands. Generated: the lecture,
role, assignment and CI columns come from each notebook's own metadata, and the purpose from
`notebooks` in `roadmap.yaml`, so the table cannot fall out of step with the notebooks.

| notebook | lecture | role | starts | purpose | CI |
|---|---|---|---|---|---|
| `lecture-03-word-embeddings` | 3 | activity | A3 | does the shared multilingual space line up across your languages | no: download, pip |
| `lecture-04-regex-refresher` | 4 | reference | | fifteen-minute regex tour; a worksheet with blanks | no: blanks |
| `lecture-04-tmx-cleaning` | 4 | activity | A4 | planted TMX problems, two parsers, a checker that names the cleaning step | no: pip |
| `01-data-and-vocab` | 4, 9 | reference | A9 | the library's view of corpus loading, from A5's two files; word-level vocab, stops at `<unk>`, which Lecture 9 takes up | yes |
| `lecture-05-sentence-alignment` | 5 | activity | A5 | estimate Gale-Church parameters on your language | no: pip |
| `lecture-06-mt-evaluation` | 6 | activity | A6 | BLEU and chrF on your data; BLEU returns zero on ten short sentences | no: exercises |
| `lecture-06-mt-evaluation-homework` | 6 | homework | A6 | the assignment's scaffold over your uploaded A5 corpus | no: colab |
| `02-evaluating-translations` | 6 | reading | A6, A8 | the library's out-of-class treatment of evaluation | yes |
| `lecture-07-toy-model` | 7 | activity | A7 | the three-part toy-model exercise; the student picks the held-out phrase, predicts, and a contrast phrase runs; resume demo | yes |
| `03-train-tiny-model` | 7 | reading | A8 | what a checkpoint holds and what it does not; a rebuild that fails loudly, and one that loads cleanly and translates nonsense | yes |
| `lecture-08a-a8-kickoff` | 8a | activity | A8 | from your A5 corpus to a running, checkpointed, bucketed training job | no: colab, pip |
| `04-attention-and-alignment` | 8a, 8b | reading | | with and without attention; did it learn the right alignment; Part 7 is the Transformer | yes |
| `lecture-08b-cross-attention` | 8b | activity | | one attention step written by the student and shown to be the trained model's, head by head; cross-attention read against a wrong translation | no: exercises |
| `05-transformer-attention` | 8b, 8c | reference | A8 | both of Lecture 8b's and 8c's notebooks in one worked example, without the exercises | yes |
| `lecture-08c-attention-heads` | 8c | activity | | the student's head split shown to be the model's in every encoder layer; heads read one by one, and sharpness measured against averaging and depth | no: exercises |
| `06-diagnosing-failures` | 8c | reading | A8 | five questions to ask of a model that is not working | yes |
| `lecture-09-subword-tokenization` | 9 | activity | A9 | `<unk>` before and after SentencePiece (the student's counter), a round trip, the student's vocabulary size and A9 prediction; Part B is A9 itself, retrained and scored | no: colab |
| `lecture-09-subword-tokenization-new` | 9 | activity | A9 | the next version of the Lecture 9 notebook, renamed over it once A9's deadline has passed; A9 trains as many epochs as A8, from A8's checkpoint | no: colab |
| `lecture-10-comet-install` | 10 | homework | A10 | install COMET; the student's comet_inputs; two systems scored with and without a reference | no: colab, hf-token, download, pip, exercises |
| `07-real-translations` | 10 | reading | A11 | a trained model on unseen talks; a comparable BLEU; greedy against beam with decoding held fixed | yes |
| `08-inference-and-beamsearch` | 10 | reading | | greedy against beam, written out and checked against the library; why a comparison holds decoding fixed, as A9 does | yes |
| `lecture-11-comet-on-a6` | 11 | homework | A11 | COMET and COMET-QE-DA on the student's A6 sentences, beside BLEU, chrF and their ranking | no: colab, download, pip, exercises |
| `lecture-12-llm-context` | 12 | homework | A12 | in-context translation with 0 to 20 examples, on the student's A12 language | no: colab, download, exercises |
| `lecture-13-back-translation` | 13 | homework | A13 | back-translation end to end, from the A9 files and the student's own monolingual text | no: colab, exercises |

**Tutorial numbers** are in course order: 01 data, 02 evaluating, 03 checkpoints, 04 attention,
05 transformer-attention, 06 diagnosing, 07 real-translations, 08 beam search.

Every notebook opens with the same two code cells: the install from the `course` branch, and
`torchlingo.colab.setup(...)`, which prints versions and device, checks the GPU, mounts Drive
and downloads data. Notebook text carries no weekdays, dates or due dates, so the notebooks
outlive one term's calendar.

---

## Tooling the semester depends on

| Tool | First needed | Used for |
|---|---|---|
| Google Colab, paid plan | Lecture 3 | every in-class activity and most assignments; the paid plan is required from 8a, where A8 needs an A100, L4 or G4 (a T4 does not hold the course model at long lengths) |
| Python TMX libraries | Lecture 4 | extracting segment pairs |
| TMX editors (Olifant, Heartsome) | Lecture 4 | inspection only, never in the pipeline |
| grader.exe | Lecture 4 | checking cleaned output. Binaries only, no source, no repository; see Open questions |
| MTEval (mteval.matrix.byu.edu) | Lecture 6 | human ranking; students self-register |
| SacreBLEU, chrF | Lecture 6 | automatic scoring, and again in 8, 9, 12, 13, 14 |
| TorchLingo | Lecture 6 | every model the students train (7, 8, 9, 13, 14), and the metrics and Colab setup in every course notebook that uses it. Notebooks install it from the repository's `course` branch, not PyPI, so a notebook and the library code it calls reach students together; `setup()` prints the exact version, which is what to ask for with a bug report |
| SentencePiece | Lecture 9 | subword tokenization |
| Hugging Face account and token | Lecture 10 | COMETKiwi, the reference-free COMET model, is gated; the reference-based COMET model, COMET-QE-DA and Lecture 12's default LLM need no token |
| COMET / COMET-QE / xCOMET | Lecture 10 | neural evaluation, and again in 11, 13 and 14 |
| Azure free tier | Lecture 15 | ASR, MT and TTS for the speech pipeline |

---

## The TorchLingo pivot: what is settled

The course trains models with TorchLingo, an educational PyTorch NMT library built in the
MATRIX Lab. Five assignments train a model: the toy model at Lecture 7 (A7), A8, A9, A13 and
A14. They are cumulative in the artifact, so this is one crossing rather than five, and every
assignment after A9 starts from A9. A7, A8, A9 and A13 have TorchLingo notebooks; A14's is to be
built. The A13 and A14 Learning Suite texts and decks still describe OpenNMT.

**A8's settings**, the course model: d_model 512, 8 heads, 6 + 6 layers, feed-forward 2048 (the
2017 base configuration, 56.4M parameters at an 8K vocabulary); a 100-token cap applied before
the split; about 35 epochs, and keep going while validation loss falls; checkpoints to Drive; an
A100, L4 or G4 runtime, not a T4. The measurements behind each follow.

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
trains and scores on A8's files and split unchanged, with the SentencePiece model and decode
limit the Lecture 9 notebook saved; both models are decoded greedily and scored by the same
code. The split is never redone with the subword tokenizer: text fed through the word
vocabulary would make A8's model output pieces, and BLEU on pieces is not comparable.

**Post-norm.** `SimpleTransformer` is the 2017 paper's architecture, post-norm by default,
pinned by a test whose failure message names the two 8b slides that depend on it. Pre-norm is
available with `norm_first=True`.

**Notebooks live in the repository**, `torchlingo/docs/docs/course/`, each with a Colab badge,
metadata declaring which lecture it serves and what it starts, and CI that executes every
notebook it can and tests the rest from their own cells. Students install TorchLingo from the
`course` branch, which moves only when `main`'s checks have passed. The two sessions exchange files under `notes/handoff/`, one file per
hand-off; course-side items are tracked in `CS479-TASKS.md`, repository items in the
repository's `notes/TASKS.md`.

---

## Open questions

- **Does "about 35 epochs" hold for a student with 30K pairs?** The 30-epoch convergence was
  measured at 100K pairs with a subword vocabulary. Revisit once the A5 audit says how many
  students are well below 100K; the audit itself still needs the submissions staged somewhere
  readable. The instruction's escape hatch is "keep going while validation loss falls".
- **Does the grader survive?** `grader.exe` is four unsigned PyInstaller binaries in a PhD
  student's OneDrive with no source, no repository, no license. Ammon has been asked for the
  source. Fallback: the checks are documented in Lecture 4 and belong in `torchlingo.diagnostics`.
- **An NLLB spotlight lecture** (Eric, Sep 27): what makes NLLB special beyond its data, namely
  the sparsely gated mixture-of-experts architecture with its regularisation against low-resource
  pairs overfitting, and the deliberate training curriculum, which has no coverage in the course.
  Home is Lecture 13 or 14. The paper's specifics are verified before any number reaches a slide.
- **Learning Suite's assignment texts.** A13 and A14 still carry last year's OpenNMT directions;
  each is rewritten against its notebook. A7's due time differs between Learning Suite and the
  deck.
- **Where does A13's monolingual text come from?** A8 used all of each student's cleaned data, so
  the A13 notebook asks students for new sentences in their language. A course-provided source
  per language would remove that step.
- **Lecture 6 is overloaded** and nothing has been done about it.
- **Lecture 19 needs rescoping** around the material now in Lecture 5, and has no notebook.
- **Deck links to personal Google Drive URLs.** Lecture 4 links the regex notebook twice and its
  own activity once; Lecture 12 links its assignment notebook. All die with the account. Each is
  now a Colab badge in the repository, but the Drive copies must stay alive while students are in
  them.

---

## Decisions log

Newest first. Each is a decision that changed the schedule, an assignment, or a notebook's
place, with who made it.

- **Oct 13.** Every assignment after A9 starts from A9 (Eric): its split, SentencePiece model,
  configuration and test set, not "A8 or A9, whichever was best". A comparison then changes one
  thing, and subwords handle the new words that back-translated and multilingual data bring. The
  A13 notebook is built this way; A14's follows, and the A13 and A14 handouts drop the choice.
  Tutorial 5 split into two course notebooks with exercises (Eric): cross-attention for 8b,
  heads for 8c, where the deck moved the multi-head slides. The tutorial stays for now as
  reference, pointing at both, so a student who opens it is not lost.
- **Oct 8.** Assignments renamed "Assignment N" and the grade category "Assignments" (Eric). The SLT
  pipeline is A15, set at Lecture 15 and still due Wed Nov 11, so the numbering has no gap and Lecture
  16 sets nothing. A10 gains a 3-point rubric and counts in the grade. Every hand-in list mirrors its
  rubric (A5 fixed after a student's report).
- **Oct 7, late.** Content pages never name reading-report presenters (Eric): the sign-up sheet is the
  official schedule, and a page that names students is a second copy to maintain. Pages link the
  sheet instead. Decks still name presenters, since they are built the morning of from the sheet.
- **Oct 7, evening, later.** Content pages carry the reading for the *next* lecture, not the day's
  (student feedback relayed by Eric: advance warning is the point). Each page: slides; "Before
  <next session>" with the reading, the assignment due, the quiz and the reading reports, mirroring
  the deck's before-next-time slide; then any in-class material; then, at the end, the reading that
  prepared the lecture. Applied to the 8c and 9 pages; every page from here follows.
- **Oct 7, evening.** Lecture 9 as taught becomes Lecture 8c and Lecture 9 moves to Mon Oct 12
  (Eric, after walking the Oct 7 deck slide by slide): 8c keeps the by-hand solution, the rest of
  8b, pros and cons, the A8 debrief and the subword activity; 9 takes deployment, positional
  encoding, the fixed vocabulary, subwords and terminology. Everything after shifts one session;
  Lectures 19 and 20 merge on Nov 30 to pay for it (Eric, on this side's recommendation). A9 due
  Wed Oct 14 (Eric). A10 Oct 19, A12 Oct 28, A13 Nov 2, A14 Nov 4, A16 Nov 11 follow the shift;
  A11 Oct 26 keeps a seven-day window (this side's call). Proposal week Nov 9, 11, 16.
  Reading-report dates stand. Today's attendance quiz and Content page are 8c.
- **Oct 7, after Lecture 9's rebuild.** Eric walked the 8b deck slide by slide, marking each
  discussed or not on Oct 5. Five were not reached (Multi-Head Attention, The Whole Transformer,
  Inside One Encoder Block, The Residual Stream, The Decoder Block): they move to the Lecture 9
  deck after the by-hand solution, preceded by a copy of the self-attention map slide; 8b `_v4`
  hides them and its objectives shrink to the three things taught. Rule implied for future
  passes: what a session did not reach moves forward rather than being assumed.
- **Oct 7.** Six de-noise rules for every deck (Eric; the header lists them), applied to
  Lecture 9, 8b `_v4` and decks 3 to 8a the same day. Protocol change: the course side edits the
  course notebooks directly in the repository tree and the repository session reviews, commits,
  pushes and promotes; first use, the Lecture 9 notebook's Part B (A9 in the same notebook).
- **Oct 6.** Slides never state how many questions a quiz has (Eric). A9's Learning Suite text
  follows the one-notebook design; the transcript changes in the same pass.
- **Oct 5, evening.** Two standing deck rules and one quiz rule (Eric): every deck carries a
  Reading Reports slide and a closing before-next-time slide synced to Learning Suite; every
  attendance quiz asks one question per required reading or tutorial, scored 2 / 1 / 0. The 8b
  `_v2` deck adopted (35 slides). Tutorial-4 recap slides added because the reading cannot be
  assumed internalised; the same will hold for Lecture 9's reading.
- **Oct 5.** Three decisions (Eric), all handed to the repository: the Lecture 7 notebook's
  held-out phrase becomes the student's choice with a required prediction and contrast; the
  Lecture 9 notebook makes `unknown_rate` the student's, the vocabulary size a reasoned choice,
  and asks for an A9 prediction before the settings print, plus a Report back; the tutorials are
  renumbered into course order (07→02, 02→03, 08→05, 05→07, 03→08) with redirect stubs for the
  term. Tutorial 5 stays at Lecture 10 and becomes tutorial 7, closing its open question. The
  Lecture 7 deck's Part B slide and the task list were updated the same day; other course-side
  references flip when the merges land.
- **Oct 1.** Learning-outcomes section added after Eric asked whether the notebooks do so much that learning by doing drops off. Four flags with proposals (the A8 write-up questions, the A9 prediction and `unknown_rate` TODO, the Lecture 7 held-out choice, the A13/A14 scaffold rule); nothing changes in a notebook or an assignment until Eric decides.
- **Sep 30, late.** A9 due Mon Oct 12, not Oct 14: an assignment is due on the day of the next
  lecture, and A9 and A10 do not share a day (Eric, to Coulson). Proposal week is one day of
  outline reviews and two of presentations, because of the class size (Coulson proposed, Eric
  agreed); days pending. Learning Suite now carries the shifted lecture rows, all assignment due
  dates, the quiz renames and a cleaned Lecture 7 page (Coulson). Remaining there: Quiz 8b's
  questions, the 8b, 9 and 10 Content pages, the A9, A13 and A14 texts, the proposal sign-up
  sheet.
- **Sep 30.** The A8 "What To Do" slide caps sentence length before the split, matching the
  kickoff; capping after the split shrinks a training set already sized to the floor
  (repository found it; course-side fixed it). Tutorial 4's Part 8 becomes tutorial 8, 8b's
  reading (Eric; repository PR #193). A9's call as first handed over failed on every Colab
  runtime; fixed in 0.2.4 and the notebook prints the model path twice (repository). Notebooks
  carry no due dates at all, not even in purpose cells (Eric). "Instructor" is not a banned
  word; the notebook check for it is removed (Eric). The assignment transcripts belong in
  `torchlingo-private` (Eric). Notebook audit: the job of each notebook is to teach; every
  notebook runs end to end; no project history in notebooks (Eric); nine of the audit's
  changes merged, Lecture 10's three and lecture-12 remain (repository).
- **Sep 29.** Lecture 7 taught from the SDL tic-tac-toe block with six transfer slides; the
  full rebuild archived (Eric). Every assignment description transcribed from Learning Suite
  (Eric asked; course-side), since moved to `torchlingo-private/notes/assignments/`. The repository's copy of
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

Two notebook families, numbered differently on purpose:

| family | numbered by | who writes it |
|---|---|---|
| `docs/docs/tutorials/` | library topic | the repository session |
| `docs/docs/course/` | lecture | both sessions, together; the repository session commits |

So which notebook serves which lecture is not derivable from a filename. Each notebook
**declares it** in its own `torchlingo` metadata, and this table is generated from those.
Dates and deadlines are in the schedule above; lecture numbers are the join.

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
| 6 | Automatic and Human MT Evaluation | `lecture-06-mt-evaluation-homework` (homework) [6], `lecture-06-mt-evaluation` (activity) [7] | `02-evaluating-translations` (reading) [2] |
| 7 | Research Paper Reviews; Intro to Neural Networks | `lecture-07-toy-model` (activity) [8] | `03-train-tiny-model` (reading) [3] |
| 8a | Neural MT: Encoder-Decoder, and Why Attention Was Invented | `lecture-08a-a8-kickoff` (activity) [9] | `04-attention-and-alignment` (reading) |
| 8b | Neural MT: The Transformer | `lecture-08b-cross-attention` (activity) [10] | `04-attention-and-alignment` (reading), `05-transformer-attention` (reference) |
| 8c | Finishing the Transformer, the A8 Debrief, and the Subword Activity | `lecture-08c-attention-heads` (activity) [11] | `05-transformer-attention` (reference), `06-diagnosing-failures` (reading) |
| 9 | Handling Morphology and Terminology in NMT | `lecture-09-subword-tokenization-new` (activity) [12], `lecture-09-subword-tokenization` (activity) [13] | `01-data-and-vocab` (reference) [1] |
| 10 | Overview of MT Quality Estimation | `lecture-10-comet-install` (homework) [14] | `07-real-translations` (reading) [4], `08-inference-and-beamsearch` (reading) |
| 11 | Neural Quality Estimation and Evaluation | `lecture-11-comet-on-a6` (homework) [15] | — |
| 12 | Using LLMs for MT; Expanding Context Awareness | `lecture-12-llm-context` (homework) [16] | — |
| 13 | Strategies for NMT of Low-Resource Languages | `lecture-13-back-translation` (homework) [17] | — |
| 14 | Multilingual NMT and "Zero-shot" NMT | — | — |
| 15 | Overview of Speech-to-Speech MT | — | — |
| 16 | Automatic Dubbing and Interpretation | — | — |
| 17 | Multimodal NMT | — | — |
| 18 | HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation | — | — |
| 19, 20 | Word Alignment, and What We Can Learn from Previous MT Methods: RBMT, KBMT, EBMT, PBSMT, SBSMT | — | — |
| 21 | Writing Research Articles; MATRIX Lab research | — | — |
| 22, 23 | MT Applications and Jobs; Opportunities for Further MT Research | — | — |

Notes on the rows that are not simple:

- [1] **`01-data-and-vocab`** — Loading and cleaning a parallel corpus is Lecture 4's
  subject from the library side, and the vocabulary half belongs to Lecture 9, which
  starts where this stops, at <unk>.
- [2] **`02-evaluating-translations`** — The out-of-class treatment of evaluation.
  lecture-06-mt-evaluation is the in-class activity and owns the teaching; this one makes
  the student choose between two systems.
- [3] **`03-train-tiny-model`** — What a checkpoint holds and what it does not: weights
  and config, not a word vocabulary. Rebuilds a saved model right, then shows the loud
  failure (one more word, a size mismatch) and the silent one (the same sentences
  reordered, which loads and translates nonsense). Lecture 7 owns resuming a run and the
  A8 notebook owns save_dir and reloading; this is the mechanism behind them.
- [4] **`07-real-translations`** — Read at Lecture 10. Its substance is evaluating and
  decoding a real model, held-out BLEU and greedy against beam search, which Lecture 10's
  decoding outcome and A11's metric work build on.
- [5] **`lecture-04-regex-refresher`** — A fifteen-minute tour of Python regular
  expressions, which the sixteen cleaning steps are written in. Lecture 4's deck links it
  and Learning Suite posts it on the Lecture 3 tab, so it is offered ahead of Lecture 4
  rather than used inside it.
- [6] **`lecture-06-mt-evaluation-homework`** — The A6 homework: ten sentences from the
  student's own Assignment 5 corpus, uploaded through the Colab file picker, translated by
  a web MT system and pasted in, ranked by hand into MY_RANKING (which stops the notebook
  while empty) before any score prints, then scored with BLEU, chrF and TER, with
  sentence-level chrF set beside the ranking. Needs the Colab picker, so CI does not
  execute it; tests/test_lecture_06.py runs its cells with the upload skipped and files,
  translations and a ranking written in.
- [7] **`lecture-06-mt-evaluation`** — Lecture 6's in-class activity on supplied data:
  BLEU's zero on two short sentences, then the student writes bleu_from_parts (BLEU from
  bleu.precisions and bleu.bp, checked on a case worked on paper, on a brevity penalty and
  on Part 1's zero, then against BLEU's own score on both Part 2 systems); reads five
  pairs and records which system they would rank higher and which BLEU will prefer before
  any score; and predicts BLEU at three words kept before the brevity-penalty table. Each
  prediction stops the notebook while empty. The homework on the student's own data is
  lecture-06-mt-evaluation-homework. CI does not execute it as shipped;
  tests/test_lecture_06.py runs its cells with an answer and predictions written in.
- [8] **`lecture-07-toy-model`** — Lecture 7's three-part exercise: Part A in class
  (install, train the toy model, read the loss curve against ln(V)), Part B outside class
  (hold out a phrase of the student's choice, predict, run a contrast phrase, score seen
  against unseen; watch a run resume after a faked disconnect), Part C the report handed
  in as A7. Trains a 64-dimensional model in seconds, so it runs in CI.
- [9] **`lecture-08a-a8-kickoff`** — The in-class start of Assignment 8: from the
  student's Assignment 5 corpus to a training run already going when class ends. The
  handout is the authority; this is the executable path through it, and every threshold is
  quoted from the handout in one cell. Cannot run in CI: it mounts Drive and needs a GPU.
- [10] **`lecture-08b-cross-attention`** — Lecture 8b's activity, the cross-attention half
  of what was tutorial 5. The student writes attention, one scaled dot-product step,
  checked on a case worked on paper (one that also catches a missing √d_k) and then
  against the trained model's own weights in all four heads of the last decoder layer's
  cross-attention; then predicts where three output pieces will look and reads the
  averaged map against the translation: the right words found, the wrong Spanish written.
  tests/test_lecture_08b.py runs it with an answer written in.
- [11] **`lecture-08c-attention-heads`** — Lecture 8c's activity, the self-attention half
  of what was tutorial 5, on the same model and sentence as Lecture 8b's. The student
  writes split_heads, checked on numbers readable by eye and then against the model's own
  weights in every encoder layer, and sharpness, checked on paper cases; predicts, then
  reads the first layer head by head and measures how averaging and depth blur attention.
  tests/test_lecture_08c.py runs it with answers written in.
- [12] **`lecture-09-subword-tokenization-new`** — The next version of
  lecture-09-subword-tokenization, renamed over it once A9's deadline has passed (Eric,
  Oct 13), so students mid-assignment keep the notebook they started. Part B reads the
  epoch count from A8's checkpoint instead of a hard-coded 35; unknown_rate is checked on
  three cases; Step 6 shows A8's BLEU before the prediction, which has its own cell; the
  opening states abilities; Part B tells students to disconnect the GPU when done.
  tests/test_lecture_09_part_a.py and _part_b.py run it.
- [13] **`lecture-09-subword-tokenization`** — Lecture 9's in-class activity and, in Part
  B, Assignment 9 itself. Part A (CPU, in class): SentencePiece fit on the student's A8
  training split, the student's own unknown-token counter under words and under pieces, a
  round trip, a vocabulary size chosen with a reason, an A9 BLEU prediction, and the
  settings Part B needs saved to Drive (a9-settings.json). Part B (GPU, outside class):
  retrains the A8 model with the pieces under its own run names (assignment-9, best-a9/,
  test-a9.hyp), re-scores the saved A8 model with the same code, and prints the write-up
  numbers. Reads the A8 split from Drive, so it cannot run in CI. Part B added so that A9
  never sends a student back to the A8 notebook.
- [14] **`lecture-10-comet-install`** — A10: installs COMET; the student writes
  comet_inputs, the records COMET scores (reused in A11 on the A6 sentences), checked on
  known cases, and a prediction, which stops the run while empty; then the reference-based
  model and COMETKiwi (whose license the student accepts on Hugging Face, then logs in
  with a token) score two systems through COMET's Python API; the last cell prints what
  A10 grades, followed by three write-up questions: the prediction against the result,
  what a COMET score means, and evaluation against estimation. tests/test_lecture_10.py
  runs its cells with COMET stubbed.
- [15] **`lecture-11-comet-on-a6`** — A11: COMET (reference-based) and COMET-QE-DA on the
  student's A6 sentences beside BLEU and chrF. The student pastes comet_inputs from A10
  and writes pairwise_agreement, the sentence-by-sentence measure of whether two metrics
  prefer the same system, each checked on a known case; a prediction, which stops the run
  while empty; the notebook says why COMET-QE-DA rather than the gated COMETKiwi. The
  report gives system scores, each metric's preferred system against the A6 human average
  ranks, and agreement counts; four summary questions follow, the last on LLM-as-judge and
  system- against segment-level agreement. tests/test_lecture_11.py runs its cells with
  COMET stubbed.
- [16] **`lecture-12-llm-context`** — Assignment 12, runnable end to end. Runs on a
  built-in English-Spanish sample from data/example.tsv until the student sets LANGUAGE
  and DATA_DIR, together. The student writes split_pool_and_test (checked on the alphabet,
  including that a different seed gives a different split, then by check_contamination on
  the real split) and build_prompt (checked on a known case), then a prediction, which
  stops the notebook while empty; an open instruction-tuned LLM (default
  Qwen2.5-1.5B-Instruct, greedy) translates 50 test sentences at 0, 5, 10 and 20
  in-context examples, scores each with BLEU and its signature, charts it, and writes the
  upload files (context-k.tsv, test.tsv, bleu-report.tsv, bleu-by-context.png) to Drive.
  The write-up adds the document-level problems no example pair fixes. Needs a GPU and
  Drive, so CI does not execute it; tests/test_lecture_12.py runs its cells with the model
  stubbed.
- [17] **`lecture-13-back-translation`** — Assignment 13 end to end, starting from what
  Lecture 9's Part B leaves on Drive (A9's split, spm.model, best-a9/, test-a9.hyp, and
  A9's training checkpoint, from which both trainings take their epoch count) plus the
  student's monolingual mono.txt, which Step 1 says where to look for and which stops with
  instructions if missing. The student writes reverse, which turns the split around for an
  X-to-English model, and combine, which joins the original and synthetic pairs and drops
  any whose source is held out; each is checked on a known case, and combine's result by
  check_contamination against validation and test. The notebook trains X-to-English with
  A9's configuration and vocabulary, back-translates with translate_to_file (resumable),
  takes a prediction that stops the run while empty, retrains English-to-X on the combined
  pairs, scores both systems with A9's BLEU, runs COMET on both, and prints the write-up
  block from Drive alone (a13-notes.json keeps the prediction and the back-translator's
  BLEU), so it survives a dropped session. The write-up asks what makes the language
  low-resource beyond pair counts. Needs a GPU and Drive, so CI does not execute it;
  tests/test_lecture_13.py runs its cells at toy size.

<!-- END generated:notebook-map -->

### Why lectures that have run are still listed

A student revisiting a past lecture needs to know what to open, later assignments refer back to
earlier ones, and a Fall 2027 offering should not have to rediscover the mapping. Where a tutorial
would have improved a past lecture it is marked `reference` rather than dropped.

### What the table cannot say

The table lists notebooks that exist; a notebook not yet written cannot declare anything.

- **Lecture 14 needs a notebook**: A14 builds a two-direction model from A9, to the rule in
  "Where the line is crossed".
- **Ten sessions pair with nothing else**: Lectures 1, 2, 15 to 18, 19 and 20 (merged), 21, and 22
  and 23 (merged). Not every lecture wants a notebook, and inventing one to fill a row would be
  the redundancy `CLAUDE.md` rules out.

**Three notebooks declare no `leads_to`, deliberately**, so nobody fills the blank in: a notebook
that teaches a prerequisite is not recorded as covering the thing.

- **`04-attention-and-alignment`** makes A8's training make sense; it gives no head start on a
  trained model of the student's own.
- **`lecture-04-regex-refresher`** teaches the regular expressions A4's cleaning is written in.
- **`08-inference-and-beamsearch`** is read the day A9 is due; it explains why A9 holds decoding
  fixed.

**Tutorial 4 serves Lectures 8a and 8b and stays one notebook**: its Part 7, "You have already
seen the Transformer's mechanism", depends on the attention the reader ablated in Parts 1 to 6.

## What exists beyond the notebooks

The notebooks themselves, what each is for and the lecture it serves, are in the notebook index
and the map above, both generated from the notebooks' own metadata. Every notebook runs on its
own, and a course notebook never depends on a tutorial (`CLAUDE.md`).

### Concept pages

`what-is-nmt.md`, `data-pipeline.md`, `vocabulary.md`, `models.md`, `training.md`,
`decoding.md`, `evaluation.md`, `when-it-fails.md`.

### Measured artifacts

Generated into `docs/docs/_generated/`, each from a script, none hand-typed:
`decode_bench`, `decoding_sweep`, `alignment_diagnosis`, `realign_report`,
`metric_comparison`, `checkpoint_comparison`. Experimental reports, also generated, are in
`notes/reports/`.

---

## Where learning outcomes live

**A lecture's outcomes** are in "Learning outcomes, and who does the work" above. **A
notebook's outcomes** are stated in its own opener, under "By the end", and nowhere else, so
they cannot drift from the notebook they describe. The notebook ratings above say who does the
work in each.

Two of Eric's rules, in `CLAUDE.md`, govern how notebook outcomes relate: every
notebook has a purpose and outcome no other notebook shares, judged by how it looks as well as
by what it teaches; and a course notebook never assumes a tutorial was read.

---

## Gaps

What the material still does not teach, ordered by how much it costs a student. Three gaps
listed here before are closed: tutorial 6 teaches diagnosis by breaking a model on purpose and
watching each check fire; tutorial 2 teaches where BLEU misleads; the Lecture 8b and 8c
notebooks read a Transformer's attention.

**Training dynamics.** `concepts/training.md` covers the loop, optimizers, schedulers,
clipping and early stopping. It does not cover what a student does when training goes wrong:
batch size against learning rate, or what overfitting looks like on a small corpus.

**How much data is enough.** Still open in general. The material has a measured answer to a
better question, *which lever are you actually pulling?*: at this scale, training longer
helped and more data did not, and the first attempt to answer it was wrong in a way only a
control run caught. The numbers are in `notes/reports/training-budget.md`. The episode is worth
teaching directly.

**Inference cost in practice.** `concepts/decoding.md` covers beam search's cost and why a
GPU at batch size 1 is mostly overhead. Nothing covers model size against latency, which is
what a student meets when a Colab session is slow.
