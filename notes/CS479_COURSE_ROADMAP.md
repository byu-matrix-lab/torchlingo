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
line. One Learning Suite text is still last year's, written for OpenNMT: A14's, rewritten
against its notebook once that lands. A13's was rewritten against its notebook on Oct 22.

---

## Semester at a glance

| # | Date | Lecture | Assignment due that day | Deck |
|---|---|---|---|---|
| 1 | Wed Sep 2 | Course Overview and History of MT | | F2026 |
| 2 | Wed Sep 9 | Translation Challenges for MT | **A1** MT Topics and History | F2026 |
| 3 | Mon Sep 14 | Introduction to Word Embeddings | **A2** Translation Challenges | F2026 |
| 4 | Wed Sep 16 | Data Preparation for MT Training | **A3** Word Embeddings | F2026 |
| 5 | Mon Sep 21 | Data Preparation for MT Training, Part 2 | **A4** Data Cleaning, First Steps | F2026 |
| 6 | Wed Sep 23 | Automatic and Human MT Evaluation | **A5** Data Cleaning Pipeline | F2026 |
| 7 | Mon Sep 28 | Research Paper Reviews; Intro to Neural Networks | **A6** Human vs. Automatic Evaluation | F2026 |
| 8a | Wed Sep 30 | Neural MT: Encoder-Decoder, and Why Attention Was Invented | **A7** A Toy Model in TorchLingo | F2026 |
| 8b | Mon Oct 5 | Neural MT: The Transformer | | F2026 |
| 8c | Wed Oct 7 | Finishing the Transformer, the A8 Debrief, and the Subword Activity | **A8** NMT on Your Data | F2026 |
| 9 | Mon Oct 12 | Handling Morphology and Terminology in NMT | | F2026 |
| 10 | Wed Oct 14 | Overview of MT Quality Estimation | **A9** Subwords with SentencePiece | F2026 rebuilt; redo planned |
| — | Mon Oct 19 | Review Day (Lecture 11 moved to Oct 21 and taught live; Eric, Oct 21) | **A10** COMET Setup and QE Reading | |
| 11 | Wed Oct 21 | LLM Judges, Sentences and Spans | | F2026 rebuilt (taught live) |
| 12 | Mon Oct 26 | Using LLMs for MT; Expanding Context Awareness | **A11** Evaluating with COMET | F2026 draft |
| 13 | Wed Oct 28 | Strategies for NMT of Low-Resource Languages | | F2026 |
| 14 | Mon Nov 2 | Multilingual NMT and "Zero-shot" NMT | **A12** Context in LLM Translation | F2025 |
| 15 | Wed Nov 4 | Speech-to-Speech MT, with Automatic Dubbing and Interpretation | **A13** Back-Translation (due Fri Nov 6, 11:59 pm) | F2025, merged at rebuild |
| — | Mon Nov 9 | Proposal week: outline reviews with the professor (Eric and Coulson, Sep 30) | **A14** Multilingual NMT, Both Directions (due Tue Nov 10, 11:59 pm) · **Final Project 1** proposal outline review (5), by appointment (due 11:59 pm) | |
| — | Wed Nov 11 | (proposal week) proposal presentations | **Final Project 2** proposal (20), at 10:00 am, the first presentation day | |
| — | Mon Nov 16 | (proposal week) proposal presentations | | |
| 16 | Wed Nov 18 | Multimodal NMT; then Writing the Final Project Report (15 minutes) | | F2025, trimmed, plus the writing slides of last year's Lecture 21 |
| — | Mon Nov 23 | Project checkpoint reviews with the instructor, by appointment; no class meets (Eric, Oct 23) | **Final Project 3** checkpoint review (5), by appointment (due 11:59 pm) | |
| — | Wed Nov 25 | **No class**, Thanksgiving Eve | | |
| 17 | Mon Nov 30 | Research in the MATRIX Lab, and Open Problems in MT | | F2025, from last year's Lectures 21 and 23 |
| 18 | Wed Dec 2 | Word Alignment, and What We Can Learn from Previous MT Methods: RBMT, KBMT, EBMT, PBSMT, SBSMT | | F2025, merged at rebuild |
| 19 | Mon Dec 7 | MT in Use: Applications, Jobs and Post-Editing | | F2025, from last year's Lecture 22, with Lecture 18's post-editing |
| — | Wed Dec 9 | Final project presentations, day 1 of 2 (Steve Richardson and Eric, Oct 26) | **Paper review assignment** | |
| — | Thu Dec 10 | Last day of class | **Final Project 4** written report (40, ACL template) (due 11:59 pm) · **Final Project 5** submission (35) (due 11:59 pm) | |
| — | Wed Dec 16 | Final exam slot, 11:00 to 14:00; final project presentations, day 2 of 2 | **Final Project 6** presentation (20; a 10-minute slot: 7 to talk, 2 for questions, 1 to hand off), Dec 9 or the exam slot (due 2:00 pm) · **Final Project 7** peer feedback (5), a form on every other presentation (due 2:00 pm) | |

An assignment appears on the row of the day it is **due**, which is how Learning Suite lists
it, at 10:00 unless said otherwise (Learning Suite has A7 at 11:59 pm; the deck says 10:00,
and the two should agree). A7 is new this year: a short report from Lecture 7's toy-model
exercise. The SLT pipeline was A15, set at Lecture 15; it is dropped this term and folded into
the Final Project as a Path B example (Eric, Oct 22), so the numbering runs 1 to 14. One due
between lecture days is listed on the row before, with its day and time. Lectures 16 to 19
carry no assignments; the final project has taken over, in seven parts with dates of their own
(Eric, Oct 23). Lectures from 15 on were renumbered Oct 26 (Eric), one number per session:
last year's 15 and 16 are now 15; 17, with the writing slides of last year's 21, is 16; last
year's 21 (its lab research half) and 23 are 17; 19 and 20 are 18; 22, with 18's post-editing,
is 19. Last year's Lecture 18 is no longer a lecture. Lecture quizzes run through Lecture 18,
and 8a and 8b each have their own.

**Paper reviews** run alongside: 24 papers on the sign-up sheet (Google Sheet "CS 479 Fall
2026", tab "Reading Report"), presented Oct 5 through Dec 2 at one or two per session, each
placed next to the lecture its topic belongs with, and the sheet's LECTURE TOPIC ALIGNMENT
column says which. From Oct 12 on, that column is one lecture off from the schedule (the QE
papers of Oct 12 land on Lecture 9, COMET on Lecture 10); the unassigned Och and Ney slot on
Nov 23 belongs on Dec 2, the merged alignment session, since no class meets on Nov 23. Dec 7 is an open slot if a
student has to move; Dec 9 is final project presentations.

---

## How the second half was scheduled

Lecture 8 is three sessions: 8a on Wed Sep 30, 8b on Mon Oct 5 and 8c on Wed Oct 7. The first
extra session (Sep 27) was Mon Nov 16, which last year held a report on the TAUS conference, a
2025 event that does not recur; the second (Oct 7, when 8b's unreached slides and the A8 debrief
filled the Oct 7 session) comes from merging last year's Lectures 19 and 20 (now Lecture 18, on Wed Dec 2 since Oct 23): Lecture 19 had
already lost its sentence-alignment half to Lecture 5, and its word-alignment half, the IBM
models, is the natural opening of the previous-paradigms lecture. Lectures 9 to 18 each sit two
sessions later than last year's calendar. Lecture *numbers* did not change then (those from 15 on were renumbered Oct 26, and no notebook
declares one of them), so every notebook's declared lecture stays valid; the Lecture 9 notebook's Part A ran in 8c.

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
| A11 | 11, Wed Oct 21 | **Mon Oct 26** | 5 since the Oct 21 shift (set at the live Lecture 11; due date kept, Eric, Oct 21) |
| A12 | 12, Mon Oct 26 | **Mon Nov 2** | 7, Mon Oct 26 to Mon Nov 2 (Eric, Oct 26: moved from Fri Oct 30, which had moved from Wed Oct 28 on Oct 22); the A12 page and notebook were open from Oct 21 |
| A13 | 13, Wed Oct 28 | **Fri Nov 6, 11:59 pm** | 9, Wed Oct 28 to Fri Nov 6, overlapping A12 until Nov 2: start the training run early and let it train while finishing A12 (Eric, Oct 26: due moved from Wed Nov 4, and windows may overlap) |
| A14 | 14, Mon Nov 2 | **Tue Nov 10, 11:59 pm** | 8, Mon Nov 2 to Tue Nov 10, overlapping A13 until Nov 6: it starts from the A13 reverse model, so it can begin once that model is trained (Eric, Oct 26: windows may overlap); may overlap the Nov 9 outline review, and ends before the Nov 11 proposal presentations (Eric, Oct 22) |
| A15 | 15, Wed Nov 4 | dropped | this term, folded into the Final Project (Eric, Oct 22): its page unpublished and kept for next year, its 20 points dropped, the speech pipeline named as a Path B example |

Costs accepted knowingly: Lecture 16 (Multimodal NMT) sits two weeks after Lecture 15 and
after the proposal presentations, the least damaging place for a gap since it is the most
self-contained lecture in the run; there is no catch-up session if the proposal presentations
overrun Nov 16; and the merged Dec 2 session has to be built from two 40-slide decks.

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
6. **Lecture 15** steps outside text into speech.
7. **Lecture 16 onward** is context, research, history and the final project.

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

What a student who did the work can do afterwards, and what shows it. From Lecture 13 on, the
lecture's objectives, one line each, the same words as its slide and its Learning Suite page;
the scope notes below carry the content, this carries the point.

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
| 8c | Finishing the Transformer | Describe multi-head attention, the encoder block and residual stream, and the decoder's masked self-attention and cross-attention; say how order gets into a model with no recurrence; place their own A8 BLEU on the measured capacity curve and say what capacity buys and when; weigh NMT's strengths against its deployment costs. | A8 report back; Quiz 8c |
| 9 | Morphology, subwords, terminology | Explain the fixed-vocabulary problem and how morphology makes vocabularies explode; segment a sentence with BPE by hand; measure `<unk>` before and after SentencePiece; retrain with one variable changed and attribute the BLEU difference to it. | A9 comparison |
| 10 | Quality estimation | Distinguish evaluation from estimation; say what COMET and COMET-QE score and what the number means; explain greedy against beam search and why a comparison must pin decoding. | A10; Quiz 10 |
| 11 | Neural QE and toolkits | Run COMET and COMETKiwi on their own A6 data and on their own A8 and A9 test output; reconcile them with BLEU, chrF and their human ranking; state what LLM-as-judge scores and its system- against segment-level caveat. | A11 |
| 12 | LLMs for MT; context | Prompt a decoder-only model to translate a low-resource language from a few example pairs, with no training; measure how much random and nearest examples improve translation, and tell a real difference from the spread between random draws; explain how examples in the prompt help or hurt when the model's weights never change, and what sentence-by-sentence translation cannot fix; give the model the sentence before, with its translation, and measure whether it resolves a pronoun or a repeated term the sentence alone leaves ambiguous, and why a score over the whole test set can miss it. | A12 chart and analysis |
| 13 | Low-resource strategies | Explain what makes a language qualify as low-resource; explain how back-translation turns target-side text into training pairs; explain sampling, and when it makes better back-translations than beam; name three other ways to augment scarce data; explain how Assignment 13 separates the effect of data from that of decoding. | A13 |
| 14 | Multilingual and zero-shot | Explain how a tag tells one model which language to write; explain why a tagged model can translate pairs it never trained on; explain how a low-resource language borrows from related ones in one model; explain when fine-tuning a pretrained model beats training from scratch; explain how Assignment 14 compares one two-way and two one-way models. | A14 |
| 15 | Speech-to-speech MT, dubbing and interpretation | Explain how cascade and end-to-end speech translation differ; explain where errors compound in an ASR, MT and TTS pipeline; explain what dubbing adds to translation: timing, length and voice; explain why live interpretation trades quality for latency. | Quiz 15; the Final Project's Path B example (A15 dropped this term) |
| 16 | Multimodal NMT; writing the report | Explain when an image helps translation, and how to test that it did; explain what an ablation with the wrong image, or none, shows; explain what each section of the ACL-style report must show. | Quiz 16; the Final Project's written report |
| 17 | Research and open problems | Explain an open problem in MT and the kind of work it needs; explain how one MATRIX Lab project frames and tests its question. | Quiz 17 |
| 18 | Word alignment and earlier methods | Explain how EM learns word alignments from sentence pairs; explain how word alignments relate to attention in NMT; explain what rule-based, example-based and phrase-based MT did; explain what NMT replaced from earlier methods, and what it kept. | Quiz 18 |
| 19 | MT in use | Explain where MT is used and what work it creates; explain how edit distance measures post-editing effort; explain when post-editing helps a translator, and when it does not. | class discussion; edit distance available as a Final Project measure |

### Notebook outcomes, and who does the work

For every notebook: the lesson, what the student must do by hand, what runs for them, and a
rating. **Low** means the result depends on something the student wrote, chose or predicted.
**Watch** means the notebook runs itself and the learning rests on the questions around it.
**High** means a student can produce the assignment's deliverable without understanding the
steps.

| notebook | the lesson | the student must | the notebook does | rating |
|---|---|---|---|---|
| `lecture-03-word-embeddings` | a shared multilingual space can be tested, not assumed | pick five words, read the matrix row by row, answer four questions; A3 adds the heat-map code, which is deliberately not given | embed and print the matrix | low |
| `lecture-04-regex-refresher` | read the patterns the cleaning steps are written in | write nine patterns, each stopping the notebook until written | check each against an expected output, refusing a greedy tag pattern and an unanchored start | low |
| `lecture-04-tmx-cleaning` | find data problems with code, through `repr` | predict two counts, write the line-break regex, write two cleaners, defend a keep-or-drop call | parse, write files, run the checker | low. The model for the course |
| `lecture-05-sentence-alignment` | length can align sentences and cannot see a deletion | change the priors and watch it fail, upload their A4 files, report c | aligner, scoring, the c and s² formulas | low |
| `lecture-06-mt-evaluation` | what BLEU and chrF count | write `bleu_from_parts`, record which system they prefer and which BLEU will before any score, predict at Parts 1 and 3 | the scoring, and the checks against SacreBLEU's own score | low |
| `lecture-06-mt-evaluation-homework` | the metrics on their own data | upload, sample, paste in ten web MT translations, rank them in `MY_RANKING` before any score prints | three scoring calls, and sentence-level chrF set beside the ranking | low; A6's substance is MTEval and the analysis |
| `02-evaluating-translations` (reading) | which metric to believe when they disagree | decide before scoring; four "Your turn" exercises | the examples | low |
| `lecture-07-toy-model` | a model can score perfectly on what it memorised and know nothing | choose the held-out phrase, predict, run Parts A and B, report back in class, write A7's two sentences; a contrast phrase runs as well | corpus, model, training, both scorings, the resume demonstration, the report printout | low |
| `03-train-tiny-model` (reading) | a saved model is weights without a vocabulary, so the vocabulary must be rebuilt exactly | read; run the three rebuilds | everything | low, as reading |
| `lecture-08a-a8-kickoff` | from cleaned corpus to a running, checkpointed model | type two file names; predict the first logged loss from ln(V), and the curve's last epochs; raise `EPOCHS` if validation loss is still falling | cap, dedupe, split, verification, nine files, the configuration, the training call, the resume, the scoring | **high** |
| `04-attention-and-alignment` (reading) | attention is measurable alignment | read Parts 1 to 5 and 7; run them | the ablation and the alignment accuracy | low, as reading |
| `lecture-08b-cross-attention` | attention can find the right words and the decoder still write the wrong ones | write `attention`, predict where three output pieces will look, say which prediction missed | the model, the capture of each head's inputs and weights, the map | low |
| `lecture-08c-attention-heads` | each head is sharp and the average is not; deeper layers spread out | write `split_heads` and `sharpness`, predict before the measurement, say which part of the prediction was wrong | the model, the attention step, the capture, the maps | low |
| `lecture-09-subword-tokenization` | a fixed vocabulary cannot say unseen words; subwords can | write the `<unk>` counter, pick the vocabulary size with a reason, predict A9's BLEU, report back, and reconcile the prediction in the A9 write-up | the SentencePiece training, the round trip, saving the settings to Drive, and in Part B the retraining and both scorings | low |
| `06-diagnosing-failures` (reading) | a procedure for a model that is not working | read once, then look symptoms up | five planted bugs and their checks | low; it is a lookup table |
| `01-data-and-vocab` (reference) | the library's view of a corpus | read | everything | low, as reference |
| `lecture-10-comet-install` | a learned metric, with and without a reference, and the records it scores | write comet_inputs (checked on known cases), predict which system each model prefers, set up the token | the install, both models' scoring, the summary A10 grades | low |
| `lecture-11-comet-on-a6` | whether metrics agree with each other and with a human, sentence by sentence | reuse comet_inputs, write pairwise_agreement (checked on a known case), predict, and explain the agreement in the summary | loading, all four metrics on every sentence of A6 and of the A8 test set, the two reports | low |
| `09-fine-tuning-nllb` (reading; GPU, and more memory than a T4) | a learning curve, and where more steps stop paying; forgetting, in the direction and the language not trained; what a learning rate set too high looks like | give A14's code and a third language they can read; predict the chrF after 100, 300 and 1,000 steps, and what one-way training does to the rest; read the results | the data, NLLB, scoring both ways, one resumable fine-tuning run scored at three budgets, the forgetting checks and the high-learning-rate run | low, as reading |
| `08-inference-and-beamsearch` (reading) | decoding is part of the system being measured | read | two decoders written to be read | low, as reading |
| `07-real-translations` (reading) | what an undertrained model looks like on unseen text | read; one cell trains a small model | everything | low, as reading |
| `lecture-12-llm-context` | in-context examples and what they buy, how many and which; then the sentence before, context for disambiguation, against a placebo pair | write split_pool_and_test and build_prompt (the method, each checked for sizes, reproducibility and contamination, and for the prompt's required contents, with and without the sentence before, its wording their own); predict the best context size and whether nearest examples beat random ones, then what the sentence before changes; reconcile both predictions with the results and the spread over three draws; find two of their own test sentences whose pronoun or term refers into the sentence before; upload the completed notebook | the LLM, three random draws and the nearest examples at each size, NLLB where it covers the language (Telugu), the sentence-before and placebo runs at k = 5 with the pronoun sentences marked, BLEU and chrF, the chart, the rerun record and the upload files | low |
| `lecture-13-back-translation` | synthetic pairs whose target side is real, and whether they help; how much of a change came from the data and how much from decoding; a difference tested, not only measured | extract monolingual text from their Church TM's unused segments, topped up from public text if short; choose their own reverse model or NLLB as back-translator, which back-translates by sampling (one retrain, Eric, Oct 22); write reverse and combine (the method, checked against known answers and the contamination check); state an "If ..., then ..." hypothesis and a prediction before retraining; say why sampling, from two of their own back-translated sentences; add Google back-translations to two helped and two hurt examples; upload the completed notebook | the reverse training (none with NLLB), the sampled back-translation with five greedy ones beside it, one retraining, A9 and A13 decoded greedily and with beam 5, BLEU, chrF, the paired bootstrap, COMET, the rerun record and the upload files | low |
| `lecture-14-multilingual` | a language code tells one model which way to translate, and why that makes zero-shot possible; whether one shared model matches two one-way models, decoding held fixed and the difference tested; what fine-tuning a pretrained model buys over the same model off the shelf and over a model trained from scratch | choose their language's code (borrowing one, with a reason, if NLLB does not cover it); write tag and both_directions (checked on a known case, for intermingling, and by the contamination check in both languages); state an "If ..., then ..." hypothesis and a prediction, and set a step budget, before training; choose a third language for the zero-shot test; explain the code from the model's output under the wrong one; upload the completed notebook | NLLB off the shelf, the fine-tuning (torchlingo_harness.nllb.fine_tune), each direction decoded greedily and with beam 5 against NLLB off the shelf and its one-way model (A9; the A13 reverse model), BLEU, chrF, the paired bootstrap, COMET, the wrong-code and third-language outputs, the rerun record and the upload files | low |

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
| A12 | `split_pool_and_test`, `build_prompt` (with the sentence before); two predictions | known cases; `check_contamination` on the split |
| A13 | `reverse`, `combine`; a prediction | known cases; `check_contamination` on both held-out sets |
| A14 | the language's code (a reasoned borrow if NLLB lacks it); `tag`, `both_directions`; a hypothesis, a prediction and a step budget; a third language for zero-shot | a known case; intermingling on the real split; `check_contamination` in both languages; codes checked against NLLB's list |

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

### 8c. Finishing the Transformer, the A8 Debrief, and the Subword Activity (F2026 `_v13`, 38 slides shown)
The by-hand solution; multi-head attention, the whole Transformer, the encoder block, the
residual stream and the decoder block; how order gets into a model with no recurrence (the
position is added to the embedding, once); deployment challenges, then pros and cons of NMT; the
A8 debrief around the capacity table; Part A of the Lecture 9 notebook, in class.
**Notebooks:** `lecture-08c-attention-heads` (activity): the student splits the projections into
heads, shown to be the model's in every encoder layer, and measures how averaging and depth blur
attention. `lecture-09-subword-tokenization` Part A (activity); `06-diagnosing-failures`
(reading): five questions to ask of a model that is not working.
**Sets:** A9.

### 9. Handling Morphology and Terminology in NMT (F2026 `_v16`, 38 slides shown)
The fixed vocabulary; morphology, one figure each for inflection, derivation, compounding and
agglutination; BPE and SentencePiece on real sentences, with the two pipelines as pseudocode;
characters; what changes in the model's shapes; terminology and getting terms into a neural
system. A pencil-and-paper "Subwords by Hand" exercise on a real 3,000-piece English and Spanish
BPE model, with its handout, is in the deck for next year or a Lecture 10 opener.
**Notebooks:** `lecture-09-subword-tokenization` (activity, and A9 itself). Part A, on a CPU:
the student writes the `<unk>` counter, trains SentencePiece on the training split, chooses the
vocabulary size with a reason, and predicts A9's BLEU; the settings go to Drive. Part B, on a GPU:
retrains with the pieces under A9's own run names and scores A8 and A9 with the same code.
`01-data-and-vocab` (reference): its vocabulary half stops at `<unk>`, where this lecture starts.
**Sets:** A9 (set at 8c): the same split as A8, the tokenizer the only change, both decoded and
scored the same way; the write-up reconciles the prediction with the result.

### 10. Overview of MT Quality Estimation (F2026 `_v2`, 37 slides shown; a redo is planned)
Evaluation against estimation, QE features and shared tasks, COMET and COMET-QE and how to read a
score; and the decoding block: greedy, beam search with a worked example, and why a comparison
must pin decoding. Taught Oct 14 from the `_v2` rebuild; Eric judged it text-heavy, out of order
and repetitive, and a redo for next year is planned around seven themes (decisions log, Oct 14;
the plan is `Working Notes/Lecture 10 Redo Plan_v1.md` in the course folder).
**Notebooks:** `lecture-10-comet-install` (homework, A10): the student writes `comet_inputs`,
predicts, and scores two systems with and without a reference. `08-inference-and-beamsearch`
(reading): greedy and beam search written out, and why a comparison holds decoding fixed.
`07-real-translations` (reading): a trained model on unseen text, greedy against beam.
**Sets:** A10 (6 points): the Hugging Face setup, including the COMETKiwi license; run the
notebook end to end; three write-up questions; two papers, one read carefully and one skimmed,
which prepare Lecture 11.

### 11. LLM Judges, Sentences and Spans (F2026, taught live Wed Oct 21; planned as a recording)
Renamed from last year's "Neural Quality Estimation and Evaluation Toolkits" (Eric, Oct 20): the
COMET material it shared with Lecture 10 stays in Lecture 10. The one sentence: every judge of
translation quality, an LLM included, is a metric, and it has to be tested against people. A
review of three Lecture 10 slides; an LLM judge as a metric (GEMBA-DA); the viewer as judge on
eight Spanish-to-English TED sentences against COMET, COMET-QE and an LLM, with and without the
reference (a fluent hallucination, a wrong source); people still deciding what ships (Kocmi et al.
2021); systems against sentences (GEMBA; Sun, Guzmán and Specia 2020); xCOMET's error spans and
its sentence-level evidence; AutoMQM. Closes by introducing the final project. Quiz 11 reads GEMBA
and xCOMET, one carefully and one skimmed.
**Notebooks:** `lecture-11-comet-on-a6` (homework, A11): COMET and COMETKiwi on the student's
A6 sentences beside BLEU and chrF; the student writes `pairwise_agreement`, which counts how
often two metrics prefer the same system, sentence by sentence. Step 6 turns the same metrics on
the student's own A8 and A9 test output.
**Sets:** A11 (30 points): the default COMET model and COMETKiwi on the A6 outputs, compared
with the BLEU, chrF and human-ranking numbers already collected, and on the student's own A8 and
A9 systems, compared with A9's BLEU; one page answering the notebook's five summary questions.
Extra credit for xCOMET.

### 12. Using LLMs for MT and Expanding Context Awareness (F2025, 32 slides)
Encoder-only, decoder-only and large language models; MT with decoder-only models;
document-level and context-aware MT; dropped and ambiguous pronouns; contrastive test sets.
**Notebooks:** `lecture-12-llm-context` (homework, A12): the student writes the context/test
split and the prompt; an open chat model translates at 0, 5, 10 and 20 examples.
**Sets:** A12: a non-MT model from the Hugging Face Hub, one of Efik, Kosraean, Palauan,
Pohnpeian, Telugu or Yapese (the data folder's list, which Learning Suite now matches), BLEU
against context size, and an analysis. The deck's own slide still lists last year's languages;
fix it at the rebuild.

### 13. Introduction to Low-Resource NMT Strategies (F2025, 30 slides)
What counts as low-resource, mitigation with and without LLMs, evaluation for low-resource
languages, data sources, data augmentation. The deck's back-translation workflow is written for
OpenNMT; the notebook replaces it. Candidate home for the NLLB spotlight (Open questions).
**Notebooks:** `lecture-13-back-translation` (homework, A13): from the A9 files, the student
writes the reversal and the combination; the reverse model's training (none with NLLB), the
sampled back-translation and one retraining run for them.
**Sets:** A13: back-translation from A9, compared with A9 on BLEU, chrF and COMET, decoded
greedily and with beam, with the paired bootstrap.

### 14. Multilingual NMT and Zero-shot NMT (F2025, 23 slides)
Bilingual against multilingual, how zero-shot works, tagging, NLLB, complete MNMT, why the
Church data suits it. The notebook replaces its handout, "MNMT Guide Using OpenNMT.docx".
**Notebooks:** `lecture-14-multilingual` (homework, A14), rebuilt on fine-tuning NLLB-200 (Eric,
Oct 25: go): from the A9 split, the student chooses their language's code (borrowing one if NLLB
does not cover it) and writes the tagging and the intermingling; the library fine-tunes NLLB
both ways at once, and the scoring runs for them. `09-fine-tuning-nllb` (reading, after A14):
how much fine-tuning is enough, and what it costs, a purpose of its own (Eric, Oct 25).
**Sets:** A14: NLLB fine-tuned in both directions on A9's split, directions intermingled,
separate test sets per direction, each direction compared with NLLB off the shelf and with its
one-way model (A9; the A13 reverse model) on BLEU, chrF and COMET, decoded greedily and with
beam, with the paired bootstrap; the wrong code, and a third language (zero-shot). The Final
Project page's Path A
example grows it (Eric, Oct 24): one multilingual model over English, the student's language
and two or three others from the Church data, against the one-way A9 model, with a zero-shot
comparison between two of the others.

Lectures from 15 on were renumbered Oct 26 (Eric), one number per session; last year's numbers
are given for finding the F2025 decks.

### 15. Speech-to-Speech MT, with Automatic Dubbing and Interpretation (F2025 Lectures 15 and 16, 33 and 34 slides; Wed Nov 4)
Spoken language translation, cascade against end-to-end, Whisper, speech data for projects; then
dubbing and live interpretation as a segment, not a lecture of its own: neural voices, automatic
video dubbing, Wav2Lip, HeyGen.
**Sets:** nothing this term. A15, an ASR, MT and TTS pipeline on a free Azure account, is
dropped and folded into the Final Project, whose Path B names a speech-to-speech pipeline as an
example (Eric, Oct 22); its page is kept for the next offering.

### 16. Multimodal NMT; then Writing the Final Project Report (F2025 Lecture 17, 42 slides, trimmed; plus last year's Lecture 21 writing slides; Wed Nov 18)
Whether visual context helps: the incongruent-image test, the MATRIX Lab's video-guided MT paper
(Sorenson and Shurtz, LREC-COLING 2024) and its ablation, and simultaneous translation, where an
image can help. The two reading reports that day are on this topic. Last year's logistics slides
come out. Then 15 minutes on writing, while students start their reports: Overleaf, the ACL
template, and what each section must show for the rubric, with the lab paper as the worked
example.

### 17. Research in the MATRIX Lab, and Open Problems in MT (F2025 Lectures 21 and 23; Mon Nov 30)
The lab research half of last year's Lecture 21 (Lawry Sorenson's CTC forced alignment, Ammon
Shurtz's phonetic transliteration for Thai, Lao and Khmer) and last year's Lecture 23, open
problems. No reading reports that day.

### 18. Word Alignment, and Previous MT Paradigms (F2025 Lectures 19 and 20, 39 and 44 slides; Wed Dec 2)
The IBM models, then the Vauquois triangle, RBMT, EBMT and phrase-based SMT, which word alignment
was built for; IBM Model 1 by hand and the full Vauquois tour shrink to examples. One deck of
about 30 slides at the rebuild; the Och and Ney and the phrase-based reading reports belong here.

### 19. MT in Use: Applications, Jobs and Post-Editing (F2025 Lecture 22, with Lecture 18's post-editing; Mon Dec 7)
Where MT is used, the industry and jobs; about 10 minutes from last year's Lecture 18 on
post-editing as how translators now work, and normalized edit distance as a measure a project
can use. The post-editing reading report belongs here. Last year's Lecture 18 (HAMT vs. MAHT,
interactive and adaptive MT, the in-class activity) is not taught this term (Eric, Oct 26).
Final project presentations close the semester, Dec 9 and the exam slot.

**The final project** is one Learning Suite page, "Final Project", in four sections (Proposal,
Presentation, Write-up, Submission), each linked from its gradebook entry. A student takes one of
two paths: **investigate a question** (a hypothesis, and one change against a named baseline) or
**build a component** (a user, a job, and what that user does today as the baseline). Both need:
- **people and a measure:** for text output, at least three people each judge the same 100 test
  sentences against the baseline in MTEval, as in A6, beside BLEU or chrF and COMET, and the
  write-up says whether the people and the metric pick the same winner;
- **a fair test:** held out, checked for overlap the way A8 checks, and the same for every
  system;
- **where it helped and where it hurt;**
- **reproducibility:** versions, seeds and decoding settings named.

All project work is due Thu Dec 10 at 11:59 pm; the presentations take the last class meeting, Wed Dec 9, and the final
exam slot, Wed Dec 16, 11:00 to 14:00.

---

## Notebook index

Every notebook in the repository, what it is for, and where it lands. Generated: the lecture,
role, assignment and CI columns come from each notebook's own metadata, and the purpose from
`notebooks` in `roadmap.yaml`, so the table cannot fall out of step with the notebooks.

| notebook | lecture | role | starts | purpose | CI |
|---|---|---|---|---|---|
| `lecture-03-word-embeddings` | 3 | activity | A3 | does the shared multilingual space line up across your languages | no: download, pip |
| `lecture-04-regex-refresher` | 4 | reference | | twenty-minute regex tour, ending with re.sub; nine exercises that stop until written | no: exercises |
| `lecture-04-tmx-cleaning` | 4 | activity | A4 | planted TMX problems, two parsers, a checker that names the cleaning step | no: pip, exercises |
| `01-data-and-vocab` | 4, 9 | reference | A9 | the library's view of corpus loading, from A5's two files; word-level vocab, stops at `<unk>`, which Lecture 9 takes up | yes |
| `lecture-05-sentence-alignment` | 5 | activity | A5 | estimate Gale-Church parameters on your language | no: pip, exercises |
| `lecture-06-mt-evaluation` | 6 | activity | A6 | BLEU's zero on two short sentences; BLEU rebuilt from its parts; your preference against BLEU's, on supplied data | no: exercises |
| `lecture-06-mt-evaluation-homework` | 6 | homework | A6 | the assignment's scaffold over your uploaded A5 corpus | no: colab |
| `02-evaluating-translations` | 6 | reading | A6, A8 | the library's out-of-class treatment of evaluation | yes |
| `lecture-07-toy-model` | 7 | activity | A7 | the three-part toy-model exercise; the student picks the held-out phrase, predicts, and a contrast phrase runs; resume demo | no: exercises |
| `03-train-tiny-model` | 7 | reading | A8 | what a checkpoint holds and what it does not; a rebuild that fails loudly, and one that loads cleanly and translates nonsense | yes |
| `lecture-08a-a8-kickoff` | 8a | activity | A8 | from your A5 corpus to a running, checkpointed, bucketed training job | no: colab, pip |
| `04-attention-and-alignment` | 8a, 8b | reading | A8 | with and without attention; did it learn the right alignment; Part 7 is the Transformer | yes |
| `lecture-08b-cross-attention` | 8b | activity | | one attention step written by the student and shown to be the trained model's, head by head; cross-attention read against a wrong translation | no: exercises |
| `05-transformer-attention` | 8b, 8c | reference | A8 | both of Lecture 8b's and 8c's notebooks in one worked example, without the exercises | yes |
| `lecture-08c-attention-heads` | 8c | activity | | the student's head split shown to be the model's in every encoder layer; heads read one by one, and sharpness measured against averaging and depth | no: exercises |
| `06-diagnosing-failures` | 8c | reading | A8 | five questions to ask of a model that is not working | yes |
| `lecture-09-subword-tokenization` | 9 | activity | A9 | `<unk>` before and after SentencePiece (the student's counter), a round trip, the student's vocabulary size and A9 prediction; Part B is A9 itself, retrained for as many epochs as A8 and scored | no: colab |
| `lecture-10-comet-install` | 10 | homework | A10 | install COMET; the student's comet_inputs; two systems scored with and without a reference | no: colab, hf-token, download, pip, exercises |
| `07-real-translations` | 10 | reading | A11 | a trained model on unseen talks; a comparable BLEU; greedy against beam with decoding held fixed | yes |
| `08-inference-and-beamsearch` | 10 | reading | A13, A14 | greedy against beam, written out and checked against the library; why a comparison holds decoding fixed, as A9 does | yes |
| `lecture-11-comet-on-a6` | 11 | homework | A11 | COMET and COMETKiwi on the student's A6 sentences, beside BLEU, chrF and their ranking, then on their own A8 and A9 systems | no: colab, hf-token, download, pip, exercises |
| `lecture-12-llm-context` | 12 | homework | A12 | in-context translation with 0 to 20 examples, on the student's A12 language | no: colab, download, exercises |
| `lecture-13-back-translation` | 13 | homework | A13 | back-translation end to end, from the A9 files and the student's own monolingual text | no: colab, exercises |
| `lecture-14-multilingual` | 14 | homework | A14 | NLLB-200 fine-tuned both ways on A9's split, against NLLB off the shelf and the student's own one-way models | no: colab, exercises |
| `09-fine-tuning-nllb` | 14 | reading | FP | how much fine-tuning is enough and what it costs, on the student's A8 split, after A14 | no: colab, download, exercises |

**Tutorial numbers** are in course order: 01 data, 02 evaluating, 03 checkpoints, 04 attention,
05 transformer-attention, 06 diagnosing, 07 real-translations, 08 beam search, 09 fine-tuning.
The "starts" column's `FP` is the Final Project, which has no assignment id.

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
| grader.exe | Lecture 4 | checking cleaned output. Students get binaries; the source is one `grader.py` in byu-matrix-lab/data-cleaning-pipeline-grader, private and unlicensed; see Open questions |
| MTEval (mteval.matrix.byu.edu) | Lecture 6 | human ranking; students self-register |
| SacreBLEU, chrF | Lecture 6 | automatic scoring, and again in 8, 9, 12, 13, 14 |
| TorchLingo | Lecture 6 | every model the students train (7, 8, 9, 13, 14), and the metrics and Colab setup in every course notebook that uses it. Notebooks install it from the repository's `course` branch, not PyPI, so a notebook and the library code it calls reach students together; `setup()` prints the exact version, which is what to ask for with a bug report |
| SentencePiece | Lecture 9 | subword tokenization |
| Hugging Face account and token | Lecture 10 | COMETKiwi, the reference-free COMET model in A10 and A11, is gated; the reference-based COMET model and Lecture 12's default LLM need no token |
| COMET / COMET-QE / xCOMET | Lecture 10 | neural evaluation, and again in 11, 13 and 14 |
| Azure free tier | Lecture 15 | ASR, MT and TTS for the speech pipeline |

---

## The TorchLingo pivot: what is settled

The course trains models with TorchLingo, an educational PyTorch NMT library built in the
MATRIX Lab. Five assignments train a model: the toy model at Lecture 7 (A7), A8, A9, A13 and
A14. They are cumulative in the artifact, so this is one crossing rather than five, and every
assignment after A9 starts from A9. All five have TorchLingo notebooks, A14's since Oct 22.
A14's Learning Suite text still describes OpenNMT, until it is rewritten against the notebook;
A13's was rewritten on Oct 22.

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
hand-off. Course work is tracked in one list both sessions share,
`torchlingo-private/notes/course/TASKS.md` (since Oct 25); library and notebook engineering in
this repository's `notes/TASKS.md`.

---

## Open questions

- **Does "about 35 epochs" hold for a student with 30K pairs?** The 30-epoch convergence was
  measured at 100K pairs with a subword vocabulary. Revisit once the A5 audit says how many
  students are well below 100K; the audit itself still needs the submissions staged somewhere
  readable. The instruction's escape hatch is "keep going while validation loss falls".
- **Can students run the grader from source?** The source exists:
  `byu-matrix-lab/data-cleaning-pipeline-grader`, one `grader.py` run as
  `python3 grader.py source target output`, which would end the download-and-override-Gatekeeper
  step. Two things block pointing the Lecture 4 and 5 decks at it, both Eric's: the repository is
  private, so students cannot clone it, and it has no license. It was last pushed in September
  2024, so its checks need comparing with what Lectures 4 and 5 ask for first. Until then
  students use the four PyInstaller binaries.
- **An NLLB spotlight lecture** (Eric, Sep 27): what makes NLLB special beyond its data, namely
  the sparsely gated mixture-of-experts architecture with its regularisation against low-resource
  pairs overfitting, and the deliberate training curriculum, which has no coverage in the course.
  Home is Lecture 13 or 14. The paper's specifics are verified before any number reaches a slide.
- **Learning Suite's assignment texts.** A14 still carries last year's OpenNMT directions,
  rewritten against its notebook once that lands (A13's was, Oct 22). A7's due time differs between Learning Suite and the
  deck.
- **Lecture 6 is overloaded** and nothing has been done about it.
- **Lecture 18 (word alignment and earlier methods) needs rescoping** around the material now in
  Lecture 5, and has no notebook.
- **What Lectures 15 to 19 and the final project need from the repository**, held until Lectures
  11 to 13 are done (Eric, Oct 19). None has a notebook, by design, but three may want code:
  Lecture 19's post-editing segment measures normalized edit distance (a small notebook, or
  TorchLingo's TER?); Lecture 18's IBM Model 1 worked by hand, set beside 8a's attention
  alignment, once the lecture is rescoped; and a "starting a project from your own A9 or A14
  model" page for the final projects, proposed Nov 16.
- **Do lectures after 12 have students prompt an LLM?** Eric, Oct 19: "almost certainly yes", to
  be confirmed against these scope notes. If a second notebook does, A12's prompting code becomes
  a TorchLingo helper; until then it stays in A12's notebook.
- **Deck links to personal Google Drive URLs.** Lecture 4 links the regex notebook twice and its
  own activity once; Lecture 12 links its assignment notebook. All die with the account. Each is
  now a Colab badge in the repository, but the Drive copies must stay alive while students are in
  them.

---

## Decisions log

Newest first. Each is a decision that changed the schedule, an assignment, or a notebook's
place, with who made it.

- **Oct 28.** Each lecture's outcomes are its objectives slide, one line each, worded "Explain
  ...", and the lecture's Learning Suite page lists the same words; the Lecture outcomes table
  above holds them for Lectures 13 to 19 (13 and 14 as taught or built; 15 to 19 drafts, refined
  as each lecture is built) (Eric). Lecture quizzes have at most 8 questions, in generic
  wording, following the lecture's current outcomes, this offering and the next; the quiz review
  in class is one slide on the most-missed important question (Eric). Lecture Quiz 12's Q5
  thrown out, two options being untrue (Eric).
- **Oct 26, late.** Lectures from 15 on renumbered, one number per session (Eric): 15 speech
  with dubbing and interpretation as a segment (Nov 4); 16 Multimodal NMT, trimmed, then 15
  minutes on writing the report, moved up from last year's Lecture 21 so it lands before students
  write (Nov 18); 17 MATRIX Lab research and open problems (Nov 30); 18 word alignment and
  earlier methods (Dec 2); 19 MT in use, applications, jobs and post-editing (Dec 7). Last year's
  Lecture 18 (HAMT vs. MAHT) is dropped as a lecture (Eric), its post-editing kept as a segment
  of 19. Reading reports follow their topics: the alignment report moves from Nov 30 to Dec 2,
  the post-editing report to Dec 7. Quizzes renamed to match.
- **Oct 26, evening.** Final project presentations take two days (Steve Richardson: they need
  the last class day; Eric: make room on Dec 9 and give each student more time): the last class
  meeting, Wed Dec 9, and the final exam slot. Each slot is 10 minutes, a 7-minute talk, 2 for
  questions and 1 to hand off (Eric), so 7 talks fit Dec 9 and 18 the exam slot, for 24
  students. Lectures 22 and 23 move to Dec 7 with Lecture 21 (superseded the same day by the
  renumbering, the entry above). The presentation part stays due
  with the exam slot; Dec 9 presenters upload slides by 10:00 am that day.
- **Oct 26, later.** A12 is due Mon Nov 2 at 10:00 and A13 Fri Nov 6 at 11:59 pm; A14 stays
  Tue Nov 10 (Eric). A12 and the Lecture 12 activity use non-reasoning LLMs only (Eric): a
  reasoning model's chain of thought is a more powerful mechanism than in-context learning and
  would confound it. Assignment windows may now overlap (Eric, replacing Oct 22's "no two windows
  overlap"): A13 is open from Lecture 13 (Oct 28) and A14 from Lecture 14 (Nov 2), so each gets
  eight or nine days instead of four. Students start A13's training run while finishing A12, and
  A14 once A13's reverse model is trained. The A15 gradebook item is deleted (Eric; no scores or
  submissions). Roadmap A1 to A3 due dates corrected to Learning Suite's (Sep 9, 14, 16).
- **Oct 26.** The tutorials are **required reading** (Eric). They still carry no points, so
  the assignment notebooks stay self-sufficient, as the safety net for a student who skips
  them. Also Oct 26 (Eric): this roadmap is the source for each assignment's points and
  stretch points, written in `roadmap.yaml` before the Learning Suite page and the gradebook;
  and for the Assignments page's two charts, now generated from it rather than hand-kept.
  Tutorial 4 leads to A8, Tutorial 8 to A13 and A14, and Tutorial 9 to the Final Project.
- **Oct 25.** One task list for course work (Eric): `torchlingo-private/notes/course/TASKS.md`,
  shared by both sessions, with the course side's log and next-offering notes beside it. The
  public `notes/TASKS.md` keeps library and notebook engineering.
- **Oct 25.** A14 is rebuilt on fine-tuning NLLB-200, both directions in one run (Eric: go, with
  one Colab run before A14 opens Nov 4). Two short real runs here made the case: German 25.8 to
  32.4 BLEU, and Efik, borrowing Igbo's code, 3.6 to 18.5, after 300 steps each. A14 keeps its
  three outcomes, with NLLB's language codes as the tags, and adds a fourth, what fine-tuning
  buys over the model off the shelf and over A9 trained from scratch; COMET is in. Tutorial 9,
  first written as fine-tuning one direction step by step, was given a purpose no assignment has
  the same day, so that it and A14 do not look alike: how much fine-tuning is enough, and what
  it costs (Eric). Tutorials carry no points, so students pressed for time skip them, and the
  assignment notebooks are load-bearing (Eric). Eric's frame for the notebooks: not a
  deep-learning class but an application of deep learning, so students are told the ingredients
  while their decisions and code stay on the translation problem.
- **Oct 24.** A12 gains a fourth outcome, context for disambiguation beside in-context learning
  (Eric: "both in-context learning and (separately) contextual disambiguation"). The Lecture 12
  notebook's new Step 6 gives the model the sentence before each test sentence, with its
  translation, against a placebo pair in the same slot; nine questions; A12 is 30 points (Eric).
  Also Oct 24 (Eric):
  - **Final Project, Path A example:** the page's Path A takes one from Lecture 14 (Eric: "a
    great final project idea. path A even"). One multilingual model over English, the student's
    language and two or three others from the Church data, against the one-way A9 model, with a
    zero-shot comparison between two of the others.
  - **Presentations stay in the final exam slot.** Moving them into class would cost Lectures 21
    to 23 and a week of project time.
  - **Shorter assignment names** in Learning Suite, numbers unchanged: "Assignment 12 - Context in
    LLM Translation", "Assignment 14 - Multilingual NMT, Both Directions", and so on.
  - **A stretch task** on each assignment, about 20% of its points, with AI at Level 2 (Reviewer)
    for the stretch task only. Live for A12, A13 and the Paper Review; A14's follows the Oct 29
    decision below.
  - **A fine-tuning tutorial**, Tutorial 9 on NLLB, to fill the course's largest gap: nothing
    teaches fine-tuning a pretrained model, which is how most Final Projects will get their
    results. A14 is rebuilt on it if it is ready by Thu Oct 29 (go or no-go then).
- **Oct 23.** The Final Project has seven parts, each its own gradebook entry, "Final Project 1"
  to "Final Project 7" (Eric):

  | Part | Points | When |
  |---|---|---|
  | 1. Proposal outline review | 5 | Nov 9 |
  | 2. Proposal | 20 | Nov 11, 10:00 |
  | 3. Checkpoint review | 5 | Nov 23 |
  | 4. Written report | 40 | Dec 10 |
  | 5. Submission | 35 | Dec 10 |
  | 6. Presentation | 20 | Dec 16 |
  | 7. Peer feedback | 5 | Dec 16 |

  - The project itself is 100: parts 3 to 6. Parts 1, 2 and 7 are graded separately.
  - The proposal moves from Nov 9 to Nov 11.
  - The written report (was "write-up") uses the ACL template only.
  - Presentations are at most 6 minutes, 2 of them for questions.
  - Peer feedback is new: four questions on every other presentation, by name, on a Google Form.
  - The optional LLM-judge prompts moved to their own page.
  - The checkpoint reviews take Mon Nov 23, with no lecture. Lecture 18 moves to Nov 30, and
    Lectures 19 and 20 to Dec 2.
  - No A14 timing run is needed: "plan for about twice your A9 training" stays (Eric).
- **Oct 22.** No two assignment windows overlap (Eric). A12 is due Fri Oct 30 (moved for
  breathing room, Eric), A13 Wed Nov 4 and A14 Tue Nov 10, all at 11:59 pm, each opening when
  the one before closes; A14 may overlap the Nov 9 outline review (Eric) and ends before the
  Nov 11 proposal presentations. The Final Project Proposal is due Mon Nov 9 at 10:00, the
  outline-review day, with the presentations on Nov 11 and Nov 16. A15 is folded into the Final
  Project (Eric): its page unpublished and kept for next year, its 20 points dropped, the speech
  pipeline named as a Path B example. A13 is simplified after Eric asked whether any assignment
  had grown too complicated for its outcomes: one retrain on sampled back-translations instead
  of greedy and sampled (the Oct 21 entry's three trainings become two, one with NLLB), the
  MTEval partner exchange dropped, seven questions, 40 points to 30 (Eric). A12 keeps its work;
  its outcomes go from five to three, and both pages' outcomes now state the low-resource lesson
  (Eric). The assignment arc plan is retired into the roadmap (Eric).
- **Oct 21, evening.** The assignment arc plan for A12 to A15 goes ahead, every addition
  required (Eric: "All of our additions are important and required"), and every notebook
  assignment submits its completed notebook (Eric). A12's notebook adds chrF beside BLEU, three
  seeded draws of the examples with their mean and spread, a nearest-examples condition from a
  500-pair pool, NLLB as a comparison row where it covers the language (Telugu, of the six), a
  pronoun-or-repeated-term question, and a rerun record; a student who already finished reruns
  with it (Eric, in the plan). TorchLingo gains sampled and noised-beam decoding for A13's
  back-translation (Eric: "Then it can be an option in A13"). For A13's rewrite: the embedding
  filter is cut (Eric); comparing two back-translation methods is required, three trainings in
  all, two with NLLB (Eric), and the two are greedy and sampling, for every student: beam
  back-translates one sentence at a time, about 10 hours for 100,000 (Eric: "greedy against
  sampling for the sake of time"); beam stays in scoring the test set; NLLB is called from `torchlingo_harness`, an accessory shipped with
  TorchLingo and kept out of the library (Eric: "keep TorchLingo lean and pedagogical"). The A15
  notebook is deferred to `TASKS.md` (Eric).
- **Oct 21, afternoon.** A11 also scores the student's own A8 and A9 systems (Eric: "add the
  requirement to score on A8 and A9"): COMET, COMETKiwi, BLEU and chrF on their A8 test set, in a
  new Step 6 of the notebook, and a fifth summary question on whether COMET agrees with BLEU that
  the pieces helped. Two rubric rows at 2 points each; A11 goes from 26 points to 30. A9's Part B
  now saves its re-translation of the A8 model, `test-a8-again.hyp`, which A11 prefers over A8's
  `test.hyp` (Eric).
- **Oct 21, later.** A11's estimator switches from COMET-QE-DA to COMETKiwi (wmt22-cometkiwi-da),
  the model A10 already runs (Eric): one estimator across both assignments, the newer and stronger
  one; A11 builds on the A10 Hugging Face setup. The in-class judges in Lecture 11 stay COMET-QE.
  Content quizzes are renamed "Lecture Quiz N" in Learning Suite (Eric). Every deck's objectives
  slide, and each content page, now states learning outcomes ("By the end of class, you can:")
  rather than a list of topics (Eric).
- **Oct 21.** Lecture 11 is taught live today instead of recorded, with in-class human scoring
  (the room is the human judge), and the calendar shifts one class: Oct 19 becomes a review day;
  Lecture 12 moves to Mon Oct 26, 13 to Wed Oct 28, 14 to Mon Nov 2, and Lectures 15 and 16
  merge on Wed Nov 4, so proposal week and everything after it stay put. Assignment due dates
  stay as they were, so A12 and A14 now have two days after their lectures (their pages are open
  earlier). Quizzes follow the lectures: Quiz 11 closes Oct 26, Quiz 12 Oct 28, Quiz 13 Nov 2,
  Quiz 14 Nov 4, and Quizzes 15 and 16 merge into one closing Nov 9; no content quiz allows late
  submission (Eric: results are discussed in class the day they are due). Attendance quizzes are
  renamed to follow the lectures. Reading reports keep their sign-up-sheet dates. The final
  project allows AI use Level 2, Reviewer, and Path B becomes "a meaningful application of MT
  for a chosen audience" (Eric).
- **Oct 20.** The final project is one Learning Suite page with two paths, investigate a question
  or build a component, and both judge their result with people as well as a metric (see
  Lectures 21 to 23's scope note). The four separate project pages are gone. Work is due Thu Dec
  10; the presentations move to the final exam slot, Wed Dec 16, 11:00 to 14:00 (Cowork, with
  Eric).
- **Oct 19.** Lecture 11 is recorded, not taught in the room, and its "be the judge" activity
  turns around to Spanish into English, with the English original as the reference, so every
  student judges as a monolingual rater against a reference (Eric: not every student reads
  Spanish). Its message: every judge of translation quality, an LLM included, is a metric and has
  to be tested against people. Both sessions stay on Lectures 11 to 13 and A11 to A13 for now;
  what later lectures need goes into the open questions below as notes (Eric).
- **Oct 15.** A13's monolingual text comes from the student's Church translation memory first
  (Eric: "they have more data from the Church TMs"): the target side of segments A5 left out,
  topped up from public text (Wikipedia, MADLAD-400 or CC-100, OPUS) only if that runs short.
  Both are deduplicated against the A8/A9 split, never carved from A8's training targets. There
  is no minimum; the student reports how many came from where. The A13 notebook follows on Oct 19.
- **Oct 14, evening.** Lecture 10, taught today from its rebuild, is to be redone for next year
  (Eric: text-heavy, snapshots of old slides, out of order, repetitive). The redo's themes, from
  an interview: open on "how do you improve on your A9 results?" with decoding as the first
  answer; evaluation needs a reference and estimation does not, evaluation taught first; both are
  plain supervised learning with a human label as the supervision signal, shown with one native
  regressor diagram; three ways to get a label from a human (post-edits, DA, MQM), each setting a
  granularity, each introduced by the decision it serves; a feature-based estimator worked on
  English and Spanish slides after a class brainstorm, with no system names; a COMET score is a
  predicted human judgment; and evaluating metrics and estimators as the close, the bridge to A11.
  About 50 to 55 minutes. Lecture 11 does not repeat what Lecture 10 now owns, and its rebuild
  waits for its own themes interview.
  A10 gains the Hugging Face setup with the COMETKiwi license and the notebook's three write-up
  questions, 6 points (Eric: 3 + 3). A11 is rewritten against its notebook: "Install COMET" goes
  from the directions and the title, since A10 did it (Eric); the notebook's four summary
  questions are added, the last three at 2 points each, 26 in all (Eric: add points). The two A10
  papers prepare Lecture 11, one read carefully and one skimmed, the student's choice (Eric, as he
  told the class); Quiz 11's two papers, GEMBA and xCOMET, follow the same rule, one question each.
  The desktop roadmap is retired (Eric: one roadmap, in the repository).
- **Oct 12.** Three slide moves across decks (Eric): "The Model Behind the Attention Maps" from 8b
  to 8a, after the maps it describes; how order gets into a model with no recurrence, and the
  deployment challenges, from 9 to 8c. The course runs at Level 1, Reference, of the CS
  department's AI policy, stated in the syllabus and on Lecture 1's assignments slide and not
  repeated on later decks (Eric); the lecture quizzes are open book, now in the syllabus too. The
  syllabus gains a Colab computing note (Colab Pro's compute units; disconnect the GPU runtime
  when a run is done), also sent to the class as an announcement.
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
| 4 | Data Preparation for MT Training | `lecture-04-regex-refresher` (reference) [6], `lecture-04-tmx-cleaning` (activity) [7] | `01-data-and-vocab` (reference) [1] |
| 5 | Data Preparation for MT Training, Part 2 | `lecture-05-sentence-alignment` (activity) | — |
| 6 | Automatic and Human MT Evaluation | `lecture-06-mt-evaluation-homework` (homework) [8], `lecture-06-mt-evaluation` (activity) [9] | `02-evaluating-translations` (reading) [2] |
| 7 | Research Paper Reviews; Intro to Neural Networks | `lecture-07-toy-model` (activity) [10] | `03-train-tiny-model` (reading) [3] |
| 8a | Neural MT: Encoder-Decoder, and Why Attention Was Invented | `lecture-08a-a8-kickoff` (activity) [11] | `04-attention-and-alignment` (reading) |
| 8b | Neural MT: The Transformer | `lecture-08b-cross-attention` (activity) [12] | `04-attention-and-alignment` (reading), `05-transformer-attention` (reference) |
| 8c | Finishing the Transformer, the A8 Debrief, and the Subword Activity | `lecture-08c-attention-heads` (activity) [13] | `05-transformer-attention` (reference), `06-diagnosing-failures` (reading) |
| 9 | Handling Morphology and Terminology in NMT | `lecture-09-subword-tokenization` (activity) [14] | `01-data-and-vocab` (reference) [1] |
| 10 | Overview of MT Quality Estimation | `lecture-10-comet-install` (homework) [15] | `07-real-translations` (reading) [4], `08-inference-and-beamsearch` (reading) |
| 11 | LLM Judges, Sentences and Spans | `lecture-11-comet-on-a6` (homework) [16] | — |
| 12 | Using LLMs for MT; Expanding Context Awareness | `lecture-12-llm-context` (homework) [17] | — |
| 13 | Strategies for NMT of Low-Resource Languages | `lecture-13-back-translation` (homework) [18] | — |
| 14 | Multilingual NMT and "Zero-shot" NMT | `lecture-14-multilingual` (homework) [19] | `09-fine-tuning-nllb` (reading) [5] |
| 15 | Speech-to-Speech MT, with Automatic Dubbing and Interpretation | — | — |
| 16 | Multimodal NMT; then Writing the Final Project Report (15 minutes) | — | — |
| 17 | Research in the MATRIX Lab, and Open Problems in MT | — | — |
| 18 | Word Alignment, and What We Can Learn from Previous MT Methods: RBMT, KBMT, EBMT, PBSMT, SBSMT | — | — |
| 19 | MT in Use: Applications, Jobs and Post-Editing | — | — |

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
- [5] **`09-fine-tuning-nllb`** — How much fine-tuning is enough, and what it costs (Eric,
  Oct 25: a purpose no assignment has; A14 keeps the language codes, borrowing, baseline
  and comparisons). On the student's A8 split, or a built-in English-Spanish sample, with
  the NLLB code A14 used and a third language the student can read. NLLB-200 is scored
  before training (chrF both ways, five sentences into the third language); the student
  predicts the chrF after 100, 300 and 1,000 steps and what one-direction training will do
  to the other direction and the third language, which stops the notebook while empty;
  torchlingo_harness.nllb.fine_tune trains English into the student's language, resuming
  one run through the three budgets, scored at each; forgetting is measured on the
  direction not trained and read in the third language; a 100-step run at a learning rate
  thirty times higher, from the original weights, shows what too high looks like. Results
  persist in t9-results.json on Drive. Read after A14, whose code it uses, and before a
  Final Project that fine-tunes, so it leads to FP, the Final Project. Needs a GPU with
  more memory than a T4, and Drive, so CI does not execute it; tests/test_tutorial_09.py
  runs its cells with NLLB stubbed.
- [6] **`lecture-04-regex-refresher`** — A twenty-minute tour of Python regular
  expressions, which the sixteen cleaning steps are written in, ending with re.sub for
  collapsing whitespace and stripping tags. Lecture 4's deck links it and Learning Suite
  posts it on the Lecture 3 tab, so it is offered ahead of Lecture 4 rather than used
  inside it. Each exercise stops until its pattern is written, then checks it;
  tests/test_lecture_04.py runs it with answers in.
- [7] **`lecture-04-tmx-cleaning`** — In-class dry run of A4 on a ten-unit TMX sample with
  planted problems. It stops at a line-count prediction and at the line-break PATTERN
  until the student writes them; then a checker names the remaining problems by cleaning
  step and the student writes two string cleaners. Installs Translate Toolkit, so CI does
  not execute it; tests/test_lecture_04.py runs its cells with answers written in and the
  parser stubbed by xml.etree.
- [8] **`lecture-06-mt-evaluation-homework`** — The A6 homework: ten sentences from the
  student's own Assignment 5 corpus, uploaded through the Colab file picker, translated by
  a web MT system and pasted in, ranked by hand into MY_RANKING (which stops the notebook
  while empty) before any score prints, then scored with BLEU, chrF and TER, with
  sentence-level chrF set beside the ranking. Needs the Colab picker, so CI does not
  execute it; tests/test_lecture_06.py runs its cells with the upload skipped and files,
  translations and a ranking written in.
- [9] **`lecture-06-mt-evaluation`** — Lecture 6's in-class activity on supplied data:
  BLEU's zero on two short sentences, then the student writes bleu_from_parts (BLEU from
  bleu.precisions and bleu.bp, checked on a case worked on paper, on a brevity penalty and
  on Part 1's zero, then against BLEU's own score on both Part 2 systems); reads five
  pairs and records which system they would rank higher and which BLEU will prefer before
  any score; and predicts BLEU at three words kept before the brevity-penalty table. Each
  prediction stops the notebook while empty. The homework on the student's own data is
  lecture-06-mt-evaluation-homework. CI does not execute it as shipped;
  tests/test_lecture_06.py runs its cells with an answer and predictions written in.
- [10] **`lecture-07-toy-model`** — Lecture 7's three-part exercise: Part A in class
  (install, train the toy model, read the loss curve against ln(V)), Part B outside class
  (hold out a phrase of the student's choice, predict, run a contrast phrase, score seen
  against unseen; watch a run resume after a faked disconnect), Part C the report handed
  in as A7. Part B stops while HELD_OUT is unset or the prediction is empty, so CI does
  not execute it; tests/test_lecture_07.py runs its cells with both filled in.
- [11] **`lecture-08a-a8-kickoff`** — The in-class start of Assignment 8: from the
  student's Assignment 5 corpus to a training run already going when class ends. The
  handout is the authority; this is the executable path through it, and every threshold is
  quoted from the handout in one cell. Cannot run in CI: it mounts Drive and needs a GPU.
- [12] **`lecture-08b-cross-attention`** — Lecture 8b's activity, the cross-attention half
  of what was tutorial 5. The student writes attention, one scaled dot-product step,
  checked on a case worked on paper (one that also catches a missing √d_k) and then
  against the trained model's own weights in all four heads of the last decoder layer's
  cross-attention; places it among the three attentions and their weight shapes; then
  predicts where three output pieces will look, which stops the notebook while empty,
  reads the averaged map against the translation (the right words found, the wrong Spanish
  written) and records which prediction missed. tests/test_lecture_08b.py runs it with an
  answer written in.
- [13] **`lecture-08c-attention-heads`** — Lecture 8c's activity, the self-attention half
  of what was tutorial 5, on the same model and sentence as Lecture 8b's, with Lecture
  8b's attention step given. The student writes split_heads, checked on numbers readable
  by eye and then against the model's own weights in every encoder layer, and sharpness,
  checked on paper cases; predicts, which stops the notebook while empty, then reads the
  first layer head by head, measures how averaging and depth blur attention, and records
  which part of the prediction missed. tests/test_lecture_08c.py runs it with answers
  written in.
- [14] **`lecture-09-subword-tokenization`** — Lecture 9's in-class activity and, in Part
  B, Assignment 9 itself. Part A (CPU, in class): SentencePiece fit on the student's A8
  training split; the student's own unknown_rate counter, checked on three cases, under
  words and under pieces; a round trip; a vocabulary size chosen with a reason; A8's BLEU
  from test.hyp, then an A9 BLEU prediction in its own cell; the settings Part B needs
  saved to Drive (a9-settings.json). Part B (GPU, outside class): trains for as many
  epochs as A8 did, read from A8's checkpoint, under its own run names (assignment-9,
  best-a9/, test-a9.hyp); re-scores the saved A8 model with the same code, saving its
  translations as test-a8-again.hyp, which A11 scores with COMET; prints the write-up
  numbers; says to disconnect the GPU. Reads the A8 split from Drive, so it cannot run in
  CI; tests/test_lecture_09_part_a.py and _part_b.py run it.
- [15] **`lecture-10-comet-install`** — A10: installs COMET; the student writes
  comet_inputs, the records COMET scores (reused in A11 on the A6 sentences), checked on
  known cases, and a prediction, which stops the run while empty; then the reference-based
  model and COMETKiwi (whose license the student accepts on Hugging Face, then logs in
  with a token) score two systems through COMET's Python API; the last cell prints what
  A10 grades, followed by three write-up questions: the prediction against the result,
  what a COMET score means, and evaluation against estimation. tests/test_lecture_10.py
  runs its cells with COMET stubbed.
- [16] **`lecture-11-comet-on-a6`** — A11: COMET (reference-based) and COMETKiwi, the
  estimator from A10, on the student's A6 sentences beside BLEU and chrF, then on their
  own A8 and A9 systems. Step 1 reads the student's MTEval results file from A6 (a header
  naming the two systems, a rank per system per ranked sentence, ties allowed), joins it
  to the sources on whitespace-normalized text, and computes the average ranks. The
  student pastes comet_inputs from A10 and writes pairwise_agreement, the
  sentence-by-sentence measure of whether two metrics prefer the same system, each checked
  on a known case; a prediction, which stops the run while empty. COMETKiwi is gated, so a
  login cell reads the A10 Hugging Face token from Colab Secrets and, when it is missing,
  names Notebook access, which is set per notebook; the scoring cell names the license
  when the download fails. The report gives system scores, each metric's preferred system
  against the A6 human average ranks, and agreement counts, between metrics and between
  each metric and the student's ranks, human ties left out, read from a11-notes.json,
  which the scoring cell saves to Drive so the report survives a dropped session, and
  which it refuses if saved before the switch to COMETKiwi. Step 6 scores the student's A8
  and A9 translations of their A8 test set (MyDrive/CS479/assignment8: test.tsv,
  test-a9.hyp, and test-a8-again.hyp if A9 saved it, else test.hyp) with all four metrics,
  BLEU as A9 scored it, saved to assignment11/a11-own-notes.json; its report prints which
  system each metric prefers, COMET-chrF sentence agreement and the first three
  disagreements. Five summary questions follow, the fourth on LLM-as-judge and system-
  against segment-level agreement, the fifth on the student's own systems.
  tests/test_lecture_11.py runs its cells with COMET stubbed.
- [17] **`lecture-12-llm-context`** — Assignment 12, runnable end to end. Runs on a
  built-in English-Spanish sample from data/example.tsv until the student sets LANGUAGE
  and DATA_DIR, together. The student writes split_pool_and_test (checked on the alphabet,
  including that a different seed gives a different split, then by check_contamination on
  the real split) and build_prompt (checked on a known case), then a prediction, which
  stops the notebook while empty; an open instruction-tuned LLM that does not reason
  (default Qwen2.5-1.5B-Instruct, greedy; the model cell refuses the known reasoning
  models by name, turns Qwen3's thinking off and stops on a reply holding <think>)
  translates 50 test sentences with no examples, then at 5, 10 and 20 in-context examples
  drawn from a 500-pair pool three ways at random (three seeds, nested) and once as the
  pool pairs nearest each test sentence (character-trigram overlap, after Moslem et al.);
  where NLLB-200 covers the language (of the A12 languages, Telugu only) it translates
  them too, greedily. Every run is scored with BLEU and chrF and their signatures into
  scores.tsv; the report gives each size's mean and spread over the draws,
  scores-by-context.png charts both metrics, and a12-rerun.json records the model, the
  non-reasoning check, seeds, decoding and versions. Step 6 then adds context of the other
  kind: after a second prediction, the 5 examples of draw 1 plus the sentence before each
  test sentence (the line above it in the files, with its translation; build_prompt's
  before argument, checked in Step 2) against a placebo (a different pool pair for each
  sentence, the next in draw 1's order after its five, in the same slot, same label),
  written with every pronoun-bearing test sentence marked to before-5.tsv. The write-up
  asks whether a difference between sizes exceeds the spread, nearest against random, LLM
  against NLLB, whether the sentence before beat the placebo by more than the spread, and
  whether it settled a pronoun or term in two of the student's own test sentences; the
  completed notebook is uploaded with it. Needs a GPU and Drive, so CI does not execute
  it; tests/test_lecture_12.py runs its cells with both models stubbed.
- [18] **`lecture-13-back-translation`** — Assignment 13 end to end, as a small Path A,
  starting from what Lecture 9's Part B leaves on Drive (A9's split, spm.model, best-a9/,
  test-a9.hyp, and A9's training checkpoint, from which every training takes its epoch
  count) plus the student's monolingual text: mono-tm.txt from their Church TM's unused
  segments, topped up by an optional mono-public.txt (OPUS, Wikipedia, MADLAD-400,
  CC-100), with a note on domain; Step 1 removes sentences in the split and repeats, runs
  check_contamination against the test set, and saves how many came from each file. The
  student chooses the back-translator, their own reverse model or NLLB-200 (through the
  accessory torchlingo_harness.nllb, where it covers the language, one training fewer),
  and writes reverse, which turns the split around. Every student back-translates once, by
  sampling (top_k 10 advised with NLLB; Eric, Oct 22: one retrain, not one per method),
  written resumably to Drive, with five sentences also made greedily and printed beside
  the samples. They write combine, checked on a known case and by check_contamination
  against validation and test; state an "If ..., then ..." hypothesis and a prediction,
  which stop the run while empty; and the notebook retrains English-to-X once: two
  trainings, one with NLLB. A9 and A13 decode the test set greedily and with beam 5, four
  outputs; BLEU (A9's) and chrF with sacreBLEU's paired bootstrap against A9 (identical
  outputs reported at p = 1); COMET on all four. Seven write-up questions, no partner
  step. The write-up block reads from Drive alone (a13-notes.json), prints two sentences
  where back-translation helped most and two where it hurt, and saves a13-rerun.json; the
  completed notebook is uploaded. Needs a GPU and Drive, so CI does not execute it;
  tests/test_lecture_13.py runs its cells at toy size, with NLLB stubbed.
- [19] **`lecture-14-multilingual`** — Assignment 14, rebuilt on fine-tuning NLLB-200
  (Eric, Oct 25: go), starting from what Lecture 9's Part B leaves on Drive (A9's split,
  spm.model, a9-settings.json, best-a9/) and, for the comparison into English, A13's
  best-a13-reverse/ when present. The student sets LANGUAGE (its NLLB code from a table
  checked against NLLB's tokenizer; Hmong is the uncovered A8 language, and the student
  then borrows a covered code with a reason and the chrF of each candidate off the shelf),
  writes tag and both_directions (each pair once each way with src_code and tgt_code,
  shuffled with the seed), checked on a known case, for intermingling and by
  check_contamination of both languages' test sentences; NLLB off the shelf decodes both
  test sets before training; an "If ..., then ..." hypothesis, a prediction and a step
  budget stop the run while empty; torchlingo_harness.nllb.fine_tune trains both
  directions at once (Adafactor, accumulation, bfloat16 where available, a
  TrainingCheckpointer on Drive, best by validation loss, a Drive-space check). Each
  direction compares A14, NLLB off the shelf and the student's one-way model (A9; the A13
  reverse model) greedy and beam 5 by BLEU and chrF with sacreBLEU's paired bootstrap
  against A14, then COMET (wmt22-comet-da through its Python API, last because its install
  pins transformers below 5). Step 6 gives the wrong code both ways and asks for a third
  language the student chooses (zero-shot). Seven write-up questions; the write-up block
  reads from Drive alone (a14-notes.json) and saves a14-rerun.json; the completed notebook
  is uploaded. Tutorial 9, after it, asks how much fine-tuning is enough and what it
  costs. Needs a GPU with more memory than a T4, and Drive, so CI does not execute it;
  tests/test_lecture_14.py runs its cells with NLLB and COMET stubbed.

<!-- END generated:notebook-map -->

### Why lectures that have run are still listed

A student revisiting a past lecture needs to know what to open, later assignments refer back to
earlier ones, and a Fall 2027 offering should not have to rediscover the mapping. Where a tutorial
would have improved a past lecture it is marked `reference` rather than dropped.

### What the table cannot say

The table lists notebooks that exist; a notebook not yet written cannot declare anything.

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

### Practice before the Final Project

Each Final Project requirement is met at least once in an assignment first. Fall 2026:

| Final Project requirement | Practiced in |
|---|---|
| A held-out test with an overlap check | A8, A9, A14 |
| BLEU or chrF plus COMET | A6, A10, A11, A13 |
| One change against a named baseline (Path A) | A9, A12, A13, A14; the page's Path A example grows A14 to more languages |
| A hypothesis and a prediction before any score | A9 to A14 |
| Is the difference real (paired bootstrap) | A13, A14 |
| Where it helped and where it hurt | A13 (A9 next offering) |
| Rerunnable (versions, seeds, decoding) | A12 to A14 |
| The completed notebook submitted | A7, A11 to A14 (A8 to A10 next offering) |
| At least 3 raters, and their agreement | A6 (one rater); first met in the project itself |
| An application for an audience (Path B) | the speech-pipeline example on the Final Project page |

The gaps (raters, Path B) are where the project asks for something new;
`torchlingo-private/notes/course/NEXT-OFFERING.md` carries the fixes.

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
