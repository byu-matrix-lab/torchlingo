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
the plan is `Roadmap/Lecture 10 Redo Plan_v1.md` in the course folder).
**Notebooks:** `lecture-10-comet-install` (homework, A10): the student writes `comet_inputs`,
predicts, and scores two systems with and without a reference. `08-inference-and-beamsearch`
(reading): greedy and beam search written out, and why a comparison holds decoding fixed.
`07-real-translations` (reading): a trained model on unseen text, greedy against beam.
**Sets:** A10 (6 points): the Hugging Face setup, including the COMETKiwi license; run the
notebook end to end; three write-up questions; two papers, one read carefully and one skimmed,
which prepare Lecture 11.

### 11. Neural Quality Estimation and Evaluation Toolkits (F2025, 19 slides; rebuild pending)
What the QE datasets taught (skewed HTER, partial-input baselines, artifacts; Sun, Guzmán and
Specia 2020), xCOMET's error spans, and LLM-as-judge (GEMBA, AutoMQM) with its system- against
segment-level caveat. It does not repeat Lecture 10's COMET material. Quiz 11 reads GEMBA and
xCOMET, one carefully and one skimmed.
**Notebooks:** `lecture-11-comet-on-a6` (homework, A11): COMET and COMET-QE-DA on the student's
A6 sentences beside BLEU and chrF; the student writes `pairwise_agreement`, which counts how
often two metrics prefer the same system, sentence by sentence.
**Sets:** A11 (26 points): the default COMET model and COMET-QE-DA on the A6 outputs, compared
with the BLEU, chrF and human-ranking numbers already collected; one page answering the
notebook's four summary questions. Extra credit for xCOMET.

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

All project work is due Thu Dec 10 at 11:59 pm; the presentations take the final exam slot, Wed
Dec 16, 11:00 to 14:00.
