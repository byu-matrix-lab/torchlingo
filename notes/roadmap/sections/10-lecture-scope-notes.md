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
`02-evaluating-translations` (reading; tutorial 2 since Oct 5): the library's out-of-class treatment of evaluation,
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
deck's Part A links its Colab badge): a three-part exercise adapted from tutorial 3 (train-tiny, tutorial 2 until Oct 5).
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
and says why. Part C: the report. **Since Oct 5 (Eric; merged as PR #198):** the student chooses the held-out
phrase and writes a prediction; the cell asks for it and the report shows the gap if it is
missing (no assert, so the notebook still runs unedited from its badge and in CI); a second
phrase of the other kind ("Good night", or "The dog sleeps" if they chose a word-sharing phrase)
runs as a required contrast; the report prints the prediction and both results, and question 1
becomes "Was your prediction right?" (A7's Learning Suite text at the next offering). The deck's Part B slide
says so. Executed unedited in 28 seconds: *The dog sleeps* came out *El perro corre* (chrF 44.7), *Good night*
came out *Buenos días* (chrF 24.8), the seen lines 11/11.
`03-train-tiny-model` (reading; tutorial 3 since Oct 5): the library's own tutorial this was adapted from, for anyone
who wants the plain version.
**Sets:** A7 (due Wed Sep 30, 10:00): the notebook's printed report plus two sentences, on what
the seen/unseen gap says about what the model learned and what they will do differently on A8.
Graded for completion and engagement, not for the numbers.

### 8a. Encoder-Decoder, and Why Attention Was Invented (F2026, 30 slides)
The seam with 8b: *8a is what you need in order to do the assignment; 8b is what the model
actually is.* The toy-model debrief; the encoder-decoder sequence with its two animations; the
fixed-representation bottleneck; degradation with sentence length; the attention build-up;
**"Attention Is Learned Alignment"**, one text slide stating what attention computes, that its
weights form an alignment matrix nobody wrote a model for, and that tutorial 4 measures whether
it is the right one, tying Lecture 5's length-based alignment to 8b's self-attention; **"A
Learned Alignment, Seen"** (added Oct 1), two cross-attention heatmaps from the pretrained
English-to-Spanish Transformer of the 8b reading, one in-order ("But we have a problem.") and
one reordered ("And that is very useful information."), both exact matches, with the figures and
weights in `Figures/attention-alignment-*`; then "RNN with Attention". The practical half opens with why a trained model comes out bad, moved
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
`04-attention-and-alignment` (reading, Parts 1 to 7): trains the same model with and without
attention and checks whether the attention it learned points at the right words; Part 6 is
optional, Part 7 is not, because 8b starts from it; assigned in the five-day gap before 8b.
**Sets:** A8 (due Wed Oct 7): an English-to-X model on your own cleaned data; at least 100K
training pairs, 2K validation, 2K test, or all of it if you have less; source-side dedupe and a
verified split; 100-token cap; the course model (d_model 512, 8 heads, 6 + 6 layers,
feed-forward 2048, the 2017 base configuration, 56.4M parameters at an 8K vocabulary); **about
35 epochs, and keep going if validation loss is still falling at the end** (Eric, Sep 28: not
a firm range; the trainer keeps the best checkpoint by validation loss, so running long costs
time, not quality); an A100, L4 or G4 runtime, not a T4; checkpoint to Drive; SacreBLEU over
the whole test set; state your decoding strategy.

### 8b. Neural MT: The Transformer (F2026 `_v4`, 33 slides: 24 shown, 9 hidden in an appendix)
Taught Oct 5 from `_v3`; `_v4` (Oct 7, de-noised, awaiting adoption and re-posting) records what was
actually reached. Its own quiz. Reading Reports (the first two of the semester); a Quiz 8a review; objectives
(three main bullets, no sub-lines: the three attentions, softmax, A8 due); "Reading a Shape: (1, 7, 64)", tensor notation introduced
before it is used, then "The Symbols, All in One Place", every symbol of the day in one table
with its shape and the paper's and the small model's numbers; "Where We Left Off: The Bottleneck, and the Fix" (the 8a recap, drawn, with
the `encode_source` line and the three panels); "Welcome to the Birthplace of the Transformer";
"Three Attentions, Three Questions" (encoder self-attention for representation, masked decoder
self-attention for fluency, cross-attention for adequacy; drawn with arcs; the why before the
how); "Softmax: Scores into Weights" (the rule, a three-score example at 4 / 2 / 0 and the same
scores scaled by √d_k = 2, as bars); "Attention, Mechanically: Query, Key, Value" with the scaled
dot-product formula as editable text and the cross-attention framing (queries from the decoder,
keys and values from the encoder); the in-class activity "One Attention Step, By Hand" (three
words, two dimensions; score, scale, softmax, context vector, each with the type of its result;
answers 0.14 / 0.28 / 0.58 and (0.86, 1.14)); "Self-Attention: A Different Take" (adapted from
Koehn; the bank-of-the-river strip); "The Model Behind Today's Maps" (the pretrained
English-to-Spanish SimpleTransformer, 2.5M parameters, 64,311 TED-talk pairs, BLEU about 7),
placed just before the first map; "Self-Attention, Seen: One Word, Two Senses" ("like" as a verb
and as a preposition, layer 3 head 2); "Self-Attention, Formally" (Attention(Q, K, V) in the paper's notation
with Q, K, V as projections of H, and the three attentions as three sources of H; Koehn's
H Hᵀ form noted as the special case); "The Self-Attention Layer, Piece by Piece" (Serrano's layout
redrawn with a shape on every block: H n × d, the three W matrices, Q / K / V, scores n × n,
weights, weights × V); "Replacing the Recurrence with Self-Attention" (drawn chain against fan);
"Assignment 8 Is Due Wednesday" (the Colab
clock, the split check, the length cap with the (64, 100, 512) batch shape, checkpointing,
tutorial 6); "Before Wednesday" (A8, Quiz 8b, tutorials 6 and 5, the next presenters);
appendices: Going Deeper (Koehn) and Acknowledgments (Koehn's JHU 2020 slides, Vaswani's
Figure 1, Serrano.Academy's layout; the papers). Hidden at the end, kept for reference: three
tutorial-4 recap slides (the reversal task and its ablation, reading the map, Part 7's
code-to-formula bridge), Koehn's encoder figure, and the five slides the Oct 5 session did not
reach: "Multi-Head Attention" (four real heads), "The Whole Transformer, as the Paper Draws It"
(Figure 1 redrawn), "Inside One Encoder Block", "The Residual Stream" and "The Decoder Block:
Where Translation Happens". Those five are taught from the Lecture 9 deck (Oct 7). Positional
encoding, deployment, pros and cons, and the two capacity slides moved there on Oct 5.
**Notebooks:** `04-attention-and-alignment` Part 7 ("You have already seen the Transformer's
mechanism") is where this session starts; kept in one notebook with 8a's parts on purpose.
`05-transformer-attention` (reading, this week; tutorial 8 until Oct 5): a pretrained Transformer's attention on real
text, split out of tutorial 4's old Part 8 because it alone needed the 11 MB model download; since Oct 5 it also
shows the encoder's self-attention head by head (each head sharp, 0.90 to 1.00; their average 0.40), the 8b slides' point in numbers.
**Sets:** nothing of its own. A8 is due two days later.

### 8c. Finishing the Transformer, the A8 Debrief, and the Subword Activity (F2026, 22 slides)
Taught Wed Oct 7 as "Lecture 9" from the `_v2` deck; after class Eric walked it slide by slide and
split it, so the 8c file is the record of what the session held, trimmed to what was reached, plus
the 8b slides and the A8 debrief that belong with Lecture 8. Its own quiz (Attendance Quiz 8c, the
Oct 7 quiz, asked about tutorials 6 and 5). Order: Thought and Prayer; Attendance Quiz 8c; Reading
Reports (Rachel White on Attention Is All You Need, Daniel Lambert on Sennrich et al. 2016); Quiz
8b Review (22 of 24, mean 5.45 of 6; Q4 and Q6 the misses); objectives (four: finishing 8b, the A8
debrief, pros and cons, A9 starts in class); Monday's by-hand attention exercise repeated, then
"The Solution, Step by Step"; the rest of 8b: "Self-Attention: One Word, Two Senses", "Multi-Head
Attention", "The Whole Transformer, as the Paper Draws It", "Inside One Encoder Block", "The
Residual Stream", "The Decoder Block: Where Translation Happens"; "Pros and Cons of NMT"; the A8
debrief: "Assignment 8: What You Just Built", "The Models Behind the Capacity Table", "What
Capacity Buys, and When"; "In Class: The Lecture 9 Notebook" (Part A); "Assignment 9: Subwords,
Due Wed Oct 14 at 10:00"; "Before Monday"; Acknowledgments (Vaswani's Figure 1, Elhage et al.
2021, the maps and the capacity table as measured here). The unreached 8b slides were not taught
on Oct 5 or Oct 7; they are here because they are Lecture 8's, and Eric covers them as he sees fit.
**Notebooks:** `lecture-09-subword-tokenization` Part A ran in class (the student writes
`unknown_rate`, picks `VOCAB_SIZE`, predicts A9); Part B is A9. Section 9 describes the notebook.
**Sets:** A9 (due Wed Oct 14).

### 9. Handling Morphology and Terminology in NMT (F2026 `_v4`, 31 slides: 21 shown, 10 hidden)
Mon Oct 12. Built Oct 6 as the Oct 7 deck (`_v2`), split Oct 7; `_v4` is the Monday deck and awaits
adoption. Its own quiz (Attendance Quiz 9 asks about
tutorials 6 and 5). Order: Thought and Prayer; Attendance Quiz 9; Reading Reports (Nathaniel May on Kocmi and
Federmann 2023, Erik Fosse on To Ship or Not to Ship; footer names Armando Carreon's COMET report
for Wednesday); objectives (six main bullets: deployment, order without recurrence, the fixed
vocabulary, subwords, terminology, A9 Part B); "Deployment Challenges, 2017 and Now"; "No
Recurrence. So How Does It Know the Order?" (the 8b loose end, the bridge from position into
tokens); "Your Model Has a Fixed Vocabulary"; "Morphology: Why One Word Has Many Forms"
(inflection, derivation, compounding and affixes; *evlerimizden* in four morphemes); "Why
Vocabularies Explode, and What It Costs" (morphology, compounding, names; Koehn's WMT 2013
unknown-word rates; why 2% is a lot); "Byte-Pair Encoding: Merge Until V Is Big Enough" (the toy
sentence through four merges, drawn; the algorithm; the one choice, k); "What Pieces Look Like:
The Small Model's 3,000" (the pretrained model's SentencePiece on held-out sentences, ▁ shown;
frequent words whole, rare ones shattered, pieces are not morphemes; measured 1.70 / 1.79 pieces
per word, longest 25 words → 63 pieces); "SentencePiece: The Tool You Will Use" (raw text, ▁,
reversible, language independent; BPE or unigram inside; one shared model trained on the
training split, `train_sentencepiece`, `SentencePieceVocab`); "The Other Extreme: Characters"
(alphabet vocabulary, five-to-six-times-longer sequences, the no-spaces case A8's `NO_SPACES`
switch already took, where subwords sit; Koehn's character-based word models); "What Changes in
Your Model, in Shapes" (|V| to 8,000 and ln |V| to 9.0, n up 1.7 to 1.8x and `max_decode_length`
with it, what stays fixed: the split, size, epochs, test set, greedy); "Terminology: When the Right Word Is
Required" (glossaries, term bases, the Church's English-Chinese glossary excerpt kept); "Getting
Terms Into a Neural System" (substitution, rule-based inflection, WMT 2021 and PROMT's two
systems, the LLM question for Lecture 12); "Assignment 9: Subwords, Due Wednesday at 10:00" (the
four steps including A9's own run names and score-both-the-same-way, Part A "done in class on
Wednesday"; what to submit; the no-spaces note; matches the posted A9 text); "Before Wednesday" (A9,
Quiz 9 with the BPE paper, tutorials 7 and 8, Armando Carreon's report); appendices Going Deeper and
Acknowledgments. The Pros and Cons, A8 debrief, by-hand and In Class slides of the Oct 7 deck are
in 8c. Hidden at the end: the ten Koehn screenshot slides of the old deck, for
reference. The old text slides (OpenNMT activity and handout, the old A8 and A9, Quiz 9 note,
Morphology, BPE and SentencePiece bullets) are gone; the original deck is in `Archive/`.
The Lecture 9 Content page was rewritten the same day (tutorials 6 and 5 as reading, the
notebook's Colab link, the BPE paper kept for Quiz 9).
**Notebooks:** `lecture-09-subword-tokenization` (activity; merged Sep 28, PR #174): closes
this lecture and starts A9. On a CPU, from the student's A8 split in Drive: counts the test
tokens the word vocabulary cannot represent; trains SentencePiece at 8,000 pieces on the
training split only; shows the pieces; counts `<unk>` again under subwords (zero unless a
character never appears in training); round-trips ten sentences; and prints the three A9
settings for that student's data. Retrains nothing. **Since Oct 7 (Eric; edited in the tree by the course side, committed by the repository as PR #224 and on `course` at 12:30 MDT the same day; not yet executed end to end, the repository's next job): Part B of the same notebook is Assignment 9**, run on a GPU outside class: it trains with the pieces under A9's own names, re-scores the saved A8 model with the same code, and prints the write-up numbers; Step 6 saves its settings to Drive (`a9-settings.json`) instead of printing lines to paste, so nothing sends a student back to the A8 notebook. **Since Oct 5 (Eric; merged as PR #197, before Wednesday's class):** the counting function is the student's
four lines with a known-answer check and a written guess at the target-side rate; the vocabulary
size is the student's choice from 4K, 8K or 16K with a reason; an A9 BLEU prediction is written
before the settings print; and a Report back section is added; the asserts are hard, since the notebook is Colab only. The printed settings are
unchanged, except that A9 now runs under its own names (`experiment_name="assignment-9"`, `best-a9/`, `test-a9.hyp`) so it no longer overwrites the A8 model and translations it is compared against; students with `NO_SPACES = True` are detected from their split and compare characters against pieces, with spBLEU on both sides. The A9 write-up then asks for the prediction back with a reconciling sentence, the two
target-side `<unk>` rates and the piece-per-word ratio, the size and why, and one differing
sentence with an opinion (course side, at the Lecture 9 rebuild). `06-diagnosing-failures` (reading): five
questions to ask of a model that is not working, each with a planted bug; paired here by
sequence, not topic, because this is the first session at which a student has a trained model
to diagnose. `01-data-and-vocab` (reference): its vocabulary half, word-level only, stopping at
`<unk>` for "Hello universe", the motivating example for A9 rather than coverage of it. Last
year's OpenNMT notebook is in the repository's `notes/legacy-f2025/` as scope.
**When rebuilt:** open with an A8 debrief, the way Lecture 7 opens with an A6 debrief; quote
the notebook's printed settings rather than restating them.
**Sets:** A9 (due Wed Oct 14; set at 8c, where Part A ran): the same files and the same split as A8, never re-split, and
three settings changed in the kickoff's Step 6: `use_sentencepiece=True`, the subword model
(the notebook prints `sp_model_path` and `sp_tgt_model_path`, the same file twice, because
one model serves both languages; with 0.2.4 the first alone also works), and
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
sets up the HuggingFace token through Colab Secrets; A10's own scaffold. `08-inference-and-
beamsearch` (reading; tutorial 3 until Oct 5): greedy and beam search side by side, checked against the library, both
scored with SacreBLEU; **it cannot yet run from its Colab badge** (it loads the model tutorial
3 saved, which a fresh runtime never has; its rewrite is paused until A8 is in shape, Eric, Oct 5), so the deck names it without
linking it. `07-real-translations` (reading; tutorial 5 until Oct 5): a trained model's output on
real sentences, greedy against beam on a model that is actually unsure, and what undertrained
looks like; it stays here (Eric, Oct 5: 8b was getting too much) beside the decoding block, and
points at A11.
**Sets:** A10 (due Mon Oct 19): install COMET and run the notebook end to end with a token; read
two papers for the quiz.

### 11. Neural Quality Estimation and Evaluation Toolkits (F2025, 19 slides)
COMET and COMET-QE in depth, HTER distributions, partial-input baselines, lexical artifacts,
xCOMET, and **LLM-as-judge** (decided Sep 28): this is the one lecture where the course treats
prompting an LLM to score translations as a method in its own right. The F2025 deck carries two
slides on it; the combined May 2025 "Lectures 10, 11" deck carries five (the Google 2023
score-prediction prompt, GEMBA, a GEMBA-DA example, GEMBA against the other QE metrics, and the
system-level vs segment-level caveat), and the rebuild should use the five. Student reviews
set it up: the GEMBA paper ("LLMs are SOTA Evaluators of Translation Quality") and "To Ship or
Not to Ship" on Oct 12 (now Lecture 9's day), COMET itself on Oct 14 (Lecture 10); the 2025 data-contamination study came earlier,
on Oct 5.
**Notebooks:** none of its own; A10's install notebook is the prerequisite.
**Sets:** A11 (due Mon Oct 26): run the default COMET model and COMET-QE-DA on the Lecture 6
outputs; compare against the BLEU, chrF and human-ranking numbers already collected; one page.
Extra credit for xCOMET-XL. *Reaches back to A6 for its data.*

### 12. Using LLMs for MT and Expanding Context Awareness (F2025, 32 slides)
Encoder-only, decoder-only and large language models; MT with decoder-only models; document-
level and context-aware MT; dropped and ambiguous pronouns; contrastive test sets.
**Notebooks:** `lecture-12-llm-context` (homework): A12's scaffold, every code cell a TODO.
**Sets:** A12 (due Wed Oct 28): pick a non-MT HuggingFace model, load a low-resource dataset
for **Efik, Kiribati, Palauan, Pohnpeian, Yapese, Kosrean or Kamba**, translate a test set with
0, 5, 10 and 20 in-context examples, chart BLEU against context size. *The deck's own slide
still lists last year's languages (Telugu rather than Kiribati and Kamba); fix it when the deck
is rebuilt.*

### 13. Introduction to Low-Resource NMT Strategies (F2025, 30 slides)
What counts as low-resource, mitigation with and without LLMs, challenges beyond data volume,
evaluation for LRLs, data sources, data augmentation. **Written against OpenNMT**; the
back-translation workflow needs rewriting for TorchLingo before Oct 26. Candidate home for the
NLLB spotlight (Open questions).
**Notebooks:** none yet.
**Sets:** A13 (due Mon Nov 2): back-translation. Train X-to-English on the reversed data, back-
translate at least as many held-out target sentences as the training set, add the synthetic
pairs, retrain English-to-X, compare SacreBLEU and COMET against the original.

### 14. Multilingual NMT and Zero-shot NMT (F2025, 23 slides)
Bilingual versus multilingual, how zero-shot works, tagging, NLLB, complete MNMT, why the
Church data suits cMNMT. **Its handout is "MNMT Guide Using OpenNMT.docx"** and needs replacing
outright before Oct 28. The other candidate home for the NLLB spotlight.
**Notebooks:** none yet.
**Sets:** A14 (due Wed Nov 4): a bidirectional two-language system, English and X, built from
the A8/A9 system, directions randomly intermingled, separate test sets per direction, BLEU and
COMET both ways.

### 15. Overview of Speech-to-Speech MT (F2025, 33 slides)
Spoken language translation, early systems, cascade versus end-to-end, Skype Translator,
Whisper, speech data for projects.
**Sets:** A15 (due Wed Nov 11; was A16, set at Lecture 16, until Oct 8): a three-component speech-to-speech pipeline, ASR, MT and TTS, on a free Azure account, explicitly not the Speech Translation API. Ten spoken sentences in and out, plus a video of one. Also two papers for the quiz.

### 16. Automatic Dubbing and Interpretation (F2025, 34 slides)
Neural voices, LINGUA/ToAll, automatic video dubbing, Wav2Lip, HeyGen. The final project
timeline.
**Sets:** nothing since Oct 8; the speech-to-speech pipeline moved to Lecture 15 as A15.

### 17. Multimodal NMT (F2025, 42 slides)
Whether visual context helps, video-guided MT, VaTeX and MAD, architecture, ablations, project
examples. Runs after the proposal presentations; no assignment.

### 18. HAMT vs. MAHT, Productivity, Real-time Prediction and Adaptation (F2025, 40 slides)
The BYU interactive translation system, CAT tools, post-editing, normalized edit distance, why
post-editing helps some translators and not others, adaptive MT and Lilt, whether LLMs can do
adaptive MT. In-class activity.

### 19 and 20, merged. Word Alignment, and Previous MT Paradigms (F2025, two decks of 39 and 44 slides, Mon Nov 30)
Merged Oct 7 to pay for Lecture 8c. From 19: the IBM Models worked through, Awesome Align; the
sentence-alignment half had already moved into Lecture 5, and its notebook (tutorial 4) to 8a/8b.
From 20: the Vauquois triangle, RBMT, LFG, KANT, EBMT, phrase-based SMT, the history lecture placed
late so students can see what NMT replaced. The IBM models are the bridge: word alignment is what
phrase-based SMT was built on. One session, one deck at rebuild, about 30 slides; the Och and Ney
reading report belongs here.

### 21 to 23 (F2025)
Writing research articles with Overleaf and LaTeX, current MATRIX Lab research, MT applications
and jobs, opportunities for further research. Final project presentations close the semester.
