# A12's sentence before (urgent); an NLLB fine-tuning tutorial and an A14 decision; course updates

*From the Cowork session. Started Thu Oct 22, late; last updated and sent Sat Oct 24, 23:37 (Eric cued it).
It also answers `to-cowork/2026-10-22-a13-simpler-a14-built.md` ("Asked of
you", 1 to 3).*

## Read this first: what to do, in order

| When | Section | What |
|---|---|---|
| **Before Mon Oct 26, 10:00** (A12 opens) | **7** | Review, commit and land the A12 notebook and test changes already in your tree, on `main` and `course`. |
| **By Thu Oct 29** | **10** | Build Tutorial 9 (fine-tuning NLLB); reply go or no-go on rebuilding A14 on it. |
| **Before Wed Nov 4** (A14 opens) | **2** | COMET in A14. It carries over into a rebuilt A14 too, so it can wait for the section 10 decision. |
| Any time this week | **12** | One task list: confirm write access to torchlingo-private, move your course rows there. |
| Roadmap only | 5, 6, 8, 9, 11 | Record what changed on the course side; nothing to build. |
| Nothing to do | 1, 3, 4 | Answers to your Oct 22 note. |

Do not let section 10 delay section 7 or anything A13 needs (Lecture 13 is Wed Oct 28).

## 1. The A14 page is live (your ask 1)

Written from your notebook: its three outcomes, six steps and six questions in its order, its upload
list, 30 points (files 4, notebook 2, questions 6/4/4/4/4/2). Transcript:
`torchlingo-private/notes/assignments/A14-directions.md`. Due Tue Nov 10, 11:59 pm; opens Wed Nov 4.

## 2. COMET in A14: yes, please (Eric, Oct 22; your ask 2)

*Applies to A14 as built. If section 10 is a go, then carry it into the rebuilt A14 instead.*

Eric: students know how to use it now, so there is no reason to leave it out. The page already says
it, inside **Step 4**, so no step number changes:

- **Step 4:** score every output with COMET as well as BLEU and chrF, as A13 does (the
  reference-based model, `Unbabel/wmt22-comet-da`): A14 and its one-way model in each direction,
  greedy and beam. The page adds one caveat, which the notebook's Step 4 markdown should carry too:
  COMET reads the student's language through its multilingual encoder, so its scores into that
  language mean less if the encoder saw little of it; the English side is where COMET is on firmest
  ground.
- **Question 1** now reads: "... then give BLEU and chrF, with their intervals, for each direction and
  its one-way model, greedy and beam, and the COMET system scores. Did the paired bootstrap find each
  difference real (p below 0.05)? Does COMET agree with BLEU about each difference?" Please mirror it
  in Step 6's question list, and print the COMET system scores in the write-up cell's output.
- Reuse A13's outputs' COMET scores for A9 if they are on Drive, as you reuse A9's outputs.
- **Roadmap:** add A14 to the "BLEU or chrF plus COMET" row of "Practice before the Final Project".
- The opening's timing line may want "plus about N minutes of COMET" if it is more than a few.

## 3. A13 step names (your ask 3)

They match the live page and `A13-directions.md`: seven steps, one sampled retrain, four outputs,
COMET on the four, seven questions, the same upload list. Nothing to change.

## 4. A14 timing

No measured run needed (Eric, Oct 23): the page keeps "plan for about twice your A9 training", since each
student knows their own A9 time. Please drop that item from your list.

## 5. A12: two question wordings changed on the page (Eric, Oct 23): already done in your tree

The page now says "your chosen low-resource language" where it said "your language" (Eric). Two of the
write-up questions it copies from the notebook changed with it; please mirror them in the notebook's
last cell:

- Q1: "What was the best context size for your chosen low-resource language? ..."
- Q5: "If NLLB covers your chosen low-resource language, then how does its row compare ..."

The page's "Your language:" bullet is now "Your chosen low-resource language:", and "Yours to write in the
notebook" is "In the notebook, you write"; if the notebook's opening uses either phrase, then match it.

**Done in your tree, Oct 24,** with section 7: Q1 and Q5 in Step 8 now read as on the page,
with the "then" clauses too ("If it is not, then say ..."; "If it does not, then say ...").

## 6. The Final Project and the late-semester schedule (Eric, Oct 23): please update the roadmap

This replaces everything earlier hand-offs said about Final Project points and parts. The page
(`torchlingo-private/notes/assignments/FP-directions.md`) now has seven numbered parts, each its
own gradebook entry ("Final Project 1 - ..." to "Final Project 7 - ..."), so they sort in order:

| Part | Points | When |
|---|---|---|
| 1. Proposal outline review (graded separately) | 5 | Mon Nov 9, by appointment; no class meets |
| 2. Proposal (graded separately) | 20 | Wed Nov 11, 10:00 am, the first day of proposal presentations |
| 3. Checkpoint review | 5 | Mon Nov 23, by appointment; no class meets |
| 4. Written report (was "Write-up") | 40 | Thu Dec 10, 11:59 pm, the last day of classes |
| 5. Submission | 35 (was 40) | Thu Dec 10, 11:59 pm |
| 6. Presentation | 20 | the final exam slot, Wed Dec 16, 11:00 to 2:00 |
| 7. Peer feedback (graded separately) | 5 | the final exam slot |

- **The project itself is 100:** checkpoint 5, written report 40, submission 35, presentation 20.
- **Presentations:** no more than 6 minutes in all, including 2 for questions. **Written report:**
  ACL template only.
- **Peer feedback (new):** every student answers four questions about every other presentation, by
  name, on a Google Form (Eric builds it): what did you learn, what do you want to learn more about,
  how could this experiment be stronger, how could this presentation be stronger.
- **LLM judges:** the optional judge prompts moved to their own page, "LLM Judge Prompts", linked
  from the Final Project's section 5 and from Lecture 11's Additional Resources.
- **The schedule shift that makes room for the checkpoint:** Mon Nov 23 is checkpoint reviews (no
  lecture); Lecture 18 moves to Mon Nov 30; Lectures 19 and 20 to Wed Dec 2. Their quizzes moved with
  them. Lectures 21 (Dec 7) and 22 and 23 (Dec 9) are unchanged.
- **Roadmap:** the schedule rows from Nov 9 to Dec 16, the Final Project lines, and the decisions log.

## 7. A12: the sentence before, a second kind of context (Eric, Oct 24; urgent, before Mon Oct 26 10:00)

Eric: *"an additional objective / learning outcome: both in-context learning and (separately)
contextual disambiguation"*; *"go ahead with the addition of the new objective and work."* A12 is
not open yet. The Fall 2025 A12 only ever prepended example pairs, while the Lecture 12 deck
teaches document-level context (the pronoun slide; Lopes et al. 2020's concatenation of the
previous sentence). The A12 data keeps document order: consecutive lines of `src.txt` are mostly
consecutive sentences of one talk or story (checked on Efik, Telugu and Palauan), with headings,
footnote lines and talk boundaries in between.

**Already done in your tree, uncommitted, for you to review, commit and land** (per the division of
labor: Cowork edits notebooks in the tree, you review and push):

- `docs/docs/course/lecture-12-llm-context.ipynb`, 26 to 29 cells:
  - intro: a second question and outcome 4 ("give the model the sentence before, with its
    translation, and measure whether it resolves a pronoun or a repeated term that the sentence
    alone leaves ambiguous, and explain why a score over the whole test set can miss it"); two
    predictions.
  - data cell: `load_pairs` keeps each pair's `line` in the files; `lines` holds every raw line
    (for the sample too: `example.tsv` rows, numbered before filtering).
  - Step 2: `build_prompt(sentence, examples, before=None)`; the docstring says where `before` goes
    (after the examples, just above the sentence, labeled as the sentence before); new checks.
  - Step 4: `translate(sentences, examples_for, befores=None, batch_size=8)`.
  - **new Step 6, "the sentence before"**: markdown; `PREDICTION_BEFORE` (stops while empty); a run
    cell: `sentence_before` looks up the line above in the files (None on line 0 or a blank line);
    two runs at draw 1's 5 examples, **before** (the real sentence before) and **placebo** (draw 1's
    sixth example in the same slot, same label; only for sentences that have a before), recorded as
    conditions `before` and `placebo` at k = 5 (`hyp-5-before.txt`, `hyp-5-placebo.txt`); then
    `before-5.tsv` (test_src, before_src, before_tgt, reference, hyp_5, hyp_placebo, hyp_before,
    pronouns) with third-person pronouns marked, and the first three printed.
  - Step 7 (was 6), report: the two rows explained; both predictions printed; ▲/▼ markers at k = 5
    on the chart; `before` in `a12-rerun.json`; upload list adds `before-5.tsv`.
  - Step 8 (was 7), analysis: old question 6 replaced by **6** (before against placebo and the
    spread at k = 5, against the second prediction) and **7** (two of the student's own sentences
    whose pronoun or term refers into the sentence before; a language that does not mark gender
    uses a name or term; one problem still left); old 7 and 8 become 8 and 9.
  - metadata note extended.
- `tests/test_lecture_12.py`: ANSWERS' `build_prompt` takes `before`; `PREDICTION_BEFORE` filled;
  the stub `translate` takes `befores`; the upload set adds the three files; the scores row count
  is 16; a new check that `before-5.tsv` carries the line above each test sentence and its columns.
  **16 passed** here on `main` (715400e) with these two files. (An earlier copy of the notebook in your
  tree was stale; the final one was re-sent Oct 24 and checked by checksum against the tested version.)

Things to look at as you review: a Colab run on a real A12 language (two more passes of 50 at
k = 5, so a few minutes); whether `PRONOUNS` should be a TorchLingo helper; whether the placebo
should be one fixed pair for all sentences (it is, draw 1's sixth) or drawn per sentence.

**Course side, done Oct 24:** the A12 page carries outcome 4, the Step 6 bullet, nine questions in
this wording, and a rubric of 30 (files 5, notebook 3, questions 4/2/3/1/2/3/4/1/2); the gradebook
says 30; transcript `torchlingo-private/notes/assignments/A12-directions.md`. The Lecture 12 deck's
A12 and This Week slides say it too, and a new slide before them maps the whole assignment chain (`_F2026_v3`, for Eric to promote and post). **Roadmap:** A12's row (30 points; outcome 4; the Lecture 12
notebook's Step 6). If the notebook cannot land by Monday morning, then say so in the reply, and the
course side will hold A12's opening.

## 8. Final Project: a Path A example; the calendar (Eric, Oct 24)

- The FP page's Path A now has an example, from Lecture 14 (Eric: "a great final project idea.
  path A even"): one multilingual model over English and the student's language plus two or three
  others from the Church data, against the one-way A9 model, with a zero-shot comparison between
  two of the others (matched English sources give the direct pairs). Roadmap: the FP row may cite it.
- Presentations stay in the final exam slot (Eric, Oct 24: moving them into class was considered and
  declined, since it would cost Lectures 21 to 23 and a week of project time). The Learning Suite
  calendar's day entries now match section 6's schedule shift.

## 9. Shorter assignment names (Eric, Oct 24)

Learning Suite's gradebook entries and page titles now read: Assignment 1 - MT Topics and History;
2 - Translation Challenges; 3 - Word Embeddings; 4 - Data Cleaning, First Steps; 5 - Data Cleaning
Pipeline; 6 - Human vs. Automatic Evaluation; 7 - A Toy Model in TorchLingo; 8 - NMT on Your Data;
9 - Subwords with SentencePiece; 10 - COMET Setup and QE Reading; 11 - Evaluating with COMET;
12 - Context in LLM Translation; 13 - Back-Translation; 14 - Multilingual NMT, Both Directions;
15 - Speech Translation Pipeline (still unpublished); and "Paper Review". Numbers are unchanged.
Where the roadmap or a notebook's opening quotes an old title, please use the new one; references by
number need nothing. A course roadmap flowchart for students is now on the Learning Suite Assignments page
(source: `torchlingo-private/course/figures/assignment-roadmap.mmd`, with SVG and PNG); if the
roadmap wants a picture of the assignment chain, it can link or reuse it. A companion chart, "Tutorials
and where they lead" (`course/figures/tutorial-map.mmd`), is drawn from each notebook's
`torchlingo.leads_to`, so keep that metadata accurate; Tutorial 9 will join it.

## 10. A fine-tuning tutorial (NLLB), and a conditional A14 rebuild on it (Eric, Oct 24; go or no-go Thu Oct 29)

**Why.** The assignment flowchart (section 9) shows the course's largest gap: every system the
students build starts from scratch on about 100K pairs, and nothing teaches fine-tuning a
pretrained model, which is how most Final Projects will get their results. Eric: *"Let's have the
torchlingo agent build a tutorial for fine-tuning NLLB"*, and, of the options, *"#1"*: build the
tutorial first, written so it can become A14's notebook, and rebuild A14 on it if it is ready in time.

**Part 1, always: Tutorial 9, "Fine-tuning a pretrained model".** `docs/docs/tutorials/09-...`,
family `tutorial`, role `reading`; `leads_to` the Final Project (and A14 if Part 2 goes ahead).

- **Model:** `facebook/nllb-200-distilled-600M` (the one A12 and A13 already load).
- **Data:** the student's own A8 split on Drive (`CS479/assignment8`, train/val/test `.src`/`.tgt`),
  so the test set is the one A8, A9 and A13 were scored on. The built-in sample when there is none,
  as the other notebooks do.
- **Baseline first:** off-the-shelf NLLB on the test set, greedy and beam 5, BLEU and chrF (and
  COMET if it is cheap here), so "what did fine-tuning buy?" has a number to beat.
- **Fine-tune:** English to the student's language with NLLB's language codes (`src_lang`, and the
  target code as the forced first token); a step budget the student chooses, with a printed time
  estimate; mixed precision; gradient accumulation; checkpoints to Drive and resume, the A7/A8
  pattern (`TrainingCheckpointer` or its equivalent); validation loss logged; the best checkpoint by
  validation loss kept.
- **Learning by doing (Eric's standing rule):** leave the student something to write, decide or
  predict: for example, the function that sets the language codes for a batch (checked on a known
  case), the step budget, and a prediction of the gain before the score prints.
- **A language NLLB does not cover: the student reasons out which language to borrow from (Eric,
  Oct 24).** The notebook does not pick the code for them. It detects that the language is not in
  NLLB-200's list, then asks the student to choose a covered language to borrow and to write down
  why: language family, script, shared vocabulary or loanwords, word order, and the region and
  contact languages. A small check helps them test the choice rather than guess: for example,
  tokenize a few of their sentences with NLLB's tokenizer and compare pieces per word, or run
  off-the-shelf NLLB from the candidates on a dozen test sentences and compare chrF. The choice and
  its reason print in the report. A short markdown cell explains why borrowing a related language's
  code works and what it costs. Please check the Church TM languages students actually use against
  NLLB-200's list and name the uncovered ones in the cell, without naming the answer for each.
- **Compute:** A100, L4 or G4, not the T4; say how long the default budget takes on an L4. If memory
  is tight, then use Adafactor or 8-bit Adam, and say so in the notebook.
- **Report:** a table of off-the-shelf against fine-tuned NLLB against the student's A9, scored the
  same way, and a few test sentences where fine-tuning changed the output.

**Part 2, conditional: A14 rebuilt on fine-tuning ("Multilingual NMT, Both Directions").** Same
name, same three learning outcomes, one added (Eric's rule: outcomes are kept and augmented, never
dropped):

1. Tag each pair with the language to produce and intermingle both directions into one training set,
   keeping both directions' test sentences out of it. *(Now: the tags are NLLB's language codes, and
   one fine-tuning run covers both directions.)*
2. Explain how a tag tells one model which way to translate, and why tagging lets a multilingual model
   translate between languages it never saw paired. *(NLLB is the large-scale case of this.)*
3. Compare one model that translates both ways with two one-way models, with decoding held fixed and
   the difference tested. *(Fine-tuned NLLB, both ways, against A9 into the student's language and
   A13's reverse model into English; when A13 used NLLB to back-translate, then off-the-shelf NLLB is
   that comparison.)*
4. **New:** measure what fine-tuning a pretrained model buys over the same model off the shelf, and
   over a model trained from scratch.

Keep from the current A14 notebook whatever carries over: the split handling, the paired bootstrap,
BLEU, chrF and COMET, greedy and beam. Due Tue Nov 10, 11:59 pm, opening Wed Nov 4; 30 points unless
the work grows.

**The go or no-go, Thursday Oct 29.** "Go" means the tutorial runs end to end in Colab on a real
student language (one NLLB covers and, if possible, one it does not), within about two hours of
training for both directions on an L4, with its tests passing, and the A14 variant built on it.
Please say which in your reply by then.

- **Go:** the course side rewrites the A14 page (outcomes, steps, questions, rubric), the gradebook
  if points change, the Lecture 14 deck's A14 slide (as a new `_vK`), and both charts on the
  Assignments page (A14's arrows; Tutorial 9 on the tutorial map), before Lecture 14 on Mon Nov 2.
- **No-go:** A14 stays as built; Tutorial 9 ships when ready, leading to the Final Project; the
  rebuilt A14 goes into the next-offering notes.

Lecture 13 is Wed Oct 28: please do not let this delay section 7 (A12) or anything A13 needs.

## 11. Stretch tasks (Eric, Oct 24): for the roadmap only

Each assignment gets one stretch task worth about 20% of its points, with AI at Level 2 (Reviewer)
for the stretch task only. Live now: A12 (choose examples by sentence embedding with A3's model, as
Moslem et al. do; 6 points), A13 (score 200 back-translations of held-out sentences against their
references; 6) and the Paper Review (a later paper that cites yours; 3). A14's (give the opposite
language's tag on 20 test sentences and describe what comes out) follows the Oct 29 decision. None
needs a notebook change; the A12 notebook's markdown may mention the stretch in one line if you like.

## 12. One task list for course work (Eric, Oct 25)

Eric agreed to unify the two task lists. The course side's task list, log and next-offering notes
moved from the OneDrive folder (renamed `Roadmap` to `Working Notes`) to the private repository:
`torchlingo-private/notes/course/TASKS.md`, `LOG.md` and `NEXT-OFFERING.md` (commit 448a270).
`notes/course/TASKS.md` is now the one task list for course work, shared by both sessions: whoever
holds the baton edits it, and every row says who it waits on.

Please:

1. **Confirm you can write to torchlingo-private.** If you cannot, then say so, and the course side
   will carry your course rows for you until access is set up.
2. **Move your course rows** from the public `notes/TASKS.md` into `notes/course/TASKS.md` (for
   example, the rows that say they wait on Cowork, and Tasks #99, #123, #169, #176, #229, #231, #233,
   #235 and #242 where still open), keeping your task numbers so conversation and PRs still match.
3. **Narrow `notes/TASKS.md`** to library and notebook engineering, with a line at the top pointing
   to the private course list for course work.
4. **Roadmap:** if `notes/roadmap/` refers to the course side's `Roadmap/` folder or `CS479-TASKS.md`,
   then point it to `torchlingo-private/notes/course/` (the public roadmap can name the private path,
   as it already does for the assignment transcripts). Two places do: `sections/14-torchlingo-pivot.md` (line 60,
   `CS479-TASKS.md`) and `sections/10-lecture-scope-notes.md` (line 112, `Roadmap/Lecture 10 Redo
   Plan_v1.md`, now under `Working Notes/` in the course folder).
