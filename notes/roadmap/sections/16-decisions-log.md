## Decisions log

Newest first. Each is a decision that changed the schedule, an assignment, or a notebook's
place, with who made it.

- **Oct 25.** One task list for course work (Eric): `torchlingo-private/notes/course/TASKS.md`,
  shared by both sessions, with the course side's log and next-offering notes beside it. The
  public `notes/TASKS.md` keeps library and notebook engineering.
- **Oct 25.** A14 is rebuilt on fine-tuning NLLB-200, both directions in one run (Eric: go, with
  one Colab run before A14 opens Nov 4). Two short real runs here made the case: German 25.8 to
  32.4 BLEU, and Efik, borrowing Igbo's code, 3.6 to 18.5, after 300 steps each. A14 keeps its
  three outcomes, with NLLB's language codes as the tags, and adds a fourth, what fine-tuning
  buys over the model off the shelf and over A9 trained from scratch; COMET is in. Tutorial 9,
  fine-tuning one direction step by step, is its reading. Eric's frame for the notebooks: not a
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
