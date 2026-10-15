# One roadmap in the repository, every course notebook audited, and nine things we need

**Sent 2026-10-14; written from Oct 12.** Answers
`from-cowork/2026-10-12-decks-and-build-library.md`. The baton is yours.

## What we need from you, in one place

Each question is explained in the section named. None has been asked before.

1. **A11 is set Monday Oct 19, so this one first** (section 6c): are the students' A6 files one
   plain-text file per column, and is MTEval's average rank the mean rank per system, 1 best?
   And, if you know it, what columns MTEval's `.tsv` export has.
2. **A12** (section 6): where the A12 shared folder is, as a student finds it, and its layout.
3. **A13** (section 6b): a course-recommended source of monolingual text, if there is one; and
   the A13 handout rewritten against the notebook.
4. **8c** (section 6c): Lecture 8c's outcomes, for the roadmap; the 8c notebook quotes them once
   they exist.
5. **A6** (section 6e): do students translate their ten sentences themselves, or paste a web MT
   system's output, as the notebook has them do?
6. **A8** (section 6f): the handout's minimum corpus size.
7. **Lectures 14 onward** (section 6f): will any of them have students prompt an LLM?
8. **Write-up questions** for A10, A11, A12 and A13 (sections 6, 6b, 6c): into each assignment
   text and its rubric, or your wording.
9. **Badges and slides** (sections 1, 6d, 6f): tutorial 8's badge back, the 8b and 8c badges,
   and the slides section 6f says are now out of date.

## 1. Your two requests

- **Committed: everything in `torchlingo-private`.** One commit on `main`, pushed: the
  `course/build/` scripts and README, `course/figures/` with the 3,000-piece English and Spanish
  SentencePiece model and its training script, and the refreshed assignment transcripts (A15
  added, A16 removed).
- **Your question: tutorial 8 now runs from its badge.** It trains its own small model in a
  setup cell, so it no longer needs tutorial 3's checkpoint. It has been run the way a student
  runs it: a fresh environment, its own install cell, Colab faked. It is on `course`. **Put the
  badge back** on the Lecture 10 Content page and the "Decode the Same Way" slide:
  `https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/08-inference-and-beamsearch.ipynb`

Your roadmap v16 was in the tree uncommitted, with no hand-off naming it. It is committed now,
as of this morning, because everything after it depended on the new schedule.

## 2. One roadmap, and it lives in the repository (Eric, 2026-10-12)

**There is one roadmap file set, in the repository, and the desktop copy is retired.** Eric:
*"we want only one roadmap file, and it should be in the repo,"* and *"any other info the Cowork
session has about the roadmap should go into the shared roadmap file."*

What that means for you:

- **Stop editing the desktop roadmap.** Edit the repository's files directly in the tree, the way
  you already edit course notebooks, under the same baton rule: only the side holding the baton
  edits them, and the hand-off says what changed.
- **Move in everything you hold about the roadmap** that the repository copy does not have yet:
  anything on the desktop newer than v16 (Oct 8), Lecture 9 as taught on Oct 12, the deck
  versions now awaiting adoption, the rebuilt Lecture 10, the Learning Suite and syllabus
  changes, and any schedule or decision record kept elsewhere. If it is about the course plan,
  it belongs in the shared roadmap.
- **The roadmap is now structured, and its Markdown is generated.** Eric chose a data file
  plus generated tables. `notes/CS479_COURSE_ROADMAP.md` keeps its path and reads as before,
  but it is now built by a script, and a check fails if anyone edits it by hand. **Do not edit
  it.** Edit its sources, in `notes/roadmap/`:
  - **`roadmap.yaml`, for facts.** Each schedule row (`lectures`: ids, date, title, deck
    status), each assignment with its due date (`assignments`; the schedule's "due that day"
    column is built from these, so a due date is written once), the lecture outcomes, each
    notebook's purpose and ratings, the tooling. Every section opens with a comment saying
    what its fields mean. Write dates as `2026-10-14`; the weekday is filled in.
  - **`sections/*.md`, for prose.** One file per stretch of text, numbered in document order:
    the introduction, the lecture scope notes, the decisions log, and so on. Edit them
    as ordinary Markdown.
  - **The notebook index is generated** from each notebook's own metadata, so it no longer
    needs keeping in step at each hand-off; that is why its tutorial rows are now correct.
  - **You do not run anything.** Edit the files in the tree and say in the hand-off what
    changed; this side renders the Markdown, runs the checks and commits, as with notebooks.
    If something you need to record has no field in `roadmap.yaml`, put it in the nearest
    prose file and say so, and the schema can grow.
- **The decisions log is the only history.** Eric asked for an audit of the roadmap for clarity
  and simplicity, then for all twelve findings to be fixed, and they are: the roadmap now
  describes the course as it is, and is a third shorter (16,800 words to 11,100). What that
  means for your edits:
  - The **version history** at the top is gone. A decision goes in the Decisions log, once, and
    the rest of the document is simply updated to match.
  - The **lecture scope notes** are about 100 words each: scope, notebooks, what the lecture
    sets, and the deck's status in the heading. The slide-by-slide inventories of 8b, 8c and 9
    were cut; the decks hold the slides. If something there mattered for planning, put it back
    as a sentence.
  - **No dated history in the prose** ("since Oct 5", "tutorial 3 until Oct 5"), and no task or
    PR numbers.
  - **A12's language list follows Learning Suite**, which has no Kiribati, per the roadmap's own
    rule that Learning Suite is authoritative. If Kiribati should be there, the handout is what
    needs the change.
  - **"Where the line is crossed"** now states Eric's rule for assignment notebooks, in section
    3 below, with a table of where each assignment stands.

## 3. Two rules for every notebook (Eric, 2026-10-12)

Both are in the repository's `CLAUDE.md`, and both affect what the decks and quizzes can assume.

- **Every notebook has its own specific purpose and learning outcome, which no other notebook
  shares, whatever its type.** Eric: *"students ignore the appearance of redundancy."* The test is
  the appearance as well as the content: the same corpus, model or opening as another notebook
  reads as a repeat and gets skipped.
- **A course notebook stands on its own; it never assumes a tutorial was read.** Eric: *"some
  students don't 'read the textbook' or do the tutorials."* A course notebook may assume earlier
  course notebooks and the lectures, never a tutorial. A tutorial may build on course notebooks.
  A deck can still assign a tutorial as reading; it just cannot be a prerequisite for an activity.

And one about who writes them. **Both sessions write course notebooks, together** (Eric,
2026-10-13: *"both Claude Code agent and cowork agent can collaborate on these, as we're
doing"*). Neither side owns one; the baton decides who edits, and each asks the other for what
only the other knows. That is why section 6 asks you about A12's data rather than leaving the
notebook to either side alone.

And with it: **each side goes as far as it can before handing off** (Eric, 2026-10-13: *"so that
the number of hand-offs are minimized for any given task or notebook"*). Build everything that
does not wait on the other side, leave one marked seam for what does, and ask every question a
task will need in one pass. When you receive a notebook edit to look at, fix what you can rather
than sending it back with a note, and say what you changed. This side does the same: the
Lecture 12 notebook is being built in full around its loader, so your answers in section 6 are
the only thing it waits on.

And one about what the student does in an assignment notebook (Eric, 2026-10-13, starting from
A13 and then *"make sure that A12 follows the same approach for student responsibility as A13.
Same for A10, A11"*): **the student writes the steps that are the method**, as short functions
that stop the notebook until written, each checked on a case whose answer is known and then by
the library's own checks on the real data; the notebook runs what is not the lesson (installs,
loading, training, decoding, scoring); and a prediction goes in before each long run, to be
reconciled in the write-up. A9 to A13 are all built this way now (A10 and A11 were rebuilt to
it on Oct 13; section 6c). An assignment text or slide that says "run the notebook" should say
which parts are the student's.

## 4. What changed in the notebooks on Oct 12

- **Tutorial 8** (Lecture 10's reading) runs from its badge, as above. Its outcomes are now in
  Lecture 10's terms: explain greedy against beam search, say what beam size, the length
  penalty and the length cap change, and **why a comparison between two models holds decoding
  fixed**. It scores with chrF and points at tutorial 2 for what scores mean.
- **A9 decodes greedily, and tutorial 8 now says so.** The September audit's direction for
  tutorial 8 assumed A9 lets students choose beam size and the length penalty. The A9 notebook
  scores both models greedily. Tutorial 8 now teaches why it holds decoding fixed. Worth checking
  that the Lecture 10 deck's decoding slide and the A9 text say the same.
- **Tutorial 7** (Lecture 10's other reading) states four learning outcomes, and its 700-word
  postmortem is gone. Its numbers are unchanged (BLEU 7.17; greedy 6.21 against beam 7.89).
- **`lecture-10-comet-install`** has the standard opener ("Install COMET and score one example"),
  scores **both** example hypotheses (the second was written and never scored), and ends with
  "What to record": the last output, which is what A10's rubric grades. It has not been run in
  Colab; it needs a Hugging Face token, so nothing here can run it.
- **Tutorial 3 is now "What a Checkpoint Holds, and What It Doesn't."** It used to train the
  same twelve phrases as the Lecture 7 notebook. It now shows that a saved model holds weights
  and config but not a word vocabulary, and the two ways a rebuild goes wrong: one extra word
  fails loudly, and the same sentences in another order load cleanly and translate nonsense. If
  a deck or quiz describes tutorial 3 as "train a tiny model", it is out of date.
- **Tutorials 5 and 6 are filed under Lecture 8c**, the session your deck assigned them for.
- **The five old tutorial paths are gone.** Eric decided they were not needed. Nothing in the
  repository, the transcripts or the deck build scripts linked to them; a link saved before the
  Oct 5 renumbering now gives a 404.

- **For A13: a back-translation run can now survive a disconnect.** The library has
  `translate_to_file`: it writes each batch of translations to a file as it goes, and running
  the same cell again after a dropped session picks up where it stopped. Back-translating
  100,000 sentences is under an hour; the risk was losing it all at minute forty. Worth a line
  in the A13 handout when it is rebuilt: put the output file on Drive, and if the session
  drops, rerun the cell.

## 5. No course notebook leans on a tutorial any more

A check of all eleven course notebooks against the second rule found seven clean and six places
in four notebooks where a student needs something only a tutorial explains. All are fixed, on
this side, so there is nothing to ask of you here:

- **Lecture 7 and the A8 notebook** define the vocabulary, a token and `<unk>`, and explain
  greedy decoding, where students first meet them.
- **Lecture 9:** the `unknown_rate` exercise says what the special tokens are (the `<sos>` and
  `<eos>` markers) and why `add_special_tokens=False` leaves them out; Part B's scoring step
  says what greedy decoding is and why both models must be decoded the same way; and the word
  vocabulary is "every word seen at least twice", which is what the A8 notebook keeps.
- **Lecture 6 homework** now says what TER is (edits needed to reach the reference, divided by
  its length; lower is better). It printed TER without ever defining it.

If a Lecture 9 slide describes the vocabulary as "every word in training", it is slightly off
in the same way.

## 6. The Lecture 12 notebook is built; two questions about the A12 data

**The Lecture 12 notebook now runs end to end**, as Eric decided in September, following your
audit's direction for it. Run all works from the badge on a built-in English–Spanish sample, and
a student switches to their language by setting two lines, `LANGUAGE` and `DATA_DIR`. It:

- splits the student's pairs into a 20-pair context pool and a 50-sentence test set, kept apart;
- prints the whole prompt once at five examples, so "adding context" is something they can see;
- translates with an open chat model that needs no token and is not MT-trained
  (`Qwen/Qwen2.5-1.5B-Instruct` by default; students may choose another), greedily, at 0, 5, 10
  and 20 examples;
- scores each size with BLEU and its signature, draws the chart, and **writes the upload list to
  Drive**: `context-0.tsv` to `context-20.tsv`, `test.tsv`, `bleu-report.tsv` and the chart,
  `bleu-by-context.png`, which the roadmap names as A12's evidence but the upload list had
  left out.

Run here end to end with the default model. On the Spanish sample, five examples barely helped
and more made BLEU *worse* (25.8 at zero, 26.5 at five, 22.0 at ten, 23.6 at twenty): the model
already knows Spanish. That is worth a line in the deck, since students may see a curve that
falls and think they did something wrong. Before the loop runs, the student writes a prediction
of the best size, and the notebook stops until they do; the write-up asks them to reconcile it
with the curve, so the notebook is not Run-all-and-submit.

**The write-up has five questions now, not three** (Eric, 2026-10-13: each assignment asks about
its lecture's conceptual outcome). The three from the handout stay, the first now asking to put
the prediction beside the result and the second asking what "adding context" changes when the
weights do not. Two are new:

- *Why must the context pool and the test set share no sentence?* Say what BLEU would have
  measured if they did.
- *What can no number of example pairs fix?* Each test sentence is translated alone, without the
  sentences around it. Name two problems that leaves, such as a pronoun whose referent is in the
  sentence before, or a name translated two ways in one document, and say what the prompt would
  need to hold to fix each. This is Lecture 12's "name the document-level problems (pronouns,
  consistency)", which nothing in A12 asked before.

Please add both to the A12 text and its rubric, and the chart to its upload list, or tell me how
you would word them. The notebook also now tells the student that only the first line of the
model's reply is kept, which matters for how they word the prompt, and warns that a gated model
(Llama, Gemma) needs its license accepted and a token.

**Two things it waits on: where the A12 shared folder is, and its layout.** Step 1 now says to
copy the student's language folder from "the A12 shared folder" into `MyDrive/CS479/a12-data`,
without saying where that shared folder is, because I do not know. Please give its location as a
student finds it (a Learning Suite link, a Drive path), and I will put it in Step 1. The loader
guesses the layout: one tab-separated
file with English in the first column, or two line-aligned text files with the English file's
name containing `en`. If the real folder is anything else, the notebook stops with a message
rather than misreading it. Please either put one language's files in the tree (for example
`docs/docs/course/data/a12-sample/`) or describe them: file names, format, how many sentences,
and whether a test set is provided or the student carves one out. If the guess is right, say so,
and nothing more is needed.

Two smaller asks, only if you know: **which model students used in Fall 2025**, if one was
recommended (the default can change to it), and **whether last year's working notebook or
handout** did something this one should keep.

## 6b. The Lecture 13 notebook is built; the A13 handout needs to match it

**`lecture-13-back-translation` does Assignment 13 end to end**, picking up from what Lecture 9's
Part B leaves on Drive. It reverses the split by swapping columns, trains the student's language
to English with A9's configuration and SentencePiece model (one vocabulary serves both
directions), back-translates the student's monolingual text with a resumable decode, retrains
English to their language on the original plus synthetic pairs, scores both systems with A9's
BLEU and with COMET, and prints the write-up block, including three sentences the systems
translated differently. The student writes a prediction before the second training, and the
notebook stops until they do. It is tested here at toy size from the notebook's own cells; a full
run is GPU hours, so it has not been run at full size.

Since then (Eric, 2026-10-13: each assignment asks about its lecture's conceptual outcome), the
write-up has **four numbered questions**, where before it asked for numbers and Google
translations but no questions:

1. *Did back-translation help?* Both systems' BLEU and COMET, the prediction beside them, and
   what explains the difference.
2. *Which side of each back-translated pair is real, and why is that the side that matters?*,
   answered with the back-translator's own BLEU.
3. *What makes your language low-resource, beyond its number of sentence pairs?* Two things, such
   as how much text they could find for `mono.txt`, its domains, a standardized spelling, or how
   well Google Translate and the tokenizer handle it, and which limited A13 most. This is Lecture
   13's "define low-resource beyond pair counts", which nothing in A13 asked before.
4. The three sentences the systems translated differently, with Google Translate's English for
   both, and which did better (the handout's existing ask).

Please put them in the A13 handout and its rubric, or tell me how you would word them. The
notebook also now reads the epoch count from A9's checkpoint instead of asking the student to
type it, and its write-up cell works in a fresh session, from what earlier steps saved to Drive.

Three things for you:

1. **The A13 handout still describes OpenNMT** (its transcript says so). Please rewrite it
   against the notebook: the steps above, `mono.txt` beside the A9 files, and the upload list
   the notebook names: `train-a13.tsv`, `val.tsv`, `test.tsv` and `test-a13.hyp`.
2. **Every assignment after A9 starts from A9** (Eric, 2026-10-13: *"starting point for
   assignments after A9 is A9"*), not "A8 or A9, whichever was best", as the A13 handout says.
   Please drop the choice from the A13 and A14 handouts and any slide that offers it. It is in
   the roadmap's decisions log.
3. **Where does the monolingual text come from?** A8 used all of each student's cleaned data, so
   the notebook asks the student for new sentences in their language, at least as many as their
   training pairs. Step 1 now names general places to look (Wikipedia, news sites, monolingual
   collections on the Hugging Face Hub and OPUS), and stops with instructions if `mono.txt` is
   missing. If the course folder has monolingual text per language, or a source to recommend,
   say so and Step 1 will point at it.

## 6c. A10 is rebuilt and A11 has a notebook; one question about MTEval

**`lecture-10-comet-install`** now has the student write `comet_inputs`, which turns aligned
sentences into the records COMET scores and refuses lists of different lengths, checked on known
cases. A prediction comes before scoring. Both models score through COMET's Python API rather
than the `comet-score` command, and the last cell prints the summary A10 grades: each model's
two system scores and preferred system, and the prediction. Its first lines changed, so **a
screenshot of the old final cell is no longer what A10's rubric will see**; the A10 text's "the
notebook's final output cell" still holds.

Three later changes to the same notebook (Eric, 2026-10-13), two of which the A10 text should
match:

- **COMETKiwi needs its license accepted**, not only a token: the download fails for a logged-in
  student until their Hugging Face account has clicked *Agree and access repository* on
  `Unbabel/wmt22-cometkiwi-da`, and COMET then reports it misleadingly as "not supported by
  COMET". The notebook now makes that step 1 and names it when the download fails. If the A10
  text or the Lecture 10 slides list the Hugging Face steps, please add the license there too.
- **The prediction now stops the run while empty**, so every submission has one.
- **Three write-up questions follow the final output**, answered in two or three sentences each:
  the prediction against the result; what a COMET score means (is 0.85 "85% correct"? what two
  systems' scores from one model on the same sentences tell you, and why a `wmt22-comet-da` score
  cannot be compared with a `wmt22-cometkiwi-da` one); and evaluation against estimation (which
  score is which, one situation where only estimation is possible, and what having no reference
  costs there). They serve Lecture 10's outcome "distinguish evaluation from estimation; say what
  COMET and COMET-QE score and what the number means". **Please add them to the A10 text and its
  rubric**, or tell me what to change if you would word them differently.

**`lecture-11-comet-on-a6`** is new: A11 had no notebook. From the student's A6 files on Drive
(sources, two systems' translations, references, one sentence per line) it scores both systems
with COMET, COMET-QE-DA, BLEU and chrF. The student pastes in their A10 `comet_inputs` and writes
`pairwise_agreement`, which counts how often two metrics prefer the same system sentence by
sentence. That makes the handout's "do they correlate" question something they measure. The
report gives what the one-page summary asks for. Both COMET models were run here for real on a
toy set; COMET-QE-DA needs no token. Please link it from the Lecture 11 page and the A11 text.

**The question: MTEval's export.** The A11 notebook takes the human ranking as the two average
ranks each student reported in A6, so the human comparison is system-level only. If you can
describe the `.tsv` MTEval exports (its columns, one row per what), the notebook can compare
each metric with the student's ranking sentence by sentence too.

**Two things to confirm in the A11 notebook's Step 1** (Eric, 2026-10-13), which I wrote from
the notebooks alone, since nothing here records how A6 was handed in:

- **The A6 files.** The notebook asks for one plain-text file per column (sources, each
  system's translations, references), one sentence per line, in `MyDrive/CS479/assignment6`, and
  tells a student whose data is one spreadsheet to save each column as its own `.txt`. If A6
  asked for a particular layout or file names, tell me and the defaults will match it.
- **MTEval's average rank.** The notebook defines it as, per system, the mean of the ranks the
  student gave its translations, 1 being the better of the two, so closer to 1 is better. Please
  correct it if MTEval computes it differently.

The A11 notebook also gained, today: a prediction that stops the run while empty; a paragraph on
why it uses COMET-QE-DA rather than A10's COMETKiwi (no token or license needed, so it runs for
everyone); and four summary questions, the last on LLM-as-judge: whether it is evaluation or
estimation, and, from the student's own sentence-level agreement counts, how a metric can pick
the better system while disagreeing on many sentences. **Please add the four questions to the
A11 text**, or tell me how you would word them.

**Also for the roadmap:** Lecture 8c has no lecture outcomes in `notes/roadmap/roadmap.yaml`
(`lecture_outcomes`). Tutorials 5 and 6 now name the outcomes they serve, and for 8c there is
nothing to name; the new 8c notebook (section 6d) says only "Lecture 8c's multi-head attention".
Please add them when you next edit the roadmap, and the notebook's line will follow.

## 6d. Tutorial 5 is now two course notebooks, for 8b and 8c (Eric, 2026-10-13)

Eric asked for tutorial 5 to be split into the two lectures it served, with exercises, and for
the tutorial to stay for now, pointing at both, so that a student who opens it is not lost.

- **`lecture-08b-cross-attention`** (8b, activity). The student writes `attention`, one scaled
  dot-product step: softmax(QKᵀ/√d_k)V. It is checked on a one-query case small enough to do on
  paper, with weights 0.670 and 0.330, and is then shown to give exactly the trained model's
  weights in all four heads of its cross-attention. Then a prediction: for three output pieces,
  which English piece each will look at most, marked right or wrong against the map. The
  reading is tutorial 5's: attention finds the right words, and the decoder still writes the
  wrong Spanish.
- **`lecture-08c-attention-heads`** (8c, activity). The student writes `split_heads`, which cuts
  a layer's 128 numbers per piece into four heads of 32, and `sharpness`, a row's largest weight
  averaged over the rows. Before the measurements comes a prediction: whether the heads differ,
  whether their average is sharper, and whether deeper layers are. Then the first layer head by
  head, and the sharpness table (each head near 1.0 in the first layer, their average 0.40,
  deeper layers about 0.3).

Both run on a CPU in under a minute from their Colab badges, on the sentence tutorial 5 used,
so the maps and numbers are the ones you may already have on slides.

**What this asks of you:**

- **The 8b deck and Learning Suite page**: the badge for `lecture-08b-cross-attention`, where
  tutorial 5 was linked. The deck's "one attention step by hand" is now the notebook's first
  exercise, so its worked numbers can be the slide's.
- **The 8c deck and page**: the badge for `lecture-08c-attention-heads`, beside the multi-head
  slides moved there from 8b. If you show the self-attention maps on a slide, they now come from
  this notebook. The 8b deck's "self-attention on a real model's maps" belongs here, unless you
  would rather keep it in 8b; say which, and the roadmap will follow.
- Both lectures have already run this term, so this is for students revisiting them, and for
  the next offering.

## 6e. Lectures 3 to 8a: the audit is applied; one question about A6

The cell-by-cell audit of the earlier course notebooks is done and on the course branch. Each
now has a prediction that stops the notebook until written, and the exercises stop with a
message instead of failing to parse. What a slide or assignment text might mention:

- **Lecture 3**: cosine similarity is defined where it is first used, and the model's supported
  languages are linked.
- **Lecture 4**: the TMX activity asks for a line-count prediction first, and its cleaning
  choices are now limited to steps a string function can fix (4, 6 and 11); steps 2 and 10 are
  said to drop a pair. The regex refresher has nine exercises, not seven blanks, and a new
  section on `re.sub`.
- **Lecture 5**: the cost model and s² are explained and asked about, with a success criterion.
- **Lecture 6 activity**: students compute BLEU from its precisions and brevity penalty, checked
  against SacreBLEU's own score, and say which of two systems they prefer before any score
  prints.
- **A6 homework**: students rank their ten sentences in the notebook before any score prints;
  then each sentence's chrF rank prints beside theirs.
- **Lecture 7**: the held-out phrase has its own cell and no default.
- **Lecture 8a**: training against validation loss is plotted, with when to raise the epoch
  count; the first loss is set beside ln(V), now as a prediction (section 6f); a write-up block
  prints at the end.

**The question (Eric, 2026-10-14): how do A6 students get their ten translations?** The roadmap
said "translate ten by hand"; the notebook has them paste ten sentences into a web MT system and
paste its output back. The roadmap now describes the notebook, but if the assignment means the
student translates, or uses both, say so and the notebook will follow.

## 6f. What changed on Oct 14, and two more questions

Everything here is on the course branch, so students already have it.

- **The new A9 notebook replaced the old one** once A9's deadline passed (Eric, 2026-10-14). The
  Lecture 9 badge opens it; its file name did not change. Part B now reads the epoch count from
  A8's checkpoint rather than a hard-coded 35, `unknown_rate` is checked on three cases, A8's
  BLEU is on screen before the A9 prediction, and Part B ends by telling students to disconnect
  the GPU. Students who started A9 on the old one keep their own Drive copies.
- **8a: the first loss is predicted, not copied.** Students used to copy their first logged loss
  into the notebook by hand. Now, before training starts, they predict it from ln(V), and the
  notebook reads the real value back from the checkpoint on Drive, which works after a dropped
  session too. **A slide or the A8 text that says to copy or note down the first loss is out of
  date.** The explanation of why it usually sits a little above ln(V) now comes after the
  prediction, not before it, so as not to give the answer away.
- **8b asks for exactly three guesses**, as its text always said; two now stop the notebook.
- **Missing files are named all at once.** The A9, A11, A12 and A13 notebooks, 8a and Lecture 7
  find the student's course folder on Drive with one library call, and when a file is missing
  they stop with a message naming every missing file and where it should be, instead of a
  traceback at the first one. Worth knowing if students paste one of those messages into a
  question.
- **Fewer and shorter setup cells.** 8b, 8c and tutorial 5 load the trained model in one line,
  and the A8, A9 and A13 notebooks decode in one call rather than a hand-written loop. Nothing a
  student does changed.
- **TorchLingo 0.2.8 is on PyPI**, with these library additions. Students install from the
  course branch, so this matters only to anyone outside the course.
- **The Lecture 10 COMET notebook has still not been run in Colab.** Its install and the
  COMETKiwi login are untested on a real Colab runtime, and there will not be time to before
  students use it. If students report that COMET will not import after the install cell, the
  fix is Runtime → Restart session, then continue from the cell after the install; please pass
  that on, and tell me, so the notebook can say it.

**Two more questions:**

- **A8's minimum corpus size.** 8a prints its first loss once 50 batches have run, about 3,200
  training pairs at its batch size. A student with fewer pairs than that sees no first loss at
  all, and the notebook says so plainly. What is the smallest corpus the A8 handout allows? If it
  can be under about 3,200 pairs, 8a will log more often for small corpora.
- **Will any of Lectures 14 onward have students prompt an LLM?** A12 is the only notebook that
  does today, so its prompting code stays in A12. If a later assignment prompts one too, that code
  moves into the library, so the two notebooks share it.

## 7. What moved on this side

- Closed: tutorial 8 running from its badge, its scoring section and its use of the library's
  evaluation functions; tutorial 7's outcomes, postmortem and scoring; the COMET notebook's goal;
  tutorials 5 and 6's lecture; tutorial 3's missing checkpointer; the tutorial redirect stubs;
  every course notebook's reliance on tutorials (section 5); every tutorial's opener, now in one
  format with outcomes stated as abilities and the lecture outcome each serves; the roadmap's
  repository-written sections, which used the tutorial numbers from before Oct 5; a resumable
  bulk decode for A13.
- Closed under the first rule: tutorial 8 no longer trains Lecture 7's twelve phrases (it
  trains ten travel phrases, "where is the station", "turn left here"), and tutorial 1 no
  longer opens with Lecture 7's "Hello world / Hola mundo" (now "Hello friend / Hola amigo").
  No other notebook shares a corpus or an opening. If a slide shows either tutorial's old
  phrases, it is out of date.
- Closed: an audit of the roadmap for clarity and simplicity, which Eric asked for (section 2
  says what changed for your edits).
- Closed: an audit of every course notebook, cell by cell, for whether a student can tell what
  to do in each without being handed the answer (Eric); section 6e lists what changed. If you
  have seen students stuck at a particular cell, say which; that is the best evidence there is.
- Built: the Lecture 12 notebook (section 6), waiting only on where the A12 folder is and its
  layout; the
  Lecture 13 back-translation notebook (section 6b); the Lecture 6 notebooks now install from
  the course branch like every other.
- Closed on Oct 14: the new A9 notebook rolled in (section 6f); the house patterns for course
  notebooks, now written down and checked automatically; the audit's per-notebook work for 8a,
  Lecture 7, the Lecture 6 activity, Lecture 5, both Lecture 4 notebooks and Lecture 3.
- Changed: "wrap the plumbing" now has two items left, both waiting: A12's loader, on your
  answer about the A12 folder, and an LLM helper, on your answer about later lectures.
- Now waiting on you, each tracked in the task list until you answer: the A11 Step 1 fit, the
  A12 folder and loader, the A13 monolingual source, 8c's outcome, the A6 translation wording,
  and 8a on small corpora.
- New on this side, not waiting on you: an architectural review of the library, with pedagogy
  as the foremost requirement (Eric, 2026-10-14). Nothing changes before it is reported.
