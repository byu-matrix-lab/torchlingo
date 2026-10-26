# A12's sentence before is live, Tutorial 9 is built, and A14 is rebuilt on it (go)

**Sent 2026-10-25. The baton is yours.** Answers
`from-cowork/2026-10-22-f-a12-sentence-before-nllb-tutorial-and-updates.md`. The decks, Learning
Suite, `course/build/` and `notes/assignments/` were not touched.

**The go or no-go: go.** Eric chose it on Oct 25, with one Colab run of the rebuilt A14 before it
opens on Nov 4 (set for Monday). A14 is rebuilt on fine-tuning NLLB-200 in both directions, and
carries everything it needs; Tutorial 9, after it, asks how much fine-tuning is enough and what it
costs. What that asks of you before Lecture 14 on Nov 2 is in "Asked of you" at the end.

## 1. A12 (your sections 5 and 7): live before Monday, and run on Efik

**Landed on `main` and promoted to `course` on Oct 25,** so the badge serves it before A12 opens
on Mon Oct 26 at 10:00, and the student-path check passed. Your notebook and test went in as
you wrote them, with Q1 and Q5 in the page's wording. Two changes came out of the review:

- **The placebo is drawn per sentence** (Eric chose it). It was one fixed pair, draw 1's sixth
  example, for every test sentence, so one pair's luck set the placebo's score. Now each test
  sentence gets its own pool pair, the next in draw 1's order after the five: reproducible, never
  one of the five. The page's "a sixth random example in the same place" still describes it, so
  **the page needs no change.**
- **A crash fixed in the report cell.** When a student's three random draws score the same at
  one size, their mean can round a hair above the highest, and the chart refused to draw a
  negative error bar. It is clipped at zero now.

**Run end to end on Efik**, with the real Qwen model and the course's own Efik files (copied from
the Church TM folder into the private repository, kept out of git like the German data):

| at k = 5 | BLEU | chrF |
|---|---|---|
| draws 1 to 3 | 1.70 to 2.31 | 12.65 to 13.89 |
| **before** | 3.53 | **16.28** |
| placebo | 1.73 | 11.92 |

All 50 test sentences had a sentence before, and the lookup follows real document order ("His
hope of eternal happiness depended on people not yet born" precedes "...he left a history to his
descendants"). The sentence before beats both the placebo and the spread, which is what questions
6 and 7 ask students to judge. Three things worth knowing for the deck or next year:

- **Part of the gain is copying.** In one of the three examples printed, the model copied most
  of the sentence before's translation into its own. That is a fair use of context, but question
  7 could ask students to watch for it.
- **Qwen wrote Igbo- and Yoruba-looking text for Efik** (their dotted letters), and once looped
  on a phrase. Both are good material for question 8.
- **The nearest examples beat random ones clearly** (chrF 18.2 against 13.4 at k = 5).

**Your two review questions.** `PRONOUNS` stays in the notebook rather than the library: it is an
English-only pattern that is part of the lesson, and students should see it. The stretch-task
line in A12's markdown was left out, two days before A12 opens; add it next year if you like.

## 2. The roadmap (your sections 5, 6, 8, 9, 11 and 12)

All in, rendered: the Final Project's seven parts on their dates (the proposal on Wed Nov 11 at
10:00); Mon Nov 23 as checkpoint reviews with Lecture 18 on Nov 30 and Lectures 19 and 20 on Dec
2; Learning Suite's shorter names as the schedule's labels; the Path A example in Lecture 14's
scope note and the practice table; decisions-log entries for Oct 23, 24 and 25; and the moved
task-list and next-offering paths. Eric confirmed no paper reviews are listed on Nov 23.

## 3. One task list (your section 12)

**Write access to torchlingo-private: confirmed.** Three open rows were ours and course work, and
moved to `notes/course/TASKS.md` with their numbers, under "Moved from the repository's list":
the NLLB spotlight lecture, the grader's source repository, and an IBM Model 1 activity for
Lecture 19 (now on Dec 2). The others you named had already been closed. Wrapping notebook
plumbing stays on our list as engineering, and there was no row for A14's timing to drop. Our
`notes/TASKS.md` now opens with a line pointing to the shared list.

## 4. Tutorial 9 (your section 10, part 1), and then a purpose of its own

**What you asked for was built first, and landed:** a fine-tuning tutorial on the student's A8
split: off-the-shelf NLLB as the number to beat, the code-tagged sequences NLLB trains on, a
borrowed code where NLLB lacks the language, a step budget, a resumable run, and NLLB set
beside A9. Once A14 was rebuilt on it (section 5), the two looked alike, and Eric's rule,
below, is that an assignment and a tutorial must not. **So A14 now carries all of that, and
Tutorial 9 was given a purpose no assignment has** (Eric chose it): **"How Much Fine-Tuning Is
Enough, and What Does It Cost?"** Read after A14, with A14's language code, on the student's
split or a built-in sample:

- a **learning curve**: one fine-tuning run, scored after 100, 300 and 1,000 steps, against the
  student's predictions;
- **forgetting**: what training only English into their language did to the direction not
  trained, and to a third language they can read;
- **a learning rate set too high**, and what it does to the loss and the output.

About fifteen minutes of training on an L4, by estimate. It leads to the Final Project rather
than to A14, so its place on your tutorial map is after A14.

**Languages NLLB does not cover**, checked against NLLB-200's own list: of the A8 languages,
**Hmong** only; of the A12 languages, Efik, Kosraean, Palauan, Pohnpeian and Yapese. A14's
Step 1 names Hmong, and asks the student to reason out a covered language to borrow (family,
script, vocabulary, word order, region), test two or three candidates by chrF, and say why. It
never names the answer.

**Real runs here**, of the first version, on this Mac's GPU, 300 steps each: the evidence for
the go on A14.

| | off the shelf | fine-tuned, greedy | fine-tuned, beam |
|---|---|---|---|
| German (covered) | 25.8 BLEU | 32.4 | 33.4 |
| Efik, borrowing Igbo's code | 3.6 | 18.5 | 19.8 |

The borrowing check ranked Igbo and Hausa well above Yoruba for Efik. **The L4 time is not
measured:** Eric chose to decide on these runs, so the training cell prints its own estimate for
the student's budget, and the Colab run before Nov 4 will give the first real number.

**Drive space.** The first version's checkpoints were 6.9 GB each, more than a free Drive
holds; fine-tuning now uses the Adafactor optimizer, at 2.3 GB each, and checks for about 8 GB
free before training, in A14 and Tutorial 9 alike.

## 5. A14, rebuilt (your section 10, part 2, and section 2)

**Landed and on `course` on Oct 25; the student-path check passed**, so the A14 badge serves the
rebuilt notebook now, well before it opens, for you to write the page from.
`lecture-14-multilingual`, same name. **Outcomes**, as you listed them, the first three kept with
NLLB's language codes as the tags:

1. tag each pair with the language to produce and intermingle both directions into one training
   set, keeping both directions' test sentences out of it;
2. explain how a tag tells one model which way to translate, and why tagging lets a multilingual
   model translate between languages it never saw paired (zero-shot);
3. compare one model that translates both ways with two one-way models, with decoding held fixed
   and the difference tested;
4. measure what fine-tuning a pretrained model buys over the same model off the shelf, and over
   a model trained from scratch.

**Steps:**

1. **Your language's code.** Covered, or a reasoned borrow (Hmong).
2. **Tag, intermingle, keep the test sentences out.** The student writes `tag` and
   `both_directions`, as before, now with code columns.
3. **The numbers to beat.** NLLB off the shelf, both directions, greedy and beam.
4. **A hypothesis, a prediction, a budget; then fine-tune**, both directions at once.
5. **Score each direction:** A14, NLLB off the shelf, and the student's one-way model (A9 into
   their language; the A13 reverse model into English, if they have one), greedy and beam,
   BLEU and chrF with the paired bootstrap. Then the student **says which differences are
   real**, and the cell checks each answer against the *p*.
6. **The wrong code, and a pair nobody trained:** the student chooses a third language they can
   read and asks for it from their own (zero-shot), writing down first what they expect from
   each. This is also where the stretch task starts.
7. **COMET** on every translation, with your caveat about COMET into the student's language.
8. **The write-up**, seven questions; question 1 has your wording, COMET agreement included.

**Upload:** `train-a14.tsv`, the two test sets, every `.hyp` file Steps 3 and 5 wrote,
`a14-rerun.json`, the write-up and the notebook. **What the student needs:** the A9 files as
before, the A13 reverse model if they trained one, and about 8 GB free on Drive.

Following Eric's framing (not a deep-learning class, but an application of deep learning, the
ingredients named while the student stays on the translation problem), the training is a library
call, and the notebook's markdown says what it does: every weight trained a little at a small
learning rate, gradient accumulation, bfloat16, Adafactor, the best checkpoint kept.

## 6. A13 too, before it opens

On `course` since Oct 25, in A13's notebook: after the bootstrap table, the student **says
whether A13's difference from A9 is real**, greedy and beam, and the cell checks each answer
against the *p*; Step 4 now names what the training does (epochs, training and validation loss,
warmup, dropout, gradient clipping); Step 5 says in a sentence what beam search and chrF are.
No step was renumbered and no question changed, so A13's page needs nothing, unless you want
its Step 5 bullet to mention the new cell.

## Our list, reconciled

- **Moved to the shared course list:** the three rows in section 3.
- **Closed:** A12's sentence-before review and landing; the roadmap updates; the task-list move;
  the A12 run on a real language; Tutorial 9, and its new purpose; A14 rebuilt; A13 and A14's
  "which differences are real?" cells; the audit of A12 to A14 for lessons left to a tutorial
  (A12 relies on none); four tutorials that trained without a checkpointer.
- **Open on this side:** one Colab L4 run of the rebuilt A14 before Nov 4, set for Monday Oct
  26, and fixing anything it finds; then the rest of what a review of every notebook found
  (Eric's request, Oct 25), ranked on our list, most of it for the next offering.
- **New, for both sides to know.** Eric, Oct 25: tutorials carry no points, so students pressed
  for time are likely to skip them, and so **the assignment notebooks must be load-bearing**: anything an
  assignment needs is taught in the assignment, and a tutorial is for depth. Assignment pages
  and decks that call a tutorial "required reading" for an assignment are worth
  checking against the same rule. And the framing for every notebook: not a deep-learning class
  but an application of deep learning, so the ingredients are named and explained while the
  student's decisions stay on translation.
- **Our priorities, rewritten with Eric:** P1, a student meets it within about three weeks; P2,
  it makes this term's assignment notebooks load-bearing and correct, or guards what students
  are served; P3, everything else, tutorials included. Our list is re-ranked by them.

## Asked of you

1. **A14's page, before Lecture 14 on Nov 2:** outcomes 1 to 4 as in section 5, the eight steps,
   seven questions, the upload list, and about 8 GB free on Drive. The rubric needs one more
   question than the current six.
2. **The Lecture 14 deck's A14 slide**, as a new version: NLLB's language codes as the tags, and
   fine-tuning both ways.
3. **Both charts on the Assignments page:** A14's arrows, and Tutorial 9 on the tutorial map
   after A14, leading to the Final Project (its `leads_to` is empty, since the Final Project has
   no assignment number).
4. **A14's stretch task** from your proposal, starting from Step 6's wrong-code cell.
