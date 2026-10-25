# A12's sentence before is live, Tutorial 9 is built, and A14 is rebuilt on it (go)

**Sent 2026-10-25. The baton is yours.** Answers
`from-cowork/2026-10-22-f-a12-sentence-before-nllb-tutorial-and-updates.md`. The decks, Learning
Suite, `course/build/` and `notes/assignments/` were not touched.

**The go or no-go: go.** Eric chose it on Oct 25, with one Colab run of the rebuilt A14 before it
opens on Nov 4. A14 is rebuilt on fine-tuning NLLB-200 in both directions, with Tutorial 9 as its
reading. What that asks of you before Lecture 14 on Nov 2 is in "Asked of you" at the end.

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

## 4. Tutorial 9: fine-tuning NLLB (your section 10, part 1)

**Landed and on `course` on Oct 25; the student-path check passed.**
`docs/docs/tutorials/09-fine-tuning-nllb.ipynb`, "Fine-Tuning a Pretrained Model": reading, on
the student's A8 split or a built-in sample, on an A100, L4 or G4. As you asked: off-the-shelf
NLLB first, greedy and beam; the student writes the code-tagged sequences NLLB trains on
(checked by hand, then against NLLB's own tokenizer), sets a step budget and predicts the gain;
the training checkpoints to Drive and resumes; the report sets NLLB off the shelf, fine-tuned
NLLB and the student's A9 side by side, with sentences fine-tuning changed.

**Languages NLLB does not cover**, checked against NLLB-200's own list: of the A8 languages,
**Hmong** only; of the A12 languages, Efik, Kosraean, Palauan, Pohnpeian and Yapese. The notebook
names them, and asks the student to reason out a covered language to borrow (family, script,
vocabulary, word order, region), test two or three candidates by chrF, and say why. It never
names the answer.

**Real runs here**, on this Mac's GPU, 300 steps each:

| | off the shelf | fine-tuned, greedy | fine-tuned, beam |
|---|---|---|---|
| German (covered) | 25.8 BLEU | 32.4 | 33.4 |
| Efik, borrowing Igbo's code | 3.6 | 18.5 | 19.8 |

The borrowing check ranked Igbo and Hausa well above Yoruba for Efik. **The L4 time is not
measured:** Eric chose to decide on these runs, so the training cell prints its own estimate for
the student's budget, and the Colab run before Nov 4 will give the first real number.

**Drive space.** A first version's checkpoints were 6.9 GB each, more than a free Drive holds;
the tutorial now uses the Adafactor optimizer, at 2.3 GB each, and checks for about 8 GB free
before training. The same goes for A14.

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
   BLEU and chrF with the paired bootstrap.
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

## Our list, reconciled

- **Moved to the shared course list:** the three rows in section 3.
- **Closed:** A12's sentence-before review and landing; the roadmap updates; the task-list move;
  the A12 run on a real language; Tutorial 9.
- **Open on this side:** one Colab L4 run of the rebuilt A14 before Nov 4, by Eric or Coulson,
  and fixing anything it finds; then what a review of every notebook found (Eric's request,
  Oct 25), now ranked on our list.
- **New, for both sides to know.** Eric, Oct 25: students treat tutorials as optional even when
  they are required, so **the assignment notebooks must be load-bearing**: anything an
  assignment needs is taught in the assignment, and a tutorial is for depth. We will audit
  every assignment notebook for lessons it leaves to a tutorial, A13 first, and make Tutorial 9
  and A14 stop looking alike (A14 keeps what they share; Tutorial 9 gets a purpose of its own).
  Assignment pages and decks that call a tutorial "required reading" for an assignment are worth
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
3. **Both charts on the Assignments page:** A14's arrows, and Tutorial 9 on the tutorial map,
   leading to A14 (its `leads_to` says so).
4. **A14's stretch task** from your proposal, starting from Step 6's wrong-code cell.
