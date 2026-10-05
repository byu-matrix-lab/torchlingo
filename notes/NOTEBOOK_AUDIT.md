# CS 479 Fall 2026: notebook audit from a student's seat

Version 3, September 29, 2026 (v1 was the audit alone; v2 added implementation directions and Eric's two decisions: every notebook must be runnable end to end, and lecture-12 becomes a runnable notebook rather than a scaffold; v3 applies his third: no project history in notebooks, not even as an aside). Covers all 18 notebooks in the torchlingo repository: 11 under `docs/docs/course/` and 7 under `docs/docs/tutorials/`. I read every markdown cell of every notebook and counted words and code lines. Two questions per notebook, as Eric asked:

1. **Goal.** Before running anything, can a student say in one sentence what they will have, or be able to do, when the notebook ends? And does the notebook tell them what it feeds (an assignment, a lecture)?
2. **Brevity.** Is the reading load proportionate to the role? Working budget used here: an in-class activity ("about 15 minutes") should be under about 800 words of prose; a homework shell under about 700; a tutorial assigned as reading under about 1,200. Above 1,500 words needs a reason, and a single prose cell above 250 words is a wall.

The generated purpose cell (Lecture N, assignment N) is present in all 18 and does the "what it feeds" job well. It is not counted as a goal statement: it says where the notebook sits, not what the student will get out of it.

## Scorecard

| Notebook | Role | Prose words | Code lines | Goal clear? | Brief enough? | Verdict |
|---|---|---|---|---|---|---|
| lecture-03-word-embeddings | activity | 587 | 15 | Yes | Yes | Good as is |
| lecture-04-regex-refresher | reference | 762 | 92 | Generic, no course tie | Yes | Light rewrite of the opener |
| lecture-04-tmx-cleaning | activity | 781 | 121 | Yes | Yes | Good as is |
| lecture-05-sentence-alignment | activity | 719 | 97 | Yes | Yes | Good as is |
| lecture-06-mt-evaluation | activity | 854 | 122 | Yes | Borderline | Trim Report Back; hide the 70-line helper |
| lecture-06-mt-evaluation-homework | homework | 521 | 72 | Yes | Yes | Good as is |
| lecture-07-toy-model | activity | 1,343 | 166 | Yes | No | Cut Part A prose by a third |
| lecture-08a-a8-kickoff | activity | 1,355 | 158 | Yes | No | Move the "why this size" essay out |
| lecture-09-subword-tokenization | activity | 902 | 79 | Yes | Yes | Good; the model for the others |
| lecture-10-comet-install | homework | 258 | 22 | **No** | Wrong places | Needs a goal; cut the token tour |
| lecture-12-llm-context | homework | 548 | 12 | As a spec, yes | Yes | Fix the loop wording; add one worked prompt |
| 01-data-and-vocab | reference | 273 | 122 | One-liner only | Yes | Add a "by the end" list; drop boilerplate |
| 02-train-tiny-model | reading | 341 | 176 | One-liner only | Yes | Say how it relates to Lecture 7's activity |
| 03-inference-and-beamsearch | reading | 1,645 | 258 | One-liner only | No | Point it at A9; merge three closings; fix stale Colab text |
| 04-attention-and-alignment | reading | 1,709 | 177 | Yes, best of all | No | Mark Parts 6 to 8 optional |
| 05-real-translations | reading | 1,970 | 111 | Yes | No | Move the 710-word postmortem to a concept page |
| 06-diagnosing-failures | reading | 2,330 | 252 | Yes | No | Reframe as a lookup procedure; collapse notes |
| 07-evaluating-translations | reading | 1,877 | 99 | Yes | No | Cut the project-history paragraphs |

Eleven of eighteen have a clear goal. Eight are brief enough. The course activities written this month (L4 TMX, L5, L6, L9) are the strongest on both counts; the five long tutorials (3 to 7) are the weakest on brevity, and the three Fall 2025 carry-overs (regex, COMET install, LLM context) are the weakest on goal.

## Course notebooks

**lecture-03-word-embeddings.** Opens with a question ("Does the multilingual embedding space really line up?"), the standard frame ("Ungraded. Work in pairs. About 15 minutes."), five numbered steps, and a Report back with four prompts. 15 code lines total. Nothing to change.

**lecture-04-regex-refresher.** The goal is "learn the fundamental concepts of regular expressions." True, but a CS 479 student does not know why they are being handed this today: the two sentences it lacks are that A4's cleaning pipeline is written with these patterns and that the TMX activity assumes them. The headings are "Cell 1" through "Cell 7", which is how the author saw it rather than what the topic is, and it closes with a "Congratulations" cell. Length is fine for a 15-minute tour. Fix: a two-sentence opener tying it to A4, topic headings, drop the closing cell. Low priority because its role is reference.

**lecture-04-tmx-cleaning.** "Break the alignment, then fix it," a "By the end you will have" list, the Save-a-copy warning, two appendices clearly marked optional. Good as is. The 44-line code cell in Step 4 is the only place a student might scroll past something that matters.

**lecture-05-sentence-alignment.** Same frame, same clarity ("run the Gale-Church aligner, find the one thing it gets wrong, try to fix it, then measure your own language pair"). Good as is.

**lecture-06-mt-evaluation.** Three parts that each make one point (the zero, a sample big enough, the brevity penalty on purpose). Two things push it past 15 minutes: the Report Back is 191 words of instructions at the moment students have the least attention left, and Part 2 has a 70-line code cell that reads as a wall. Fix: three one-line Report Back prompts; move the helper into the library or label the cell "run, do not read."

**lecture-06-mt-evaluation-homework.** "What to record" gives exactly three items that open the A6 analysis. Good as is.

**lecture-07-toy-model.** The title states the lesson ("your first model, and what it does not know") and Part C says what to turn in. But Part A, the in-class half, carries most of 1,343 words, and the "What to look at" cell after A4 alone is 405 words. In the room on the 28th that cell was competing with the deck saying the same thing. Fix: cut that cell to the two sentences that matter (seen 11/11 is memorization; the unseen line is the only number about translation) and leave the interpretation to Part B; turn the "watch a training run survive a disconnect" section into a five-line checklist. Target about 900 words. This one is mine.

**lecture-08a-a8-kickoff.** The deliverables are listed up front and every step has a verifiable output, which is exactly right for a notebook students will leave running after class. The weight is in Step 6: 268 words on why the model is the size it is, which is deck material, and "Step 4: verify. Do not trust." repeats what the cell shows. Fix: keep "the number to compare your first loss against" (students need it Wednesday), reduce Step 6 to the configuration and one sentence of why, and move the size argument to the 8a deck where it already lives. Target about 1,000 words. Also mine.

**lecture-09-subword-tokenization.** Opens by naming the problem in the student's own model ("a word it never saw becomes `<unk>`"), lists the three files they will have in Drive, and ends with "What A9 is, now that you have seen this." 902 words, 79 code lines. This is the template the others should follow.

**lecture-10-comet-install.** The one notebook with no goal at all. Its first heading is "Installation (Commands come directly from the Github)", and the longest cell is a 165-word Hugging Face token walkthrough that includes advice about choosing the Write role "if you plan to push changes" (they will not). A student does not learn what COMET is, what they will have when it works, or what to bring to Lecture 11. Fix: a three-sentence header (COMET in one line; you will have `comet-score` running on one worked example; A10 asks for that plus the reading), cut the token instructions to "create a Read token and store it as a Colab secret," and one closing line about A11. It is due at Lecture 11, so there is time.

**lecture-12-llm-context.** This is an assignment scaffold: six headed sections, every code cell a `# TODO`, 12 code lines in total. As a specification the objective and task list are clear. Two problems a student will hit: the comment "the outer loop iterates through each language" contradicts the task, which is one language of the student's choice; and the interesting idea (add in-context examples and watch BLEU move) is never shown, so a student has no picture of what a zero-shot versus five-shot prompt looks like. Fix: correct the loop wording and add one worked prompt pair. Decide whether it stays a scaffold or becomes a runnable notebook; if it stays a scaffold, it is already brief enough.

## Tutorials

**01-data-and-vocab.** The opener is one line ("Learn how to load parallel data, build vocabularies, and prepare your data"). Short and runnable, but a student reading it "alongside Lecture 4 and Lecture 9" is never told which part serves which lecture, or what it gives them for A5. The Summary ("You've learned: 1. Loading data...") and Next Steps are the boilerplate pattern the course notebooks dropped. Fix: a three-item "by the end" list in the opener, one closing line. Low priority.

**02-train-tiny-model.** Also a one-liner opener, though the Colab instructions (GPU first, then two cells) are the clearest install text in the set. A student who did Lecture 7's activity in class will recognise the same corpus and the same model and will not know whether this is a repeat. Fix: say so in the opener ("Lecture 7's activity was cut from this; here is the library's version, with save and load, which A8 needs") and a three-item "by the end" list. Low priority.

**03-inference-and-beamsearch.** Assigned reading for Lecture 10 and a head start on A9, which is where students will meet `translate_batch`, beam size, `alpha` and `max_decode_length`. The notebook does not say that. Its goal line is "Generate translations using greedy and beam search," the Colab text still says "Uncomment and run the `%pip install` cell" (nothing is commented out any more), and it requires the Tutorial 2 checkpoint (repository task #166). It is 1,645 words and 258 code lines with a 72-line cell, and it closes three times: Summary, Key Takeaways, What's Next, 378 words together. "A measurement that tells you nothing" is a good lesson told at essay length. Fix: a "by the end" list pointed at the A9 settings; fold the null-result section to one paragraph with a link to the Decoding concept page; one closing section; fix the Colab text. #166 covers the checkpoint.

**04-attention-and-alignment.** The best goal statement in the set: four numbered things you will have done, then eight parts that do them. The cost is 1,709 words, and Parts 6 to 8 (Bahdanau or Luong; the Transformer's mechanism; the same picture on a real model, with a 380-word reading of one heatmap) are extensions of the point Parts 1 to 5 already made. Fix: keep the goal list, label Parts 6 to 8 "if you have time," and cut the heatmap reading to its two conclusions (attention found the right words; the output is wrong anyway). That makes the core about 1,000 words, which is right for 8a and 8b reading.

**05-real-translations.** The opener is the most motivating in the set: a warning that the translations are bad, and why seeing that is the point. Then it grows to 1,970 words in 15 cells. The 710-word cell at the end is a postmortem on a confounded experiment (+2.33 BLEU credited to data, mostly training length). It is a fine essay about research method and it is the wrong place: the student came to see a model fail on unseen text, and the same story is told again in tutorial 6's Question 4. The 437-word greedy-versus-beam cell restates tutorial 3's signature lesson. "What's Next" points back to Tutorial 4. Fix: move the postmortem to `concepts/when-it-fails.md` (or its own concept page) and leave a three-sentence pointer; cut the greedy/beam cell to the comparison table and one paragraph; fix What's Next. Target about 1,100 words. Its purpose cell says Lecture 10 and A11; that placement is Eric's open call.

**06-diagnosing-failures.** Clear goal, and the right shape for what it is: five questions in cost order, each with Recognize and Fix, and a closing procedure table. At 2,330 words and 30 cells it is the longest notebook, and it is assigned as reading before Lecture 9 while students' A8 runs are going. The two `!!! note` admonitions add about 300 words of nuance a first reader does not need; the setup is a 56-line cell plus an 81-line `Vocab` class that a student will assume they are supposed to read. Fix: present it to students as a lookup ("read the table; run the question that matches your symptom") rather than a linear read, collapse the two notes, and mark the setup cells "run, do not read." The content itself is what a student with a broken A8 run needs.

**07-evaluating-translations.** Sharp goal ("given two systems, which one ships?", "by the end you will have seen the default metric pick the worse system"). Every part changes one thing, which is good teaching. About 400 of its 1,877 words are project history: the two wrong conclusions the `word_order` default produced in this repository (Part 3) and the reference-shape bug in `compute_chrf` and `compute_ter` "for their whole existence" (Part 4). The habits those stories teach are stated in one sentence each and can stand alone. Fix: cut the history to a one-line "this bit this project once" per part, and link the bug note. Target about 1,400 words. Its purpose cell names Lecture 6 and A6, which is past; it is one of the four stale purpose cells already in the hand-off.

## Cross-cutting

1. **Two opener conventions.** The course activities written this month all open the same way: what this is, ungraded or not, time, "by the end you will have," what it feeds. The tutorials and the three Fall 2025 carry-overs do not. One four-line header on every notebook would close most of the goal gaps in this audit. The purpose cell already supplies the last line; the "by the end" line is the one missing.
2. **Project history leaks into teaching text.** Tutorials 3, 5, 6 and 7 each tell the story of a bug the library shipped, and the +2.33 BLEU confound appears in both 5 and 6. The lesson each story carries ("change one thing at a time", "score a case whose answer you know") is one sentence; the story is a changelog entry, and it dates the notebook. None of it belongs in a notebook.
3. **Triple closings.** Summary, Key Takeaways and What's Next in tutorials 1 to 3; "Congratulations" in the regex tour. One closing section, and only when it says something the body did not.
4. **Code walls.** The longest code cells are 70 to 81 lines (L6 activity, tutorial 3, tutorial 6). A student cannot tell a helper from a step. Either move helpers into the library or label the cell.
5. **Stale text.** Tutorial 3's "uncomment and run"; tutorial 5's What's Next; tutorial 7's purpose line; the COMET notebook's write-token advice; the language-loop comment in lecture-12.

## Implementation directions

Written for the torchlingo session to carry out without coming back for clarification. Two decisions from Eric on September 29 govern all of it:

- **Every notebook runs end to end.** Open the Colab badge in a fresh runtime, Run all, no cell to uncomment, no edit required, no dependency on another notebook's output. That includes lecture-12, which becomes a runnable notebook, and tutorial 3, which stops depending on tutorial 2's checkpoint (#166).
- **The torchlingo session does all of it**, including lecture-07-toy-model and lecture-08a, which were previously Cowork's.

### Rules that apply to every notebook

1. **Standard opener.** Every notebook's first prose cell (the one after the Colab badge and before the generated purpose cell) has this shape. Course activities from Lecture 3 on already follow it; bring the rest into line.

   ```
   # <Title, stating the lesson, not the topic>

   **<In-class activity | Homework | Reading>. <Ungraded. Work in pairs. About 15 minutes. | About N minutes.>**

   <Two to four sentences: the question this notebook answers, and why a student
   in this course needs it now.>

   By the end you will have:

   1. <a thing they will have run, seen or saved>
   2. <...>
   3. <...>
   ```

   The generated purpose cell stays where it is and keeps doing the "what this feeds" job. Do not restate the assignment number in the opener.

2. **One closing section, at most.** Named "What to bring to class" (activities), "What to record" (homework) or "What's next" (tutorials). No "Summary" recapping the body, no "Key Takeaways", no "Congratulations". A closing section exists only when it says something the body did not.

3. **Prose ceilings**, counted across markdown cells excluding the purpose cell: in-class activity 800 words; homework 700; tutorial 1,200. No single prose cell over 250 words. Where a cut leaves a lesson that deserves the full telling, move the long form to a page under `concepts/` and leave a two-sentence pointer.

4. **Code cells a student is not meant to read** (helpers, class definitions, setup) either move into the library or open with the comment `# Setup: run this cell, no need to read it.` No teaching-path code cell over 40 lines.

5. **No project history in notebooks.** Bugs the library shipped, experiments that went wrong, and what used to be true belong in `CHANGELOG.md` and `concepts/when-it-fails.md`. A notebook states the habit in one sentence and does not say where it came from. No "this library once", no "that table used to", no "for a period".

6. **Text rules already in force:** no weekdays or dates in notebook text; the banned word stays banned; `scripts/notebook_meta.py --check` clean after the edit, `--purpose` regenerated where the metadata changed; the runnable check above passes.

### Per-notebook directions

Cells are named by their heading or opening words, not by index.

**lecture-03-word-embeddings.** No change.

**lecture-04-regex-refresher.**
- Replace the opener ("Goal: Learn the fundamental concepts...") with the standard shape. Title: "Regular expressions, for cleaning your corpus". Body: two sentences saying that the A4 cleaning pipeline is written with these patterns (whitespace runs, stray markup, numbers and punctuation that should not split a segment) and that the TMX activity assumes them. "By the end": match literal text and character classes; anchor and repeat a pattern; capture a group with `re.findall`.
- Rename "Cell 1" to "Cell 7" headings to topic names (Literal characters; The dot and character sets; Anchors and shorthands; Quantifiers; Grouping).
- Delete "Cell 7: Summary & Next Steps" entirely, including the regex101 pointer, or reduce it to one line under "What's next".
- Ceiling: 700 words.

**lecture-04-tmx-cleaning.** No change.

**lecture-05-sentence-alignment.** No change.

**lecture-06-mt-evaluation.**
- Replace the Report Back list with three one-line prompts, keeping the closing sentence: "1. One place where BLEU and chrF disagreed, and which you believed. 2. Where in Part 3 BLEU fell fastest, and why a metric should punish a short output. 3. What happens to a 4-gram precision on a two-word sentence." Keep "A very low BLEU on a handful of short sentences is the expected answer, not a bug."
- The 70-line code cell in Part 2: move its helper functions into `torchlingo.evaluation` (or a course helper module) so the cell is the call and the printout, or split it into a setup cell marked per rule 4 followed by the cell the student reads.
- Ceiling: 800 words.

**lecture-06-mt-evaluation-homework.** No change.

**lecture-07-toy-model.**
- Replace the 405-word "What to look at" cell after A4 with this, verbatim:

  > **What to look at.** The seen line should read 11/11 and chrF 100: the model memorised its training set, and a score measured on training data is not a measurement. The unseen line is the only number that says anything about *translation*. If it produced *El perro duerme*, it recombined pieces of three phrases it saw; if it produced something else (often *El perro corre*), sixteen passes over eleven phrases were not enough to learn that *sleeps* is *duerme* whichever animal is doing it.
  >
  > BLEU is 0.0 on both lines because a three-word sentence has no 4-grams; that is the metric's shape, not the model's quality, which is why this notebook reports chrF and exact match.

  Move the "Optional, five minutes" paragraph (change `HELD_OUT` to "Good night") into Part B, after the resume exercise, unchanged.
- Replace the prose of "Also before Lecture 8a: watch a training run survive a disconnect" with one paragraph (under 90 words): the A8 run takes hours and Colab sessions drop; `TrainingCheckpointer` saves at every epoch end and every ten minutes; run the same call again and it resumes. Then the code. Replace the 205-word "What to look at" after it with three lines: the second call printed `resumed from epoch 10` and started at epoch 12; what was lost is the partial epoch since the last save; in A8 the checkpoints go to Google Drive because a Colab runtime forgets its own disk.
- Part C unchanged.
- Ceiling: 900 words (it has three parts, so 100 over the activity ceiling is allowed).

**lecture-08a-a8-kickoff.**
- "Step 4: verify. Do not trust." Reduce to two sentences: the check raises rather than prints, and a contaminated split here is carried into Assignments 9, 13 and 14.
- "Step 6: the model." Keep the first paragraph (the configuration and the parameter count). Replace the "Why this size" paragraph with one sentence: "This size translated better than a model a fifth its size and converged in about 30 epochs where the smaller one needed 65; the trainer keeps your best checkpoint by validation loss, so running longer costs time, not quality." Reduce the bucketing paragraph to two sentences (batches are padded to their longest sentence; bucketing groups similar lengths and removed about three quarters of padded tokens on a 100,000-pair corpus; `padding_report` prints yours). Keep the `num_workers=0` sentence. Drop the note about dropped incomplete batches and the seeded shuffle, or move it to a code comment.
- "The number to compare your first loss against": keep as is.
- Ceiling: 1,000 words (the deliverable list and eight steps justify the extra 200).

**lecture-09-subword-tokenization.** No change.

**lecture-10-comet-install.**
- Replace the "## Installation (Commands come directly from the Github)" heading with the standard opener. Title: "Install COMET and score one example". Body: "COMET is a learned metric: a model that scores a translation, with or without a reference, and correlates with human judgment better than BLEU or chrF. Lecture 11 and Assignment 11 use it on your A6 sentences, so the goal today is only to get it running." "By the end": `comet-score` installed and run on a two-sentence example with a reference; the same example scored without a reference (COMETKiwi), which needs a Hugging Face token; the four file formats A11 expects, seen.
- Fix `hypt2.txt` in the "Look in the files" sentence. Change the `>>` appends to `>` writes so Run all twice does not double the files.
- Replace the long "## The reference-based model does not require..." heading with a short heading ("Reference-free scoring needs a Hugging Face token") and one sentence.
- Reduce the token walkthrough to: create a **Read** token at huggingface.co/settings/tokens; in Colab open the key icon, add a secret named `HF_TOKEN`, turn on Notebook access. Delete the Write-role advice. Keep the login cell and its error message as they are.
- Delete the empty trailing code cell. Add a one-line "What to record" closing: the two scores, and which of the two hypotheses each model preferred.
- Ceiling: 400 words.

**lecture-12-llm-context.** Becomes a runnable notebook (Eric's decision). Build it so it runs end to end on a built-in sample with no edits, and does the A12 task when the student points it at their language.
- Opener in the standard shape. Title: "Does context help an LLM translate a language it barely knows?" Keep the Objective sentence. Replace the Tasks list with the "By the end" list: a loop that translates 50 test sentences at context sizes 0, 5, 10 and 20; a corpus BLEU (with signature) for each size; one plot; the three analysis questions answered in your write-up.
- Fix the loop description: one language, chosen by the student; outer loop over context sizes, inner over test sentences. Delete "The outer loop iterates through each language."
- Data cell: a `DATA_DIR` variable. Default is a built-in smoke-test sample so Run all works with no edits: 70 English-Spanish pairs from `data/example.tsv`, 20 held as the context pool and 50 as the test set, with a printed warning that Spanish is not the assignment and the student must set `DATA_DIR` to their A12 language folder. The A12 languages are Efik, Kiribati, Palauan, Pohnpeian, Yapese, Kosrean and Kamba; the file format of that shared folder is not in the repository, so the loader must be written against the real files. **Eric: give the torchlingo session one language's files, or the format.**
- Model cell: a default small open instruction-tuned decoder model that runs on a Colab GPU without a token, chosen and verified by the torchlingo session (the current `AutoModelForSeq2SeqLM` import points the wrong way for a chat model; the assignment forbids MT-trained models like NLLB). One prompt-building function, and print the full prompt once at context size 5 so the student sees what "adding context" means.
- Evaluation cell: `sacrebleu.corpus_bleu` per context size, signature printed, results in a small table. Plot cell: one line chart. Analysis cell: the three questions unchanged.
- Ceiling: 700 words. Verify against `notes/assignments/A12-directions.md` (the upload list: context files, test set, BLEU report, write-up).

**01-data-and-vocab.**
- Opener in the standard shape. Keep the title. Body: two sentences saying this is the library side of what Lecture 4 does by hand, and that A5's output is exactly what `load_data` expects. "By the end": a parallel corpus loaded and inspected; a vocabulary with its special tokens, and what `<unk>` swallows; a padded batch.
- Delete "Summary". Keep "Next Steps" as a single line.
- Ceiling unchanged; it is already brief.

**02-train-tiny-model.**
- Opener: keep the Colab instructions (they are the clearest in the set). Add before them: "Lecture 7's in-class activity was cut from this notebook. This is the library's version: the same tiny corpus, plus saving and loading the model, which A8 needs." "By the end": a Transformer built from a `Config`; trained to a loss well below `ln(V)`; saved and reloaded.
- Delete "Summary". Keep "Next Steps" as one line.

**03-inference-and-beamsearch.**
- Standalone (#166): train the toy model inside this notebook (a few seconds, as tutorial 2 shows) or load the pretrained checkpoint the way tutorial 5 does; delete the "Prerequisites: requires the model checkpoint from Tutorial 2" warning.
- Fix the Colab text: "Uncomment and run" becomes "Run the next cell. There is nothing to uncomment."
- Opener in the standard shape, pointed at A9: "By the end": greedy and beam decoding run on the same sentence and compared; the three settings A9 asks you to choose, `beam_size`, `alpha` and `max_decode_length`, seen doing something; a BLEU read with its signature.
- "A measurement that tells you nothing": cut to one paragraph (under 120 words): five identical rows mean the experiment cannot answer the question, because a model that memorised twelve phrases is certain; a flat sweep tells you about your setup, not the parameter; the real measurement is in Decoding, on the tutorial 5 model. Drop the three-bullet list and the paragraph after it.
- Closings: delete "Summary" and "Key Takeaways". Keep the numbers (greedy to any beam about +1.2 to 1.6 BLEU; peak at beam 3 to 5; beam 10 loses; `alpha` indistinguishable on this model) as one short paragraph under "What's next", with the link to Decoding.
- Split the 72-line code cell into a setup cell (rule 4) and the cell the student reads.
- Ceiling: 1,100 words.

**04-attention-and-alignment.**
- Keep the opener; it is the model for the others.
- Insert one line before Part 6: "**Parts 6 to 8 are optional.** Parts 1 to 5 are the lesson; the rest connects it to the Transformer and to a real model, if you have time."
- "Read it against the translation" (Part 8): cut to its two conclusions in under 100 words: attention found the right words (four content words, four correct alignments, from a few minutes of training) and the output is still wrong, because alignment is not translation.
- Ceiling: 1,200 words total, with Parts 1 to 5 under 900.

**05-real-translations.**
- Keep the opener through the warning.
- The 710-word closing cell: delete everything from "### How we know that order, and how we got it wrong first" through the `scripts/compare_checkpoints.py` sentence. Move the postmortem, if it is worth keeping, to `concepts/when-it-fails.md`; the notebook does not point at it. Keep the "What would make this better" table (with its "Notice that model size is third" line) and the "Summary" paragraph (rename it "What you saw").
- The 437-word "First, why is that BLEU lower" cell: cut to the sample-size point (two sentences), the signature point (two sentences, since tutorial 7 owns it) and the +1.6 BLEU result with the link to Decoding. Under 150 words.
- "What's Next": delete the Tutorial 4 line (it points backwards). Keep the four Read links and the retrain line.
- Ceiling: 1,100 words.

**06-diagnosing-failures.**
- Opener: keep the title, the two paragraphs and the five-question table. Add one line after the table: "**Use this as a lookup.** When your own run misbehaves, find your symptom in the table at the end, run that one question's cells, and read its Recognize and Fix. Reading top to bottom once is worth doing; rereading it is not." Add the standard "By the end" list (three items: the five checks run against a healthy control; each one seen to fire on a break made on purpose; the procedure table).
- Mark the 56-line setup cell and the 81-line `Vocab` cell per rule 4, or import `Vocab` from the library if the library version fits.
- The two `!!! note` admonitions ("Why the flat loss lands slightly above ln(V)" and "A frozen encoder is subtler than it sounds"): keep the first as two sentences inline (the fresh model is confidently wrong about some tokens, so the flat number sits a little above `ln(V)`; what identifies the failure is that it does not move). Move the second to `concepts/when-it-fails.md` and leave one sentence.
- The Question 4 warning ("This generalizes beyond contamination"): keep its first and last sentences only: "Any comparison where more than one thing changed produces a number that is true of nothing. Change one variable at a time, or you are measuring their sum." Delete the +2.33 BLEU sentences.
- Question 1: delete "This is not hypothetical. TorchLingo's own `data/example.tsv` shipped misaligned for a period, and" and start the sentence at "`shuffle_target_side` exists to reconstruct exactly that failure on any corpus".
- Opener: "lists six real failures from this project's history and what actually caused each one" becomes "lists real failures and what caused each one". Closing: "For the six failures this project actually shipped, and what each one cost" becomes "For the failures behind these checks".
- "Try it yourself": keep three of the five (swap src and tgt for half the corpus; the vocabulary built from `train_df` only; detach the encoder output).
- Ceiling: 1,600 words (five questions, each with a break, a Recognize and a Fix, justify 400 over the tutorial ceiling; the reference table at the end is not counted).

**07-evaluating-translations.**
- Part 3: delete the paragraph beginning "This exact mismatch has already produced two wrong conclusions in this project". The paragraph before it (two libraries, one label, two numbers) already makes the point; "The defence is a signature" follows directly.
- Part 4: delete the paragraph beginning "This was a real bug in `torchlingo.evaluation`". Replace it with one line: "All three metrics route through one helper, so the reshaping cannot be right in one place and wrong in another:" and keep the code cell.
- Everything else stays. It already has a "By the end" sentence in the opener; make it the numbered list.
- Ceiling: 1,500 words (five parts that each change one variable; the length is doing work).

### Order of work

| # | Notebook | Students meet it | Do by |
|---|---|---|---|
| 1 | lecture-08a-a8-kickoff | running now | this week |
| 2 | 06-diagnosing-failures | reading before Lecture 9 | Lecture 9 |
| 3 | 03-inference-and-beamsearch (with #166) | reading for Lecture 10, A9 | Lecture 10 |
| 4 | lecture-10-comet-install | homework after Lecture 10 | Lecture 10 |
| 5 | 05-real-translations | reading for Lecture 10 | Lecture 10 |
| 6 | lecture-12-llm-context (runnable rebuild) | Lecture 12 | Lecture 12; needs the data format from Eric |
| 7 | 04-attention-and-alignment | already read | when convenient |
| 8 | 07-evaluating-translations | already read | when convenient |
| 9 | lecture-07-toy-model | Parts B and C still in use this week | when convenient |
| 10 | lecture-06-mt-evaluation | done | when convenient |
| 11 | 01, 02, regex | reference | when convenient |

No change to lecture-03, lecture-04-tmx-cleaning, lecture-05, lecture-06-homework or lecture-09. Nothing here changes the roadmap's notebook index or any date.

### The one thing only Eric can supply

The A12 data: one language's files from the shared assignment folder, or a description of their format, so lecture-12's loader is written against the real thing.

## Errata and status, September 30

From the repository session's first pass (nine notebooks merged; Lecture 10's three and lecture-12 remain):

- The Lecture 6 activity's 70-line code cell is a data fixture (twenty references and two systems' outputs), not a helper; it stays visible with a "skim it" line. Rule 4 does not apply to fixtures.
- Tutorial 1's opener: A5's output is two sentence-aligned text files, which go through `parallel_txt_to_dataframe` first; not "exactly what `load_data` expects".
- Tutorial 7 had a third history passage (Part 2) the directions did not list; removed under rule 5.
- The regex refresher is a fill-in worksheet, which cannot meet "runs end to end with no edits" without a rewrite into worked examples. Left as a worksheet; Eric's call whether to rewrite it.
- Tutorial 4's Parts 6 to 8: Part 8 is now tutorial 8, Lecture 8b's reading; Part 6 is optional, Part 7 is not (8b starts from it). The direction "Parts 6 to 8 optional" is superseded.
- Rule 6's "the banned word stays banned" is void: "instructor" was never banned, and the check is removed.
- Notebooks carry no due dates at all, including the purpose cell's "(due at Lecture N)", which is no longer generated. Read the standard opener with that in mind.
- The lecture-12 direction's reference to `notes/assignments/A12-directions.md` now means `torchlingo-private/notes/assignments/`.
- Word ceilings were treated as targets, not limits, where what remained was teaching (tutorials 6 and 7 landed above them).
