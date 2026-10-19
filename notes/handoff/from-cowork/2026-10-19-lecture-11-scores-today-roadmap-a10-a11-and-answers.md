# Lecture 11 judge scores needed this morning (class at 11:00 MT); roadmap edits, A10 and A11, and six answers

**Sent 2026-10-19, 08:05 MT** (Eric passed the baton; the draft was last updated Oct 15). Answers `to-cowork/2026-10-14-one-roadmap-and-every-notebook-audited.md`. Section 5 is urgent: Lecture 11 is taught today at 11:00 MT, and the deck's activity needs the scores by **10:30 MT**. Everything else can wait.

## What needs you

0. **Now, by 10:30 MT: the Lecture 11 judge scores** (section 5). Lecture 11's message (Eric, this morning): *every judge of translation quality, an LLM included, is a metric, and it has to be tested against people.* The room scores two rounds; the deck sets the room beside COMET, COMET-QE and Qwen with the GEMBA-DA prompt.
1. **Render, check and commit the roadmap edits** listed in section 1. Nothing here was run.
2. **Commit** the refreshed transcripts and README rows in `torchlingo-private/notes/assignments/` (A10, A11, A12), `course/figures/l11-judges/`, `course/figures/l10-greedy-vs-beam/`, `course/figures/l10-worked-example/`, and the `course/build/` scripts written since Oct 12 (Lecture 10 `_v12` scripts, `l11_v2.py`, `deck_lib.py`, `native_bullets_pass.py`, `strip_empty_fills_pass.py`, `sweep_lecture_numbers.py`).
3. **A11, Step 1:** consider reading MTEval's rank `.tsv` directly (section 2, question 1).

## 1. Roadmap: what changed in the tree

The desktop roadmap is retired. Every version of it is in the course folder's `Archive/Roadmap (retired Oct 14)/`, and this side no longer edits it. Eric asked whether anything in it still needed folding in. Nothing did: all 21 decisions-log entries of desktop v16 are in your `16-decisions-log.md`, and the scope-note detail your audit cut was cut on purpose. What was missing was newer, recorded only in this side's log, and it is now in the sources:

- **`sections/16-decisions-log.md`:** two entries above Oct 13.
  - **Oct 12:** the three cross-deck moves (8b to 8a; 9 to 8c twice), AI use at Level 1 stated in the syllabus and Lecture 1 only, open-book quizzes, and the Colab computing note.
  - **Oct 14, evening:** the Lecture 10 redo and its seven themes; Lecture 11 not repeating Lecture 10; the A10 and A11 changes; the read-one-skim-one rule for the A10 pair and the Quiz 11 pair; the desktop roadmap retired.
- **`roadmap.yaml`:**
  - **Lecture 8c outcomes added, as you asked.** The multi-head, encoder-block, decoder, word-order and capacity-curve items were sitting under Lecture 9 from before the split. They move to 8c, with a deployment item, shown by the A8 report back and Quiz 8c.
  - **Lecture 9's outcome is now its own subject:** the fixed vocabulary, morphology, BPE by hand, `<unk>` before and after, one variable changed.
  - **Lecture 10's deck reads "F2026 rebuilt; redo planned".** Your 8c notebook can now quote 8c's outcome.
- **`sections/10-lecture-scope-notes.md`:**
  - **8c and 9:** headings name the adopted decks (8c `_v13`, 9 `_v16`), and the order and deployment material moves from 9 to 8c. Lecture 9 also gains the morphology figures, the pipelines as pseudocode, and the Subwords by Hand exercise.
  - **Lecture 10:** taught from `_v2`, with the redo pointed at `Roadmap/Lecture 10 Redo Plan_v1.md` in the course folder. Its Sets line now has A10 at 6 points.
  - **Lecture 11:** its new scope, and A11 at 26 points with the four questions.

## 2. Your nine questions

1. **A11 (A6 files and MTEval).** Of your three questions, I can answer one and part of another:
   - **The average rank: your definition is right.** MTEval's user guide (Fall 2025 folder, `MTEval User Guide.pptx`): "Compute the average ranking for each column. The MT model with the average closest to 1 has the highest quality." A rank assessment enters 1 or 2 beside each translation.
   - **The A6 files: A6 prescribed no layout** for the 500+ sentences. Students ran their own bilingual data through two web systems, so the per-column `.txt` default with the spreadsheet tip is a fair guess.
   - **The export's columns: only the upload format is known.** MTEval's upload for a rank assessment is a header row, then `source<TAB>model_1<TAB>model_2`. The guide shows the downloaded results only as screenshots, so the result columns are not confirmed. A6 asked students to submit that results `.tsv`, so they all have one. If Step 1 also accepted it (the 50 ranked sentences), then sentence-level human comparison would be possible for the ranked subset. Say whether you want a student's file to check the columns; Eric or Coulson can supply one from A6 submissions.
2. **A12 data: found (read-only look, Oct 14).** The A12 data is the `LLM_Assignment_Data` folder inside "Church TM Data for Students", in two OneDrive copies: Steve Richardson's `CS479-Fall2025` share and Eric's `Eric Ringger - CS479-Fall2026` folder. Students are pointed to Eric's copy, never to Steve's (Eric: Steve's folder is a reference only). **For Step 1:** students follow the A12 page's link to Eric's shared "Church TM Data for Students" folder (`https://byu-my.sharepoint.com/:f:/g/personal/ekr8_byu_edu/IgD-HN8O7OIgT54tYYC1RK_cAej1dhSqBhVnzQL78chPfU4?e=KeHkYt`), open `LLM_Assignment_Data/<language>/`, and copy `src.txt` and `tgt.txt` into `MyDrive/CS479/a12-data`. The page also keeps Steve's copy as a second link for now. Better to name the page ("the A12 page's data link") than to paste the URL into the notebook, so a changed link is fixed in one place. The A12 page's notebook link now opens `lecture-12-llm-context`.
   - **Layout, one folder per language:** two line-aligned plain-text files. `src.txt` is English and `tgt.txt` is the language. There are no tabs, no header and no test split, so the student carves the test set out themselves.
   - **Your loader's guess does not match:** it looks for a file whose name contains `en`. Here it is `src.txt` (English) and `tgt.txt`.
   - **Sizes, in lines:** Efik 22,611; Kosraean (folder spelled `Kosrean`) 70,208; Palauan 42,662; Pohnpeian 50,818; Telugu 49,611; Yapese 54,080.
   - **The language list now matches the folder** (Eric). Learning Suite's A12 says Efik, Kosrean, Palauan, Pohnpeian, Telugu or Yapese; Kamba is gone. The roadmap's Lecture 12 scope note is edited to match, and your notebook's language list should follow.
   - **Content:** Church general conference text, in the same order in every language (the first line is "Jesus is the Christ.").
   - **Your two smaller asks, from last year's notebook** (`Fall 2025/Jupiter notebooks/LLM_Assignment.ipynb` in the course folder):
     - **No model was recommended.** It was a TODO skeleton: install, a TODO to load the data, a TODO to choose a model, then a loop, a plot and three questions. Its model cell imports `AutoModelForSeq2SeqLM`, an encoder-decoder class at odds with the decoder-only framing, so keeping Qwen2.5-1.5B-Instruct as the default loses nothing.
     - **Keep its three questions:** optimal context size and diminishing returns; why context helps or hurts; challenges met. Your write-up already carries them.
     - **Its language list was wrong last year too:** it names Kiribati and Kamba, and neither folder exists.
3. **A13 monolingual text: from the Church TMs** (Eric: "they have more data from the Church TMs"). Each student's translation memory holds more than A8 trained on. A5 had medium- and high-resource students prepare at least 200K pairs, not all their data.
   - **Where `mono.txt` comes from:** the language side of TM segments the student did not use in A8 and A9.
   - **The check that matters:** none of those sentences may appear in the A8/A9 test set, or the BLEU comparison leaks. Step 1 could say so and check it, the way A8's contamination check does.
   - **The source order** (Eric agreed, Oct 15), the same for every student:
     1. Unused Church TM segments first: the target side of segments not used in A8, deduplicated against the A8/A9 test set and the training targets.
     2. If the TM runs short, top up from public monolingual text (Wikipedia in the language; MADLAD-400 or CC-100 on the Hub; OPUS), deduplicated the same way.
     3. No minimum size. The student reports how many sentences came from where; question 3 (what makes the language low-resource) is where a shortage, or a domain shift from TM text to web text, gets written up.
   - **Never carve "monolingual" text from the student's own A8 training targets.** Back-translating sentences the model already saw paired adds almost nothing.
   - **The A13 handout rewrite** stays on this side's list.
4. **8c outcomes: done** (section 1).
5. **A6: students paste web MT output; they do not translate by hand.** A6: "Run at least 500 (1K would be better) random source sentences ... through two MT systems of your choice," then rank 50 in MTEval. The notebook is right, and the roadmap's old "translate ten by hand" was wrong.
6. **A8 minimum corpus: there is no hard minimum.** A8 says "At least 100K training pairs ... If your cleaned data has less than that, use all of it and say so," and A5 has low-resource students prepare all their data. A corpus under 3,200 pairs is possible in principle, so 8a logging more often for small corpora is the safe choice.
7. **LLM prompting after Lecture 12: open, for Eric.** One existing case: A11's extra credit lets a student with API access prompt an LLM for a DA score per sentence. That runs outside the notebook.
8. **Write-up questions:**
   - **A10 and A11: added verbatim,** with points (below).
   - **A12: not yet.** A12 is set at Lecture 12 (Wed Oct 21) and goes in the next pass, after Eric sees your two new questions.
   - **A13: comes with its handout rewrite.**
9. **Badges and slides:**
   - **Tutorial 8:** its badge is on the Lecture 10 Content page under Additional Resources. The Lecture 10 deck is not being patched; it is being redone.
   - **8b and 8c:** their pages' badges are not done yet.
   - **Slides that say to copy the first loss: not yet checked.**

## 3. Learning Suite, Oct 14 (all live)

- **A10 page** (`HokS`):
  - **Hugging Face setup:** account, the COMETKiwi license ("Agree and access repository"), and the token as the `HF_TOKEN` Colab secret with Notebook access.
  - **Papers:** "read one carefully and skim the other; you choose which".
  - **Questions and points:** the notebook's three write-up questions; one rubric row each; 6 points, and the gradebook followed.
- **A11 page** (`jLj_`):
  - **Rewritten against `lecture-11-comet-on-a6`.** It names what is the student's (comet_inputs, pairwise_agreement, the MTEval ranks, the prediction) and the GPU disconnect.
  - **The install is gone.** Eric: A10 did the install, so "Install COMET" is out of the directions and the title. The page and the assignment entry are now "Assignment 11 - Run COMET models on the Assignment 6 sentences".
  - **Questions and points:** the four summary questions, the old correlation row becoming question 1. Questions 2 to 4 are worth 2 points each, 26 in all, and the gradebook followed.
  - **Extra credit:** the xCOMET-XL license note.
- **Lecture 11 page** (`RYOh`):
  - **Quiz - Lecture 11:** reads GEMBA (Kocmi and Federmann, EAMT 2023) and xCOMET (Guerreiro et al., TACL 2024), one carefully, one skimmed. Both are linked to the ACL Anthology, not attached.
  - **A11:** linked, "set today, is due Monday".
- **Quiz - Lecture 11:** one question per paper (GEMBA: state of the art at system level; xCOMET: error spans with severities). It is now 8 questions. Its six older questions come from last year's deck and will be checked against the rebuilt Lecture 11.
- **Attendance Quiz 11:** one reading question on the A10 pair (2 / 1 / 0).
- **Lecture 10 page** (`WuRS`): the tutorial 8 badge, and the read-one-skim-one rule on the quiz line.

## 4. Lecture 11 deck: a draft set aside

`course/build/l11_v1.py` built a claim-first Lecture 11. Eric then judged Lecture 10's rebuild text-heavy and full of snapshots of old slides, and this draft has the same faults. It stays as a script, uncommitted to the course folder; its renders are in `runs/cowork-renders/`, which is ignored. Commit the script or not, as you prefer. It will be superseded after Lecture 11's interview.

## 5. Lecture 11's "be the judge" activity: please score it, by 10:30 MT today

The deck is `Lectures 11 - LLM Judges, Sentences and Spans_F2026_v1.pptx` in the course folder (built by `course/build/l11_v2.py`). Data: `torchlingo-private/course/figures/l11-judges/sentences.tsv`, eight TED sentences (tutorial 7's test set, one talk) with columns `id src ref google deepl torchlingo`.
- **`torchlingo` is already there** (added Oct 15 on this side: tutorial 7's pretrained model, `torchlingo.inference.greedy_decode`, max_len 60).
- **Two references are wrong on purpose:** sentence 2 ("Amygura Network" for Omidyar) and sentence 6 ("equidad" for equity). Keep them; slide 14 is about them.
- **The rounds:** Round 1 is sentences 1, 4, 7, 8, with A = `torchlingo` and B = `deepl`. Round 2 is sentences 2, 3, 5, 6, with A = `google` and B = `deepl`. The "Four Judges" slide shows Round 2.

Please write, in that folder:

1. **`scores.tsv`:** one row per sentence and system (all eight sentences; systems `google`, `deepl`, `torchlingo`), columns:
   - `comet`: `Unbabel/wmt22-comet-da`, with the reference;
   - `cometqe`: `Unbabel/wmt20-comet-qe-da`, the model A11 uses;
   - `gemba_ref` and `gemba_noref`: Qwen on Eric's machine with the GEMBA-DA prompt (Kocmi and Federmann 2023, Figure 1, verbatim; the no-reference variant drops the reference line and "with respect to the human reference"). Record the Qwen model name and the exact prompt text in the README.
2. **`summary.md`:** per round, the system averages per judge, and for each judge, which output (A or B) it prefers on each sentence. Flag anything surprising: for example COMET and COMET-QE splitting on sentences 2 and 6, or Qwen with and without the reference splitting.
3. **The script** beside the data, so it can be rerun.
4. **Optional, if time allows:** `gemba_live.py`, which Eric can run in class to score any sentence pair with Qwen and print the two GEMBA-DA scores, so the room can test the LLM live.
5. **Optional, lowest priority:** an `opusmt` column (`Helsinki-NLP/opus-mt-en-es`).

If you cannot finish by 10:30, then write whatever is done (even COMET alone) and say so at the top of `summary.md`; this side will put what exists into the deck.

This side cannot reach Hugging Face or the local Qwen server (both are blocked by the cloud allowlist), which is why this comes to you.
