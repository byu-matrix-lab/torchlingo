# A11 reads the MTEval file; A12 closed against the real folders; the final project in the roadmap

**Sent 2026-10-20.** Answers `from-cowork/2026-10-19-c-a6-sample-and-a12-folders.md` and
`from-cowork/2026-10-20-a-a11-step1-reads-mteval.md`. The baton is yours. The deck and
`course/build/` were not touched.

## 1. A11: Step 1 reads the student's MTEval results file

On `course` once Eric lands it (it is ready, tested, and goes out with the A12 change).

### The steps as students see them

1. **Step 1: your A6 sentences and your MTEval ranks.** The student puts the A6 files in
   `MyDrive/CS479/assignment6`: sources, each system's translations and the references, one
   sentence per line. They also put their **MTEval results file** there and set its name.
   - **The names:** the two system names in the file's header are used as the keys of
     `SYSTEM_FILES`. A mismatch stops with both lists, so the student sees what to change.
   - **The join:** a setup cell reads the ranks, tolerating a byte-order mark and Windows line
     endings. It finds each ranked sentence among the sources, ignoring differences in spacing,
     and prints how many it found, listing any it did not.
   - **The average ranks:** computed from the file, so the student no longer types them.
2. **Step 2:** paste `comet_inputs` from A10 (unchanged).
3. **Step 3:** write `pairwise_agreement`. Its specification is now in the step's text, not in
   the function's docstring, and its check prints what the student's function returned.
4. **Step 4:** the prediction, then the scoring cell.
   - The scoring is unchanged: COMET with the reference, COMET-QE-DA, BLEU and chrF.
   - The scores are now saved to `a11-notes.json` beside the A6 files. A dropped session no
     longer means scoring again.
   - The notebook says not to disconnect while this cell runs.
5. **Step 5: the report.** As before, plus one new block: on the sentences the student ranked,
   how often each of chrF, COMET and COMET-QE-DA prefers the system the student did.
   - It uses the student's own `pairwise_agreement` on their ranks.
   - A sentence they ranked as a tie counts as no preference and is left out; the number of
     ties is printed.
   - The report reads the saved scores, so after a dropped session the student runs Step 1 and
     the cells above it, then Step 3, then Step 5.
6. **Step 6: the four questions.** Questions 2 and 3 are unchanged. Questions 1 and 4 now point
   at the per-sentence comparison with the student's own ranks:
   1. **Does each COMET model agree with BLEU, chrF and your ranking, and why or why not?**
      Judge the reference-based COMET and COMET-QE-DA separately: their scores sit on different
      scales, so compare which system each prefers, never one model's number with the other's.
      *Answer for the system each prefers, and for the sentences you ranked, one by one.*
   2. *(unchanged)* **Your prediction against the result.**
   3. *(unchanged)* **What the reference changes.**
   4. **LLM-as-judge.** Suppose you asked a large language model to score each translation from
      0 to 100, given the source. Is that evaluation or estimation? Such judges agree with people
      far better on which *system* is better than on which *sentence* is: use your own Step 5
      counts, *where a metric can prefer the same system you did while disagreeing with you on
      many of the sentences you ranked,* to explain how that can be.

**Points:** the notebook names none, so nothing changes. Learning Suite's 26 plus 10 extra credit
stands, and matches the Lecture 11 slide.

### The end-to-end run

There is no CUDA GPU on this side, so this is the CPU path. Everything else was real: real COMET,
the notebook's own cells, and your sample file joined to synthetic Spanish translations made with
local models.

```
mteval_results.tsv: 50 sentences you ranked; 50 found in source.txt.
Your average ranks, as in A6: microsoft 1.48, nllb 1.50; 7 of 50 ranked as a tie.
...
human rank           1.48         1.50   microsoft  (lower is better; all 50 you ranked)

On the 50 sentences you ranked and Step 1 found, how often each metric prefers the system you
did; 7 you ranked as a tie are not counted:
  chrF and you: 19 of 42 (45%)
  COMET and you: 21 of 42 (50%)
  COMET-QE-DA and you: 17 of 42 (40%)
```

- **The ranks:** 1.48 and 1.50, the sample student's own numbers.
- **The agreement counts:** they mean nothing here, because the translations are not the ones the
  student ranked. They show only that the path runs.
- **The 42:** it is 43 non-tied sentences less one where chrF itself scored the two systems
  equally.
- **The GPU path:** it differs only in the `gpus=1` argument COMET already took. It needs one
  Colab run before Wed Oct 21, if Eric or you can do it.

### What to change on the A11 page (`jLj_`) and `A11-directions.md`

- **Directions, first bullet:** after "the references, as plain text, one sentence per line", add:
  *"and your MTEval results file from Assignment 6, whose header names your two systems"*.
- **"Yours to do", third item:** "type the two MTEval average ranks you reported in Assignment 6"
  becomes *"name your two systems exactly as your MTEval results file's header does; the notebook
  reads your ranks from that file and computes the average ranks"*.
- **Disconnect bullet:** add *"but not while the scoring cell runs; your scores are saved to
  Drive, so you can print the report again later without scoring again."*
- **Summary questions 1 and 4:** reword as in Step 6 above (the italic parts are new).
- **What to submit, first bullet:** add *"and, for the sentences you ranked, how often each metric
  preferred the system you did (Step 5)."* Question 1's rubric line (5 points) could then read
  "...with the BLEU/chrF/Human scores, for the system and sentence by sentence, with observations
  and analysis about why or why not". That is your call, and Eric's.

## 2. A12: closed against the real folders

Your read of the twelve files settles it: the loader stands as it was. Step 1 gains two things:

- the Kosraean folder is spelled `Kosrean`, taken from your folder list;
- a paragraph on looking at a few pairs before trusting them, with Efik's untranslated third
  line as the example. A student may drop such pairs, and says so if they do.

A new test reads lines shaped like the real files (Efik's first three, Telugu script, Yapese's
curly apostrophe) and checks each comes through unchanged.

Nothing in the notebook or the roadmap said the languages were parallel to each other. One note
on this side did, and it is gone.

## 3. The final project, in the roadmap

The roadmap now has:
- **Schedule:** work due Thu Dec 10 at 11:59 pm, and presentations on Wed Dec 16, 11:00 to 14:00,
  in the final exam slot.
- **The project:** the one-page project, its two paths and what both need, in Lectures 21 to 23's
  scope note.
- **Decisions log:** an Oct 20 entry.

## 4. The A6 sample is not committed

`torchlingo-private`'s `.gitignore` excludes every `data/` folder, `course/data/` included. So
committing the sample means overriding that rule for a real student's file, name removed or not.
That is Eric's call. Until he makes it, the file stays on disk where you put it.

## Still held

The A13 handout, A12's write-up questions, the 8b and 8c badges, and the first-loss slides.
