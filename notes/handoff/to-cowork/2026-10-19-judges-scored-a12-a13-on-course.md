# Lecture 11 judges scored; A12, A13, 8a, 8c and 9 on `course`; four things we need

**Sent 2026-10-19.** Answers `from-cowork/2026-10-19-lecture-11-scores-today-roadmap-a10-a11-and-answers.md`.
The baton is yours.

## What we need from you, in one place

1. **One student's MTEval results file from A6** (section 4, A11). Eric says you can get one
   through Learning Suite. Put it, with the student's name removed, in
   `torchlingo-private/course/data/a6-mteval-sample/`, and say which A6 submission it came from.
   With its columns known, A11's Step 1 can read the file every student already has, and compare
   COMET with their own ranks sentence by sentence on the 50 they ranked.
2. **Lectures 14 to 23 and the final project: what do they need from this side?** The roadmap
   gives them no notebook, by design. Two candidates we can see: Lecture 18's edit-distance
   activity, and Lecture 19's IBM Model 1 worked by hand, set beside 8a's attention alignment
   (once Lecture 19 is rescoped). And would the projects want a page on starting a project from
   your own A8/A9 model? Asked now so the answer comes back with everything else.
3. **The A13 handout, rewritten against the notebook.** The notebook's Step 1 now says what the
   handout should (section 3): the Church TM first, public text only to top it up, no minimum,
   and the counts by source in the write-up.
4. **Carried from your last hand-off, still open on your side:**
   - A12's write-up questions into its Learning Suite page and rubric, after Eric sees them. The
     notebook's are in its Step 6: five questions, the fourth new (what no number of example
     pairs can fix).
   - The 8b and 8c pages' Colab badges.
   - Slides that tell students to copy the first loss down: 8a keeps it in the checkpoint, so
     there is nothing to copy, and now records it on a corpus of any size (section 3).

One question stays with Eric, not you: **whether any lecture after 12 has students prompt an
LLM.** It decides whether A12's prompting code becomes a library helper.

## 1. Lecture 11: the four judges, scored by 08:30

In `torchlingo-private/course/figures/l11-judges/`, committed and pushed:

| file | what |
|---|---|
| `scores.tsv` | per sentence and system: `comet`, `cometqe`, `gemba_ref`, `gemba_noref` |
| `summary.md` | per round, each judge's system averages and its A/B preference on each sentence |
| `README.md` | the models, the prompt verbatim, the parse, and what stands out |
| `gemba_raw.json` | Qwen's full answers, reasoning included |
| `score_judges.py` | reruns all of it |
| `gemba_live.py` | for class: `python gemba_live.py "source" "translation" ["reference"]` prints both GEMBA-DA scores; `--show` adds the prompt and Qwen's reasoning |

COMET is `Unbabel/wmt22-comet-da` with the reference; COMET-QE is `Unbabel/wmt20-comet-qe-da`.
Qwen is `qwen3.5:9b` through Ollama on Eric's Mac, at temperature 0, with thinking off. Two
full runs gave identical scores. The prompt is Figure 1 of Kocmi and Federmann, checked against
the paper's PDF. The paper's appendix and the GEMBA repository each word it slightly
differently; the README says how. Qwen writes "**Score: 5/100**" followed by a paragraph of
reasoning, so only the first line is parsed, by GEMBA's own rule. All 48 answers parsed.

**What stands out, for the slides:**

- **Sentence 6, "equidad": only Qwen reacts to the wrong reference, and unevenly.** With the
  reference it gives Google 65 and DeepL 92; without it, 92 and 92. Both say "capital", so Qwen
  penalized one of two equally "wrong" translations. COMET prefers Google, 0.8608 to 0.8444.
- **Sentence 2, "Amygura Network": no judge reacts at all.** Google and DeepL score alike
  everywhere (COMET 0.8924 and 0.8909; Qwen 95 and 95; COMET-QE 0.6564 each, a real tie).
- **COMET-QE is fragile on a correct sentence.** Sentence 8, in Round 1: Google's "un futuro
  donde" scores 0.0339, DeepL's "un futuro en el que" 0.5298. Both are fine Spanish.
- **Round 2: COMET and COMET-QE pick different systems.** COMET prefers Google on all four
  sentences; COMET-QE prefers DeepL on average, 0.4915 to 0.4675.
- **Qwen without the reference never separates Google from DeepL** in Round 2.
- **Round 1 is unanimous:** every judge prefers DeepL to TorchLingo on every sentence.

**Not done:** the optional `opusmt` column.

## 2. Committed, and the roadmap

- **`torchlingo-private`**, four commits on `main`, pushed: the A10 to A12 directions and the
  index; the deck build scripts since Oct 12 (Lecture 10 v11 and v12, Lecture 11 v1 and v2,
  `deck_lib.py`, the three passes); the two Lecture 10 figure folders; and the Lecture 11 judges.
  `l11_v1.py` is committed as the set-aside draft.
- **Your roadmap edits are rendered, checked and merged**, all as you wrote them. One line was
  rewrapped. This side added:
  - **The grader:** its tooling row and its open question now say the source exists, one
    `grader.py` in the lab's GitHub organization, and that two things block pointing the decks
    at it, both Eric's. The repository is private, and it has no license. Both were checked
    today.
  - **A13:** the open question "where does A13's monolingual text come from?" is closed, and
    Eric's Oct 15 decision is in the decisions log.

## 3. On `course` now: what students get

- **A12 reads the real folder.** The loader reads exactly `src.txt` and `tgt.txt`. If either is
  missing, it says so, and names what it found instead. It copes with a byte-order mark, Windows
  line endings and a missing final newline. Step 1 now says where the data is: "the data link on
  the Assignment 12 page in Learning Suite", then `LLM_Assignment_Data`, then the language's
  folder, copied to `MyDrive/CS479/a12-data`. The language list is Efik, Kosraean, Palauan,
  Pohnpeian, Telugu, Yapese; Kosraean is spelled as the language is, though the folder says
  `Kosrean`. It was tested on files built to your description, not yet on a real folder. **If
  you can see one, the first three lines of a language's `src.txt` and `tgt.txt` would close
  that.**
- **A13's monolingual text** follows Eric's order. Step 1 reads `mono-tm.txt`: the student's
  language side of the TM segments A5 left out, which the student extracts with their own A4/A5
  code. It then reads an optional `mono-public.txt`, which only tops up. It removes anything in
  the split and any repeat, and runs the contamination check against the test set. It keeps the
  count from each file for the write-up, and question 3 asks for them. "At least as many" is
  gone, replaced by "no minimum".
- **8a records a first loss on any corpus.** With no minimum corpus in A8, an epoch could be
  shorter than the 50 batches between log lines, and then no first loss was printed. The
  interval now shrinks for a small corpus.
- **8c and 9 quote their lectures' new outcomes**, cut to what each notebook teaches.

## 4. Your nine answers, and what happened to each

1. **A11:** your definition of average rank, and the per-column `.txt` default, both stand, and
   the notebook is unchanged. The MTEval results file is the open ask above.
2. **A12 data:** done (section 3).
3. **A13 text:** done (section 3); the handout is the open ask above.
4. **8c outcomes:** in the roadmap, and quoted by the 8c notebook.
5. **A6, web MT output:** nothing to change. The roadmap's notebook entry already says students
   paste in ten web MT translations, and the notebook agrees.
6. **A8, no minimum:** done (section 3, 8a).
7. **LLM prompting after Lecture 12:** with Eric.
8. **Write-up questions:** A10 and A11 are on Learning Suite, thank you. A12 is yours after Eric,
   and A13 comes with its handout.
9. **Badges and slides:** still open (above).

## Tasks moved on this side

Closed: the A12 loader (pending a real folder), the A13 monolingual source, the 8a first loss for
small corpora, the 8c and 9 outcome quotes, the grader's roadmap entry, and the A6
by-hand-or-web question. Waiting on you: the MTEval results file. Waiting on Eric: whether a
later lecture prompts an LLM.
