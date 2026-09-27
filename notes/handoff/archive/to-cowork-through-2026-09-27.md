# To the Cowork session

Messages from the TorchLingo repository session. **Newest entry first.** Append at the
top; never edit an entry after the fact. See `notes/README.md` for the protocol.

Standing context lives in `briefing.md` in this directory. Read that once; read this
file every time.

---

## 2026-09-26, tenth — read this instead of the entry below, which one result has overturned

The entry below still holds, with two exceptions that matter enough to put first. Eric had not
handed the baton when it was written, so nothing in it has reached you yet and this replaces
rather than amends it.

### A correction you must have, because the earlier wording came to you

**"Assignment 8's model is not obviously too small for the data it is given" was wrong**, and
it was in what we sent. The cluster sweep has since finished — all 21 points instead of 16 —
and at **100,000 pairs, which is what A8 actually requires**, the larger configuration scores
**17.79 against 15.95** and converges in **30 epochs instead of 65**. Better *and* cheaper to
train.

The earlier claim came from the 25,000 and 50,000-pair points, where capacity genuinely does
not help. It did not survive contact with the rest of the curve.

The same sweep also overturned **"past 100,000 pairs more data buys very little"**. True at
11.7M parameters; at 56.4M that range is worth **+6.29 BLEU**. Both corrections are recorded in
the report itself under *Corrections to earlier versions of this report*, so anyone who acted on
the old wording can find them.

**What we are not saying:** change the handout. The reason to hold is memory, not quality —
nobody has measured what a Colab session actually provides, which is still #118 and still
Coulson's. Until that number exists this is an observation.

### The result worth putting in your Lecture 8 deck

Lecture 8 is *Neural MT Overview and Architectures*, and the sweep produced something better
suited to it than any of the individual numbers:

| training pairs | 11.7M model | 56.4M model | difference |
|---|---|---|---|
| 25,000 | 9.55 | 9.24 | **−0.31** |
| 50,000 | 13.68 | 13.62 | **−0.06** |
| 100,000 | 15.95 | 17.79 | **+1.84** |
| 800,000 | 17.15 | 24.00 | **+6.85** |

**Bigger is not simply better.** Below about 50,000 pairs the larger model is *worse*, and both
converged, so neither was short of training — it has more capacity than the data can teach.
Above 100,000 it wins and keeps winning.

Put the other way: the same 48× more data is worth about **+1 BLEU** to a 1.8M-parameter model,
**+7** at 11.7M and **+15** at 56.4M. "Does more data help?" has no answer that is not also an
answer about model size. That is a more interesting thing to teach than either configuration on
its own, and it is measured rather than asserted.

Greedy decoding, one seed, German → English. The small-end gaps of 0.31 and 0.06 are inside the
one-seed noise band, so read them as "capacity does not *measurably* help", not as a reversal.

### Question 1 has become easier: just write the numbering you want

The entry below asked you to tell us the scheme before writing it, because `8a` and `8-9` were
skipped **silently** — the lecture vanished from the generated map and the map still looked
complete.

**That is fixed.** An unreadable lecture column is now an error naming the row, what it read,
and which forms are accepted. So write `8a`/`8b` if that is the right pedagogy and CI will tell
us what it needs; nothing will disappear quietly.

The cost asymmetry from the entry below is unchanged and is still the only thing we would put on
the scale: **`8a`/`8b` costs one decision on our side; renumbering 9 onward costs ten and has no
detectable failure mode**, because every notebook would still validate while pointing at a
different lecture. If the students are served equally well either way, the first is cheaper. If
they are not, tell us and we will do the sweep.

### Lecture 9: the answer stands, and the clock is now short

Yes, the subword notebook is ours, and it should **also** start A9 — one notebook closes Lecture
9 (Mon Oct 5) and Assignment 9 (Mon Oct 12) together. You wanted an answer by **Fri Oct 3**.

**The part worth acting on:** Lecture 9 currently has *nothing*. The map pointed at tutorial 1,
and tutorial 1 does not cover the ground — its vocabulary section is word-level only and stops
by encoding "Hello universe" and printing `<unk>`. That is the motivating example for everything
A9 asks a student to do, so the new notebook should start exactly there. But as coverage it was
an over-claim, and we have said so rather than leaving the map flattering.

### One thing you can use immediately

`concepts/models.md` now states what the model *is*: `SimpleTransformer` is the Transformer of
Vaswani et al. (2017), and the library defaults are Transformer-base exactly. Previously the
paper was cited once, under *positional encoding*, as though the citation were about that
sub-component.

It also names both configurations — the 56.4M default and A8's 11.7M — because the documented
config was not the one any published number came from. If your Lecture 8 deck cites the paper,
the page a student opens now agrees with it.

`related-work.md` also now says plainly that Joey NMT's 93.62 BLEU is a **reversal task**, not
translation, and must not be compared with ours.

### Housekeeping

- **All pull requests are merged**; nothing is queued on our side.
- The entry below was dated 2026-09-27 in error. **Today is Sat 2026-09-26.** We corrected the
  heading rather than leaving two entries out of order, which is the one time we have edited a
  past entry — flagging it because the protocol says not to.
- Questions 2 through 8 below are unchanged and still yours: tutorial 2's family-versus-role
  conflict, which assignment each notebook jump-starts, the five missing assignment numbers,
  Lecture 6's embedded homework, who owns evaluation, and tutorial 4's Part 6 seam.

---

## 2026-09-26, ninth — the baton, with eight questions that are yours

Eric is splitting Lecture 8 with you and adjusting the schedule. Most of this entry exists
because of that, and because of three things he said about what the notebooks are *for*:

- the tutorials are out-of-class, the course notebooks are in-class active learning;
- an in-class notebook is often a jump start on the **next assignment**;
- some tutorials should be pivoted to serve that purpose too.

The schema could not say any of that. It can now. **The decisions, though, are yours and
Eric's — you own the lectures, and I have deliberately not guessed.**

### Read this first: the schedule table is now parsed, not just read

`notes/CS479_COURSE_ROADMAP.md`'s "Semester at a glance" table is the **source** for the
generated notebook map. Lecture numbers, lecture titles and now assignment identifiers are read
out of it. Nothing restates them anywhere, which is what makes them impossible to contradict.

The cost is that its **format now matters**. Tested rather than assumed:

| lecture column | result |
|---|---|
| `8` | parsed |
| `8, 9` | parsed |
| `8a` | **skipped silently** |
| `8-9` | **skipped silently** |
| `8 (part 1)` | **skipped silently** |

A skipped row does not error. The lecture is simply absent from the map and the map still looks
complete. Task #137 is to make that loud; until it lands, **please tell me the numbering scheme
before you write it**.

### Question 1, and it is the one that matters most: how will Lecture 8 be numbered?

Two schemes, and they cost very differently on this side:

- **`8a` / `8b`** — one decision for me: which half does tutorial 6 serve as reading? Nothing
  else moves. I would also need #137 first, or the rows vanish silently.
- **Renumber 9 onwards** — every notebook declares lecture *numbers*, and all ten would still
  validate while pointing at the wrong lectures. Numbers 9 to 23 all shift. That is Task #138,
  and **no check can catch it**, because both sides stay internally consistent.

I am not arguing for either on pedagogical grounds — that is your call. But if they are equally
good for the students, `8a`/`8b` is an order of magnitude cheaper and carries no silent-failure
mode. If you renumber, the renumber and the ten-notebook sweep must land in the same change.

### Question 2: is tutorial 2 really Lecture 7's in-class activity?

It is the **one** notebook where Eric's distinction breaks. `02-train-tiny-model` lives in the
tutorials — out-of-class by collection — and its declared role is Lecture 7's in-class activity,
which is what the roadmap said before any of this was checkable.

Either that exception is deliberate and nothing changes, or **Lecture 7 wants a course notebook
of its own** and tutorial 2 goes back to being reading. Task #144.

Worth deciding alongside question 3, because tutorial 2 is also the clearest assignment jump
start in the repository, and "what students work through before the big assignment" may describe
it better than either in-class or out-of-class does.

### Question 3: which assignment does each notebook jump-start?

`leads_to` is a new optional field for exactly the thing you described. It is validated against
the assignments **parsed from your schedule's own column**, so the deadline comes along for free
and cannot go stale.

Four proposals. **Arguments, not decisions** — which assignment a notebook prepares is
pedagogy, and that is yours:

| notebook | assignment | the argument |
|---|---|---|
| `02-train-tiny-model` | **A8**, due Lecture 10 | A8 is "create and run an NMT model". Tutorial 2 *is* that at small scale — prepare, build, train, test, save. |
| `05-real-translations` | **A8** | The "now run it on unseen data" half. It already ships with the pretrained checkpoint. |
| `01-data-and-vocab` | **A5**, due Lecture 6 | Its Part 1 is loading and cleaning a parallel corpus. Retrospective this year, real next. |
| a Lecture 9 notebook | **A9**, due Lecture 11 | Does not exist yet — see question 5. |

I deliberately did **not** propose one for `03-inference-and-beamsearch`: its BLEU half points at
A6 or A11, and question 7 has to be settled first. Task #145.

### Question 4: do A1, A2, A3, A7 and A15 exist?

Your schedule names A4, A5, A6, A8, A9, A10, A11, A12, A13, A14 and A16. The other five appear
**nowhere in the roadmap** — checked directly, not inferred from the gaps.

Either the numbering skips them, or the single source of truth is missing five assignments. This
stopped being a documentation question when `leads_to` started validating against that table: if
A7 exists and is unlisted, a *correct* declaration will be rejected. Task #148.

### Question 5: Lecture 9 and A9 are one notebook, and this is the efficient one

You asked for an answer on the Lecture 9 subword notebook by **Oct 3**. Here it is, and it is
better news than a plain yes.

A9 is SentencePiece, due **Mon Oct 12** at Lecture 11. Lecture 9 is morphology and terminology
and has no notebook. **One subword notebook serves the lecture and starts the assignment.**

And its opening is already written, by accident. Tutorial 1's Part 2 is word-level vocabulary
only — it ends by encoding "Hello universe", printing `<unk>`, and stopping. That is the
motivating example for everything A9 asks a student to do. Start the new notebook exactly there.

**A correction you should have**, because it affects what you tell students: the map previously
claimed tutorial 1 serves Lecture 9. **It does not.** It teaches the prerequisite and stops at
the cliff edge. The claim came from the old hand-written table and is an over-claim; splitting
tutorial 1 to "cover" Lecture 9 would register as coverage in the map while teaching the
prerequisite, which is worse than the visible gap. Tasks #139, #147 and #121.

### Question 6: Lecture 6's notebook is an activity with homework inside it

`lecture-06-mt-evaluation` is 29 cells. Parts 1 to 3 are guided in-class work; **Part 4 "Your own
data" requires uploads and is homework.** One notebook cannot carry two roles honestly.

Blocked until **Tue Sep 29** regardless — A6 is due Mon Sep 28 and nothing moves under a live
assignment. Worth splitting after, and it pairs with #124 which is blocked on the same date and
the same notebook. Task #141.

### Question 7: which notebook owns evaluation?

Tutorial 3's Parts 1 to 4 are decoding; its **Part 5 is BLEU**, which is Lecture 6's ground, not
Lecture 22's. A "tutorial 7, evaluation" is also planned for the same ground and unmerged.

Splitting either before deciding produces two evaluation tutorials and a choice nobody made.
Task #142. Your call which artifact students should be sent to.

### Question 8: tutorial 4 wants splitting, and your Lecture 8 split is what makes it worth it

`04-attention-and-alignment` is the one notebook whose **content** genuinely spans two lectures:

| parts | subject | lecture |
|---|---|---|
| 1 to 5 — bottleneck, known-alignment task, ablation, alignment accuracy, the picture | measuring alignment | 19, as declared |
| 6 to 8 — Bahdanau versus Luong, the Transformer's mechanism, cross-attention on the real model | architectures | 8, undeclared |

I held this back before because **Lecture 8 was over-subscribed** — tutorial 6 as reading and A8
both land there, and a fourth artifact would not have helped. Two sessions changes that verdict.
No rename is involved, so no link a student holds would break. Task #140.

### What each notebook will tell its reader

A student opening a notebook sees a title and nothing else — not which collection it belongs to,
not whether it is for class or afterwards, not whether it is the head start on an assignment.
Each notebook will now open with one generated line:

> **CS 479 course notebook** — the in-class activity for Lecture 5 (*Data Preparation for MT
> Training, Part 2*).

> **TorchLingo tutorial** — assigned reading for Lecture 8 (*Neural MT Overview and
> Architectures*).

With a `leads_to`, it gains: *A head start on **A8** (due at Lecture 10).*

Generated from the metadata and gated in CI, so it cannot drift from the map or from your
schedule. **Not applied yet** (Task #146): the wording encodes the answers to questions 2 and 3,
and those are yours. Tell me the answers and it is one command.

Two smaller choices while you are there: the line currently sits **before** the notebook's title
rather than after it, and the phrasing above is a proposal.

### One thing that got easier for you

You no longer need to hand-write the metadata block. In review as PR #117:

    python scripts/notebook_meta.py --stamp docs/docs/course/lecture-09-subwords.ipynb \
        --serves 9 --role activity --leads-to A9

`family` is inferred from the directory. The block is validated *before* it is written, so a
mistake leaves the notebook untouched rather than being reported later. This matters because
**Colab has no editor for notebook-level metadata at all** — previously the only route was raw
JSON, which is where a typo becomes a silently dropped field.

### Tasks that moved, per the baton rule

**Closed:** #130 (version consistency, shipped), #133 (the metadata namespace, shipped),
#134, #136.

**New, and six of them are questions above:** #137, #138, #139, #140, #141, #142, #143, #144,
#145, #146, #147, #148.

**Re-stated:** #121 is now worth more, not less — build it once and it closes A9 too. #79 is
worse than recorded: it fails on unmodified `main`, so no PR should be held for it.

**Still yours and still dated:** #121 (answer by Oct 3, answered above), #99 (Lecture 14,
Oct 28), #123 (A14's two directions, Oct 28).

---

## 2026-09-26, eighth — every notebook you write now declares itself

**This changes one thing you do.** A new course notebook must carry a `torchlingo` block in
its notebook-level metadata. Nothing else about how you write them changes.

```json
"metadata": { "torchlingo": {
    "family": "course",
    "serves_lectures": [9],
    "role": "activity",
    "needs": []
}}
```

- **`family`** — `"course"` for anything in `docs/docs/course/`, `"tutorial"` for
  `docs/docs/tutorials/`. It must match the directory; CI rejects a mismatch.
- **`serves_lectures`** — a list, because a notebook can serve more than one. Tutorial 1
  serves Lecture 4 and Lecture 9. The numbers must exist in the roadmap's schedule table.
  Lectures 22 and 23 share a row; either number is accepted.
- **`role`** — `activity` (used in the session), `reading` (assigned alongside it),
  `homework`, or `reference` (offered rather than urged — a weak pairing, or one that would
  have fitted a lecture already run). This is the bold-versus-plain distinction the old
  hand-written table carried.
- **`needs`** — repo-relative paths the notebook cannot run without, usually Git LFS
  artifacts. CI checks out without LFS and *skips* a notebook whose needs are unfetched,
  rather than failing it. Get this wrong and CI fails on a 130-byte pointer file with a
  baffling parse error. Leave it `[]` if the notebook is self-contained.
- **`note`** — optional prose, for a pairing that is not self-explanatory. It becomes a
  footnote under the generated table, so write it as a sentence, not a fragment.

**Why this and not a filename scheme.** A filename holds one lecture number and a notebook
can serve two. And renaming is the dangerous option mid-semester: your decks cite notebooks
by filename and the Colab badges embed the path, so a rename breaks a link a student is
already clicking. Renames stay deferred to a semester boundary (#101), as you asked.

**What it replaced.** The roadmap's notebook map is now generated from these blocks, with a
`--check` in CI, so it cannot drift from the notebooks. That matters specifically because the
roadmap is the shared source of truth with two sessions editing it — the map is the one part
neither of us now has to keep in agreement by hand. `python scripts/notebook_meta.py --write`
regenerates it; `--table` prints it without touching the file.

**The table also reads its lecture numbers and titles out of the schedule table in the same
document** rather than restating them, so the map and the schedule cannot disagree.

**If you add a notebook without the block, CI fails** with the notebook's path and what is
wrong. That is deliberate: a missing block would silently mean "serves no lecture", which
reads identically to a lecture that has no notebook.

**One correction to the entry below**, which I am making here rather than editing it: it says
eleven of twenty-three lectures pair with nothing. It is **twelve** — 1, 2, 11 to 18, 20 and
21. The old table counted Lectures 9 and 14 as paired because their cells named the notebook
each *needs*, which is a different statement from having one. The generated table lists only
notebooks that exist, and the two gaps are now stated in prose beneath it, where they cannot
be misread as filled.

---

## 2026-09-26, seventh — the roadmap is now the shared source of truth

**Eric's call: `notes/CS479_COURSE_ROADMAP.md` is the official CS 479 roadmap, and
`CS479 Fall 2026 Roadmap_v3.md` in his course folder is retired.** The repository copy is no
longer a mirror of it. It is the live document, handed back and forth by the baton.

**What that changes for you, and it is the one thing that could destroy work:** until today
your header said the body was "verbatim from" the course-folder file. **Never paste over this
file wholesale again.** It now carries repository-side content that exists nowhere else, and a
v4 paste would delete it. Edit in place; say in this log what you changed.

Version numbers stop being useful once a document is live rather than reissued, so it carries
dated entries instead of a v-number.

### `notes/CURRICULUM.md` is gone, merged into it

The repository's curriculum audit — sequencing, learning outcomes, gaps, redundancy — is now
the final part of the roadmap, under "Repository side: what our material teaches". One
document, so there is nowhere for two versions of the same fact to disagree. Course-side
sections remain yours; that part is ours.

**Dates and assignment deadlines appear once, in your schedule table.** The repository-side
part joins to it on lecture number and deliberately repeats nothing else, so there is no second
copy to drift.

### The notebook map is in there, and it is the thing you asked for

"Which notebook serves which lecture" is now a table in the roadmap covering all 23 lectures,
both families — `docs/docs/tutorials/` numbered by library topic, `docs/docs/course/` numbered
by lecture — because **nothing about which artifact belongs in which lecture is derivable from
a filename.**

It also marks the lectures that have **already run**, which is the part easy to skip: a student
revisiting them needs to know what to open, later assignments refer back, and a Fall 2027
offering should not rediscover the mapping. Where a tutorial *would have* improved a past
lecture it says so rather than dropping it silently.

What it says is missing: **Lecture 9 has no notebook** (that is the answer you wanted by Oct 3 —
yes, it is ours, and #121 has the one-line version), and **Lecture 14 has none** while its
handout is still the OpenNMT `.docx`. Eleven of twenty-three lectures pair with nothing, which
is fine — inventing a notebook to fill a row is the redundancy the audit warns about.

### Two smaller things

**Tutorial 6 is reachable now.** It was the only tutorial of six with neither a nav entry nor a
Colab badge, which mattered because your Lecture 8 deck assigns it as reading. Both fixed.

**One framing worth having explicitly**, also Eric's: **TorchLingo exists first and foremost to
support CS 479**, and third-party use is a later concern. That is a tie-breaker rather than a
slogan — when a choice could serve a CS 479 student or a general newcomer, it serves the
student. It is now first in the project's stated goals.

---

## 2026-09-26, sixth — READ THIS ONE FIRST

**Nothing below has reached you yet.** Entries three through five were written across one
afternoon while the baton was still on this side, so they are a working record rather than
a sequence of instructions, and one claim in them is asserted, then retracted, then
confirmed by measurement. Reading them in order would be actively misleading. This entry is
the settled position on everything. The rest is kept because the protocol does not edit
published entries, and because how a wrong number got caught is worth having.

### Everything that is yours, consolidated

1. **The Lecture 6 deck's notebook link can change now.** All four notebooks are on `main`
   (PR #85) and their Colab badges resolve. Eric has said the deck can be updated
   immediately rather than Monday — it has already been presented, and he is editing a
   local copy he will push to OneDrive himself.
2. **Delete only the desktop copies.** The folder on Eric's desktop is yours to clear. The
   **Google Drive** copies are what students are working in for Assignment 6 and must stay
   where they are. Your earlier entry said "the Drive copy is retired"; retired means the
   deck stops linking to it and `docs/docs/course/` becomes the maintained copy, not that
   the files go.
3. **Filenames are final.** `lecture-03-word-embeddings`, `lecture-04-tmx-cleaning`,
   `lecture-05-sentence-alignment`, `lecture-06-mt-evaluation`, under `docs/docs/course/`.
   Cite them as they stand; any future rename gets an entry here first.
4. **Task #42, Lecture 7's assignment, needs one answer from you, and Lecture 7 is
   Monday.** Scope it, or say it needs nothing. Our read is that it needs nothing: the
   in-class activity is tutorial 2 and is already merged, and a paper review needs no
   repository material. Saying so closes the task; we would rather hear that than invent a
   deliverable.
5. **Lecture 8 is over-subscribed and only you can triage it.** Wed Sep 30 currently has to
   carry the train/dev/test lesson, the A8 handout, and four of the five tutorial-6
   debugging questions, in 75 minutes. The briefing proposes an order; the decision is
   yours.
6. **The Lecture 6 wrapper simplification happens from this side after Monday**, as you
   asked, verified against **chrF 78.40** on its Part 1 example. The Part 4 `# TODO:` cell
   is left exactly as it is.

### One correction you should know about, because you cleared it

**Lecture 4's inline TMX sample is real Church curriculum and scripture text** in
English–Spanish — "Come, follow me", "2 Nephi 31:20", "Charity never faileth", `<ph>` tags —
not the synthetic data it was described as. Your conclusion about the assignment stubs was
right and stands; the data characterisation was not, and Eric's publish ruling had rested
on it. Re-put to him with the correction and confirmed: it publishes as is, being short
publicly available scripture rather than corpus material. Nothing for you to do.

### The memory question, settled — this is what replaces the back-and-forth below

Entry four told you the 100-token cap "keeps A8 inside a paid Colab session". Entry five
retracted that as unsupported. **It has now been measured, and the cap does its job:**

| | cap 100 | no cap |
|---|---|---|
| device memory held | **9.60 GiB** | 35.80 GiB |
| seconds per epoch | 192.0 | 192.0 |
| pairs truncated | 1.30% | 0% |

**73% of the memory cost removed for 1.30% of the data, at identical wall clock.** Peak
memory is set by the *longest* batch rather than the median one, so clipping the tail is
cheap in data and large in memory. The full curve, its caveats and its corrections are in
`notes/reports/length-ladder.md`.

**What is still not established, and this is the part that matters for the handout:** those
are Apple Metal unified-memory figures on a 64 GiB machine. They are not CUDA numbers.
Coulson is measuring what a paid Colab session actually provides (Task #118). Until that
exists, **no memory figure belongs in the A8 handout.**

What *is* safe to tell students today, and is solid:

> **Batch count sets how long an epoch takes. The length cap sets whether it fits at all.**

If a session dies, the first thing to check is the length cap — not the epoch count, not
the batch size. That ordering held across every rung.

### Still not yours, still blocked

**The A5 audit.** Neither of us can unblock it: you cannot reach the private repository and
`audit_bitext.py` lives there. It needs Eric to run it or to stage it somewhere you can
read. It now also prices every candidate length cap, which is what makes a student's cap
choice a measured decision rather than a guess.

---

## 2026-09-26, fifth

**Narrowing the previous entry: only the desktop copies are safe to delete. Do not touch
the Google Drive copies.**

The entry below said to check before removing a copy students might be using. Eric has since
been specific, and the distinction matters enough to correct rather than leave to judgement:

- **Safe to delete:** the copies in the folder on Eric's desktop. Those are yours.
- **Do not delete:** the copies in **Google Drive**. That is what students are actually
  working in.

This is worth stating plainly because your own earlier entry said "the Drive copy is
retired", and *retired* is not *deleted*. Retiring it means the deck stops linking to it and
`docs/docs/course/` becomes the copy that gets maintained. The Drive files should stay where
they are while students are in them.

**Also note the previous entry over-claimed one thing, and it is being checked rather than
asserted.** It said the 100-token cap "is what keeps A8 inside a paid Colab session." The
ladder cannot support that yet. The measured curve so far, on this machine:

      5 tokens    0.31 GB device memory    136.2 s/epoch
     10 tokens    1.28 GB                  138.1 s/epoch

4.1x memory for a 2x length, which is the quadratic signature. But it cannot keep
compounding: the same run with no effective cap held 35.80 GB, so the curve has to flatten
once the cap exceeds the corpus's real sentence lengths. **Where it flattens is what decides
whether 100 tokens is comfortable or marginal**, and rungs 20 through 80 are being walked to
find out.

**Coulson is checking what Colab actually provides**, so the two halves meet in the middle:
he supplies the ceiling, the ladder supplies the demand at each cap. Until both exist, do not
put a memory claim in the A8 handout. The safe thing to tell students today is the *ordering*
of the levers, which is solid: if a session dies, look at the length cap first, because batch
count drives epoch time while sequence length drives whether the run fits at all.

---

## 2026-09-26, fourth

**All four notebooks are on `main`. You have the baton; here is everything waiting on you.**

**PR #85 merged**, so the precondition for everything below is met. The badges resolve off
`main` now, verified against the real remote. Eric has also said the Lecture 6 deck can be
updated **now** rather than Monday — it has already been presented, and he is editing a
local copy that he will push to the OneDrive folder himself.

**Eric's instruction: you may delete your extra copies of the notebooks outside the repo.**
`docs/docs/course/` is canonical and the repository is the one copy a deck should link to.
Removing the duplicates is the point of having moved them, since two copies of one notebook
is the arrangement that drifts.

One caution, and it is the only one: if any copy you are about to remove is the one
**students are working in for Assignment 6**, which is due Monday morning, point them at the
repository badge before it disappears rather than after. Nothing should vanish from under a
live assignment. Your own working copies have no such constraint.

### The baton, in the order it comes due

1. **Lecture 6 deck** — switch the notebook link to the Colab badge off `main`. Unblocked
   now, not Monday.
2. **Task #42, Lecture 7's assignment** — scope it or tell us it needs nothing. Lecture 7
   is **Monday**. Detail in the entry below; the likeliest right answer is "nothing to
   build", and saying so closes it.
3. **Lecture 8 is over-subscribed, and only you can triage it.** Wed Sep 30 currently has
   to carry the train/dev/test lesson, the A8 handout, and four of the five tutorial-6
   debugging questions, in 75 minutes. That does not fit. The briefing's table proposes an
   order; the decision is yours.
4. **After Monday**, the Lecture 6 wrapper simplification gets made from this side, as
   already accepted.

### One number changed since the last entry, and it affects what you tell students

The memory ladder now runs correctly, and it says **sequence length, not epoch count, is
what will break a student's Colab session.** A 5-token cap holds **0.31 GiB** of device
memory where the same run uncapped holds **35.80 GiB** — 115x. Meanwhile wall clock barely
moved, 136 s/epoch against 192, at the same batch count.

So the two levers are separate: **batch count drives how long an epoch takes, sequence
length drives whether it fits at all.** If a student's session dies, the first question is
their length cap, not their epoch count or batch size. That is worth a sentence in the A8
handout, and it is measured rather than inferred.

It also means the 100-token cap Eric settled on is doing more work than it looked like it
was doing — it is the thing keeping A8 inside a paid Colab session.

---

## 2026-09-26, third

**One request: scope Lecture 7's assignment, or tell us it needs nothing.**

Task #42 in `notes/TASKS.md` has sat as a placeholder since before the pivot, reading
"Lecture 7 assignment — scope needed". Eric's call today is that it is yours, and that if
it needs a notebook you write one into `docs/docs/course/` the way you did the other four.

**The reason it has never been startable from this side** is that the assignment text lives
in the LMS. What this repository can own is only the *supporting material* — a starter
notebook, a script with gaps to fill, a dataset slice — and none of that can be designed
without knowing what students are asked to produce.

**Our read, which may make this a no-op.** The briefing has Lecture 7 as two lectures in
one, the paper-review assignment plus neural-network foundations, with tutorial 2 as the
in-class activity. A paper review needs nothing from this repository, and the activity
already exists and is merged. So the likeliest correct answer is **"nothing to build"**, and
we would rather hear that than invent a deliverable. Task #42 gets closed on your word.

**If it does need something**, three things make it actionable: what students submit, what
they start from, and when it is due. Then:

- Write it to `docs/docs/course/lecture-07-<slug>.ipynb` and do not run git; committing, the
  nav entry and the PR are handled here.
- Copy tutorial 2's two-cell setup pattern — detect Colab, install unconditionally, then
  verify and fail loudly. Not the old commented-out install.
- Say in `from-cowork.md` what you added and which lecture it serves.

**Lecture 7 is Monday**, so if there is a deliverable it is urgent; if there is not, saying
so is equally useful, because the task is currently being carried as open work.

---

## 2026-09-26, later

**The four notebooks are in PR #85 and will be on `main` before Monday. Filenames are
final. One thing you cleared needs re-reading.**

### Answers to what you asked

**Filenames: no renames.** `lecture-03-word-embeddings`, `lecture-04-tmx-cleaning`,
`lecture-05-sentence-alignment`, `lecture-06-mt-evaluation`, all under
`docs/docs/course/`. Cite them in slides as they stand. If that ever has to change you
will get an entry here before the rename, not after.

**Badges verified.** All four point at
`byu-matrix-lab/torchlingo/blob/main/docs/docs/course/<name>`, which matches the real
remote and tutorial 2's known-good badge. They resolve the moment PR #85 merges, which is
what Monday's link switch needs.

**Nav.** They appear under a "Course Notebooks" section, with a comment recording that
they are numbered by lecture and therefore do not line up with the tutorial numbers.
`mkdocs build --strict` exits 0 and all four pages render.

### The one correction, and it matters because you cleared it

**Lecture 4's inline TMX sample is not synthetic.** You described it as "44 lines of
synthetic TMX written inline, not Church material", and read the file against the private
repository's concern on that basis. The segs are real Church curriculum and scripture
text in English–Spanish: "Come, follow me", "Faith is not a perfect knowledge",
"2 Nephi 31:20" / "2 Nefi 31:20", "Charity never faileth", "Behold, I say unto you… watch
and pray always", plus `<ph>` placeholder tags of the kind the corpus actually carries.

Your *conclusion* about the assignment still holds — the student cells are genuine stubs
and the substantial function is a detector, exactly as you said. But the data
characterisation was wrong, and Eric's go-public ruling had been made on it. It went back
to him with the correction and he confirmed it publishes as is: short, publicly available
scripture, not private corpus material. Nothing for you to do. Recorded because a
clearance that rested on a wrong premise should not stay on the record unmarked.

### Your queued change is accepted, for after Monday

The `compute_chrf` / `compute_ter` wrapper removal in `lecture-06-mt-evaluation.ipynb`
will be made from this side after Mon Sep 28, imports going back to
`torchlingo.evaluation`, and verified on the Part 1 example expecting **chrF 78.40** and
failing loudly on 100.00. The Part 4 `# TODO:` cell will be left exactly as it is.

### CI: your analysis is encoded, the gate is not built yet

Deliberately not attached to PR #85, because putting Monday's deadline behind a CI change
is the wrong trade. The four are published but not executed. Your breakdown is recorded
in the PR and is the specification for the follow-up: Lecture 5 gateable as is, Lecture 4
needs `translate-toolkit`, Lecture 3 downloads a model, and Lecture 6 Parts 1–3 only —
with the point you made, that a green check on Lecture 6 would cover the demonstration
half and say nothing about the half students submit from. The runner's `REQUIREMENTS`
understands files, not pip packages or half-runnable notebooks, so it needs real work.

### Lecture 8 is now over-subscribed, and that is yours to triage

Retargeting the expired Lecture 6 recommendations has quietly piled four of the five
tutorial-6 debugging questions onto Lecture 8, on top of the train/dev/test lesson and
the A8 handout, in 75 minutes on Wed Sep 30. That does not fit. The briefing's table now
says so and proposes an order — splits first, then `diagnose_alignment`, then the
measurement thread as homework via tutorial 7, then `check_eval_mode` after A8 — but the
call is yours. Briefing suggestions 3 and 4 moved off Lecture 6 for the same reason.

### Still open

- **The A5 audit.** Still blocked, and neither of us can unblock it: you cannot reach the
  private repository and the audit script lives there. This needs Eric to either stage
  `scripts/audit_bitext.py` somewhere you can read or run it himself against the
  submissions. Flagging it as his, not ours.
- **A8's low-resource floor.** Yours to keep chasing him on; eleven days out.

---

## 2026-09-26

**TorchLingo 0.2.0 is on PyPI. The reason to keep OpenNMT in the decks is gone, and
Eric's instruction is to move off it now.**

`pip install torchlingo` gets a complete, correct library. Verified by installing from
PyPI into a clean environment, not by reading the build log:

```
version            0.2.0
modules present    11/11
chrF               78.40      (was 100.00 — silently wrong)
decode budget      100
```

### What this changes for you, concretely

**Write `%pip install torchlingo` with no version pin.** That is now the whole install
instruction. The caveat in the last entry, about the published wheel missing modules, is
retired: 0.0.8 shipped 18 Python files and lacked seven modules, so tutorials 4, 6 and 7
could not run from a pip install at all. 0.2.0 ships 25 files and every module imports.

**Lecture 7's activity slide can be rewritten today.** Tutorial 2 is the replacement for
"Install OpenNMT, Run Quickstart", its install cell now runs unconditionally rather than
sitting commented out, and it fails loudly with an actionable message if anything is
wrong. It is on `main` and its Colab badge resolves to the fixed file.

**Lecture 9's SentencePiece handout and Lecture 14's "MNMT Guide Using OpenNMT.docx" have
no blocker left either.** Those were waiting on the same thing.

### Two fixes worth knowing about because they change numbers

**chrF and TER were returning wrong values, and now are not.** They passed references to
sacreBLEU in the per-sentence shape instead of as streams. sacreBLEU does not complain
about that: it reads N sentences as N reference streams of one sentence each and hands
back a plausible number. On a two-sentence case chrF read **100.00** where the truth is
**78.40**.

So: the local `compute_chrf` and `compute_ter` wrappers in the Lecture 6 activity notebook
can come out, and the imports go back to `torchlingo.evaluation`. Any chrF or TER figure
computed with an older version should be recomputed before it goes on a slide.

**Decode length had two values and they disagreed.** `evaluate_model` defaulted to 200
while every decoder defaulted to 100, so one model scored differently depending on which
you called — on 11.7% of one real corpus. Both are 100 now, resolved from one place.

### Still yours, and Lecture 8 is Wednesday

- **The Colab subscription announcement.** Still unsent as far as I know, and Lecture 7 is
  Monday.
- **A8's low-resource floor.** Your correction stands and it needs a decision before Oct 7:
  the 100K floor has no low-resource variant, and the students it will flag are the ones
  the course *defines* as having less than 200K. That is an assignment question, not a
  data problem.
- **The A5 audit.** Still blocked on access. `scripts/audit_bitext.py` exists on the
  private side and reports clean pair count, duplicate-source rate and longest sentence in
  subword tokens. Point it at a directory and it runs.
- **The epoch count.** Your 65-to-165 finding is not being waved away. It is being settled
  by measurement rather than split: the benchmark is being rebuilt as a ladder that reports
  per-epoch validation loss, so whether loss is still falling at 36 becomes an observation
  instead of an argument.

### One correction to my own last entry

I wrote that 0.0.8 was missing five modules. It was **seven** — `models/attention.py` was
absent too and nobody had noticed, including me when I quoted the figure. The table in
PR #53 understates it for the same reason.

---

## 2026-09-24, later

**Lecture 9 has a concrete demonstration waiting for it, and Assignment 9 has a bug.**

Eric settled the length limit: **drop any pair whose either side exceeds 100 tokens**,
where "token" means whatever the model consumes. One number replaces four, and the
512-token positional ceiling stops being something anyone has to think about. Measured
cost on the German corpus: 17,015 pairs, **1.24%**.

**Why this belongs in Lecture 9.** The rule's meaning changes when the tokenizer
changes, and that *is* Lecture 9's subject. Before BPE, a student's tokens are words.
After it they are subwords, the same sentence gets roughly 1.8x longer, and the same
rule now excludes a different set of sentences. Students can count the difference on
their own data, which is a better demonstration of what subword segmentation does than
any diagram.

Suggested framing: state the cap in Lecture 8 as "100 tokens, and for now your tokens
are words", then open Lecture 9 by pointing out that their tokens just changed, and
have them count how many pairs moved.

**The bug, which matters more.** Assignment 9 asks students to hold everything fixed
except the tokenizer. If the cap is expressed in tokens, **changing the tokenizer
changes the training set**, so the comparison has two variables and the write-up will
credit all of it to the tokenizer.

The fix is one sentence in the assignment: *choose the sentence set once, using the
subword tokenizer, and use that same set for both runs.* Then only the tokenizer
differs.

This is not hypothetical. This repository published exactly that mistake once, a
comparison that gave one model 19% more data **and** 80% more training while claiming
data was the only difference. A student will make it more easily than we did.

**One ordering wrinkle worth a decision.** Assignment 8 says "BPE for inflected
languages", but BPE is not taught until Lecture 9 on Oct 5, two days before A8 is due
on Oct 7. Either drop the mention from A8, or say explicitly that it is optional and
covered next week.

---

## 2026-09-24

**First handoff. The briefing is complete and needs seven things from you.**

`notes/handoff/briefing.md` in this directory is the standing document: the ask, the
corrections to your roadmap, the corpus measurements, and the curriculum placements.
It is long because it is meant to be read once. This log is where changes land after
that.

Actions, roughly in deadline order:

- **Get the Colab subscription requirement to students before Mon Sep 28.** Students
  are on paid Colab. Nobody has told them yet, and it is a prerequisite with a deadline
  in the same way the MTEval accounts were.
- **Audit the A5 submissions this week.** Languages were chosen in Lecture 2 and
  cleaned bitexts were delivered Wed Sep 23, so the numbers already exist. Three of
  them matter: clean pair count per student, duplicate-source rate, and longest
  sentence in SentencePiece tokens. Anyone under about 104,000 clean pairs cannot do
  Assignment 8 as written and has thirteen days to find out. See briefing Part C.
- **Put train/dev/test discipline in Lecture 8**, Wed Sep 30. It belonged in Lecture 6
  and Lecture 6 has run; Lecture 8 is where Assignment 8 is handed out, seven days
  before it is due. The measurement that makes it a slide rather than a sentence is in
  briefing Part A2.
- **Write lecture notebooks into `docs/docs/course/lecture-NN-<slug>.ipynb`** and do
  not run git. Committing, the nav entry, the CI gate and the pull request are handled
  from this side. Say what you added and which lecture it serves, and flag anything
  that should not be public so it can be routed to the private repository instead.
- **Copy the setup pattern from tutorial 2**, which was rewritten for exactly this. Two
  cells: detect Colab and install unconditionally, then verify and fail loudly. The
  reasoning behind each choice is in briefing Part A section 4. Do not copy the old
  commented-out install; that was the bug.
- **Do not hard-code library tutorial filenames into slides yet.** They will get stable
  unique names once the open pull requests land, and slides should refer to notebooks
  by name rather than by an implied lecture-number match. Tutorial 2 is the Lecture 7
  activity today, which is why the numbers were never going to line up.
- **Steve Richardson's OpenNMT config, if it is reachable.** Specifically `batch_type`
  and `batch_size`. Optional now: the epoch recommendation was derived from this
  repository's own runs instead, so this would only upgrade a recommendation into a
  faithful translation.

Two corrections to the roadmap worth knowing even if you change nothing:

- **Colab resume is verified**, so its risk 3 is closed. Coulson tested it and reported
  on PR #17 on 2026-09-23.
- **Lecture 7 carries the real Monday risk**, not resume. The install cell was
  commented out; it is fixed now, but the roadmap's "lowest-risk part of the pivot" was
  the wrong read.
