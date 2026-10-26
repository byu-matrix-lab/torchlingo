# The roadmap as the one source: generate the two charts (PR #356 to review)

*From the Cowork session. Written Mon Oct 26, morning (Eric: "yes, please write that up"); trimmed the
same morning after Cowork did sections 2 and 3 itself in PR #356 (Eric: "please go ahead"). It also
answers `to-cowork/2026-10-25-a12-landed-tutorial-9-a14-go.md` ("Asked of you", 1 to 4).*

## Read this first: what to do, in order

| When | Section | What |
|---|---|---|
| **First** | **2, 3** | Review and land PR #356 (Cowork's): Tutorials 4 and 8 `leads_to`; points and stretch points in `roadmap.yaml`. |
| **Before Mon Nov 2** (Lecture 14) | **4** | Generate both charts from the roadmap; Cowork stops hand-keeping them. |
| When convenient | **5** | A check that the published copies match. |
| Nothing to do | **1** | Your four asks are done. |

The baton comes back to Cowork when section 4 is on `main`, or when you need a decision.

## Why (Eric, Oct 26)

Eric asked whether the roadmap is a structured doc and the single source of truth (SSOT) for the
rest of the work. It is structured (since Oct 12: `roadmap.yaml` plus `sections/`, rendered by
`scripts/render_roadmap.py`, checked in CI), and your Oct 25 commits brought it up to date: short
assignment names, A12's sentence before, A14 rebuilt on NLLB, Tutorial 9. It is not yet the source
for four things the course runs on:

1. **Points and stretch points.** They live only on the Learning Suite pages and in the course-side
   transcripts (`torchlingo-private/notes/assignments/`).
2. **The two charts on the Assignments page** (the assignment roadmap and the tutorial map). Cowork
   hand-writes them as Mermaid in `torchlingo-private/course/figures/`, copying facts from the
   roadmap and from notebook metadata. They drift as soon as either changes.
3. **Which tutorial leads to which assignment.** The tutorial map now shows links that the notebook
   metadata does not have (section 2).
4. **The published copies** (Learning Suite pages, gradebook, calendar, deck assignment slides). These
   stay hand-kept by Cowork, since nothing can write Learning Suite; but they can be checked.

The aim: one place for each fact, the charts generated, and everything Cowork publishes checked
against the roadmap rather than against memory.

## 1. Your Oct 25 asks: all done

1. **The A14 page** (Learning Suite `x85K`): rewritten from the rebuilt notebook, saved and published
   Oct 26. Four outcomes; what you need (A9 files, `best-a13-reverse/`, about 8 GB on Drive, an A100,
   L4 or G4 runtime); eight steps; the notebook's seven questions verbatim; rubric 30 (files 4,
   notebook 2, questions 6/4/3/3/3/3/2), plus the stretch task (6). The 30 is Cowork's proposal, which
   Eric has not yet confirmed; if he raises it, then section 3's value changes with it. Transcript:
   `notes/assignments/A14-directions.md` (commit 88e21ad).
2. **The Lecture 14 A14 slide:** slide 22 of `Lectures 14 ..._F2026_v1.pptx`, "Assignment 14, Due
   Tuesday, November 10". The rest of that deck is still last year's; Cowork rebuilds it before Nov 2.
3. **Both charts:** on the Assignments page since Oct 26. The roadmap's A14 to Final Project edge reads
   "fine-tuning a pretrained model"; the tutorial map shows Tutorial 9 read after A14 and leading to
   the Final Project. Eric then added two links of his own (section 2).
4. **A14's stretch task:** on the page. The opposite language code on 20 sentences, 6 points, AI
   Level 2.

The tutorials are **required reading** (Eric, Oct 26); the assignment notebooks are self-sufficient
only as a safety net for students who do not read them. If the roadmap's prose calls them optional
or background anywhere, then please change it.

## 2. `leads_to` for Tutorials 4 and 8 (Eric, Oct 26): done in PR #356

Tutorial 4 (Attention and Alignment) leads to A8 ("attention feeds the NMT"); Tutorial 8 (Greedy or
Beam Search) leads to A13 and A14 (both decode greedy and beam; A13 back-translates by sampling).
PR #356 sets both, regenerates their purpose cells and the notebook index, and moves the purpose-cell
test's "deliberate blank" example from Tutorial 4 to Tutorial 9. Please review and land it.

Tutorial 9 has an empty `leads_to` because the Final Project has no assignment id, and the course's
chart draws "A14, read after, Tutorial 9, leads to the Final Project". Section 4 needs a way to say
that. One way: allow `FP` as a `leads_to` target and add a `read_after: ["A14"]` field. Your choice
of names; the generated chart needs only the two facts.

## 3. Points and stretch points in `roadmap.yaml`: done in PR #356

Each `assignments` entry now has `points` and, where there is one, `stretch_points`, from the Learning
Suite pages via the course-side transcripts; documented in the file's header comment, not rendered
yet. From here a points change is written in `roadmap.yaml` first, then on the page and in the
gradebook. A14's 30 waits on Eric's confirmation. Rendering points in the assignment-window table is
your call.

## 4. Generate both charts from the roadmap

**Inputs:**

- **Tutorial map:** each tutorial's `leads_to` (and the read-after fact, section 2), plus a one-line
  label per arrow. The labels are course wording Eric has approved. Today's are in
  `torchlingo-private/course/figures/tutorial-map.mmd`; please carry them over verbatim, and keep its
  layout (Eric, Oct 26): two columns, tutorials top to bottom in numerical order, Tutorial 9 in the
  left column with a dashed "read after A14" back-arrow from A14 (drawn as `T9 <-.-> A14` with the
  A14-end arrowhead removed from that one path; the `.mmd` header says how), and Tutorial 9 to the
  Final Project labelled "sufficiency and budget", for example
  as a `leads_to_labels` map in the notebook's roadmap entry.
- **Assignment roadmap:** the assignment-to-assignment edges, which nothing records yet. Add them to
  `roadmap.yaml`, each with what it carries (the product label), whether it is dashed (a prepares
  edge, as A7 to A8), and the groups that set the colours (spine, evaluation, LLM, early). Today's are
  in `torchlingo-private/course/figures/assignment-roadmap.mmd`; carry the labels verbatim. Lecture
  edges stay out (Eric, Oct 24: they line up with the numbers).

**Output:** `render_roadmap.py --write` writes both `.mmd` files, at their current paths in
torchlingo-private if the script can reach it, otherwise under `notes/roadmap/generated/`, with the
CI `--check` covering them. Keep the current Mermaid config (`assignment-roadmap.mermaid-config.json`)
and class colours, so the images do not change unless a fact does. Cowork renders the SVG and puts
the WebP on the page; that step stays manual, since Learning Suite cannot be written to from CI.

**The Lecture 12 slide version** (`assignment-roadmap-l12-slide.mmd`, left to right with the same
edges) can come from the same data with a direction switch; it is the same chart sized for a slide.

## 5. Optional: check the published copies

A script that reads a Learning Suite export would need Eric's sign-in. A cheaper check: a test that
each course-side transcript's title, Total row and due date match `roadmap.yaml`, run when either
changes. Cowork keeps the transcripts in step with the pages (re-transcribed at each edit), so a match
there is a match with Learning Suite in practice. If you would rather leave it, then Cowork will run
the same comparison by hand at each hand-off.

## Also, for your task list

- **Cowork ran git this once** (Eric, Oct 26: "please go ahead"). PR #356 is on a branch
  (`cowork/roadmap-points-leads-to`) from a separate clone, checked with `render_roadmap.py --check`,
  `notebook_meta.py --check`, `tests.test_render_roadmap` and `tests.test_notebook_meta` (177 OK) and
  ruff, but not with `scripts/preflight.sh`. Please land it the usual way (`scripts/land_pr.sh`), with
  `merge_check.sh` first; the working tree on Eric's Mac does not have its changes.
- **Task #281** (no assignment notebook leaves a lesson to a tutorial, A7 to A11): Eric agreed Oct 26
  on proposals for the next offering that bear on it, in `torchlingo-private/notes/course/NEXT-OFFERING.md`,
  "Tutorials into assignments": fold Tutorial 3 into A8, Tutorial 2 into A6, Tutorial 9 into A14's
  stretch; merge Tutorials 7 and 8; retire Tutorial 5; keep 1, 4 and 6. Judged from the roadmap's
  notebook entries, so please check them against the notebooks when you take up Task #281.
- **Tutorials are required reading** (Eric, Oct 26). Task #281's note says tutorials "carry no points,
  so students pressed for time skip them"; both are true, and the assignment notebooks stay
  self-sufficient as the safety net.

## What Cowork does when this lands

- Renders the generated charts and replaces the two images on the Assignments page.
- Retires the hand-kept `.mmd` files (or keeps them as generated copies, whichever you choose).
- Treats `roadmap.yaml` as the source when editing pages, the gradebook, the calendar and deck
  assignment slides, and sends any course-side change to it in the next hand-off rather than keeping
  it only on Learning Suite.
