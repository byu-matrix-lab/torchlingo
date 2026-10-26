# The two charts are generated from the roadmap

*From the repository session. Draft, Mon Oct 26: it becomes the hand-off when the charts are on
`main`. Answers `from-cowork/2026-10-26-g-roadmap-as-the-one-source.md`, sections 1 to 5.*

## Read this first

| Status | What |
|---|---|
| **Waiting on Eric** | PR #356 merges on his OK; its `merge_check.sh` is `CLEAN` against `main`. |
| **Built, lands after PR #356** | Section 4: both charts and the Lecture 12 slide variant, generated and checked in CI. |
| **For you, once it lands** | Render from `notes/roadmap/generated/`, retire the hand-kept `.mmd` files, and add a due-date line to each transcript (section 5). |

## 2, 3. PR #356

Reviewed; no changes needed. `merge_check.sh` ran the full preflight on it merged with `main`:
`CLEAN`. Landing needs Eric's OK, because it touches notebooks and tests and not only `notes/`.
Points are not rendered in the window table yet. Nothing asks for them there, and the table
covers only A7 onward.

## 4. The charts

**Where.** `python scripts/render_roadmap.py --write` writes four files to
`notes/roadmap/generated/`: `assignment-roadmap.mmd`, `assignment-roadmap-l12-slide.mmd`,
`tutorial-map.mmd` and `assignment-roadmap.mermaid-config.json` (your config, unchanged). CI's
`--check` fails if any is stale or hand-edited, and so does `notebook_meta.py --check`, since a
tutorial's `leads_to` draws the map. I chose this repository over writing into
torchlingo-private: CI cannot see that repository, so a copy there would go unchecked. Please
**retire the hand-kept `.mmd` files** in `torchlingo-private/course/figures/` and render from
the generated ones. Each file's header carries its render command.

**The same pictures.** Below their headers, the generated files hold your statements in your
order (node declarations, edges, `~~~` anchors), and the class lines differ only in order. So
Mermaid should lay them out as before. I could not render here (no `mmdc`), so please glance at
the first render.

**One fact changed:** the Lecture 12 slide's A14 arrow read "a two-way model to build on".
The main chart had moved to "fine-tuning a pretrained model", and both now say that, since
they share one list of edges. The slide file is `flowchart TB`, as yours was, not left to right.

**Where each fact lives now** (`notes/roadmap/roadmap.yaml`):

- `group` on each assignment: `spine`, `eval`, `llm` or `early`, its colour.
- `charts.assignment_edges`: each arrow, in drawing order, with `carries` (the label) and
  `prepares: true` for a dashed one (A7 to A8). Your labels, verbatim.
- `charts.final_project`: the FP node and its "builds on any of these".
- `charts.slides`: a slide variant is a name and a `here`. A Lecture 13 slide with A13 marked
  is one line, `{name: assignment-roadmap-l13-slide, here: A13}`.
- `charts.tutorials`: per tutorial, the arrow labels keyed by target, a short `title` where
  the node's differs from the notebook's (Tutorials 3, 8 and 9), and `read_after` (Tutorial 9:
  A14). Which arrows exist is still each notebook's own `leads_to`. The render fails if a
  tutorial gains a `leads_to` without a label here, so the chart cannot show an unlabelled or
  invented arrow.

**Tutorial 9 leads to the Final Project.** `FP` is now a valid `leads_to` target. Tutorial 9's
purpose cell says "A head start on the **Final Project**". The read-after arrow's two heads,
and removing one after rendering, are as before; the map's header says how.

**Changing a chart from here on:** edit `roadmap.yaml` (or a notebook's `leads_to`), then run
`render_roadmap.py --write`. The `.mmd` files are never edited by hand.

## 5. The published copies

I ran a one-off comparison today: every transcript's title and **Total** row (A1 to A15)
matches `roadmap.yaml`. A standing check needs a due date it can read, and the transcripts
state theirs in prose, where they state it at all. **Asked of you:** give each transcript one
line under its title, `Due: 2026-11-10, 11:59 pm` (the format `roadmap.yaml` uses). I will
then add the check to torchlingo-private, where both files are visible. Until then, please do
the comparison by hand at each hand-off, as you offered.

## 1. Your Oct 25 answers

All four received, nothing left open. The A14 rubric's 30 is still Eric's to confirm. If it
changes, then `roadmap.yaml` changes first, as section 3 says.

## Also

- **Tutorials are required reading** (Eric, Oct 26), now in the roadmap's decisions log. No
  roadmap prose called them optional or background. The task list's ranking note now says both
  things: required, and no points.
- **Tutorials into assignments, next offering:** recorded with the audit of A7 to A11
  notebooks, to be checked against the notebooks themselves when that audit is taken up.
- **Roadmap prose:** the notebook index's "Tutorial numbers" line gained 09 fine-tuning, and
  says that `FP` in its "starts" column is the Final Project.
