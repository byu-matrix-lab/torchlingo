"""Generate ``notes/CS479_COURSE_ROADMAP.md`` from its sources in ``notes/roadmap/``.

The roadmap is what both CS 479 sessions read, and it used to be one hand-kept Markdown file
of twelve hundred lines. Two things went wrong with that, repeatedly. A fact written in two
places drifted: a due date in the schedule and again in the assignment-window table, a
notebook's lecture in the notebook index and again in the generated map. And the file had two
editors, one of them working on a desktop copy that was pasted back over the repository's,
which is how half a document gets deleted. Eric, 2026-10-12: the roadmap is edited only in
the repository from now on, and its facts live in one data file.

So the sources are:

    notes/roadmap/roadmap.yaml        the facts: lectures, assignments, notebooks, tooling,
                                      and the layout, which says what goes where
    notes/roadmap/sections/*.md       the prose, one fragment per stretch between tables

and this script is the only thing that writes the Markdown. The generated file keeps its old
path because a great many files link to it.

What is generated rather than written:

- the schedule, whose "Assignment due that day" column comes from ``assignments`` grouped by
  due date, so a due date is written once;
- the assignment-window table, from the same entries, with each "set at" date read off the
  lecture that sets it;
- the lecture and notebook outcome tables and the tooling table, from the data file;
- the notebook index, from each notebook's own ``torchlingo`` metadata plus the one-line
  purpose kept here;
- the lecture-to-notebook map, by ``notebook_meta.table()``, unchanged.

Run it three ways:

    python scripts/render_roadmap.py --write    rewrite the roadmap
    python scripts/render_roadmap.py --check    exit 1 if the committed file differs
    python scripts/render_roadmap.py            print the rendered roadmap

``--check`` is the point, as it is for ``render_report.py``: without it "generated" is an
honour system, and the first hand-correction to the Markdown is silently undone by the next
render -- or, worse, never reaches the source at all.

PyYAML is the one dependency, imported where it is used, so ``notebook_meta`` (which imports
this module) stays importable by tools that never touch the roadmap.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import sys
from pathlib import Path

ROADMAP = Path("notes/CS479_COURSE_ROADMAP.md")
SOURCE = Path("notes/roadmap")
DATA = SOURCE / "roadmap.yaml"
SECTIONS = SOURCE / "sections"

# The one line a reader of the generated file needs. It replaced a paragraph explaining which
# half of the file a desktop refresh would overwrite, which is the arrangement this retired.
NOTICE = (
    "*Generated from `notes/roadmap/` by `scripts/render_roadmap.py`: "
    "edit there, never here.*"
)

# Spelled out rather than taken from strftime, whose day and month names follow the locale.
# A roadmap rendered on a machine set to German must not differ from one rendered in CI.
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

# Two assignments due the same day share one schedule cell, joined the way the hand-kept
# schedule joined them.
SAME_DAY = " · "

# What the notebook index's CI column says about a notebook's `requires`, most constraining
# first. A notebook declaring nothing is executed in CI; one declaring anything is skipped,
# and the column says why.
CI_ORDER = ("colab", "hf-token", "download", "blanks", "pip")


class RoadmapError(ValueError):
    """The roadmap's sources are inconsistent, so rendering them would mislead."""


_PARSED: dict[tuple[str, str], dict] = {}


def load(path: Path | None = None) -> dict:
    """Read the roadmap's data file.

    Cached on the file's contents, not its name: ``notebook_meta`` asks for the schedule once
    per notebook, and PyYAML parses a file this size slowly enough to notice over a test run.
    Keying on the bytes means an edited file is always re-read.

    Args:
        path (Path | None): The data file. Defaults to ``DATA``, read at call time so a test
            can point it at a scratch copy.

    Returns:
        dict: The parsed document, a fresh copy the caller may modify.

    Raises:
        RoadmapError: If PyYAML is missing, or the file is absent or not a YAML mapping. Each
            message names the file, because the reader has to go and fix it.
    """
    path = DATA if path is None else Path(path)
    try:
        import yaml
    except ImportError as error:
        raise RoadmapError(
            f"reading {path} needs PyYAML, which is not installed: pip install pyyaml"
        ) from error
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RoadmapError(f"{path}: cannot be read ({error})") from error
    key = (str(path), hashlib.sha256(raw).hexdigest())
    if key not in _PARSED:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        try:
            # yaml.load with an explicit safe loader: CSafeLoader when PyYAML was built
            # against libyaml, several times faster on a file this size.
            doc = yaml.load(raw, Loader=loader)
        except yaml.YAMLError as error:
            raise RoadmapError(f"{path}: not valid YAML ({error})") from error
        if not isinstance(doc, dict):
            raise RoadmapError(f"{path}: expected a mapping at the top level")
        _PARSED[key] = doc
    return copy.deepcopy(_PARSED[key])


def day(value: datetime.date) -> str:
    """Render a date the way the schedule always has: ``Wed Oct 14``.

    The weekday is derived, never stored. A hand-typed weekday is a second copy of the date,
    and the two disagree the first time a lecture moves.

    Args:
        value (datetime.date): The date, as YAML parses ``2026-10-14``.

    Returns:
        str: Weekday, month and day, with no year and no leading zero.

    Raises:
        RoadmapError: If the value is not a date -- typically a quoted one, which YAML reads
            as a string.
    """
    if not isinstance(value, datetime.date):
        raise RoadmapError(
            f"{DATA}: {value!r} is not a date; write it unquoted, as 2026-10-14"
        )
    return f"{WEEKDAYS[value.weekday()]} {MONTHS[value.month - 1]} {value.day}"


def row(cells: list[str]) -> str:
    """Render one Markdown table row, with an empty cell written ``| |``.

    Args:
        cells (list[str]): The cell texts.

    Returns:
        str: The row.
    """
    return "|" + "|".join(f" {c} " if c else " " for c in cells) + "|"


def table(header: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with the house's plain ``|---|`` separator.

    Args:
        header (list[str]): Column titles.
        rows (list[list[str]]): Cell texts, one list per row.

    Returns:
        str: The table, without a trailing newline.
    """
    lines = [row(header), "|" + "---|" * len(header)]
    lines += [row(cells) for cells in rows]
    return "\n".join(lines)


def _lecture_label(entry: dict) -> str:
    """The schedule's ``#`` cell: ``8c``, ``19, 20``, or an em dash for a non-lecture row."""
    return ", ".join(str(i) for i in entry.get("ids") or []) or "—"


def _due_when(assignment: dict) -> str:
    """An assignment's due day, with its time if ``due_time`` gives one: ``Fri Oct 30, 11:59 pm``."""
    when = day(assignment["due"])
    return f"{when}, {assignment['due_time']}" if assignment.get("due_time") else when


def _due_by_date(doc: dict) -> dict[datetime.date, list[str]]:
    """Group the assignments' schedule labels by the schedule row each is shown on.

    An assignment due on a lecture day is shown on that row. One due between lecture days
    (Eric, Oct 22: A12 is due on a Friday) is shown on the last row before its due date, with
    the date stated, ``(due Fri Oct 30, 11:59 pm)``, so the row does not claim it is due then;
    one on its row with a ``due_time`` states the time, the schedule's default being 10:00.
    A ``dropped`` assignment is not due at all and is left out; the window table says why.

    Raises:
        RoadmapError: If an assignment is due before the first schedule row or after the
            last. There is no row to show it on, and rendering anyway would drop it from the
            schedule without a word, which is the failure the generated column exists to
            prevent. Also if an assignment has neither ``due`` nor ``dropped``.
    """
    dates = [entry["date"] for entry in doc["lectures"]]
    repeated = sorted({d for d in dates if dates.count(d) > 1})
    if repeated:
        raise RoadmapError(
            f"{DATA}: more than one schedule row on {', '.join(map(str, repeated))}, so "
            "an assignment due that day would be listed twice"
        )
    grouped: dict[datetime.date, list[str]] = {}
    stray = []
    for assignment in doc.get("assignments", []):
        if assignment.get("dropped"):
            continue
        due = assignment.get("due")
        name = assignment.get("id") or assignment.get("label")
        if not isinstance(due, datetime.date):
            raise RoadmapError(f"{DATA}: {name} has no due date and is not marked dropped")
        before = [d for d in dates if d <= due]
        if not before or due > dates[-1]:
            stray.append(f"  {name}: {due}")
            continue
        label = assignment["label"]
        if before[-1] != due:
            label = f"{label} (due {_due_when(assignment)})"
        elif assignment.get("due_time"):
            label = f"{label} (due {assignment['due_time']})"
        grouped.setdefault(before[-1], []).append(label)
    if stray:
        raise RoadmapError(
            f"{DATA}: these assignments are due outside the schedule's dates, so the "
            "schedule would not show them:\n" + "\n".join(stray)
        )
    return grouped


def schedule_table(doc: dict) -> str:
    """The "Semester at a glance" table, with its due column built from ``assignments``.

    Args:
        doc (dict): The parsed data file.

    Returns:
        str: The Markdown table.
    """
    due = _due_by_date(doc)
    rows = []
    for entry in doc["lectures"]:
        labels = due.get(entry["date"], [])
        rows.append(
            [
                _lecture_label(entry),
                day(entry["date"]),
                entry["title"],
                SAME_DAY.join(labels),
                entry.get("deck", ""),
            ]
        )
    return table(["#", "Date", "Lecture", "Assignment due that day", "Deck"], rows)


def windows_table(doc: dict) -> str:
    """The assignment-window table: what sets each assignment, when it is due, and why.

    The "set at" date is the setting lecture's own date and the "due" date the assignment's,
    so neither is restated here. ``emphasize_due`` bolds a due date that moved, as the
    hand-kept table did. A ``dropped`` assignment keeps its row, its due cell saying
    "dropped" and its window cell the reason, so the table still accounts for it.

    Args:
        doc (dict): The parsed data file.

    Returns:
        str: The Markdown table.

    Raises:
        RoadmapError: If ``set_at`` names no lecture in the schedule.
    """
    dates = {}
    for entry in doc["lectures"]:
        for lecture in entry.get("ids") or []:
            dates[str(lecture)] = entry["date"]
    rows = []
    for assignment in doc.get("assignments", []):
        if "set_at" not in assignment:
            continue
        lecture = str(assignment["set_at"])
        if lecture not in dates:
            raise RoadmapError(
                f"{DATA}: {assignment.get('id')} is set at lecture {lecture!r}, which is "
                "not in the schedule"
            )
        if assignment.get("dropped"):
            due, window = "dropped", assignment["dropped"]
        else:
            due, window = _due_when(assignment), assignment.get("window", "")
            if assignment.get("emphasize_due"):
                due = f"**{due}**"
        rows.append(
            [
                assignment["id"],
                f"{lecture}, {day(dates[lecture])}",
                due,
                window,
            ]
        )
    return table(["assignment", "set at", "due", "window"], rows)


def lecture_outcomes_table(doc: dict) -> str:
    """The "Lecture outcomes" table.

    Its rows are not the schedule's: Lectures 19 and 20 share a session but have an outcome
    each, and 21 to 23 share one outcome across two sessions. So it is its own list rather
    than a field on each schedule row.
    """
    rows = [
        [str(o["lecture"]), o["topic"], o["can"], o["shown_by"]]
        for o in doc.get("lecture_outcomes", [])
    ]
    return table(["#", "Lecture", "A student who did the work can", "Shown by"], rows)


def notebook_outcomes_table(doc: dict) -> str:
    """The "Notebook outcomes, and who does the work" table, in the data file's order."""
    rows = []
    for stem, entry in doc.get("notebooks", {}).items():
        if "lesson" not in entry:
            continue
        name = f"`{stem}`"
        if entry.get("aside"):
            name += f" {entry['aside']}"
        rows.append(
            [
                name,
                entry["lesson"],
                entry["student_must"],
                entry["notebook_does"],
                entry["rating"],
            ]
        )
    return table(
        ["notebook", "the lesson", "the student must", "the notebook does", "rating"],
        rows,
    )


def ci_cell(requires: list[str]) -> str:
    """Say whether CI executes a notebook, and if not, which declared need stops it."""
    if not requires:
        return "yes"
    ranked = sorted(requires, key=lambda c: CI_ORDER.index(c) if c in CI_ORDER else 99)
    return "no: " + ", ".join(ranked)


def notebook_index_table(doc: dict) -> str:
    """The notebook index, from each notebook's own metadata plus its purpose line.

    Every column but the purpose is read from the notebook, so it cannot go stale the way the
    hand-kept index did. Rows are in the order a student meets them: by the first lecture the
    notebook serves, course notebooks before tutorials, then by name.

    Args:
        doc (dict): The parsed data file.

    Returns:
        str: The Markdown table.

    Raises:
        RoadmapError: If a notebook has no purpose here, or a purpose names a notebook that
            does not exist. Either way one of the two has been forgotten.
    """
    import notebook_meta as nm

    purposes = {stem: e.get("purpose") for stem, e in doc.get("notebooks", {}).items()}
    titles, aliases = nm.lectures()
    order = {lecture: i for i, lecture in enumerate(titles)}

    found = nm.notebooks()
    stems = {path.stem for path in found}
    missing = sorted(stems - {s for s, p in purposes.items() if p})
    orphaned = sorted(set(purposes) - stems)
    if missing or orphaned:
        lines = [f"  {s}: no purpose in the notebooks section" for s in missing]
        lines += [f"  {s}: has an entry but no notebook" for s in orphaned]
        raise RoadmapError(
            f"{DATA}: the notebook index and the notebooks disagree:\n"
            + "\n".join(lines)
        )

    def first_lecture(meta: dict) -> int:
        served = [nm.lecture_id(n) for n in meta.get("serves_lectures", [])]
        return min(
            (order.get(aliases.get(s, s), len(order)) for s in served), default=0
        )

    entries = []
    for path in found:
        meta = nm.read_meta(path)
        family = 0 if meta.get("family") == "course" else 1
        entries.append(((first_lecture(meta), family, path.stem), path.stem, meta))

    rows = []
    for _, stem, meta in sorted(entries):
        rows.append(
            [
                f"`{stem}`",
                ", ".join(nm.lecture_id(n) for n in meta.get("serves_lectures", [])),
                meta.get("role", ""),
                ", ".join(meta.get("leads_to", [])),
                purposes[stem],
                ci_cell(meta.get("requires", [])),
            ]
        )
    return table(["notebook", "lecture", "role", "starts", "purpose", "CI"], rows)


def tooling_table(doc: dict) -> str:
    """The "Tooling the semester depends on" table."""
    rows = [
        [t["tool"], t["first_needed"], t["used_for"]] for t in doc.get("tooling", [])
    ]
    return table(["Tool", "First needed", "Used for"], rows)


def notebook_map(doc: dict) -> str:
    """The lecture-to-notebook map, between the markers it has always had.

    Generated by ``notebook_meta.table()``, which owns it; the markers stay so a reader of the
    Markdown can still see where the build artifact starts and ends.
    """
    import notebook_meta as nm

    return f"{nm.BEGIN}\n\n{nm.table()}\n\n{nm.END}"


TABLES = {
    "schedule": schedule_table,
    "assignment-windows": windows_table,
    "lecture-outcomes": lecture_outcomes_table,
    "notebook-outcomes": notebook_outcomes_table,
    "notebook-index": notebook_index_table,
    "tooling": tooling_table,
    "notebook-map": notebook_map,
}


def fragment(name: str) -> str:
    """Return one prose fragment, without the newline its file ends with.

    The document is its pieces joined by one blank line, so a fragment must not begin or end
    with a blank line of its own: it would render as two, and the next fragment's author
    would never see why.

    Raises:
        RoadmapError: If the fragment is missing or carries blank lines at either end.
    """
    path = SECTIONS / f"{name}.md"
    if not path.exists():
        raise RoadmapError(f"{DATA}: the layout names {path}, which does not exist")
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n") or text.endswith("\n\n") or text.startswith("\n"):
        raise RoadmapError(
            f"{path}: a fragment ends with exactly one newline and starts with text"
        )
    return text[:-1]


def render(doc: dict | None = None) -> str:
    """Assemble the roadmap from its layout.

    The layout is a list. Each entry is ``rule`` (a ``---`` line), ``notice`` (the generated
    file's header line), ``{prose: NAME}`` (``sections/NAME.md``) or ``{table: NAME}`` (one of
    ``TABLES``). Pieces are joined by a blank line.

    The layout must use every table exactly once and every fragment exactly once. A table left
    out would leave its data unrendered with nothing to say so; a fragment left out is prose
    that silently stopped being published.

    Args:
        doc (dict | None): The parsed data file. Read from ``DATA`` when omitted.

    Returns:
        str: The complete Markdown document.

    Raises:
        RoadmapError: On any inconsistency in the sources, with a message naming the file.
    """
    doc = load() if doc is None else doc
    pieces: list[str] = []
    tables: list[str] = []
    prose: list[str] = []
    for entry in doc.get("layout", []):
        if entry == "rule":
            pieces.append("---")
        elif entry == "notice":
            pieces.append(NOTICE)
        elif isinstance(entry, dict) and set(entry) == {"prose"}:
            prose.append(entry["prose"])
            pieces.append(fragment(entry["prose"]))
        elif isinstance(entry, dict) and set(entry) == {"table"}:
            if entry["table"] not in TABLES:
                raise RoadmapError(
                    f"{DATA}: the layout names table {entry['table']!r}; known tables are "
                    f"{sorted(TABLES)}"
                )
            tables.append(entry["table"])
            pieces.append(TABLES[entry["table"]](doc))
        else:
            raise RoadmapError(
                f"{DATA}: layout entry {entry!r} is not rule, notice, "
                "{prose: NAME} or {table: NAME}"
            )

    problems = []
    for name in sorted(TABLES):
        if tables.count(name) != 1:
            problems.append(f"  table {name!r} appears {tables.count(name)} times")
    on_disk = sorted(p.stem for p in SECTIONS.glob("*.md"))
    for name in sorted(set(on_disk) | set(prose)):
        if prose.count(name) != 1:
            problems.append(f"  fragment {name!r} appears {prose.count(name)} times")
    if problems:
        raise RoadmapError(
            f"{DATA}: the layout must use every table and every fragment in {SECTIONS} "
            "exactly once:\n" + "\n".join(problems)
        )
    return "\n\n".join(pieces) + "\n"


def current() -> str | None:
    """Return the committed roadmap's text, or None when it does not exist yet."""
    return ROADMAP.read_text(encoding="utf-8") if ROADMAP.exists() else None


def write(text: str | None = None) -> bool:
    """Write the roadmap. The only function anywhere that does.

    ``notebook_meta.py --write`` calls this too, so the two commands cannot produce different
    files.

    Args:
        text (str | None): The rendered document; rendered here when omitted.

    Returns:
        bool: True when the file changed.
    """
    text = render() if text is None else text
    if current() == text:
        return False
    ROADMAP.write_text(text, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    """Render the roadmap, write it, or check the committed copy against its sources.

    Args:
        argv (list | None): Arguments to parse; the real command line when omitted.

    Returns:
        int: 0 when current or written, 1 when stale or the sources are inconsistent.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite the roadmap")
    mode.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the committed roadmap differs from its sources",
    )
    args = parser.parse_args(argv)

    try:
        expected = render()
    except RoadmapError as error:
        print(error, file=sys.stderr)
        return 1

    if args.check:
        if current() != expected:
            print(
                f"{ROADMAP} is out of date with {SOURCE}/, or was edited by hand.\n"
                f"Edit {SOURCE}/ instead, then run: python scripts/render_roadmap.py --write",
                file=sys.stderr,
            )
            return 1
        print(f"  current    {ROADMAP}")
        return 0
    if args.write:
        print(f"  {'written  ' if write(expected) else 'unchanged'}  {ROADMAP}")
        return 0
    sys.stdout.write(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
