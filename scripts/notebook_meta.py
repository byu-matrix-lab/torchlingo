"""One metadata namespace for both notebook families, and the map generated from it.

Two families of notebook exist and their numbering deliberately does not line up:
``docs/docs/tutorials/`` is numbered by library topic, ``docs/docs/course/`` by lecture. So
which artifact serves which lecture is derivable from nothing, and it used to be recorded in
a hand-maintained table in the roadmap.

This replaces the table with data the notebooks carry themselves:

    "metadata": { "torchlingo": {
        "family": "tutorial" | "course",
        "serves_lectures": [8],
        "role": "activity" | "reading" | "homework" | "reference",
        "needs": ["data/example.tsv"],
        "note": "optional prose, for a pairing that is not self-explanatory"
    }}

``nbformat`` permits arbitrary keys under ``metadata``, and neither Jupyter nor Colab minds
them.

``role`` also carries how strong the pairing is, which the hand-written table said with
bold and italics: ``activity`` is used in the session itself, ``reading`` is assigned
alongside it, and ``reference`` is offered rather than urged -- a weak pairing, or one that
would have fitted a lecture that has already run. ``needs`` is repo-relative paths only,
because ``execute_notebooks.py`` resolves them against the filesystem.

Why metadata rather than renaming the files. A notebook can serve more than one lecture —
tutorial 4 serves Lecture 19 and is also the natural companion to Lecture 8's attention
material — and one lecture number in a filename cannot say that. Renaming is also actively
dangerous mid-semester: decks cite notebooks by filename and Colab badges embed the path, so
a rename breaks a link a student is already clicking. That work is deliberately deferred.

The roadmap's lecture map is then generated from it, between markers, the same arrangement
``scripts/render_report.py`` uses for the reports: the file holds the artifact, the script is
the only thing that writes it, and ``--check`` in CI is what keeps "generated" from being an
honour system.

Usage:

    python scripts/notebook_meta.py --check     validate, and verify the roadmap is current
    python scripts/notebook_meta.py --write     rewrite the roadmap's map block
    python scripts/notebook_meta.py --table     print the map without touching anything
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

TUTORIALS = Path("docs/docs/tutorials")
COURSE = Path("docs/docs/course")
ROADMAP = Path("notes/CS479_COURSE_ROADMAP.md")

BEGIN = "<!-- BEGIN generated:notebook-map -->"
END = "<!-- END generated:notebook-map -->"

# Superscripts rather than Markdown footnote syntax. `[^1]` needs the footnotes extension,
# which docs/mkdocs.yml does not enable, and the roadmap is also read as plain text in a
# terminal -- where a footnote reference that never resolves is just noise.
MARKERS = "¹²³⁴⁵⁶⁷⁸⁹"

FAMILIES = {"tutorial", "course"}
ROLES = {"activity", "reading", "homework", "reference"}

# Rows whose lecture column is a number, from the "Semester at a glance" table. Rows
# numbered "—" are not lectures: no class, a project checkpoint, the final.
SCHEDULE_ROW = re.compile(r"^\| (\d+(?:, ?\d+)*) \| [^|]+ \| ([^|]+) \|")
SCHEDULE_HEADING = "## Semester at a glance"


def lectures() -> tuple[dict[int, str], dict[int, int]]:
    """Read the lecture numbers and titles out of the roadmap's own schedule table.

    Parsed rather than restated. The task that asked for this named the risk precisely:
    two tables in one document that must agree, with nothing checking, is the pattern that
    has already cost this repository three separate defects. A short hand-written copy of
    the titles here would have been a fourth. So the schedule is the source and the map
    below is derived from it, which makes disagreement impossible rather than detectable.

    Returns:
        tuple: Titles by lecture number, and an alias map folding a shared row's later
        numbers onto its first -- Lectures 22 and 23 are one session in one row.

    Raises:
        ValueError: If the schedule cannot be found, rather than generating an empty map.
    """
    titles: dict[int, str] = {}
    aliases: dict[int, int] = {}
    # Only within the schedule's own section. Without this the generated map's rows match
    # the same shape and overwrite every title with a notebook filename -- which is exactly
    # what happened the first time this ran, and is the kind of quiet wrongness that would
    # have shipped had the output not been read.
    inside = False
    for line in ROADMAP.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == SCHEDULE_HEADING
        if not inside:
            continue
        match = SCHEDULE_ROW.match(line)
        if not match:
            continue
        numbers = [int(n) for n in match.group(1).replace(" ", "").split(",")]
        title = match.group(2).strip()
        titles[numbers[0]] = title
        for extra in numbers[1:]:
            aliases[extra] = numbers[0]
    if not titles:
        raise ValueError(f"{ROADMAP} has no recognizable schedule table")
    return titles, aliases


def notebooks() -> list[Path]:
    """Return every notebook in both families, tutorials first.

    Returns:
        list: Paths, sorted within each family.
    """
    return sorted(TUTORIALS.glob("*.ipynb")) + sorted(COURSE.glob("*.ipynb"))


def read_meta(path: Path) -> dict:
    """Return a notebook's ``torchlingo`` metadata block.

    Args:
        path (Path): The notebook.

    Returns:
        dict: The block, or ``{}`` when absent.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    return nb.get("metadata", {}).get("torchlingo", {})


def problems(path: Path, meta: dict) -> list[str]:
    """Return every complaint about one notebook's metadata.

    Returns all of them rather than the first, so a run says everything that is wrong
    instead of one thing per invocation.

    Args:
        path (Path): The notebook, for the messages.
        meta (dict): Its ``torchlingo`` block.

    Returns:
        list: Human-readable complaints; empty when the block is sound.
    """
    found = []
    if not meta:
        return [f"{path}: no torchlingo metadata block"]

    family = meta.get("family")
    if family not in FAMILIES:
        found.append(f"{path}: family {family!r} is not one of {sorted(FAMILIES)}")
    elif family == "course" and path.parent != COURSE:
        found.append(f"{path}: family 'course' but it lives in {path.parent}")
    elif family == "tutorial" and path.parent != TUTORIALS:
        found.append(f"{path}: family 'tutorial' but it lives in {path.parent}")

    role = meta.get("role")
    if role not in ROLES:
        found.append(f"{path}: role {role!r} is not one of {sorted(ROLES)}")

    titles, aliases = lectures()
    declared = meta.get("serves_lectures")
    if not isinstance(declared, list):
        found.append(
            f"{path}: serves_lectures must be a list, got {type(declared).__name__}"
        )
    else:
        for n in declared:
            if n not in titles and n not in aliases:
                found.append(
                    f"{path}: serves_lectures has {n!r}, which is not in "
                    f"{ROADMAP}'s schedule"
                )
        # A course notebook is named for its lecture, so the two must agree. This is the
        # check that would have caught a lecture-06 notebook claiming to serve Lecture 5.
        if family == "course":
            stem = path.stem
            if stem.startswith("lecture-"):
                numbered = int(stem.split("-")[1])
                if numbered not in declared:
                    found.append(
                        f"{path}: filename says lecture {numbered} but serves_lectures "
                        f"is {declared}"
                    )

    needs = meta.get("needs")
    if not isinstance(needs, list) or any(not isinstance(x, str) for x in needs):
        found.append(f"{path}: needs must be a list of strings, got {needs!r}")

    if "note" in meta and not isinstance(meta["note"], str):
        found.append(f"{path}: note must be a string, got {meta['note']!r}")

    # Reject unknown keys. A typo in an optional field is otherwise invisible: the block
    # validates, the generated table silently omits what the typo was meant to say, and
    # nothing anywhere reports a problem.
    unknown = set(meta) - {"family", "serves_lectures", "role", "needs", "note"}
    if unknown:
        found.append(f"{path}: unknown key(s) {sorted(unknown)}")
    return found


def table() -> str:
    """Render the lecture-to-notebook map as Markdown.

    Rows come from the roadmap's schedule and cells from the notebooks, so the map cannot
    disagree with either. Lectures that no notebook serves are listed with an em dash rather
    than omitted: "nothing yet" is information, and an absent row reads as an oversight.

    Returns:
        str: The Markdown table.
    """
    titles, aliases = lectures()
    serves: dict[int, dict[str, list[str]]] = {
        n: {"course": [], "tutorial": []} for n in titles
    }
    notes: list[str] = []
    for path in notebooks():
        meta = read_meta(path)
        marker = ""
        if meta.get("note"):
            marker = MARKERS[len(notes)]
            # Wrapped, because this document is read in a terminal as often as rendered,
            # and the surrounding prose is hand-wrapped to the same width.
            notes.append(
                textwrap.fill(
                    f"- {marker} **`{path.stem}`** — {meta['note']}",
                    width=90,
                    subsequent_indent="  ",
                )
            )
        for n in meta.get("serves_lectures", []):
            # A lecture that shares a row is folded onto that row's number, so declaring
            # 23 lands in the "22, 23" row instead of vanishing.
            row = aliases.get(n, n)
            if row in serves:
                entry = f"`{path.stem}` ({meta['role']}) {marker}".rstrip()
                if entry not in serves[row][meta["family"]]:
                    serves[row][meta["family"]].append(entry)

    rows = [
        "| # | Lecture | Course notebook | TorchLingo tutorial |",
        "|---|---|---|---|",
    ]
    shared = {v: k for k, v in aliases.items()}
    for n, title in titles.items():
        label = f"{n}, {shared[n]}" if n in shared else str(n)
        c = ", ".join(serves[n]["course"]) or "—"
        t = ", ".join(serves[n]["tutorial"]) or "—"
        rows.append(f"| {label} | {title} | {c} | {t} |")

    if notes:
        rows += ["", "Notes on the rows that are not simple:", ""] + notes
    return "\n".join(rows)


def roadmap_with_table(current: str) -> str:
    """Return the roadmap text with the generated block replaced.

    Args:
        current (str): The roadmap's present contents.

    Returns:
        str: The same document with a freshly generated map between the markers.

    Raises:
        ValueError: If the markers are absent or out of order, rather than appending a
            second table somewhere harmless-looking.
    """
    start, end = current.find(BEGIN), current.find(END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{ROADMAP} does not contain the map markers in order")
    head = current[: start + len(BEGIN)]
    return f"{head}\n\n{table()}\n\n{current[end:]}"


def main() -> int:
    """Validate the metadata, or print the generated table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="validate and exit 1 on problems"
    )
    parser.add_argument(
        "--write", action="store_true", help="rewrite the roadmap's map"
    )
    parser.add_argument("--table", action="store_true", help="print the lecture map")
    args = parser.parse_args()

    found = notebooks()
    if not found:
        print("no notebooks found; run from the repository root", file=sys.stderr)
        return 2

    if args.table:
        print(table())
        return 0

    complaints = []
    for path in found:
        complaints.extend(problems(path, read_meta(path)))
    if complaints:
        print(f"{len(complaints)} problem(s):", file=sys.stderr)
        for c in complaints:
            print(f"  {c}", file=sys.stderr)
        return 1
    print(f"  {len(found)} notebooks, metadata sound")

    # Only reached when the metadata is sound, deliberately: regenerating the roadmap from
    # a block that failed validation would write a table with holes in it.
    current = ROADMAP.read_text(encoding="utf-8")
    expected = roadmap_with_table(current)
    if args.write:
        if current == expected:
            print(f"  unchanged  {ROADMAP}")
        else:
            ROADMAP.write_text(expected, encoding="utf-8")
            print(f"  written    {ROADMAP}")
    elif args.check and current != expected:
        print(
            f"\n{ROADMAP}'s notebook map is out of date with the notebooks.\n"
            "Run: python scripts/notebook_meta.py --write",
            file=sys.stderr,
        )
        return 1
    elif args.check:
        print(f"  current    {ROADMAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
