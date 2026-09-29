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
        "leads_to": ["A8"],
        "note": "optional prose, for a pairing that is not self-explanatory"
    }}

``leads_to`` means **read this before attempting that assignment**, not only "this advances a
step of it". Eric settled the reading on 2026-09-27, and it is the wider of the two: a
notebook that explains the machinery an assignment uses qualifies, even when it completes none
of the assignment's steps, provided it carries a small deliverable of its own.

The narrow reading was tempting because it keeps the field crisp, but it would have made
``leads_to`` mean "does part of the work" -- and the notebook that sent the question up was
tutorial 4, which explains what attention learns without advancing any step of A8. Under the
wider reading it qualifies, which matches how a student actually uses it.

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

# Plain "[N]" rather than Markdown footnote syntax. `[^1]` needs the footnotes extension,
# which docs/mkdocs.yml does not enable, and the roadmap is also read as plain text in a
# terminal -- where a footnote reference that never resolves is just noise. Superscripts
# were used first, from a fixed nine-character string, and raised IndexError the day the
# tenth notebook gained a note.
def footnote_marker(n: int) -> str:
    """Return the marker for the n-th note, 1-based: 1 -> "[1]", 12 -> "[12]"."""
    return f"[{n}]"


FAMILIES = {"tutorial", "course"}
ROLES = {"activity", "reading", "homework", "reference"}

# What a notebook needs from its ENVIRONMENT, as opposed to `needs`, which lists files in
# this repository. A closed set on purpose: `requires: ["nltk "]` with a stray space would
# otherwise read as an unknown capability that happens to gate nothing, and the notebook
# would be executed in CI when it was meant to be skipped.
#
# The four that matter here, each read off the notebooks rather than guessed:
#   pip        installs a package the CI environment does not have
#   download   fetches a model or corpus at runtime, needing network and cache
#   colab      needs the Colab runtime itself -- `files.upload()` or a Drive mount, neither
#              of which has a headless equivalent
#   hf-token   a HuggingFace token, which CI does not have and should not be given
#   blanks     a worksheet whose code cells are deliberately incomplete, so they do not
#              parse until a student fills them in. What CI lacks is the student.
CAPABILITIES = {"pip", "download", "colab", "hf-token", "blanks"}

# Which collection a notebook belongs to. This says where it lives, not how it is used --
# a distinction worth keeping separate, because they come apart: tutorial 2 lives in the
# tutorials and is used as Lecture 7's in-class activity.
FAMILY_COLLECTION = {
    "tutorial": "TorchLingo tutorial",
    "course": "CS 479 course notebook",
}

# How a notebook is used, which is where the in-class / out-of-class distinction actually
# lives. Eric, 2026-09-27: the tutorials are out-of-class and the course notebooks are
# in-class active learning. That is true of the collections as a rule and of `role` always,
# so the role is what a reader is told.
ROLE_PHRASE = {
    "activity": "the in-class activity for",
    "reading": "assigned reading for",
    "homework": "homework for",
    "reference": "offered alongside",
}

# Rows whose lecture column is a number, from the "Semester at a glance" table.
SCHEDULE_ROW = re.compile(r"^\| (\d+(?:, ?\d+)*) \| [^|]+ \| ([^|]+) \|")
SCHEDULE_HEADING = "## Semester at a glance"

# What a readable lecture column looks like: one lecture, or several sharing a session.
#
# A lecture is digits with an optional single lowercase letter -- `8`, `8a`, `8b`. The letter
# form was added on 2026-09-27, when Eric and the Cowork session split Lecture 8 into 8a
# (Wed Sep 30) and 8b (Mon Oct 5) and wrote the schedule in the form students will see.
#
# This widening is deliberate and it is the whole of Task #138. The alternative was
# renumbering 9 onward, which would have left every notebook's `serves_lectures` valid while
# silently pointing at a different lecture -- the one failure mode nothing here can detect.
# A suffix cannot collide with an existing number, so it costs one decision instead of ten.
#
# The consequence is that a lecture identifier is a STRING, not an int. `8a` has no integer
# form, so there is no version of this that keeps numeric keys.
LECTURE_CELL = re.compile(r"^\d+[a-z]?(?:, ?\d+[a-z]?)*$")

# Cells that are legitimately not lectures, and are skipped rather than reported. An em dash
# marks a row that is not a lecture at all -- no class, a project review, the final exam --
# and `#` is the table's own header. Every *other* unreadable cell is an error, because the
# whole point of parsing this table is that a lecture cannot quietly go missing from the map.
SKIPPABLE_CELLS = {"—", "–", "-", "#"}

# Assignments as the schedule writes them, in the "Assignment due that day" column:
# `**A4** initial cleaning steps`, sometimes two in one cell separated by a middot.
#
# A letter suffix is accepted for the same reason lecture identifiers take one: Eric,
# 2026-09-27, wants one assignment per lecture even when it is a stepping stone to an
# existing one, so splitting Lecture 8 into 8a and 8b implies an A8a and an A8b.
#
# Widened BEFORE the schedule gained those rows, which is the difference from the morning.
# `**A8a**` did not match this pattern, so it would have been skipped in silence and every
# notebook declaring `leads_to: ["A8a"]` rejected as naming an assignment that does not
# exist -- while the schedule plainly showed it. The lecture version of this bug was found
# after the fact; this one was found by asking the question first.
ASSIGNMENT = re.compile(r"\*\*(A\d+[a-z]?)\*\*")


def lecture_id(value: int | str) -> str:
    """Normalize anything that names a lecture to the identifier the schedule uses.

    Three spellings have to land on one key. A notebook written before Lecture 8 was split
    holds the integer ``9``; the schedule writes ``9``; and a course filename pads it to
    ``09``. Suffixed lectures add ``8a``, which is why the key is a string at all.

    Leading zeros are stripped rather than preserved, because ``lecture-09-...`` and schedule
    row ``9`` are the same lecture and a mismatch there would be a confusing way to learn it.

    Args:
        value (int | str): A lecture number, identifier, or filename fragment.

    Returns:
        str: The canonical identifier -- ``"9"``, ``"8a"``.
    """
    text = str(value).strip().lower()
    digits = text.rstrip("abcdefghijklmnopqrstuvwxyz")
    suffix = text[len(digits) :]
    return f"{digits.lstrip('0') or '0'}{suffix}"


def lectures() -> tuple[dict[str, str], dict[str, str]]:
    """Read the lecture numbers and titles out of the roadmap's own schedule table.

    Parsed rather than restated. The task that asked for this named the risk precisely:
    two tables in one document that must agree, with nothing checking, is the pattern that
    has already cost this repository three separate defects. A short hand-written copy of
    the titles here would have been a fourth. So the schedule is the source and the map
    below is derived from it, which makes disagreement impossible rather than detectable.

    Returns:
        tuple: Titles by lecture identifier, in schedule order, and an alias map folding a
        shared row's later identifiers onto its first -- Lectures 22 and 23 are one session
        in one row.

        Identifiers are **strings**, because a split lecture is `8a` and `8b` and neither has
        an integer form. Schedule order is the dictionary's insertion order, which is why
        nothing here sorts: `8b` after `8a` after `8` is only obvious to a reader, and any
        numeric sort would have to be taught the same thing.

    Raises:
        ValueError: If the schedule cannot be found, or if a row in it has a lecture column
            this cannot read. Both are loud on purpose -- see below.
    """
    titles: dict[str, str] = {}
    aliases: dict[str, str] = {}
    unreadable: list[str] = []
    # Only within the schedule's own section. Without this the generated map's rows match
    # the same shape and overwrite every title with a notebook filename -- which is exactly
    # what happened the first time this ran, and is the kind of quiet wrongness that would
    # have shipped had the output not been read.
    inside = False
    for line in ROADMAP.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == SCHEDULE_HEADING
        if not inside or not line.startswith("|"):
            continue

        columns = line.split("|")
        if len(columns) < 5:
            continue
        cell = columns[1].strip()

        # A row whose lecture column this cannot read is an error, not a row to skip.
        #
        # Skipping was the original behaviour and it was silently wrong in the worst way: the
        # lecture simply was not in the generated map, nothing reported anything, and the map
        # still looked complete. `8a`, `8-9` and `8 (part 1)` all fell through. Splitting a
        # lecture is exactly when someone reaches for one of those forms, so the failure was
        # waiting for the moment it would do most damage.
        #
        # Widening the accepted forms is a separate decision -- which scheme the course uses
        # is the instructors' call, not this script's -- so the message names what is
        # accepted today rather than guessing at what was meant.
        if cell in SKIPPABLE_CELLS or set(cell) <= set("-: "):
            continue
        if not LECTURE_CELL.match(cell):
            unreadable.append(
                f"  {line.strip()[:100]}\n    lecture column reads {cell!r}"
            )
            continue

        ids = cell.replace(" ", "").split(",")
        titles[ids[0]] = columns[3].strip()
        for extra in ids[1:]:
            aliases[extra] = ids[0]

    if unreadable:
        raise ValueError(
            f"{ROADMAP}: {len(unreadable)} schedule row(s) have a lecture column that "
            "cannot be read, so they would be missing from the generated map:\n"
            + "\n".join(unreadable)
            + "\n\n  Accepted: a number (`8`), several sharing one session (`22, 23`), "
            "or `—` for a row that is not a lecture.\n"
            "  To use another scheme, widen LECTURE_CELL in scripts/notebook_meta.py "
            "deliberately -- and note that serves_lectures holds these same numbers, so "
            "renumbering means re-checking every notebook (Task #138)."
        )
    if not titles:
        raise ValueError(f"{ROADMAP} has no recognizable schedule table")
    return titles, aliases


def assignments() -> dict[str, str | None]:
    """Read the assignment identifiers out of the schedule, with the lecture each is due on.

    Parsed for the same reason the lectures are: the schedule already names every assignment,
    and a second list here would be a second thing to keep in agreement.

    Assignments matter to notebooks because of how the course actually uses them. Eric,
    2026-09-27: an in-class notebook is often a jump start on the *next* assignment, and some
    tutorials should be pivoted to serve that purpose too. So "which assignment does this
    notebook start" is a real relation, distinct from "which lecture does it serve" -- the
    lecture is where it is used, the assignment is what it is used *for*.

    Returns:
        dict: Assignment id to the lecture number it is due on, or None when it falls on a
        row that is not a lecture, such as the last day of class.
    """
    due: dict[str, str | None] = {}
    inside = False
    for line in ROADMAP.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == SCHEDULE_HEADING
        if not inside or not line.startswith("|"):
            continue
        columns = line.split("|")
        if len(columns) < 5:
            continue
        match = SCHEDULE_ROW.match(line)
        # An identifier, not an int. `int("8a")` raises, and even for unsuffixed rows an int
        # would not compare equal to the string keys `lectures()` returns -- which is how
        # this was caught: every assignment resolved to a lecture that "did not exist".
        lecture = lecture_id(match.group(1).split(",")[0]) if match else None
        for found in ASSIGNMENT.findall(columns[4]):
            due[found] = lecture
    return due


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


def write_meta(path: Path, meta: dict) -> None:
    """Write a notebook's ``torchlingo`` block, in place, changing nothing else.

    A textual splice rather than an ``nbformat`` round trip. ``nbformat`` rewrites cell
    sources from strings into line lists and reorders keys, so a round trip to add six lines
    of metadata produces a diff of hundreds of lines across cells it did not touch -- which
    is unreviewable, and in a notebook the cells are the part that matters.

    The notebook set is open: Eric and the Cowork session add a lecture notebook whenever a
    lecture needs one. So the block has to be addable without hand-editing JSON. Colab
    exposes no editor for notebook-level metadata at all, and in Jupyter it is several clicks
    into a raw-JSON panel, which is exactly where a typo becomes a silently dropped field.

    Args:
        path (Path): The notebook to modify.
        meta (dict): The block to write. Replaces an existing one entirely.

    Raises:
        ValueError: If the notebook has no top-level ``metadata`` object to write into.
    """
    text = path.read_text(encoding="utf-8")
    body = json.dumps({"torchlingo": meta}, indent=1)
    # json.dumps(indent=1) indents the inner lines one space relative to the braces it
    # writes; the notebook's own metadata members sit at two, so one more space aligns them.
    block = "\n".join(" " + line for line in body.splitlines()[1:-1])

    existing = text.find('\n  "torchlingo": {')
    if existing != -1:
        # Scan to the matching close brace rather than pattern-matching the end, because the
        # block contains braces only at its own boundaries today and might not tomorrow.
        depth, i = 0, text.index("{", existing)
        while True:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        end = i + 1
        # Put back a comma only if one was there. The block is first in a notebook this
        # tool stamped, but an nbformat round trip sorts keys and moves it last, and an
        # unconditional comma there is a trailing comma -- invalid JSON, which is how
        # restamping tutorial 1 after a re-execution broke five tests.
        had_comma = text[end : end + 1] == ","
        if had_comma:
            end += 1
        path.write_text(
            text[: existing + 1] + block + ("," if had_comma else "") + text[end:],
            encoding="utf-8",
        )
        return

    anchor = '\n "metadata": {\n'
    # The leading newline matters: without it this also matches a cell's more deeply indented
    # "metadata": { line, and the block would land inside a cell.
    if text.count(anchor) != 1:
        raise ValueError(
            f"{path}: expected exactly one top-level metadata object, "
            f"found {text.count(anchor)}"
        )
    path.write_text(text.replace(anchor, anchor + block + ",\n"), encoding="utf-8")


# Token shapes that must never reach a committed notebook. Not a general secret scanner --
# it catches the four providers this course actually touches, which is what a course notebook
# arriving from Colab plausibly carries.
SECRET_SHAPES = re.compile(
    r"\b(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})"
)


def _cell_parses(cell: dict) -> bool:
    """Report whether one code cell is syntactically valid Python.

    IPython line magics are stripped first, per line rather than per cell. A naive
    ``compile()`` flagged two notebooks that run perfectly well, because ``!pip install``
    appears *inside* a ``try`` block in tutorial 3 and mid-cell in lecture-12 -- neither is
    Python, both are fine in Jupyter. Skipping only cells that *begin* with ``!`` or ``%``
    misses exactly those cases, which is the version that produced the false positives.

    A stripped magic leaves a blank line, so indentation-sensitive constructs still parse:
    ``try:`` followed only by a magic would otherwise become an empty block. A ``pass`` is
    substituted to keep that honest.

    Args:
        cell (dict): An nbformat code cell.

    Returns:
        bool: True when the cell parses, or is empty once magics are removed.
    """
    source = "".join(cell.get("source") or [])
    kept = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            # Preserve the indentation so a magic that is the sole body of a block still
            # leaves that block non-empty.
            kept.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            kept.append(line)
    body = "\n".join(kept)
    if not body.strip():
        return True
    try:
        compile(body, "<cell>", "exec")
    except SyntaxError:
        return False
    return True


def hygiene(path: Path) -> list[str]:
    """Check a notebook's file-level hygiene, separately from its metadata.

    These are the checks that were run **by hand** on 2026-09-27 when three course notebooks
    arrived from the Cowork session: no committed outputs, no execution counts, no tokens, a
    Colab badge on anything a student is meant to open there.

    Productized because the hand check recurs on every baton and is exactly the kind of thing
    that gets skipped on the busy pass. It also catches what actually breaks a course
    notebook: one that has been round-tripped through Drive and brought back with a 675 KB
    training log in it, or with a token pasted into a cell.

    Costs nothing in CI -- it is a JSON read, and it runs in the lint job beside the metadata
    check rather than needing the test environment.

    Args:
        path (Path): The notebook to inspect.

    Returns:
        list: Human-readable complaints; empty when the file is clean.
    """
    found = []
    text = path.read_text(encoding="utf-8")
    try:
        nb = json.loads(text)
    except json.JSONDecodeError as error:
        return [f"{path}: is not valid JSON ({error})"]

    cells = nb.get("cells", [])
    if not cells:
        found.append(f"{path}: has no cells")

    # Committed outputs are a defect in a COURSE notebook and a requirement in a TUTORIAL,
    # so this cannot be one rule.
    #
    # `docs/mkdocs.yml` configures mkdocs-jupyter with `execute: false`, so the outputs
    # committed in a tutorial ARE what the documentation site renders -- strip them and the
    # published page shows code with no results. A course notebook is opened in Colab and run
    # from the top, so its outputs are dead weight: one arrived from Fall 2025 carrying a
    # 675 KB training log.
    #
    # Nearly got this backwards. The first version of the check flagged all six tutorials.
    if path.parent == COURSE:
        outputs = sum(len(c.get("outputs") or []) for c in cells)
        if outputs:
            found.append(
                f"{path}: {outputs} committed output(s). A course notebook is run from the "
                "top in Colab, so strip them -- one arrived carrying a 675 KB training log."
            )

        counted = sum(1 for c in cells if c.get("execution_count"))
        if counted:
            found.append(
                f"{path}: {counted} cell(s) carry an execution_count, so this was committed "
                "from a run rather than cleaned"
            )

    # A notebook whose code does not parse cannot be executed by anything, and the reason
    # is never a capability you could install. Two kinds exist and they need opposite
    # treatment: a worksheet with `pattern = # fill this in` is *meant* not to parse, and a
    # truncated edit is a defect.
    #
    # Found the hard way. `lecture-10-comet-install` shipped with `else:` followed by an
    # unindented `drive` -- a bare SyntaxError in Lecture 10's own assignment notebook,
    # committed and unnoticed, which a student would have hit in the room. Nothing was
    # checking, because nothing executed course notebooks.
    #
    # So the rule runs both ways: code that does not parse must be declared, and a `blanks`
    # declaration must correspond to real blanks, or the marker rots into a licence to ship
    # broken cells.
    if path.parent == COURSE:
        declared_blanks = "blanks" in (
            (nb.get("metadata", {}).get("torchlingo", {}) or {}).get("requires", [])
            or []
        )
        broken = [
            index
            for index, cell in enumerate(cells)
            if cell.get("cell_type") == "code" and not _cell_parses(cell)
        ]
        if broken and not declared_blanks:
            found.append(
                f"{path}: code cell(s) {broken} do not parse, and the notebook does not "
                "declare requires: ['blanks']. Either it is a worksheet -- declare it -- or "
                "a cell is broken, which is how lecture-10 shipped a SyntaxError."
            )
        if declared_blanks and not broken:
            found.append(
                f"{path}: declares requires: ['blanks'] but every code cell parses, so the "
                "declaration is stale and is now only keeping the notebook out of CI"
            )

    for shape in SECRET_SHAPES.findall(text):
        # Report the prefix only. Echoing a live token into CI logs would publish it more
        # widely than the commit did.
        found.append(f"{path}: contains something shaped like a token ({shape[:6]}...)")

    # A course notebook is opened in Colab by a student, from a badge. A tutorial is read on
    # the docs site, where mkdocs-jupyter renders it, so the badge is a nicety there and a
    # requirement here.
    if path.parent == COURSE and "colab.research.google.com" not in text:
        found.append(
            f"{path}: no Colab badge, and students open course notebooks in Colab"
        )

    return found


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
            if not isinstance(n, (int, str)):
                found.append(
                    f"{path}: serves_lectures has {n!r}; entries are a number or a string "
                    "like '8a'"
                )
            elif lecture_id(n) not in titles and lecture_id(n) not in aliases:
                found.append(
                    f"{path}: serves_lectures has {n!r}, which is not in "
                    f"{ROADMAP}'s schedule"
                )
        # A course notebook is named for its lecture, so the two must agree. This is the
        # check that would have caught a lecture-06 notebook claiming to serve Lecture 5.
        if family == "course":
            stem = path.stem
            if stem.startswith("lecture-"):
                # `lecture-08a-...` and `lecture-8a-...` both name lecture 8a, and
                # `lecture-06-...` names lecture 6. Comparing identifiers rather than
                # integers is what lets a suffixed lecture have a file at all.
                named = lecture_id(stem.split("-")[1])
                if named not in {
                    lecture_id(n) for n in declared if isinstance(n, (int, str))
                }:
                    found.append(
                        f"{path}: filename says lecture {named} but serves_lectures "
                        f"is {declared}"
                    )

    needs = meta.get("needs")
    if not isinstance(needs, list) or any(not isinstance(x, str) for x in needs):
        found.append(f"{path}: needs must be a list of strings, got {needs!r}")

    # `requires` is capabilities, where `needs` is repo-relative paths. The distinction is
    # what blocked the second half of #154: a notebook can be missing nothing from the
    # repository and still be unrunnable in CI because it wants a pip package, a model
    # download or a HuggingFace token. Those are not paths, so no amount of `needs` says it.
    requires = meta.get("requires", [])
    if not isinstance(requires, list) or any(not isinstance(x, str) for x in requires):
        found.append(f"{path}: requires must be a list of strings, got {requires!r}")
    else:
        unknown_caps = sorted(set(requires) - CAPABILITIES)
        if unknown_caps:
            found.append(
                f"{path}: requires has unknown capability {unknown_caps}; known ones are "
                f"{sorted(CAPABILITIES)}. Add a new one to CAPABILITIES deliberately -- a "
                "typo here would silently make a notebook look runnable."
            )

    if "note" in meta and not isinstance(meta["note"], str):
        found.append(f"{path}: note must be a string, got {meta['note']!r}")

    leads_to = meta.get("leads_to", [])
    if not isinstance(leads_to, list) or any(not isinstance(x, str) for x in leads_to):
        found.append(f"{path}: leads_to must be a list of strings, got {leads_to!r}")
    else:
        known = assignments()
        for assignment in leads_to:
            if assignment not in known:
                found.append(
                    f"{path}: leads_to has {assignment!r}, which is not an assignment in "
                    f"{ROADMAP}'s schedule"
                )

    # Reject unknown keys. A typo in an optional field is otherwise invisible: the block
    # validates, the generated table silently omits what the typo was meant to say, and
    # nothing anywhere reports a problem.
    unknown = set(meta) - {
        "family",
        "serves_lectures",
        "role",
        "needs",
        "requires",
        "note",
        "leads_to",
    }
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
            marker = footnote_marker(len(notes) + 1)
            # Wrapped, because this document is read in a terminal as often as rendered,
            # and the surrounding prose is hand-wrapped to the same width.
            notes.append(
                textwrap.fill(
                    f"- {marker} **`{path.stem}`** — {meta['note']}",
                    width=90,
                    subsequent_indent="  ",
                    # Do not break at hyphens. The default split "Framework-independent"
                    # across two lines, which reads as a different word and made a note
                    # unsearchable in the file it was written into.
                    break_on_hyphens=False,
                )
            )
        for declared in meta.get("serves_lectures", []):
            # Normalized first, so a notebook holding the integer 9 and a schedule row
            # reading `9` meet on one key -- and a lecture that shares a row is folded onto
            # that row, so declaring 23 lands in the "22, 23" row instead of vanishing.
            n = lecture_id(declared)
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


def purpose_line(meta: dict) -> str:
    """Build the sentence a notebook opens with, saying what it is and where it fits.

    Generated from the same block the map is, so a reader and the roadmap cannot be told
    different things. Eric, 2026-09-27: each notebook should be clear about its purpose and
    place. A student opening one in Colab sees a title and nothing else -- which of the two
    families it belongs to, whether it is meant for class or for afterwards, and whether it
    is the head start on an assignment are all invisible at exactly the moment they matter.

    Args:
        meta (dict): A validated ``torchlingo`` block.

    Returns:
        str: One Markdown line.
    """
    titles, aliases = lectures()
    numbers = list(
        dict.fromkeys(
            aliases.get(lecture_id(n), lecture_id(n)) for n in meta["serves_lectures"]
        )
    )

    # Every lecture it serves, each with its own title. Naming only the first was wrong in a
    # way worth recording: for tutorial 1, which serves Lectures 4 and 9, it printed Lecture
    # 4's title as though it described the notebook.
    named = " and ".join(f"Lecture {n} (*{titles[n]}*)" for n in numbers)

    collection = FAMILY_COLLECTION[meta["family"]]
    parts = [f"**{collection}** — {ROLE_PHRASE[meta['role']]} {named}."]
    if meta.get("leads_to"):
        due = assignments()
        starts = ", ".join(
            f"**{a}** (due at Lecture {due[a]})" if due[a] else f"**{a}**"
            for a in meta["leads_to"]
        )
        parts.append(f"A head start on {starts}.")
    return " ".join(parts)


# ASCII only, deliberately. Notebook JSON is written with ensure_ascii, so an em dash here
# would be stored as — while the lookup searched for the literal character -- the find
# would miss, and every run would prepend another banner. That is exactly what happened the
# first time this ran, and the test for it is idempotence rather than correct output.
PURPOSE_MARKER = (
    "<!-- GENERATED:purpose - change the torchlingo metadata, not this cell -->"
)


def purpose_cell(path: Path, meta: dict) -> str:
    """Render the opening cell as notebook JSON, indented to sit in the cells array.

    Args:
        path (Path): The notebook, read for its format version.
        meta (dict): Its validated ``torchlingo`` block.

    Returns:
        str: One cell object, two-space indented, no trailing comma.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    cell = {"cell_type": "markdown", "metadata": {}}
    # nbformat 4.5 requires a cell id; 4.0 rejects one. The course notebooks come from Colab
    # at minor 0 and the tutorials are at minor 5, so both cases are live in this repository
    # and a single shape would fail validation on half of them.
    if nb.get("nbformat_minor", 0) >= 5:
        cell["id"] = "torchlingo-purpose"
    cell["source"] = [f"{PURPOSE_MARKER}\n", "\n", purpose_line(meta)]
    body = json.dumps(cell, indent=1)
    return "\n".join("  " + line for line in body.splitlines())


def _cell_spans(text: str) -> list[tuple[int, int]]:
    """Return the character span of every top-level object in the cells array.

    Textual rather than parsed, for the same reason the rest of this module is: an
    ``nbformat`` round trip rewrites every cell's source from a string into a line list and
    buries a three-line change in hundreds of untouched lines.

    Args:
        text (str): The notebook file's contents.

    Returns:
        list: ``(start, end)`` pairs, end exclusive, in document order.

    Raises:
        ValueError: If the cells array cannot be found.
    """
    anchor = ' "cells": [\n'
    if anchor not in text:
        raise ValueError("no cells array found")
    i = text.index(anchor) + len(anchor)
    spans, depth, start, in_string, escaped = [], 0, None, False, False
    while i < len(text):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
        elif char == "]" and depth == 0:
            break
        i += 1
    return spans


def _purpose_insert_after(text: str) -> int:
    """Choose which cell the generated purpose cell should follow.

    **After the notebook's H1 title**, when there is one in the opening cells. The two
    families lay their openings out differently and a single "insert at the top" rule reads
    wrongly in one of them: a tutorial carries its title and its Colab badge in one cell,
    while a course notebook has the badge alone and the title after it. Inserting at the top
    would give a course notebook badge, purpose, *then* title.

    Args:
        text (str): The notebook file's contents.

    Returns:
        int: Index of the cell to insert after, or -1 to insert at the very top.
    """
    spans = _cell_spans(text)
    for index, (start, end) in enumerate(spans[:3]):
        body = text[start:end]
        # The H1 as JSON-escaped source: either its own line, or the start of one.
        if '"# ' in body or "\\n# " in body:
            return index
    return 0 if spans else -1


def notebook_with_purpose(path: Path, meta: dict) -> str:
    """Return the notebook's text with its generated opening cell inserted or refreshed.

    Textual, for the same reason ``write_meta`` is: an ``nbformat`` round trip would rewrite
    every cell's source from a string into a line list and bury a three-line addition.

    Args:
        path (Path): The notebook.
        meta (dict): Its validated ``torchlingo`` block.

    Returns:
        str: The complete new file contents.

    Raises:
        ValueError: If the cells array cannot be found, rather than writing a cell into a
            place where it would not be a cell.
    """
    text = path.read_text(encoding="utf-8")
    cell = purpose_cell(path, meta)

    marker = text.find(PURPOSE_MARKER)
    if marker != -1:
        # Walk back to the brace that opens the cell holding the marker, then forward to its
        # match, so the whole object is replaced however its fields are ordered.
        start = text.rindex("\n  {", 0, marker) + 1
        depth, i = 0, start
        while True:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        # Keep the cell's own indentation. Stripping it left valid JSON that nbformat
        # accepted, but the next run's lookup walks back to "\n  {" and could no longer find
        # the cell it had just written -- a bug that only appears on the third run.
        return text[:start] + cell + text[i + 1 :]

    anchor = ' "cells": [\n'
    if anchor not in text:
        raise ValueError(f"{path}: no cells array found")

    after = _purpose_insert_after(text)
    if after < 0:
        return text.replace(anchor, f"{anchor}{cell},\n", 1)
    end = _cell_spans(text)[after][1]
    return text[:end] + ",\n" + cell + text[end:]


def late_head_starts() -> list[str]:
    """Find notebooks that prepare an assignment already due by the lecture they serve.

    A notebook declaring ``leads_to: ["A8"]`` while being read at a lecture *after* A8's
    deadline is not helping anyone start A8. That can be deliberate -- a debrief, or a
    notebook a student is meant to revisit -- which is why this reports rather than fails.

    Compares against the notebook's **earliest** lecture, since a notebook serving several
    is available from the first of them.

    Returns:
        list: One line per notebook, empty when every head start arrives in time.
    """
    titles, aliases = lectures()
    order = {key: i for i, key in enumerate(titles)}
    due = assignments()

    late = []
    for path in notebooks():
        meta = read_meta(path)
        served = [
            aliases.get(lecture_id(n), lecture_id(n)) for n in meta["serves_lectures"]
        ]
        positions = [order[s] for s in served if s in order]
        if not positions:
            continue
        first_seen = min(positions)
        for assignment in meta.get("leads_to", []):
            deadline = due.get(assignment)
            if deadline in order and order[deadline] < first_seen:
                earliest = next(k for k, i in order.items() if i == first_seen)
                late.append(
                    f"{path.stem}: read at Lecture {earliest}, but {assignment} is due "
                    f"at Lecture {deadline}"
                )
    return late


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


def main(argv: list[str] | None = None) -> int:
    """Validate the metadata, or print the generated table.

    Args:
        argv (list | None): Arguments to parse. Defaults to the real command line; passed
            explicitly by the tests, so the CLI's own behaviour -- an exit code and a message
            rather than a traceback -- can be asserted instead of assumed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="validate and exit 1 on problems"
    )
    parser.add_argument(
        "--write", action="store_true", help="rewrite the roadmap's map"
    )
    parser.add_argument("--table", action="store_true", help="print the lecture map")
    parser.add_argument(
        "--stamp",
        type=Path,
        metavar="NOTEBOOK",
        help="write this notebook's torchlingo block, then regenerate the map",
    )
    parser.add_argument(
        "--serves",
        nargs="+",
        metavar="LECTURE",
        help="with --stamp: the lecture(s) it serves, e.g. 9 or 8a 8b",
    )
    parser.add_argument(
        "--role", choices=sorted(ROLES), help="with --stamp: how it is used"
    )
    parser.add_argument(
        "--needs",
        nargs="*",
        default=[],
        metavar="PATH",
        help="with --stamp: repo-relative files it cannot run without",
    )
    parser.add_argument(
        "--purpose",
        action="store_true",
        help=(
            "insert or refresh the generated purpose cell in every notebook, so each one "
            "says on its face what it is for"
        ),
    )
    parser.add_argument(
        "--requires",
        nargs="*",
        default=[],
        metavar="CAPABILITY",
        choices=sorted(CAPABILITIES),
        help=(
            "with --stamp: environment capabilities it cannot run without, which keeps it "
            f"out of CI. One or more of {sorted(CAPABILITIES)}"
        ),
    )
    parser.add_argument("--note", help="with --stamp: prose for a non-obvious pairing")
    parser.add_argument(
        "--leads-to",
        nargs="*",
        default=[],
        metavar="ASSIGNMENT",
        help="with --stamp: assignment(s) this notebook gives a head start on, e.g. A6",
    )
    args = parser.parse_args(argv)

    found = notebooks()
    if not found:
        print("no notebooks found; run from the repository root", file=sys.stderr)
        return 2

    # Read the schedule first and report a problem with it as a message rather than a
    # traceback. The message is the whole point of this check -- it tells an author which row
    # is unreadable and what forms are accepted -- and in CI a traceback buries that under
    # Python internals, which is where nobody reads.
    try:
        lectures()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    if args.table:
        print(table())
        return 0

    if args.stamp:
        if not args.serves or not args.role:
            print("--stamp needs --serves and --role", file=sys.stderr)
            return 2
        if not args.stamp.exists():
            print(f"{args.stamp} does not exist", file=sys.stderr)
            return 2
        # family is inferred from the directory rather than asked for. It is the one field
        # that is never a judgement, and the validator rejects a mismatch anyway -- so asking
        # would only create a way to get it wrong.
        family = "course" if args.stamp.parent == COURSE else "tutorial"
        # A plain lecture number is stored as an int and a suffixed one as a string, so the
        # metadata reads the way the schedule does -- `9`, not `"9"` -- while `8a` keeps the
        # only form it has. Both normalize to the same key on the way in, so nothing
        # downstream cares which it got.
        meta = {
            "family": family,
            "serves_lectures": [
                int(s) if str(s).isdigit() else lecture_id(s) for s in args.serves
            ],
            "role": args.role,
            "needs": list(args.needs),
        }
        # Carry forward the optional fields the caller did not mention, rather than
        # dropping them.
        #
        # A stamp used to rebuild the block from the arguments alone, so restamping a
        # notebook to change one field silently deleted every other optional one. That has
        # now cost three notes: tutorials 3 and 4 lost theirs when lecture ids became
        # strings, and tutorial 1 lost a four-line note explaining a retrospective pairing
        # while this very field was being added. Each loss was invisible -- the block still
        # validated, and the generated table does not show `note`.
        #
        # Passing the field explicitly still overrides it, and `--note ''` still clears it,
        # so nothing becomes unreachable. Preserved fields are reported, because silently
        # keeping data is its own small surprise.
        previous = read_meta(args.stamp)
        carried = []
        for field, supplied in (
            ("requires", list(args.requires) if args.requires else None),
            ("note", args.note),
            ("leads_to", list(args.leads_to) if args.leads_to else None),
        ):
            if supplied == "":
                # An explicit empty value clears the field rather than storing a blank.
                continue
            if supplied is not None:
                meta[field] = supplied
            elif field in previous:
                meta[field] = previous[field]
                carried.append(field)

        # Validate before writing, not after. Stamping an invalid block and reporting it on
        # the next run would leave the notebook worse than it was found.
        bad = problems(args.stamp, meta)
        if bad:
            print("refusing to write:", file=sys.stderr)
            for c in bad:
                print(f"  {c}", file=sys.stderr)
            return 1
        write_meta(args.stamp, meta)
        print(
            f"  stamped    {args.stamp}  ({family}, L{','.join(map(str, args.serves))})"
        )
        if carried:
            print(f"  kept       {', '.join(sorted(carried))} from the existing block")
        args.write = True

    complaints = []
    for path in found:
        complaints.extend(problems(path, read_meta(path)))
        complaints.extend(hygiene(path))

    # The purpose cell, applied or checked.
    #
    # Eric, 2026-09-27: each notebook should say on its face what it is for. The text is
    # GENERATED from the metadata rather than written, so the one that students read and the
    # one the roadmap's map reads cannot disagree -- which they did for a week, with the map
    # calling tutorial 2 an activity while the notebook called itself a tutorial.
    #
    # Held until #144 and #145 were settled, because the wording encodes both: whether a
    # tutorial can be an in-class activity, and which assignment each notebook starts.
    if not complaints:
        for path in found:
            meta = read_meta(path)
            wanted = notebook_with_purpose(path, meta)
            if path.read_text(encoding="utf-8") == wanted:
                continue
            if args.purpose:
                path.write_text(wanted, encoding="utf-8")
                print(f"  purpose    {path}")
            else:
                complaints.append(
                    f"{path}: the generated purpose cell is missing or stale. Run: "
                    "python scripts/notebook_meta.py --purpose"
                )

    if complaints:
        print(f"{len(complaints)} problem(s):", file=sys.stderr)
        for c in complaints:
            print(f"  {c}", file=sys.stderr)
        return 1
    print(f"  {len(found)} notebooks, metadata sound")

    # A head start that arrives after the deadline is worth saying out loud.
    #
    # Reported rather than failed, deliberately: WHERE a notebook is read is the
    # instructors' call, and a check that blocked CI over a placement judgement would be
    # overstepping. But it is machine-detectable, and noticing it by eye is exactly what
    # does not happen twice -- `05-real-translations` sat at Lecture 10 for weeks claiming
    # A8, due at Lecture 9, an artifact of the Lecture 8 split rather than a decision
    # anyone made (fixed in Task #164).
    late = late_head_starts()
    if late:
        print(f"\n  {len(late)} notebook(s) prepare an assignment that is already due:")
        for line in late:
            print(f"    {line}")
        print(
            "    Not an error -- placement is the instructors' call. Flag it to them."
        )

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
