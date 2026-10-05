"""The notebook metadata is the source of two other things, so it is tested.

``scripts/notebook_meta.py`` generates the roadmap's lecture map, and
``scripts/execute_notebooks.py`` decides what to skip in CI from the same ``needs`` field.
A silent failure in either is expensive and quiet: a dropped table row reads as "no notebook
for that lecture", and a dropped ``needs`` entry turns a CI skip into a baffling parse error
on a Git LFS pointer.

The tests that matter here are the ones asserting a *bad* block is rejected. A validator
that only ever sees good input is not known to validate anything.
"""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import notebook_meta as nm

# nbformat is a docs extra, not a test dependency, so it is absent in CI's test job. The
# round-trip test skips there rather than failing -- the same treatment the Git LFS artifacts
# get. Everything else here is standard library, deliberately.
try:
    import nbformat
except ImportError:
    nbformat = None

# A lecture that really exists, read from the schedule rather than typed.
#
# This fixture used to hardcode `8`. On 2026-09-27 Lecture 8 was split into 8a and 8b, `8`
# stopped existing, and eighteen tests failed on a change that was correct -- they were
# asserting the old timetable, not the code. The schedule is the source of truth for the code
# under test, so it has to be the source for the fixtures too.
SOME_LECTURE = next(iter(nm.lectures()[0]))

GOOD = {
    "family": "tutorial",
    "serves_lectures": [SOME_LECTURE],
    "role": "reading",
    "needs": ["data/example.tsv"],
}


class TestEveryNotebookDeclaresItself(unittest.TestCase):
    """The shipped notebooks must all carry a sound block."""

    def test_all_notebooks_validate(self):
        problems = []
        for path in nm.notebooks():
            problems.extend(nm.problems(path, nm.read_meta(path)))
        self.assertEqual(problems, [])

    def test_both_families_are_present(self):
        """Guards against a glob that silently matches nothing.

        ``notebooks()`` returning only tutorials would make every course notebook
        unvalidated and absent from the generated table, and every other test here would
        still pass.
        """
        families = {nm.read_meta(p).get("family") for p in nm.notebooks()}
        self.assertEqual(families, {"tutorial", "course"})


class TestValidatorRejectsBadBlocks(unittest.TestCase):
    """Each field's failure mode, one test each."""

    def setUp(self):
        self.path = nm.TUTORIALS / "99-imaginary.ipynb"

    def assertRejected(self, meta, fragment):
        """Assert the validator complains, and complains about the right thing."""
        found = nm.problems(self.path, meta)
        self.assertTrue(found, f"expected a complaint about {fragment!r}")
        self.assertTrue(
            any(fragment in c for c in found),
            f"complaints {found} mention nothing about {fragment!r}",
        )

    def test_good_block_passes(self):
        self.assertEqual(nm.problems(self.path, GOOD), [])

    def test_missing_block(self):
        self.assertRejected({}, "no torchlingo metadata block")

    def test_unknown_family(self):
        self.assertRejected({**GOOD, "family": "lecture"}, "family")

    def test_unknown_role(self):
        self.assertRejected({**GOOD, "role": "lab"}, "role")

    def test_lecture_not_in_the_schedule(self):
        self.assertRejected({**GOOD, "serves_lectures": [99]}, "schedule")

    def test_a_lecture_sharing_a_row_is_accepted(self):
        """Lecture 23 shares its row with 22. Declaring it is correct, not a typo."""
        self.assertEqual(nm.problems(self.path, {**GOOD, "serves_lectures": [23]}), [])

    def test_serves_lectures_not_a_list(self):
        self.assertRejected({**GOOD, "serves_lectures": 8}, "must be a list")

    def test_needs_not_a_list_of_strings(self):
        self.assertRejected({**GOOD, "needs": ["ok", 7]}, "needs must be a list")

    def test_note_must_be_a_string(self):
        self.assertRejected({**GOOD, "note": ["a", "b"]}, "note must be a string")

    def test_typo_in_an_optional_key(self):
        """The failure this check exists for: a plural that validates and does nothing."""
        self.assertRejected({**GOOD, "notes": "oops"}, "unknown key")

    def test_family_must_match_the_directory(self):
        self.assertRejected({**GOOD, "family": "course"}, "but it lives in")

    def test_course_filename_must_match_its_declaration(self):
        meta = {**GOOD, "family": "course", "serves_lectures": [11]}
        found = nm.problems(nm.COURSE / "lecture-05-sentence-alignment.ipynb", meta)
        self.assertTrue(any("filename says lecture 5" in c for c in found), found)


class TestScheduleIsParsedNotRestated(unittest.TestCase):
    """The lecture titles come from the roadmap's schedule, so nothing can disagree.

    These tests exist because the first version of this parser was silently wrong: the
    generated map's own rows have the same shape as the schedule's, so every title came back
    as a notebook filename. It produced a plausible-looking table.
    """

    def setUp(self):
        self.titles, self.aliases = nm.lectures()

    def test_titles_come_from_the_schedule_section(self):
        """A title must appear in the schedule table, not merely somewhere in the file."""
        schedule = nm.ROADMAP.read_text(encoding="utf-8").split(nm.SCHEDULE_HEADING)[1]
        schedule = schedule.split("\n## ")[0]
        for number, title in self.titles.items():
            self.assertIn(
                title, schedule, f"lecture {number}'s title is not the schedule's"
            )

    def test_no_title_is_a_notebook_filename(self):
        """The exact symptom of the bug this guards, asserted directly."""
        stems = {p.stem for p in nm.notebooks()}
        for number, title in self.titles.items():
            self.assertNotIn(title, stems, f"lecture {number}'s title is a filename")

    def test_shared_row_is_recorded_as_an_alias(self):
        self.assertEqual(self.aliases.get("23"), "22")
        self.assertNotIn("23", self.titles)

    def test_non_lecture_rows_are_not_lectures(self):
        """Rows numbered "—" are no-class days, checkpoints and the final exam."""
        for title in self.titles.values():
            self.assertNotIn("No class", title)
            self.assertNotIn("Final exam", title)

    def test_an_unparseable_roadmap_raises(self):
        original = nm.ROADMAP
        try:
            nm.ROADMAP = Path(__file__)  # a real file with no schedule in it
            with self.assertRaises(ValueError):
                nm.lectures()
        finally:
            nm.ROADMAP = original


class TestGeneratedTable(unittest.TestCase):
    """The table is an artifact, so what it must contain is asserted, not eyeballed."""

    def setUp(self):
        self.table = nm.table()

    def test_every_lecture_gets_a_row(self):
        titles, aliases = nm.lectures()
        shared = {v: k for k, v in aliases.items()}
        for n in titles:
            label = f"{n}, {shared[n]}" if n in shared else str(n)
            self.assertIn(f"| {label} | ", self.table)

    def test_a_notebook_serving_two_lectures_appears_in_both_rows(self):
        """Tutorial 1 serves Lectures 4 and 9, which is why this is metadata and not a
        filename. Counted over table rows only: the footnote names it a third time."""
        rows = [line for line in self.table.splitlines() if line.startswith("| ")]
        serving = [n for n, line in enumerate(rows) if "`01-data-and-vocab`" in line]
        self.assertEqual(len(serving), 2)

    def test_unpaired_lectures_are_listed_rather_than_omitted(self):
        """Lecture 1 has no notebook. "Nothing yet" is information; a missing row is not."""
        row = next(
            line for line in self.table.splitlines() if line.startswith("| 1 | ")
        )
        self.assertEqual(row.count("—"), 2)

    def test_notes_are_footnoted_from_the_metadata(self):
        """Compared with whitespace collapsed, because the generator wraps the notes and a
        line break would otherwise break every substring that crosses one."""
        flat = " ".join(self.table.split())
        for path in nm.notebooks():
            note = nm.read_meta(path).get("note")
            if note:
                self.assertIn(" ".join(note.split()), flat)
                self.assertIn(f"**`{path.stem}`**", flat)


class TestRoadmapIsCurrent(unittest.TestCase):
    """What CI gates on, asserted here too so a local run catches it first."""

    def test_roadmap_matches_the_notebooks(self):
        current = nm.ROADMAP.read_text(encoding="utf-8")
        self.assertEqual(
            current,
            nm.roadmap_with_table(current),
            "notes/CS479_COURSE_ROADMAP.md is stale; "
            "run python scripts/notebook_meta.py --write",
        )

    def test_missing_markers_raise_rather_than_append(self):
        with self.assertRaises(ValueError):
            nm.roadmap_with_table("a roadmap with no markers at all")

    def test_markers_out_of_order_raise(self):
        with self.assertRaises(ValueError):
            nm.roadmap_with_table(f"{nm.END}\nbackwards\n{nm.BEGIN}")


class TestExecuteNotebooksReadsTheSameField(unittest.TestCase):
    """The absorbed table. Two declarations of one fact is what this replaced."""

    def test_requirements_table_is_gone(self):
        source = (REPO / "scripts" / "execute_notebooks.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "REQUIREMENTS = {",
            source,
            "a second place to declare what a notebook needs has come back",
        )

    def test_needs_are_resolved_from_metadata(self):
        """What a notebook needs comes from its own metadata, nowhere else.

        The declared list is read rather than restated here. An earlier version
        pinned tutorial 4's exact two entries, which made this test fail when one
        of them was corrected -- reporting a stale expectation as a regression.
        Tutorial 8 is the fixture now: it took tutorial 4's pretrained-model part,
        and the need went with it.
        """
        import execute_notebooks as en

        nb = nm.TUTORIALS / "08-transformer-attention.ipynb"
        declared = nm.read_meta(nb)["needs"]
        # Guards the subset assertion below against passing vacuously.
        self.assertTrue(declared, "the fixture notebook declares nothing to need")
        # Whatever is missing must be a subset of what is declared, never something else.
        self.assertTrue(set(en.missing_requirements(nb)) <= set(declared))


@unittest.skipIf(nbformat is None, "nbformat is a docs extra, absent in CI's test job")
class TestMetadataSurvivesJupyter(unittest.TestCase):
    """A Jupyter save must not drop the block, or it would need re-adding constantly."""

    def test_round_trip_preserves_the_block(self):
        for path in nm.notebooks():
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            restored = nbformat.reads(nbformat.writes(nb), as_version=4)
            self.assertEqual(
                dict(restored.metadata.get("torchlingo", {})),
                json.loads(json.dumps(dict(nb.metadata["torchlingo"]))),
                f"{path.name} lost its block on a round trip",
            )


class TestUnreadableScheduleRowsAreLoud(unittest.TestCase):
    """A lecture must not be able to go missing from the map quietly.

    The forms below were all skipped silently before Task #137: the lecture was absent from
    the generated map, nothing reported anything, and the map still looked complete. Splitting
    a lecture is exactly when an author reaches for one of them, so the failure was waiting
    for the moment it would do most damage.
    """

    def setUp(self):
        self.original = nm.ROADMAP.read_text(encoding="utf-8")
        # Find a real schedule row rather than naming one. Naming one meant that splitting a
        # lecture broke nine tests that had nothing to do with the split.
        self.real_row = next(
            line
            for line in self.original.splitlines()
            if nm.SCHEDULE_ROW.match(line) and line.count("|") >= 5
        )
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(setattr, nm, "ROADMAP", nm.ROADMAP)

    def parse_with(self, replacement):
        """Parse a roadmap whose first real lecture row has been replaced."""
        target = self.tmp / "roadmap.md"
        target.write_text(
            self.original.replace(self.real_row, replacement), encoding="utf-8"
        )
        nm.ROADMAP = target
        return nm.lectures()

    def assertRejects(self, replacement):
        with self.assertRaises(ValueError) as caught:
            self.parse_with(replacement)
        return str(caught.exception)

    def test_letter_suffixes_are_accepted(self):
        """Accepted since 2026-09-27, when Lecture 8 was split into 8a and 8b.

        This test asserted the opposite until then, and the inversion is the whole of Task
        #138. What #137 bought was that the change had to be *made* -- the old parser would
        have dropped both halves of the split lecture and said nothing.
        """
        titles, _ = nm.lectures()
        suffixed = [k for k in titles if not k.isdigit()]
        self.assertTrue(suffixed, "no suffixed lecture in the schedule to verify")
        for key in suffixed:
            self.assertRegex(key, r"^\d+[a-z]$")
            self.assertTrue(titles[key], f"lecture {key} parsed with no title")

    def test_a_range_is_still_rejected(self):
        self.assertRejects("| 8-9 | Wed Sep 30 | NMT Overview | | **F2026** |")

    def test_a_parenthetical_is_rejected(self):
        self.assertRejects("| 8 (part 1) | Wed Sep 30 | NMT Overview | | **F2026** |")

    def test_a_typo_is_rejected(self):
        """Not only new schemes: a stray character is caught for free."""
        self.assertRejects("| 8. | Wed Sep 30 | NMT Overview | | **F2026** |")

    def test_a_multi_letter_suffix_is_rejected(self):
        """One letter is a split; two is a typo or a scheme nobody agreed to."""
        self.assertRejects("| 8ab | Wed Sep 30 | NMT Overview | | **F2026** |")

    def test_every_bad_row_is_reported_not_just_the_first(self):
        message = self.assertRejects(
            "| 8-9 | Wed Sep 30 | NMT Overview | | **F2026** |\n"
            "| 8. | Mon Oct 5 | NMT Architectures | | **F2026** |"
        )
        self.assertIn("2 schedule row(s)", message)

    def test_the_message_says_what_is_accepted(self):
        """An error that does not say what to write instead just moves the confusion."""
        message = self.assertRejects(
            "| 8-9 | Wed Sep 30 | NMT Overview | | **F2026** |"
        )
        self.assertIn("Accepted", message)
        self.assertIn("22, 23", message)
        self.assertIn("#138", message)

    def test_valid_forms_still_parse(self):
        """The unsuffixed and comma forms must keep working, with string keys.

        Asserting `8` as an integer was the version of this that broke: identifiers became
        strings when `8a` arrived, because `8a` has no integer form.
        """
        titles, aliases = self.parse_with(self.real_row)
        self.assertTrue(titles)
        self.assertTrue(all(isinstance(k, str) for k in titles))

        titles, aliases = self.parse_with(
            "| 90, 91 | Wed Sep 30 | A shared session | | **F2026** |"
        )
        self.assertEqual(titles["90"], "A shared session")
        self.assertEqual(aliases["91"], "90")

    def test_non_lecture_rows_are_still_skipped_silently(self):
        """Em-dash rows are deliberate -- no class, project reviews, the final exam -- and
        making them errors would make the check useless."""
        titles, _ = self.parse_with(
            "| — | Wed Sep 30 | **No class**, a made-up holiday | | |"
        )
        self.assertNotIn(8, titles)

    def test_the_real_schedule_parses(self):
        """The guard must not reject the file as it actually stands."""
        titles, aliases = nm.lectures()
        self.assertGreater(len(titles), 15)
        self.assertEqual(aliases.get("23"), "22")

    def test_the_cli_reports_a_message_not_a_traceback(self):
        """The message is the whole value of this check, and a traceback buries it."""
        target = self.tmp / "roadmap.md"
        target.write_text(
            self.original.replace(
                self.real_row, "| 8-9 | Wed Sep 30 | NMT Overview | | **F2026** |"
            ),
            encoding="utf-8",
        )
        nm.ROADMAP = target
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = nm.main(["--check"])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertIn("cannot be read", stderr.getvalue())


class TestAssignmentsAreParsed(unittest.TestCase):
    """Assignments come from the schedule's own column, like the lecture titles do."""

    def setUp(self):
        self.due = nm.assignments()

    def test_assignments_were_found(self):
        self.assertGreater(len(self.due), 5)

    def test_every_id_looks_like_an_assignment(self):
        for name in self.due:
            self.assertRegex(name, r"^A\d+$")

    def test_a_known_assignment_maps_to_its_due_lecture(self):
        """A8 is the NMT model assignment. Which lecture it falls on has moved once
        already, so the test asserts that it resolves to a real lecture rather than to a
        particular one."""
        titles, aliases = nm.lectures()
        self.assertIn(self.due.get("A8"), set(titles) | set(aliases))

    def test_two_assignments_in_one_cell_are_both_found(self):
        """One cell holding two assignments, separated by a middot, yields both.

        Built from a real schedule row rather than read off the live roadmap. The live one
        held A9 and A10 in one cell until roadmap v12 (2026-10-05) moved A9 to its own
        lecture, and a test that needed that coincidence went red on a schedule change."""
        original = nm.ROADMAP.read_text(encoding="utf-8")
        row = next(
            line
            for line in original.splitlines()
            if nm.SCHEDULE_ROW.match(line) and line.count("|") >= 6
        )
        columns = row.split("|")
        columns[4] = " **A98** one · **A99** two "
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.addCleanup(setattr, nm, "ROADMAP", nm.ROADMAP)
        target = tmp / "roadmap.md"
        target.write_text(original.replace(row, "|".join(columns)), encoding="utf-8")
        nm.ROADMAP = target
        due = nm.assignments()
        self.assertIsNotNone(due.get("A98"))
        self.assertEqual(due.get("A98"), due.get("A99"))

    def test_leads_to_is_validated_against_them(self):
        meta = {**GOOD, "leads_to": ["A8"]}
        self.assertEqual(nm.problems(nm.TUTORIALS / "99-x.ipynb", meta), [])
        bad = nm.problems(nm.TUTORIALS / "99-x.ipynb", {**GOOD, "leads_to": ["A99"]})
        self.assertTrue(any("not an assignment" in c for c in bad), bad)

    def test_suffixed_assignments_are_accepted(self):
        """Eric, 2026-09-27: one assignment per lecture, so a split Lecture 8 implies an A8a
        and an A8b. Widened BEFORE the schedule gained those rows -- `**A8a**` did not match
        the old pattern, so it would have been skipped in silence while the schedule plainly
        showed it, and every `leads_to: ["A8a"]` rejected as naming something that does not
        exist. The lecture version of this bug was found after the fact."""
        self.assertEqual(nm.ASSIGNMENT.findall("**A8a**"), ["A8a"])
        self.assertEqual(
            nm.ASSIGNMENT.findall("**A8a** \u00b7 **A8b**"), ["A8a", "A8b"]
        )

    def test_a_two_letter_suffix_is_still_rejected(self):
        """One letter is a split; two is a typo or a scheme nobody agreed to."""
        self.assertEqual(nm.ASSIGNMENT.findall("**A8ab**"), [])

    def test_unsuffixed_assignments_still_parse(self):
        self.assertEqual(nm.ASSIGNMENT.findall("**A8**"), ["A8"])
        self.assertEqual(nm.ASSIGNMENT.findall("**A10**"), ["A10"])

    def test_leads_to_must_be_a_list_of_strings(self):
        bad = nm.problems(nm.TUTORIALS / "99-x.ipynb", {**GOOD, "leads_to": "A8"})
        self.assertTrue(any("leads_to must be a list" in c for c in bad), bad)


class TestPurposeLine(unittest.TestCase):
    """What a student reads at the top of the notebook, so it is asserted like output."""

    def test_names_every_lecture_it_serves(self):
        """Naming only the first was a real bug: tutorial 1 serves Lectures 4 and 9, and
        printing Lecture 4's title alone read as a description of the notebook."""
        line = nm.purpose_line(nm.read_meta(nm.TUTORIALS / "01-data-and-vocab.ipynb"))
        self.assertIn("Lecture 4", line)
        self.assertIn("Lecture 9", line)

    def test_collection_and_role_are_both_stated(self):
        line = nm.purpose_line(
            nm.read_meta(nm.COURSE / "lecture-05-sentence-alignment.ipynb")
        )
        self.assertIn("CS 479 course notebook", line)
        self.assertIn("in-class activity", line)

    def test_the_two_collections_are_distinguishable(self):
        """Eric's distinction: out-of-class tutorials versus in-class exercises. A reader who
        cannot tell which they have opened is the thing this line exists to fix."""
        tutorial = nm.purpose_line(
            nm.read_meta(nm.TUTORIALS / "05-real-translations.ipynb")
        )
        course = nm.purpose_line(
            nm.read_meta(nm.COURSE / "lecture-06-mt-evaluation.ipynb")
        )
        self.assertIn("TorchLingo tutorial", tutorial)
        self.assertNotIn("TorchLingo tutorial", course)

    def test_assignment_head_start_is_stated_without_a_due_date(self):
        """Eric, 2026-09-29: due dates live in Learning Suite, never in a notebook."""
        meta = {
            **nm.read_meta(nm.TUTORIALS / "02-train-tiny-model.ipynb"),
            "leads_to": ["A8"],
        }
        line = nm.purpose_line(meta)
        self.assertIn("A8", line)
        self.assertNotIn("due", line.lower())

    def test_no_head_start_clause_when_leads_to_is_absent(self):
        """Built from a literal rather than naming a real notebook.

        This pointed at `06-diagnosing-failures` and broke the moment that notebook was
        given `leads_to: ["A8"]` -- a test asserting the absence of a field must not depend
        on a particular file continuing to lack it.
        """
        line = nm.purpose_line({**GOOD, "leads_to": []})
        self.assertNotIn("head start", line)
        self.assertIn("head start", nm.purpose_line({**GOOD, "leads_to": ["A8"]}))

    def test_every_role_renders(self):
        """A role with no phrase would raise at generation time, on whichever notebook
        happened to use it first."""
        for role in nm.ROLES:
            meta = {**GOOD, "role": role}
            self.assertTrue(nm.purpose_line(meta).strip())


class TestWritersAreIdempotent(unittest.TestCase):
    """Both writers splice text, so running twice must not double anything."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def copy(self, source: Path) -> Path:
        target = self.tmp / source.name
        shutil.copy(source, target)
        return target

    def test_write_meta_replaces_rather_than_appends(self):
        path = self.copy(nm.TUTORIALS / "06-diagnosing-failures.ipynb")
        meta = nm.read_meta(path)
        for _ in range(3):
            nm.write_meta(path, {**meta, "role": "reference"})
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count('"torchlingo"'), 1)
        self.assertEqual(nm.read_meta(path)["role"], "reference")

    def test_write_meta_refuses_a_file_with_no_metadata_object(self):
        path = self.tmp / "broken.ipynb"
        path.write_text('{"cells": []}', encoding="utf-8")
        with self.assertRaises(ValueError):
            nm.write_meta(path, GOOD)

    def test_purpose_cell_is_written_once_however_often_it_runs(self):
        """The bug this guards appeared only on the THIRD run: the replacement stripped the
        cell's indentation, which stayed valid JSON but broke the next run's lookup.

        The cell count is compared against the count after the FIRST run rather than against
        the fixture's original count. The original version asserted ``before + 1``, which
        held only while no notebook had the cell yet; once #146 stamped all fourteen, the
        first run became a refresh and the assertion failed on a correct refresh.
        """
        for source in (
            nm.TUTORIALS / "06-diagnosing-failures.ipynb",
            nm.COURSE / "lecture-03-word-embeddings.ipynb",
        ):
            path = self.copy(source)
            meta = nm.read_meta(path)
            settled = None
            for run in range(1, 5):
                path.write_text(nm.notebook_with_purpose(path, meta), encoding="utf-8")
                cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
                if settled is None:
                    settled = len(cells)
                self.assertEqual(len(cells), settled, f"{source.name}, run {run}")
                self.assertEqual(
                    path.read_text(encoding="utf-8").count(nm.PURPOSE_MARKER),
                    1,
                    f"{source.name}, run {run}",
                )

    def test_the_marker_is_ascii(self):
        """It is looked up in the raw JSON, which is written with ensure_ascii. A non-ASCII
        character would be stored escaped, the lookup would miss, and every run would prepend
        another banner."""
        self.assertTrue(nm.PURPOSE_MARKER.isascii())

    def test_an_updated_block_rewrites_the_cell_in_place(self):
        path = self.copy(nm.TUTORIALS / "02-train-tiny-model.ipynb")
        meta = nm.read_meta(path)
        path.write_text(nm.notebook_with_purpose(path, meta), encoding="utf-8")
        count = len(json.loads(path.read_text(encoding="utf-8"))["cells"])
        path.write_text(
            nm.notebook_with_purpose(path, {**meta, "leads_to": ["A8"]}),
            encoding="utf-8",
        )
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        self.assertEqual(len(cells), count)
        # Found by its marker, not by index. The cell used to be written at the top; #146
        # moved it below the H1, because a course notebook's badge sits above its title and
        # inserting at the top gave badge, purpose, then title.
        purpose = next(c for c in cells if nm.PURPOSE_MARKER in "".join(c["source"]))
        self.assertIn("A8", "".join(purpose["source"]))

    def test_cell_id_matches_the_notebook_format_version(self):
        """4.5 requires a cell id and 4.0 rejects one. Both versions are live here: the
        course notebooks arrive from Colab at minor 0, the tutorials are at minor 5."""
        for source, wants_id in (
            (nm.TUTORIALS / "06-diagnosing-failures.ipynb", True),
            (nm.COURSE / "lecture-03-word-embeddings.ipynb", False),
        ):
            path = self.copy(source)
            cell = json.loads(
                nm.purpose_cell(path, nm.read_meta(path)).strip().rstrip(",")
            )
            self.assertEqual("id" in cell, wants_id, source.name)

    def test_notebook_with_purpose_refuses_a_file_with_no_cells_array(self):
        path = self.tmp / "nocells.ipynb"
        path.write_text('{"metadata": {}}', encoding="utf-8")
        with self.assertRaises(ValueError):
            nm.notebook_with_purpose(path, GOOD)


class TestLateHeadStartsAreReported(unittest.TestCase):
    """A head start that arrives after the deadline is worth saying out loud.

    Reported rather than failed: where a notebook is read is the instructors' call. But it is
    machine-detectable, and noticing it by eye is exactly what does not happen twice.
    """

    def test_it_finds_one(self):
        """Read at Lecture 10 while preparing A8, due at Lecture 9: tutorial 5's old shape.

        Built in a scratch directory, since the repository no longer has a live example
        (Task #164 fixed tutorial 5), and a test that needs a defect to exist would push
        someone to leave one in.
        """
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        late_copy = tmp / "99-late-reading.ipynb"
        data = json.loads(
            (nm.TUTORIALS / "05-real-translations.ipynb").read_text(encoding="utf-8")
        )
        data["metadata"]["torchlingo"]["leads_to"] = ["A8"]
        late_copy.write_text(json.dumps(data), encoding="utf-8")
        self.addCleanup(setattr, nm, "TUTORIALS", nm.TUTORIALS)
        nm.TUTORIALS = tmp

        late = nm.late_head_starts()
        self.assertTrue(any("99-late-reading" in line for line in late), late)

    def test_a_same_lecture_head_start_is_not_late(self):
        """Tutorial 6 is read at Lecture 9 and A8 is due at Lecture 9. Tight, not late --
        a student can still read it while the assignment is open."""
        late = nm.late_head_starts()
        self.assertFalse(any("06-diagnosing-failures" in line for line in late), late)

    def test_an_on_time_head_start_is_not_reported(self):
        for stem in ("02-train-tiny-model", "lecture-04-tmx-cleaning"):
            self.assertFalse(any(stem in line for line in nm.late_head_starts()), stem)

    def test_the_check_reports_without_failing(self):
        """Exit 0 with the advisory printed. Failing would block CI on a curriculum call."""
        self.addCleanup(setattr, nm, "late_head_starts", nm.late_head_starts)
        nm.late_head_starts = lambda: ["99-late-reading: read at Lecture 10, ..."]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = nm.main(["--check"])
        self.assertEqual(code, 0)
        self.assertIn("already due", stdout.getvalue())


class TestNotebookHygiene(unittest.TestCase):
    """The checks that were run by hand when course notebooks arrived, now automated.

    Task #154. Every assertion here corresponds to something that has actually happened or
    was actually checked for on 2026-09-27: a 675 KB training log committed in a Fall 2025
    notebook, a HuggingFace token read from Colab Secrets, and a tutorial with no badge.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(setattr, nm, "COURSE", nm.COURSE)
        self.source = nm.COURSE / "lecture-03-word-embeddings.ipynb"
        self.base = json.loads(self.source.read_text(encoding="utf-8"))

    def check(self, nb=None, text=None):
        """Write a notebook into a directory treated as `course/`, and check it."""
        target = self.tmp / self.source.name
        target.write_text(
            text if text is not None else json.dumps(nb, indent=1), encoding="utf-8"
        )
        nm.COURSE = self.tmp
        return nm.hygiene(target)

    def with_code_line(self, line):
        """Inject a line into a real code cell, keeping the notebook valid JSON.

        Injecting into the raw text instead would break the JSON, and the checker would
        then complain about *that* -- a pass for the wrong reason, which is how the first
        version of this verification fooled itself.
        """
        nb = json.loads(json.dumps(self.base))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["source"] = [line + "\n"] + cell["source"]
                return nb
        raise AssertionError("fixture notebook has no code cell")

    def test_the_real_course_notebooks_are_clean(self):
        for path in sorted(nm.COURSE.glob("*.ipynb")):
            self.assertEqual(nm.hygiene(path), [], path.name)

    def test_the_real_tutorials_are_clean(self):
        for path in sorted(nm.TUTORIALS.glob("*.ipynb")):
            self.assertEqual(nm.hygiene(path), [], path.name)

    def test_a_committed_output_is_caught_in_a_course_notebook(self):
        nb = json.loads(json.dumps(self.base))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = [
                    {"output_type": "stream", "name": "stdout", "text": "x"}
                ]
                break
        self.assertTrue(any("committed output" in c for c in self.check(nb)))

    def test_committed_outputs_are_REQUIRED_in_a_tutorial(self):
        """The rule inverts between the two families, which is why it is not one rule.

        `docs/mkdocs.yml` sets `execute: false`, so a tutorial's committed outputs are what
        the docs site renders. The first version of this check flagged all six tutorials.
        """
        for path in sorted(nm.TUTORIALS.glob("*.ipynb")):
            nb = json.loads(path.read_text(encoding="utf-8"))
            if sum(len(c.get("outputs") or []) for c in nb["cells"]):
                self.assertEqual(
                    nm.hygiene(path), [], f"{path.name} was flagged for outputs"
                )
                return
        self.skipTest("no tutorial currently commits outputs")

    def test_an_execution_count_is_caught(self):
        nb = json.loads(json.dumps(self.base))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = 7
                break
        self.assertTrue(any("execution_count" in c for c in self.check(nb)))

    def test_token_shapes_are_caught(self):
        for line in (
            "TOKEN = 'hf_" + "A" * 24 + "'",
            "key = 'sk-" + "B" * 24 + "'",
            "aws = 'AKIA" + "C" * 16 + "'",
            "gh = 'ghp_" + "D" * 24 + "'",
        ):
            found = self.check(self.with_code_line(line))
            self.assertTrue(
                any("shaped like a token" in c for c in found), f"missed: {line[:14]}"
            )

    def test_the_whole_token_is_not_echoed(self):
        """A complaint that prints the token publishes it more widely than the commit did."""
        secret = "hf_" + "E" * 24
        found = self.check(self.with_code_line(f"TOKEN = '{secret}'"))
        self.assertTrue(found)
        self.assertNotIn(secret, " ".join(found))

    def test_a_missing_colab_badge_is_caught_for_course_notebooks(self):
        text = self.source.read_text(encoding="utf-8").replace(
            "colab.research.google.com", "example.invalid"
        )
        self.assertTrue(any("Colab badge" in c for c in self.check(text=text)))

    def test_invalid_json_is_reported_once_and_clearly(self):
        found = self.check(text="{ not json")
        self.assertEqual(len(found), 1)
        self.assertIn("not valid JSON", found[0])

    def test_an_empty_notebook_is_caught(self):
        self.assertTrue(any("no cells" in c for c in self.check({"cells": []})))

    def test_a_syntax_error_in_a_course_notebook_is_caught(self):
        """The defect this rule exists for, reproduced.

        `lecture-10-comet-install` shipped `else:` followed by an unindented `drive` -- a
        bare SyntaxError in Lecture 10's own assignment notebook, committed and unnoticed
        because nothing executed course notebooks.
        """
        nb = json.loads(json.dumps(self.base))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["source"] = ["if True:\n", "  pass\n", "else:\n", "drive"]
                break
        complaints = self.check(nb)
        self.assertTrue(any("do not parse" in c for c in complaints), complaints)

    def test_a_worksheet_may_declare_blanks_instead(self):
        """A deliberately incomplete cell is fine once declared."""
        nb = json.loads(json.dumps(self.base))
        nb["metadata"]["torchlingo"]["requires"] = ["blanks"]
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["source"] = ["pattern = # fill this in"]
                break
        self.assertEqual(self.check(nb), [])

    def test_a_stale_blanks_declaration_is_caught(self):
        """The rule runs both ways, or the marker rots into a licence to ship breakage."""
        nb = json.loads(json.dumps(self.base))
        nb["metadata"]["torchlingo"]["requires"] = ["blanks"]
        complaints = self.check(nb)
        self.assertTrue(any("stale" in c for c in complaints), complaints)

    def test_an_inline_magic_is_not_a_syntax_error(self):
        """The false positive that the first version of this check produced.

        `!pip install` inside a `try` block is not Python and is perfectly fine in Jupyter.
        Tutorial 3 does exactly this and passes CI, so flagging it would have been wrong --
        and skipping only cells that *begin* with `!` would not have caught the case.
        """
        nb = json.loads(json.dumps(self.base))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["source"] = [
                    "try:\n",
                    "    import sacrebleu\n",
                    "except ImportError:\n",
                    "    !pip install sacrebleu\n",
                ]
                break
        self.assertEqual(self.check(nb), [])

    def test_a_magic_as_the_only_body_of_a_block_still_parses(self):
        """Stripping a magic must not leave an empty block behind."""
        nb = json.loads(json.dumps(self.base))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["source"] = ["if True:\n", "    %pip install torchlingo\n"]
                break
        self.assertEqual(self.check(nb), [])


class TestThePurposeCellIsGeneratedAndGated(unittest.TestCase):
    """Every notebook opens with a generated line saying what it is for (#146).

    The point of generating it is that the sentence a **student** reads and the row the
    roadmap's map shows come from one source. They disagreed for a week: the map called
    tutorial 2 an activity while the notebook called itself a tutorial.
    """

    def test_every_real_notebook_carries_one(self):
        for path in sorted(nm.TUTORIALS.glob("*.ipynb")) + sorted(
            nm.COURSE.glob("*.ipynb")
        ):
            self.assertIn(
                nm.PURPOSE_MARKER, path.read_text(encoding="utf-8"), path.name
            )

    def test_every_real_notebook_is_current(self):
        """What is committed must equal what the generator would write."""
        for path in sorted(nm.TUTORIALS.glob("*.ipynb")) + sorted(
            nm.COURSE.glob("*.ipynb")
        ):
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                nm.notebook_with_purpose(path, nm.read_meta(path)),
                f"{path.name} is stale; run notebook_meta.py --purpose",
            )

    def test_applying_it_twice_changes_nothing(self):
        """Idempotence, which a marker-based splice does not get for free.

        An earlier version prepended a fresh banner on every run, because the marker held
        an em dash that was stored escaped and so never matched on lookup.
        """
        path = nm.COURSE / "lecture-06-mt-evaluation.ipynb"
        meta = nm.read_meta(path)
        once = nm.notebook_with_purpose(path, meta)
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        copy = tmp / path.name
        copy.write_text(once, encoding="utf-8")
        twice = nm.notebook_with_purpose(copy, meta)
        self.assertEqual(once, twice)
        self.assertEqual(once.count(nm.PURPOSE_MARKER), 1)

    def test_it_lands_after_the_title_not_above_it(self):
        """A course notebook has its badge alone and its title after it.

        Inserting at the top would give badge, purpose, *then* title. The tutorials put
        title and badge in one cell, so a single "insert first" rule reads wrongly in one
        family or the other.
        """
        for name, expected in (
            ("lecture-06-mt-evaluation.ipynb", 2),  # badge, title, purpose
            ("lecture-03-word-embeddings.ipynb", 2),
        ):
            nb = json.loads((nm.COURSE / name).read_text(encoding="utf-8"))
            index = next(
                i
                for i, c in enumerate(nb["cells"])
                if nm.PURPOSE_MARKER in "".join(c["source"])
            )
            self.assertEqual(index, expected, name)

        # A tutorial carries its H1 and badge together, so the cell after it is index 1.
        nb = json.loads(
            (nm.TUTORIALS / "02-train-tiny-model.ipynb").read_text(encoding="utf-8")
        )
        index = next(
            i
            for i, c in enumerate(nb["cells"])
            if nm.PURPOSE_MARKER in "".join(c["source"])
        )
        self.assertEqual(index, 1)

    def test_the_cell_id_follows_the_format_version(self):
        """nbformat 4.5 requires a cell id and 4.0 rejects one; both live here."""
        for directory in (nm.TUTORIALS, nm.COURSE):
            for path in sorted(directory.glob("*.ipynb")):
                nb = json.loads(path.read_text(encoding="utf-8"))
                cell = next(
                    c for c in nb["cells"] if nm.PURPOSE_MARKER in "".join(c["source"])
                )
                if nb.get("nbformat_minor", 0) >= 5:
                    self.assertEqual(cell.get("id"), "torchlingo-purpose", path.name)
                else:
                    self.assertNotIn("id", cell, path.name)

    def test_a_stale_cell_is_reported_by_check(self):
        """The gate must fail on drift, since the cell is student-facing text."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.addCleanup(setattr, nm, "COURSE", nm.COURSE)
        source = nm.COURSE / "lecture-06-mt-evaluation.ipynb"
        target = tmp / source.name
        target.write_text(
            source.read_text(encoding="utf-8").replace(
                "the in-class activity for Lecture 6", "something else entirely"
            ),
            encoding="utf-8",
        )
        nm.COURSE = tmp
        wanted = nm.notebook_with_purpose(target, nm.read_meta(target))
        self.assertNotEqual(target.read_text(encoding="utf-8"), wanted)

    def test_a_notebook_with_no_head_start_says_nothing_about_one(self):
        """The two deliberate `leads_to` blanks must not grow an invented clause."""
        for path in (
            nm.TUTORIALS / "04-attention-and-alignment.ipynb",
            nm.COURSE / "lecture-04-regex-refresher.ipynb",
        ):
            line = nm.purpose_line(nm.read_meta(path))
            self.assertNotIn("head start", line, path.name)


class TestRequiresDeclaresCapabilities(unittest.TestCase):
    """`requires` is capabilities; `needs` is repo-relative paths.

    The distinction is what blocked the second half of Task #154: a notebook can be missing
    no file and still be unrunnable in CI because it installs a package, downloads a model,
    mounts Drive or wants a token. None of those is a path.
    """

    def sound(self, **overrides):
        """Build a metadata block that validates, with fields overridden.

        Args:
            **overrides: Keys to replace in the block.

        Returns:
            dict: The block.
        """
        meta = {
            "family": "course",
            "serves_lectures": [SOME_LECTURE],
            "role": "activity",
            "needs": [],
        }
        meta.update(overrides)
        return meta

    def test_requires_is_optional(self):
        path = nm.COURSE / "x.ipynb"
        self.assertEqual(nm.problems(path, self.sound()), [])

    def test_every_known_capability_is_accepted(self):
        path = nm.COURSE / "x.ipynb"
        for capability in sorted(nm.CAPABILITIES):
            self.assertEqual(
                nm.problems(path, self.sound(requires=[capability])),
                [],
                capability,
            )

    def test_an_unknown_capability_is_rejected(self):
        """A typo must not read as a capability that gates nothing.

        `requires: ["nltk"]` would otherwise silently mean "runnable in CI", which is the
        opposite of what whoever wrote it intended.
        """
        found = nm.problems(nm.COURSE / "x.ipynb", self.sound(requires=["nltk"]))
        self.assertTrue(any("unknown capability" in c for c in found), found)

    def test_requires_must_be_a_list_of_strings(self):
        found = nm.problems(nm.COURSE / "x.ipynb", self.sound(requires="pip"))
        self.assertTrue(any("must be a list of strings" in c for c in found), found)

    def test_the_real_notebooks_declare_only_known_capabilities(self):
        for path in sorted(nm.COURSE.glob("*.ipynb")) + sorted(
            nm.TUTORIALS.glob("*.ipynb")
        ):
            declared = set(nm.read_meta(path).get("requires", []))
            self.assertLessEqual(declared, nm.CAPABILITIES, path.name)


class TestStampingKeepsWhatItWasNotAskedToChange(unittest.TestCase):
    """Restamping one field must not delete the others.

    A stamp used to rebuild the block from its arguments alone. **That has cost three
    notes**: tutorials 3 and 4 lost theirs when lecture ids became strings, and tutorial 1
    lost a four-line note explaining a retrospective pairing while `requires` was being
    added. Every loss was invisible -- the block still validated, and the generated table
    does not render `note`.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        source = nm.TUTORIALS / "01-data-and-vocab.ipynb"
        self.path = self.tmp / source.name
        shutil.copy(source, self.path)

        # `--stamp` implies `--write`, so every stamp here REGENERATES THE ROADMAP. With
        # TUTORIALS pointed at a directory holding one notebook, that wrote a map whose
        # entire tutorials column was `—` -- into the real file, while the tests passed.
        # Caught because the CLI's own --check disagreed with a green suite.
        #
        # So ROADMAP is redirected too, and `test_the_repository_roadmap_is_untouched`
        # below asserts it, because the failure was invisible from inside the tests.
        self.roadmap_copy = self.tmp / "ROADMAP.md"
        shutil.copy(nm.ROADMAP, self.roadmap_copy)
        self.real_roadmap = nm.ROADMAP
        self.roadmap_before = nm.ROADMAP.read_text(encoding="utf-8")
        self.addCleanup(setattr, nm, "TUTORIALS", nm.TUTORIALS)
        self.addCleanup(setattr, nm, "ROADMAP", nm.ROADMAP)
        nm.TUTORIALS = self.tmp
        nm.ROADMAP = self.roadmap_copy

    def tearDown(self):
        """Fail loudly if a stamp reached the repository's roadmap."""
        self.assertEqual(
            self.real_roadmap.read_text(encoding="utf-8"),
            self.roadmap_before,
            "a --stamp in this test class rewrote the repository's roadmap",
        )

    def stamp(self, *extra):
        """Run --stamp on the fixture, suppressing its output.

        Args:
            *extra: Additional command-line arguments.

        Returns:
            dict: The resulting metadata block.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            nm.main(
                [
                    "--stamp",
                    str(self.path),
                    "--serves",
                    "4",
                    "--role",
                    "reference",
                    *extra,
                ]
            )
        return nm.read_meta(self.path)

    def test_a_note_survives_a_restamp_that_did_not_mention_it(self):
        original = nm.read_meta(self.path)["note"]
        self.assertEqual(self.stamp("--leads-to", "A5")["note"], original)

    def test_leads_to_survives_a_restamp_that_did_not_mention_it(self):
        self.stamp("--leads-to", "A5")
        self.assertEqual(self.stamp()["leads_to"], ["A5"])

    def test_requires_survives_a_restamp_that_did_not_mention_it(self):
        self.stamp("--requires", "pip")
        self.assertEqual(self.stamp()["requires"], ["pip"])

    def test_an_explicit_value_still_overrides(self):
        self.assertEqual(self.stamp("--note", "replaced")["note"], "replaced")

    def test_an_empty_note_clears_the_field(self):
        """Preserving by default must not make clearing impossible."""
        self.assertNotIn("note", self.stamp("--note", ""))


class TestTheExecutorSkipsOnCapabilities(unittest.TestCase):
    """Task #154's second half: the gate that decides what CI runs."""

    def setUp(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import execute_notebooks

        self.en = execute_notebooks

    def test_a_declared_capability_is_reported_as_missing(self):
        """Every declared capability counts as absent: CI has none of them, by decision."""
        path = nm.COURSE / "lecture-10-comet-install.ipynb"
        self.assertEqual(
            self.en.missing_capabilities(path), ["colab", "hf-token", "pip"]
        )

    def test_a_notebook_declaring_nothing_is_runnable(self):
        for path in sorted(nm.TUTORIALS.glob("*.ipynb")):
            self.assertEqual(self.en.missing_capabilities(path), [], path.name)

    def test_the_executor_looks_at_both_families(self):
        """Course notebooks were executed by nothing at all until this was widened."""
        source = (REPO / "scripts" / "execute_notebooks.py").read_text(encoding="utf-8")
        self.assertIn("COURSE", source)
        self.assertIn("args.course", source)


class TestWriteMetaWhereverTheBlockSits(unittest.TestCase):
    """Rewriting the block must leave valid JSON whether it is first or last.

    A notebook this tool stamped has the block first. An ``nbformat`` round trip -- which
    re-executing a tutorial is -- sorts the metadata keys and moves it last, and the rewrite
    used to add a comma after it unconditionally: a trailing comma, so invalid JSON.
    Tutorials 4 and 5 were left in that shape by a re-execution, one restamp from breaking.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def roundtrip(self, metadata: dict) -> dict:
        path = self.tmp / "nb.ipynb"
        path.write_text(
            json.dumps({"cells": [], "metadata": metadata, "nbformat": 4}, indent=1),
            encoding="utf-8",
        )
        nm.write_meta(path, {"family": "tutorial", "role": "reading"})
        return json.loads(path.read_text(encoding="utf-8"))["metadata"]

    def test_block_first(self):
        meta = self.roundtrip(
            {"torchlingo": {"role": "activity"}, "kernelspec": {"name": "x"}}
        )
        self.assertEqual(meta["torchlingo"]["role"], "reading")
        self.assertEqual(meta["kernelspec"], {"name": "x"})

    def test_block_last(self):
        meta = self.roundtrip(
            {"kernelspec": {"name": "x"}, "torchlingo": {"role": "activity"}}
        )
        self.assertEqual(meta["torchlingo"]["role"], "reading")
        self.assertEqual(meta["kernelspec"], {"name": "x"})


if __name__ == "__main__":
    unittest.main()
