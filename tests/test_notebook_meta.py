"""The notebook metadata is the source of two other things, so it is tested.

``scripts/notebook_meta.py`` generates the roadmap's lecture map, and
``scripts/execute_notebooks.py`` decides what to skip in CI from the same ``needs`` field.
A silent failure in either is expensive and quiet: a dropped table row reads as "no notebook
for that lecture", and a dropped ``needs`` entry turns a CI skip into a baffling parse error
on a Git LFS pointer.

The tests that matter here are the ones asserting a *bad* block is rejected. A validator
that only ever sees good input is not known to validate anything.
"""

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import notebook_meta as nm

GOOD = {
    "family": "tutorial",
    "serves_lectures": [8],
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
        self.assertEqual(self.aliases.get(23), 22)
        self.assertNotIn(23, self.titles)

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
        import execute_notebooks as en

        nb = nm.TUTORIALS / "04-attention-and-alignment.ipynb"
        self.assertEqual(
            nm.read_meta(nb)["needs"],
            ["data/example.tsv", "data/pretrained/model.pt"],
        )
        # Whatever is missing must be a subset of what is declared, never something else.
        self.assertTrue(
            set(en.missing_requirements(nb)) <= set(nm.read_meta(nb)["needs"])
        )


class TestMetadataSurvivesJupyter(unittest.TestCase):
    """A Jupyter save must not drop the block, or it would need re-adding constantly."""

    def test_round_trip_preserves_the_block(self):
        nbformat = __import__("nbformat")
        for path in nm.notebooks():
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            restored = nbformat.reads(nbformat.writes(nb), as_version=4)
            self.assertEqual(
                dict(restored.metadata.get("torchlingo", {})),
                json.loads(json.dumps(dict(nb.metadata["torchlingo"]))),
                f"{path.name} lost its block on a round trip",
            )


if __name__ == "__main__":
    unittest.main()
