"""The roadmap is generated, so what generates it is tested.

``notes/CS479_COURSE_ROADMAP.md`` is rendered by ``scripts/render_roadmap.py`` from
``notes/roadmap/``: the facts in ``roadmap.yaml``, the prose in ``sections/``. Both CS 479
sessions read the rendered file, and it was hand-kept until 2026-10-12, when a due date written
twice and a desktop copy pasted over the repository's had each cost something. These tests pin
the three properties that arrangement depends on: rendering the committed sources reproduces
the committed file, the gate catches a hand edit, and a due date is written once.
"""

import contextlib
import copy
import datetime
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import render_roadmap as rr

# The schedule rows of the real data, read rather than typed, so a schedule change does not
# break a test that is about the mechanism.
DOC = rr.load()


def scratch_doc() -> dict:
    """A fresh copy of the real data, for a test to change."""
    return copy.deepcopy(DOC)


def schedule_row(rendered: str, date: datetime.date) -> str:
    """The schedule row for one date, found by its rendered date cell."""
    cell = f"| {rr.day(date)} |"
    return next(line for line in rendered.splitlines() if cell in line)


class TestTheCommittedRoadmapIsTheRenderedOne(unittest.TestCase):
    """What CI gates on, asserted here too so a local run catches it first."""

    def test_rendering_the_sources_reproduces_the_file(self):
        self.assertEqual(
            rr.current(),
            rr.render(),
            "notes/CS479_COURSE_ROADMAP.md differs from notes/roadmap/; edit the sources "
            "and run python scripts/render_roadmap.py --write",
        )

    def test_the_file_says_it_is_generated(self):
        """The one line that tells a reader where to make an edit."""
        self.assertEqual(rr.current().splitlines()[2], rr.NOTICE)

    def test_every_fragment_is_published_verbatim(self):
        """The prose was moved, not rewritten: each fragment is in the file byte for byte."""
        rendered = rr.render()
        fragments = sorted(rr.SECTIONS.glob("*.md"))
        self.assertTrue(fragments)
        for path in fragments:
            self.assertIn(path.read_text(encoding="utf-8")[:-1], rendered, path.name)


class TestTheCheckFailsOnAHandEdit(unittest.TestCase):
    """``--check`` exits 1 when the Markdown was edited instead of its sources."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(setattr, rr, "ROADMAP", rr.ROADMAP)
        self.copy = self.tmp / "roadmap.md"
        shutil.copy(rr.ROADMAP, self.copy)
        rr.ROADMAP = self.copy

    def check(self) -> int:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return rr.main(["--check"])

    def test_an_untouched_copy_passes(self):
        self.assertEqual(self.check(), 0)

    def test_a_hand_edited_copy_fails(self):
        text = self.copy.read_text(encoding="utf-8")
        self.copy.write_text(text.replace("Wed Oct 14", "Thu Oct 15", 1), "utf-8")
        self.assertEqual(self.check(), 1)

    def test_write_repairs_it(self):
        self.copy.write_text("hand-edited\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rr.main(["--write"]), 0)
        self.assertEqual(self.check(), 0)


class TestTheDueColumnComesFromAssignments(unittest.TestCase):
    """A due date is written once, in ``assignments``, and the schedule shows it."""

    def setUp(self):
        self.doc = scratch_doc()
        self.a9 = next(a for a in self.doc["assignments"] if a.get("id") == "A9")
        self.dates = [row["date"] for row in self.doc["lectures"]]

    def test_moving_a_due_date_moves_the_cell(self):
        old = self.a9["due"]
        new = next(d for d in self.dates if d > old)
        self.a9["due"] = new
        rendered = rr.schedule_table(self.doc)
        self.assertNotIn(self.a9["label"], schedule_row(rendered, old))
        self.assertIn(self.a9["label"], schedule_row(rendered, new))

    def test_two_due_the_same_day_share_one_cell(self):
        self.doc["assignments"].append(
            {"id": "A99", "label": "**A99** another", "due": self.a9["due"]}
        )
        row = schedule_row(rr.schedule_table(self.doc), self.a9["due"])
        self.assertIn(f"{self.a9['label']}{rr.SAME_DAY}**A99** another", row)

    def test_a_due_date_off_the_schedule_is_an_error(self):
        """Rendering it anyway would drop it from the schedule without a word."""
        self.a9["due"] = datetime.date(2026, 1, 1)
        with self.assertRaises(rr.RoadmapError) as caught:
            rr.schedule_table(self.doc)
        self.assertIn("A9", str(caught.exception))

    def test_the_window_table_reads_the_same_due_date(self):
        """The second table that used to restate it now reads the one field."""
        new = next(d for d in self.dates if d > self.a9["due"])
        self.a9["due"] = new
        row = next(
            line
            for line in rr.windows_table(self.doc).splitlines()
            if line.startswith("| A9 |")
        )
        self.assertIn(rr.day(new), row)

    def test_set_at_must_name_a_lecture(self):
        self.a9["set_at"] = "99"
        with self.assertRaises(rr.RoadmapError):
            rr.windows_table(self.doc)


class TestDates(unittest.TestCase):
    def test_the_weekday_is_derived(self):
        self.assertEqual(rr.day(datetime.date(2026, 10, 14)), "Wed Oct 14")
        self.assertEqual(rr.day(datetime.date(2026, 12, 10)), "Thu Dec 10")

    def test_a_quoted_date_is_an_error(self):
        """YAML reads "2026-10-14" in quotes as a string, and a string has no weekday."""
        with self.assertRaises(rr.RoadmapError):
            rr.day("2026-10-14")


class TestTheLayoutPublishesEverythingOnce(unittest.TestCase):
    """A table or fragment the layout leaves out is content that silently stops existing."""

    def test_a_missing_table_is_an_error(self):
        doc = scratch_doc()
        doc["layout"] = [e for e in doc["layout"] if e != {"table": "tooling"}]
        with self.assertRaises(rr.RoadmapError) as caught:
            rr.render(doc)
        self.assertIn("tooling", str(caught.exception))

    def test_a_repeated_table_is_an_error(self):
        doc = scratch_doc()
        doc["layout"].append({"table": "notebook-map"})
        with self.assertRaises(rr.RoadmapError):
            rr.render(doc)

    def test_an_unknown_table_is_an_error(self):
        doc = scratch_doc()
        doc["layout"].append({"table": "no-such-table"})
        with self.assertRaises(rr.RoadmapError):
            rr.render(doc)

    def test_an_unused_fragment_is_an_error(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.addCleanup(setattr, rr, "SECTIONS", rr.SECTIONS)
        sections = tmp / "sections"
        shutil.copytree(rr.SECTIONS, sections)
        (sections / "99-orphan.md").write_text("Prose nobody renders.\n", "utf-8")
        rr.SECTIONS = sections
        with self.assertRaises(rr.RoadmapError) as caught:
            rr.render(scratch_doc())
        self.assertIn("99-orphan", str(caught.exception))

    def test_a_fragment_with_a_trailing_blank_line_is_an_error(self):
        """It would render as two blank lines, and nobody would see why."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.addCleanup(setattr, rr, "SECTIONS", rr.SECTIONS)
        sections = tmp / "sections"
        shutil.copytree(rr.SECTIONS, sections)
        title = sections / "00-title.md"
        title.write_text(title.read_text(encoding="utf-8") + "\n", "utf-8")
        rr.SECTIONS = sections
        with self.assertRaises(rr.RoadmapError):
            rr.render(scratch_doc())


class TestTheNotebookIndex(unittest.TestCase):
    """Generated from each notebook's metadata, with only the purpose kept in the data."""

    def test_every_notebook_has_a_row(self):
        import notebook_meta as nm

        index = rr.notebook_index_table(DOC)
        for path in nm.notebooks():
            self.assertIn(f"| `{path.stem}` |", index)

    def test_a_notebook_without_a_purpose_is_an_error(self):
        doc = scratch_doc()
        stem = next(iter(doc["notebooks"]))
        del doc["notebooks"][stem]["purpose"]
        with self.assertRaises(rr.RoadmapError) as caught:
            rr.notebook_index_table(doc)
        self.assertIn(stem, str(caught.exception))

    def test_a_purpose_for_a_missing_notebook_is_an_error(self):
        doc = scratch_doc()
        doc["notebooks"]["99-imaginary"] = {"purpose": "nothing"}
        with self.assertRaises(rr.RoadmapError):
            rr.notebook_index_table(doc)

    def test_the_ci_column_follows_requires(self):
        self.assertEqual(rr.ci_cell([]), "yes")
        self.assertEqual(rr.ci_cell(["pip", "colab"]), "no: colab, pip")


if __name__ == "__main__":
    unittest.main()
