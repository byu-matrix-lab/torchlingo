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
import roadmap_charts as rc

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
        # The charts too, so a --write here cannot touch the repository's.
        self.addCleanup(setattr, rc, "GENERATED", rc.GENERATED)
        shutil.copytree(rc.GENERATED, self.tmp / "generated")
        rc.GENERATED = self.tmp / "generated"

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

    def test_a_due_date_outside_the_schedule_is_an_error(self):
        """Rendering it anyway would drop it from the schedule without a word."""
        for outside in (datetime.date(2026, 1, 1), datetime.date(2027, 1, 1)):
            self.a9["due"] = outside
            with self.assertRaises(rr.RoadmapError) as caught:
                rr.schedule_table(self.doc)
            self.assertIn("A9", str(caught.exception))

    def test_a_due_date_between_rows_is_shown_on_the_row_before_with_its_date(self):
        """Eric, Oct 22: A12 is due on a Friday, a day with no lecture."""
        before = next(d for d in self.dates if d > self.a9["due"])
        between = before + datetime.timedelta(days=1)
        self.assertNotIn(between, self.dates)
        self.a9.update(due=between, due_time="11:59 pm")
        row = schedule_row(rr.schedule_table(self.doc), before)
        self.assertIn(f"{self.a9['label']} (due {rr.day(between)}, 11:59 pm)", row)
        window = next(
            line
            for line in rr.windows_table(self.doc).splitlines()
            if line.startswith("| A9 |")
        )
        self.assertIn(f"{rr.day(between)}, 11:59 pm", window)

    def test_a_due_time_on_a_lecture_day_is_stated(self):
        self.a9["due_time"] = "11:59 pm"
        row = schedule_row(rr.schedule_table(self.doc), self.a9["due"])
        self.assertIn(f"{self.a9['label']} (due 11:59 pm)", row)

    def test_a_dropped_assignment_leaves_the_schedule_and_keeps_its_window_row(self):
        old = self.a9.pop("due")
        self.a9["dropped"] = "folded into the final project"
        self.assertNotIn(self.a9["label"], rr.schedule_table(self.doc))
        window = next(
            line
            for line in rr.windows_table(self.doc).splitlines()
            if line.startswith("| A9 |")
        )
        self.assertIn("| dropped | folded into the final project |", window)
        self.assertNotIn(rr.day(old), window)

    def test_an_assignment_neither_due_nor_dropped_is_an_error(self):
        del self.a9["due"]
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


class TestTheChartsComeFromTheRoadmap(unittest.TestCase):
    """The Assignments page's two charts are generated, so they cannot drift from the facts.

    They were hand-kept Mermaid until 2026-10-26, and the tutorial map had already gained
    arrows that no notebook's ``leads_to`` had.
    """

    def test_the_committed_charts_are_the_rendered_ones(self):
        self.assertEqual(
            rr.stale_charts(rr.chart_files()),
            [],
            "a chart in notes/roadmap/generated/ differs from its sources; run "
            "python scripts/render_roadmap.py --write",
        )

    def test_every_assignment_edge_is_drawn_with_its_label(self):
        chart = rc.assignment_roadmap(DOC)
        for edge in DOC["charts"]["assignment_edges"]:
            arrow = "-.->" if edge.get("prepares") else "-->"
            self.assertIn(
                f"{edge['from']} {arrow}|{edge['carries']}| {edge['to']}", chart
            )

    def test_an_edge_to_an_unknown_assignment_is_an_error(self):
        doc = scratch_doc()
        doc["charts"]["assignment_edges"].append(
            {"from": "A4", "to": "A99", "carries": "x"}
        )
        with self.assertRaises(rr.RoadmapError) as caught:
            rc.assignment_roadmap(doc)
        self.assertIn("A99", str(caught.exception))

    def test_an_early_assignment_without_an_arrow_is_anchored_above_the_spine(self):
        self.assertIn("A1 ~~~ A4", rc.assignment_roadmap(DOC))

    def test_the_slide_marks_here_and_leaves_out_unconnected_early_ones(self):
        slide = rc.assignment_roadmap(DOC, here="A12")
        self.assertIn("you are here", slide)
        self.assertIn("style A12 stroke-width:4px", slide)
        self.assertNotIn('A1["', slide)
        self.assertIn('A7["', slide, "A7 has an arrow, so it stays")

    def test_each_leads_to_is_an_arrow_in_the_tutorial_map(self):
        import notebook_meta as nm

        chart = rc.tutorial_map(DOC)
        for path in nm.notebooks():
            meta = nm.read_meta(path)
            if meta["family"] != "tutorial":
                continue
            number = path.stem.split("-")[0].lstrip("0")
            for target in meta.get("leads_to", []):
                self.assertRegex(chart, rf"T{number} -->\|[^|]+\| {target}\n")

    def test_a_leads_to_without_a_label_is_an_error(self):
        """A notebook gaining a head start must not appear on the chart unlabelled."""
        doc = scratch_doc()
        del doc["charts"]["tutorials"]["01-data-and-vocab"]["labels"]["A9"]
        with self.assertRaises(rr.RoadmapError) as caught:
            rc.tutorial_map(doc)
        self.assertIn("01-data-and-vocab", str(caught.exception))

    def test_read_after_is_drawn_back_from_the_assignment(self):
        self.assertIn("T9 <-.->|read after A14| A14", rc.tutorial_map(DOC))

    def test_check_fails_on_a_hand_edited_chart(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.addCleanup(setattr, rc, "GENERATED", rc.GENERATED)
        shutil.copytree(rc.GENERATED, tmp / "generated")
        rc.GENERATED = tmp / "generated"
        chart = rc.GENERATED / "tutorial-map.mmd"
        chart.write_text(chart.read_text(encoding="utf-8") + "    T1 --> A6\n", "utf-8")
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(rr.main(["--check"]), 1)


if __name__ == "__main__":
    unittest.main()
