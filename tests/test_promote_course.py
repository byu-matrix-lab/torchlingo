"""``promote_course.sh`` reads the paths CI watches from the workflow, and must read all of them.

A commit changing only files ``tests_and_build.yml`` does not watch gets no run, and the
script judges it by the newest earlier commit that changed a watched file (Task #295). It
reads that list from the workflow's push trigger with ``awk``, so a reformatted workflow
could make it read too few paths, and then a library change would be promoted on an older
commit's checks. These tests hold the script's reading to PyYAML's.
"""

import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "promote_course.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "tests_and_build.yml"


def watched_by_the_script() -> list[str]:
    """The paths the script's own reader produces, run from the script's own text."""
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index("watched=()")
    end = text.index("\nfi\n", start) + len("\nfi\n")
    reader = text[start:end] + 'printf "%s\\n" "${watched[@]}"\n'
    out = subprocess.run(
        ["bash", "-c", reader], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line.removeprefix(":(glob)") for line in out.stdout.splitlines()]


class TestWatchedPaths(unittest.TestCase):
    def test_the_script_reads_every_path_the_push_trigger_lists(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        # PyYAML reads the key `on` as the boolean True.
        push = (workflow.get("on") or workflow[True])["push"]
        self.assertEqual(watched_by_the_script(), push["paths"])

    def test_the_paths_include_the_library_and_the_notebooks(self):
        """What students install and open: the two a wrong reading must never drop."""
        paths = watched_by_the_script()
        self.assertIn("src/**", paths)
        self.assertIn("docs/**", paths)


if __name__ == "__main__":
    unittest.main()
