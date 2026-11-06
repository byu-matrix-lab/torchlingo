"""scripts/tree_passes.sh: which exact trees passed a check, as preflight and merge_check use it.

These run the shell functions in a throwaway git repository, since what can go wrong is a pass
counted for code that was not the code checked: a different tree, an expired pass, another check.
"""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "tree_passes.sh"


class TestTreePasses(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.repo = Path(folder.name)
        for command in (
            "git init -q",
            "git config user.name test",
            "git config user.email test@localhost",
        ):
            subprocess.run(command.split(), cwd=self.repo, check=True)
        self.commit("first")

    def commit(self, text: str) -> None:
        (self.repo / "file.txt").write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", text],
            cwd=self.repo,
            check=True,
        )

    def shell(self, script: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", f"source {HELPER}\n{script}"],
            cwd=self.repo,
            capture_output=True,
            text=True,
            env={**os.environ, **env},
            check=False,
        )

    def tree(self) -> str:
        return self.shell("tree_of_head").stdout.strip()

    def passed(self, tree: str, kind: str = "preflight", **env: str) -> bool:
        return self.shell(f"tree_passed_recently {tree} {kind}", **env).returncode == 0

    def test_a_tree_has_not_passed_until_recorded(self):
        self.assertFalse(self.passed(self.tree()))
        self.shell(f"record_tree_pass {self.tree()} preflight")
        self.assertTrue(self.passed(self.tree()))

    def test_a_changed_tree_has_not_passed(self):
        self.shell(f"record_tree_pass {self.tree()} preflight")
        self.commit("second")
        self.assertFalse(self.passed(self.tree()))

    def test_the_same_contents_are_the_same_tree(self):
        # What lets a merge with main that changes nothing count: a new commit, the same tree.
        before = self.tree()
        self.shell(f"record_tree_pass {before} preflight")
        self.commit("first")
        self.assertEqual(self.tree(), before)
        self.assertTrue(self.passed(self.tree()))

    def test_each_check_counts_only_for_itself(self):
        self.shell(f"record_tree_pass {self.tree()} install-lecture-14")
        self.assertTrue(self.passed(self.tree(), "install-lecture-14"))
        self.assertFalse(self.passed(self.tree(), "install-lecture-13"))
        self.assertFalse(self.passed(self.tree(), "preflight"))

    def test_a_pass_expires(self):
        self.shell(f"record_tree_pass {self.tree()} preflight")
        record = next((self.repo / ".git" / "torchlingo-passes").iterdir())
        two_hours_ago = time.time() - 2 * 3600
        os.utime(record, (two_hours_ago, two_hours_ago))
        self.assertFalse(self.passed(self.tree(), TREE_PASS_MINUTES="60"))
        self.assertTrue(self.passed(self.tree(), TREE_PASS_MINUTES="180"))

    def test_a_worktree_sees_the_same_passes(self):
        self.shell(f"record_tree_pass {self.tree()} preflight")
        other = self.repo / "elsewhere"
        subprocess.run(
            ["git", "worktree", "add", "-q", "--detach", str(other)],
            cwd=self.repo,
            check=True,
        )
        tree = self.tree()
        seen = subprocess.run(
            ["bash", "-c", f"source {HELPER}\ntree_passed_recently {tree} preflight"],
            cwd=other,
            check=False,
        )
        self.assertEqual(seen.returncode, 0)


if __name__ == "__main__":
    unittest.main()
