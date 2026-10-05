"""The version in ``pyproject.toml`` and the one the package reports must agree.

``pyproject.toml`` states the intent already: it is the single source of truth, and
``torchlingo.__version__`` reads it back through installed metadata. Nothing enforced that,
and the gap is not hypothetical -- a working tree at 0.2.0 reported ``0.0.8`` for a whole
session, because the editable install predated the version bump and nothing looks at
installed metadata.

That produced no wrong results, since the code was identical either way. It produced wrong
*provenance*: a benchmark report recorded the library version it thought it was running, and
two results measured on the same code looked like they came from different releases.

Related but not the same as the release workflow's tag guard, which compares a ``v*`` git tag
against ``pyproject.toml``. That catches a mistagged release. This catches a stale
environment, which is the failure a developer actually hits, and it is invisible to the tag
check because no tag is involved.
"""

from __future__ import annotations

import importlib.metadata
import re
import unittest
from pathlib import Path

import torchlingo

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def declared_version() -> str:
    """Return the version declared in ``pyproject.toml``.

    Uses ``tomllib`` where it exists. The package supports Python 3.10, where it does not,
    and adding ``tomli`` as a test dependency to read one string would be a poor trade -- so
    the fallback scans the ``[project]`` table for its ``version`` key.

    Returns:
        str: The declared version.

    Raises:
        AssertionError: If the file has no version in its ``[project]`` table.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib

        return tomllib.loads(text)["project"]["version"]
    except ImportError:
        pass

    # Only the [project] table, so a version key under [tool.something] cannot be picked up
    # by accident.
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = re.match(r"""version\s*=\s*["']([^"']+)["']""", stripped)
            if match:
                return match.group(1)
    raise AssertionError(f"no [project] version found in {PYPROJECT}")


class VersionConsistencyTests(unittest.TestCase):
    """pyproject, installed metadata and the package attribute must all agree."""

    def test_installed_metadata_matches_pyproject(self):
        """A stale install is the common case, so the message says how to fix it."""
        declared = declared_version()
        installed = importlib.metadata.version("torchlingo")
        self.assertEqual(
            installed,
            declared,
            f"pyproject.toml declares {declared} but the installed distribution reports "
            f"{installed}. The environment is stale rather than the code being wrong: "
            f"re-run `pip install -e .` (add --no-deps to leave other packages alone).",
        )

    def test_package_attribute_matches_pyproject(self):
        """``torchlingo.__version__`` is the public surface, so check it directly.

        It reads installed metadata today, but that is an implementation detail this test
        deliberately does not assume -- if it ever becomes a hardcoded string, this is what
        catches the hardcoded string drifting.
        """
        self.assertEqual(torchlingo.__version__, declared_version())

    def test_version_is_not_unknown(self):
        """``__init__`` falls back to "unknown" when metadata is missing.

        A test suite running against an uninstalled package would otherwise pass the two
        checks above only if pyproject also said "unknown", which it never will -- but the
        failure would read as a mismatch rather than as "the package is not installed".
        """
        self.assertNotEqual(
            torchlingo.__version__,
            "unknown",
            "torchlingo.__version__ is 'unknown', which means importlib.metadata could not "
            "find an installed distribution. Install the package before running the suite.",
        )


class VersionLabelTests(unittest.TestCase):
    """``version_label()`` says which code is installed, not only which release.

    The course notebooks install from the ``course`` branch, which moves between
    releases, so ``__version__`` alone stopped identifying what a student runs. The
    label reads pip's own record of where the install came from (``direct_url.json``).
    """

    @staticmethod
    def _installed_from(record):
        """Patch the distribution so its direct_url.json reads ``record``."""
        from unittest import mock

        fake = mock.Mock()
        fake.read_text.return_value = record
        return mock.patch("torchlingo.distribution", return_value=fake)

    def test_git_install_names_the_branch_and_commit(self):
        record = (
            '{"url": "https://github.com/byu-matrix-lab/torchlingo", "vcs_info": '
            '{"vcs": "git", "requested_revision": "course", '
            '"commit_id": "11aa496ff5a85addc05dcdbcb6da365b23140742"}}'
        )
        with self._installed_from(record):
            label = torchlingo.version_label()
        self.assertEqual(label, f"{torchlingo.__version__} (course @ 11aa496)")

    def test_editable_checkout_says_so(self):
        record = '{"url": "file:///repo", "dir_info": {"editable": true}}'
        with self._installed_from(record):
            label = torchlingo.version_label()
        self.assertEqual(label, f"{torchlingo.__version__} (editable checkout)")

    def test_pypi_install_is_the_release_alone(self):
        """A wheel from an index writes no direct_url.json."""
        with self._installed_from(None):
            self.assertEqual(torchlingo.version_label(), torchlingo.__version__)

    def test_unreadable_record_falls_back_to_the_release(self):
        with self._installed_from("not json"):
            self.assertEqual(torchlingo.version_label(), torchlingo.__version__)

    def test_the_real_install_starts_with_the_release(self):
        self.assertTrue(torchlingo.version_label().startswith(torchlingo.__version__))


if __name__ == "__main__":
    unittest.main()
