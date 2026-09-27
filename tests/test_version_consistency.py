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


if __name__ == "__main__":
    unittest.main()
