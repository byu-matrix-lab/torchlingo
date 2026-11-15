"""Every ``torchlingo`` name the docs import exists, and every ``Config(...)`` field is real.

Nothing executes the Python blocks in ``docs/docs/**/*.md``, so a renamed or removed name
leaves them pointing at nothing, silently. Until Nov 2026 the preprocessing pages imported
``train_test_split`` and ``set_default_config``, neither of which ever existed, and the
multilingual page built ``Config(src_file=..., tgt_file=...)``, which raises ``TypeError``.
A student following any of them got an error on the first line. This resolves each import
and checks each keyword passed to ``Config`` literally, without running the blocks.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import unittest
from pathlib import Path

from torchlingo.config import Config

DOCS = Path(__file__).resolve().parents[1] / "docs" / "docs"
BLOCK = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)
# An import statement, with a parenthesized name list continuing over several lines.
IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+\S+[ \t]+)?import[ \t]+(?:\([^)]*\)|[^\n]*)", re.MULTILINE
)
CONFIG_FIELDS = set(inspect.signature(Config.__init__).parameters) - {"self"}
# A design page may sketch an API that was never built. Such a block says so on its first
# line, for the reader as much as for this test, and is the only kind left unchecked.
PROPOSED = "# Proposed API, not in TorchLingo"


def _parse(block: str) -> ast.Module:
    """The block as Python, or, when it does not parse alone, its import statements.

    A block may be a fragment (``...`` in a call, a ``%pip`` line, a doctest prompt), so
    imports are what can always be checked.
    """
    try:
        return ast.parse(block)
    except SyntaxError:
        statements = [
            ast.parse(match.group(0).strip())
            for match in IMPORT.finditer(block)
            if _parses(match.group(0).strip())
        ]
        return ast.Module(
            body=[node for tree in statements for node in tree.body], type_ignores=[]
        )


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _problems(tree: ast.Module) -> list[str]:
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "torchlingo"
        ):
            try:
                module = importlib.import_module(node.module)
            except ImportError as error:
                problems.append(f"from {node.module} import ...: {error}")
                continue
            for alias in node.names:
                if alias.name == "*" or hasattr(module, alias.name):
                    continue
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    problems.append(
                        f"from {node.module} import {alias.name}: no such name"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("torchlingo"):
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as error:
                        problems.append(f"import {alias.name}: {error}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Config"
        ):
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in CONFIG_FIELDS:
                    problems.append(
                        f"Config({keyword.arg}=...): Config has no such field"
                    )
    return problems


class DocsImportTests(unittest.TestCase):
    def test_every_torchlingo_import_in_the_docs_resolves(self):
        pages = sorted(DOCS.rglob("*.md"))
        self.assertGreater(len(pages), 20, "found too few pages; is DOCS right?")
        found = []
        for page in pages:
            for number, block in enumerate(
                BLOCK.findall(page.read_text(encoding="utf-8")), 1
            ):
                if block.lstrip().startswith(PROPOSED):
                    continue
                for problem in _problems(_parse(block)):
                    found.append(f"{page.relative_to(DOCS)}, block {number}: {problem}")
        self.assertEqual(found, [], "\n" + "\n".join(found))

    def test_the_check_catches_what_it_is_for(self):
        """The three kinds of breakage found in Nov 2026, each caught."""
        broken = _problems(
            _parse(
                "from torchlingo.preprocessing import load_data, train_test_split\n"
                "import torchlingo.no_such_module\n"
                "cfg = Config(src_file='a.txt', d_model=8)\n"
            )
        )
        self.assertEqual(len(broken), 3, broken)
        self.assertIn("train_test_split", broken[0] + broken[1] + broken[2])
        self.assertTrue(any("src_file" in problem for problem in broken))
        self.assertFalse(any("d_model" in problem for problem in broken))

    def test_a_fragment_still_has_its_imports_checked(self):
        """A block that does not parse, for a ``%pip`` line or ``...``, keeps its imports."""
        fragment = (
            "%pip install torchlingo\nfrom torchlingo.inference import no_such_name\n"
        )
        self.assertEqual(len(_problems(_parse(fragment))), 1)


if __name__ == "__main__":
    unittest.main()
