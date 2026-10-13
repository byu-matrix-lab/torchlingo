"""Lecture 12 (Assignment 12), executed from the notebook: Colab faked, the LLM stubbed.

The notebook declares ``colab``, so CI's notebook run skips it, and it needs ``transformers`` and
a model download that CI should not make. What these tests run is everything else, from the
notebook file itself: the data cell on the built-in sample and on each layout the loader accepts,
the prompt, the loop over context sizes, the BLEU report and the files the assignment uploads.

The model cell is left out, and a stand-in ``translate`` takes its place: it returns the
reference once the prompt holds ten or more examples, and the source otherwise, so BLEU must rise
with context and the report must show it. What is checked is the path, not an LLM's quality.

Two cells are exercises: ``split_pool_and_test`` and ``build_prompt`` are the student's, and
raise ``NotImplementedError`` until written. The tests write them in, one straightforward way
each, and check that the notebook as shipped stops at the first and that a wrong split fails.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute

NOTEBOOK = COURSE / "lecture-12-llm-context.ipynb"
MODEL_CELL = "AutoModelForCausalLM"

# A student's answers to the two exercises, in place of the line that raises.
ANSWERS = {
    'raise NotImplementedError("write split_pool_and_test, then run this cell again")': (
        "chosen = pairs.sample(pool_size + test_size, random_state=seed)\n"
        "    return chosen.iloc[:pool_size], chosen.iloc[pool_size:]"
    ),
    'raise NotImplementedError("write build_prompt, then run this cell again")': (
        'lines = [f"Translate the English sentence into {LANGUAGE}.", ""]\n'
        "    for src, tgt in examples:\n"
        '        lines += [f"English: {src}", f"{LANGUAGE}: {tgt}", ""]\n'
        '    return "\\n".join(lines + [f"English: {sentence}", f"{LANGUAGE}:"])'
    ),
}


def notebook_cells(
    data_dir: str | None = None, answered: bool = True
) -> list[tuple[int, str]]:
    """Every code cell but the install and the model, pointed at ``data_dir`` if given."""
    cells = [
        c for c in code_cells(NOTEBOOK) if "%pip" not in c[1] and MODEL_CELL not in c[1]
    ]
    swaps = {"plt.show()": "plt.close()", **(ANSWERS if answered else {})}
    if data_dir is not None:
        swaps["DATA_DIR = None"] = f"DATA_DIR = {data_dir!r}"
    return substitute(cells, swaps)


def pairs(n: int) -> list[tuple[str, str]]:
    """n distinct English and made-up target sentences, six to nine words long."""
    return [
        (
            f"sentence number {i} is here in English" + " today" * (i % 4),
            f"lino {i} ka mwahu nan lokaiahn" + " rahnwet" * (i % 4),
        )
        for i in range(n)
    ]


def stub_namespace(namespace: dict) -> dict:
    """What the model cell defines, replaced by a translator that improves with context."""
    namespace["MODEL"] = "stub"
    by_source = {}

    def translate(sentences, examples, batch_size=8):
        if not by_source:
            test = namespace["test"]
            by_source.update(zip(test.src, test.tgt))
        return [by_source[s] if len(examples) >= 10 else s for s in sentences]

    namespace["translate"] = translate
    return namespace


class NotebookCase(unittest.TestCase):
    def run_notebook(self, data_dir_contents=None):
        """Run the cells in a fresh faked Drive; return (namespace, printed, output dir)."""
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        drive = Path(scratch.name)
        if data_dir_contents is None:
            # The built-in sample's layout; in CI the real one is a Git LFS pointer.
            (drive / "data").mkdir()
            rows = [(s, t, "talk", "transcript") for s, t in pairs(90)]
            rows += [("Too short", "Corto", "talk", "transcript")] * 3
            pd.DataFrame(rows, columns=["src", "tgt", "talk", "kind"]).to_csv(
                drive / "data" / "example.tsv", sep="\t", index=False
            )
            cells = notebook_cells()
        else:
            data_dir = drive / "a12-data"
            data_dir.mkdir()
            data_dir_contents(data_dir)
            cells = notebook_cells(str(data_dir))
        model_at = next(i for i, src in code_cells(NOTEBOOK) if MODEL_CELL in src)
        before = [c for c in cells if c[0] < model_at]
        after = [c for c in cells if c[0] > model_at]
        with colab_faked(drive):
            namespace, printed = run_cells(before)
            namespace, more = run_cells(after, stub_namespace(namespace))
        return namespace, printed + more, drive / "CS479" / "assignment12"


class TestBuiltInSample(NotebookCase):
    def test_runs_end_to_end_and_writes_the_upload_files(self):
        namespace, printed, out = self.run_notebook()

        self.assertIn("Spanish is not the assignment", printed)
        uploads = {"test.tsv", "bleu-report.tsv", "bleu-by-context.png"}
        uploads |= {f"context-{k}.tsv" for k in (0, 5, 10, 20)}
        uploads |= {f"hyp-{k}.txt" for k in (0, 5, 10, 20)}
        self.assertEqual({p.name for p in out.iterdir()}, uploads)

        # Run all with no prediction still runs, and the report says it is missing
        self.assertIn("your prediction: (none written)", printed)

        # Fifty test pairs; each context file holds exactly k pairs, nested; none is a test pair
        test = pd.read_csv(out / "test.tsv", sep="\t", dtype=str, keep_default_na=False)
        self.assertEqual(len(test), 50)
        contexts = {
            k: pd.read_csv(
                out / f"context-{k}.tsv", sep="\t", dtype=str, keep_default_na=False
            )
            for k in (0, 5, 10, 20)
        }
        self.assertEqual(
            {k: len(c) for k, c in contexts.items()}, {0: 0, 5: 5, 10: 10, 20: 20}
        )
        self.assertEqual(list(contexts[5].src), list(contexts[20].src[:5]))
        self.assertFalse(set(contexts[20].src) & set(test.src))

        # The prompt shown at k = 5 has five examples and the sentence to translate
        self.assertEqual(printed.count("English: "), 6)

        # The report: one row per size, a signature, and the rise the stub was built to give
        report = pd.read_csv(out / "bleu-report.tsv", sep="\t")
        self.assertEqual(list(report.context), [0, 5, 10, 20])
        self.assertTrue(report.signature.str.contains("nrefs:1").all())
        self.assertLess(report.bleu[0], report.bleu[2])
        self.assertEqual(report.bleu[3], 100.0)


class TestStudentData(NotebookCase):
    def test_reads_one_tsv(self):
        def write(data_dir):
            frame = pd.DataFrame(pairs(80), columns=["english", "pohnpeian"])
            frame.to_csv(data_dir / "pohnpeian.tsv", sep="\t", index=False)

        namespace, printed, out = self.run_notebook(write)
        self.assertNotIn("Spanish is not the assignment", printed)
        self.assertEqual(len(namespace["pairs"]), 80)
        self.assertTrue(namespace["test"].src.str.startswith("sentence number").all())

    def test_reads_two_aligned_text_files(self):
        def write(data_dir):
            src, tgt = zip(*pairs(80))
            (data_dir / "train.en.txt").write_text(
                "\n".join(src) + "\n", encoding="utf-8"
            )
            (data_dir / "train.pon.txt").write_text(
                "\n".join(tgt) + "\n", encoding="utf-8"
            )

        namespace, _, _ = self.run_notebook(write)
        self.assertTrue(namespace["test"].tgt.str.startswith("lino").all())

    def test_stops_with_a_message_on_another_layout(self):
        def write(data_dir):
            (data_dir / "data.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(SystemExit) as stopped:
            self.run_notebook(write)
        self.assertIn(
            "expected one .tsv file or two .txt files", str(stopped.exception)
        )


class TestExercises(unittest.TestCase):
    """The checks around the two exercises, run on the data cell alone."""

    def data_cell(self, answered: bool = True) -> tuple[int, str]:
        cells = notebook_cells(str(self.data_dir), answered=answered)
        return next(c for c in cells if "def split_pool_and_test(" in c[1])

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.drive = Path(scratch.name)
        self.data_dir = self.drive / "a12-data"
        self.data_dir.mkdir()
        frame = pd.DataFrame(pairs(80), columns=["english", "pohnpeian"])
        frame.to_csv(self.data_dir / "pohnpeian.tsv", sep="\t", index=False)
        self.setup_cell = next(
            c for c in code_cells(NOTEBOOK) if "setup(gpu=True" in c[1]
        )

    def test_as_shipped_it_stops_at_the_first_exercise(self):
        with (
            colab_faked(self.drive),
            self.assertRaises(NotImplementedError) as stopped,
        ):
            run_cells([self.setup_cell, self.data_cell(answered=False)])
        self.assertIn("write split_pool_and_test", str(stopped.exception))

    def test_a_split_that_shares_a_sentence_is_rejected(self):
        """Pool and test drawn independently can overlap; the toy check names the fault."""
        index, source = self.data_cell()
        overlapping = source.replace(
            "return chosen.iloc[:pool_size], chosen.iloc[pool_size:]",
            "start = pool_size - 1  # one pair off: the right sizes, one sentence shared\n"
            "    return chosen.iloc[:pool_size], chosen.iloc[start : start + test_size]",
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells([self.setup_cell, (index, overlapping)])
        self.assertIn("in both the pool and the test set", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
