"""Lecture 12 (Assignment 12), executed from the notebook: Colab faked, the LLM stubbed.

The notebook declares ``colab``, so CI's notebook run skips it, and it needs ``transformers`` and
a model download that CI should not make. What these tests run is everything else, from the
notebook file itself: the data cell on the built-in sample and on the A12 folder's layout,
the prompt, the loop over context sizes, the BLEU report and the files the assignment uploads.

The model cells are left out, and stand-ins take their place. ``translate`` returns the
reference once the prompt holds ten or more examples, and the source otherwise, so the scores
must rise with context and the report must show it. ``nllb_translate`` gets the first half of
the test set right, for a language NLLB covers, and is absent for one it does not. What is
checked is the path, not an LLM's quality. The pool shrinks from 500 to 25, so the fixtures stay
small.

Two cells are exercises: ``split_pool_and_test`` and ``build_prompt`` are the student's, and
raise ``NotImplementedError`` until written. The tests write them in, one straightforward way
each, and a prediction, and check that the notebook as shipped stops at the first, that it stops
without a prediction, that a split sharing a sentence or ignoring the seed fails, and that
``LANGUAGE`` and ``DATA_DIR`` must be set together.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute

NOTEBOOK = COURSE / "lecture-12-llm-context.ipynb"
MODEL_CELL = "AutoModelForCausalLM"
NLLB_CELL = "AutoModelForSeq2SeqLM"
# The smallest pool Step 6 accepts: draw 1's five plus a placebo for each test sentence.
TEST_SIZE = 30
SIZES = {
    "POOL_SIZE, TEST_SIZE = 500, 50": f"POOL_SIZE, TEST_SIZE = {5 + TEST_SIZE}, {TEST_SIZE}"
}
DRAWS = (1, 2, 3)

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
        "    if before is not None:\n"
        '        lines += [f"The sentence before, in English: {before[0]}", f"In {LANGUAGE}: {before[1]}", ""]\n'
        '    return "\\n".join(lines + [f"English: {sentence}", f"{LANGUAGE}:"])'
    ),
}
PREDICTED = {
    'PREDICTION = ""': 'PREDICTION = "context helps up to 10"',
    'PREDICTION_BEFORE = ""': 'PREDICTION_BEFORE = "it settles a pronoun or two"',
}


def notebook_cells(
    data_dir: str | None = None,
    answered: bool = True,
    predicted: bool = True,
    language: str | None = "Pohnpeian",
) -> list[tuple[int, str]]:
    """Every code cell but the install and the two models, pointed at ``data_dir`` if given.

    With ``data_dir``, ``LANGUAGE`` is set to ``language`` too, unless that is None.
    """
    cells = [
        c
        for c in code_cells(NOTEBOOK)
        if "%pip" not in c[1] and MODEL_CELL not in c[1] and NLLB_CELL not in c[1]
    ]
    swaps = {
        "plt.show()": "plt.close()",
        **SIZES,
        **(ANSWERS if answered else {}),
        **(PREDICTED if predicted else {}),
    }
    if data_dir is not None:
        swaps["DATA_DIR = None"] = f"DATA_DIR = {data_dir!r}"
        if language is not None:
            swaps['LANGUAGE = "Spanish"'] = f"LANGUAGE = {language!r}"
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
    """What the model cells define, replaced by translators whose scores are known.

    ``translate`` improves with context; ``nllb_translate`` exists only for Spanish and Telugu,
    the languages the notebook's NLLB cell lists (``test_nllb_covers_only_telugu`` holds the two
    together), and gets half the test set right.
    """
    namespace["MODEL"] = "stub"
    by_source = {}

    def reference(sentence):
        if not by_source:
            test = namespace["test"]
            by_source.update(zip(test.src, test.tgt))
        return by_source[sentence]

    def translate(sentences, examples_for, befores=None, batch_size=8):
        befores = befores or [None] * len(sentences)
        return [
            reference(s) if len(examples) >= 10 else s
            for s, examples in zip(sentences, examples_for)
        ]

    def nllb_translate(sentences):
        half = len(sentences) // 2
        return [reference(s) for s in sentences[:half]] + list(sentences[half:])

    namespace["translate"] = translate
    namespace["NLLB"] = "stub-nllb"
    covered = namespace["LANGUAGE"] in ("Spanish", "Telugu")
    namespace["nllb_translate"] = nllb_translate if covered else None
    return namespace


class NotebookCase(unittest.TestCase):
    def run_notebook(self, data_dir_contents=None, **cell_options):
        """Run the cells in a fresh faked Drive; return (namespace, printed, output dir).

        ``cell_options`` go to ``notebook_cells``.
        """
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
            cells = notebook_cells(**cell_options)
        else:
            data_dir = drive / "a12-data"
            data_dir.mkdir()
            data_dir_contents(data_dir)
            cells = notebook_cells(str(data_dir), **cell_options)
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
        uploads = {"test.tsv", "scores.tsv", "scores-by-context.png", "a12-rerun.json"}
        uploads |= {"hyp-0.txt", "hyp-nllb.txt"}
        uploads |= {"before-5.tsv", "hyp-5-before.txt", "hyp-5-placebo.txt"}
        for k in (5, 10, 20):
            uploads |= {f"nearest-{k}.tsv", f"hyp-nearest-{k}.txt"}
            uploads |= {f"context-{k}-draw{d}.tsv" for d in DRAWS}
            uploads |= {f"hyp-{k}-draw{d}.txt" for d in DRAWS}
        self.assertEqual({p.name for p in out.iterdir()}, uploads)

        self.assertIn("your prediction: context helps up to 10", printed)
        self.assertIn("a12-rerun.json, then this notebook", printed)

        # TEST_SIZE test pairs. Each draw's context files hold exactly k pairs, nested within
        # the draw; the draws differ; none is a test pair.
        test = pd.read_csv(out / "test.tsv", sep="\t", dtype=str, keep_default_na=False)
        self.assertEqual(len(test), TEST_SIZE)

        def context(k, d):
            return pd.read_csv(
                out / f"context-{k}-draw{d}.tsv",
                sep="\t",
                dtype=str,
                keep_default_na=False,
            )

        for d in DRAWS:
            self.assertEqual([len(context(k, d)) for k in (5, 10, 20)], [5, 10, 20])
            self.assertEqual(list(context(5, d).src), list(context(20, d).src[:5]))
            self.assertFalse(set(context(20, d).src) & set(test.src))
        self.assertNotEqual(list(context(5, 1).src), list(context(5, 2).src))

        # The nearest examples: k per test sentence, from the pool, never a test sentence
        nearest = pd.read_csv(
            out / "nearest-10.tsv", sep="\t", dtype=str, keep_default_na=False
        )
        self.assertEqual(len(nearest), 10 * TEST_SIZE)
        self.assertEqual(set(nearest.test_src), set(test.src))
        self.assertTrue(set(nearest.src) <= set(namespace["pool"].src))

        # The prompt shown at k = 5 has five examples and the sentence to translate
        self.assertEqual(printed.count("English: "), 6)

        # Every run scored with both metrics and signatures; the rise the stub was built to give
        scores = pd.read_csv(
            out / "scores.tsv",
            sep="\t",
            dtype={"context": str, "draw": str},
            keep_default_na=False,
        )
        self.assertEqual(
            sorted(set(scores.context)), ["", "0", "10", "20", "5"]
        )  # whole numbers, not 10.0
        # The none row, three draws and nearest at each size, NLLB, and Step 6's two.
        self.assertEqual(len(scores), 1 + 3 * (3 + 1) + 1 + 2)

        # Step 6: each test sentence's sentence before is the line above it in the sample; each
        # has its own placebo, a pool pair that is not one of draw 1's five, and only those
        # with a sentence before get one; the table marks the pronoun sentences.
        five, placebos = namespace["five"], namespace["placebos"]
        self.assertEqual(
            len(placebos), len(test), "every test sentence needs a placebo slot"
        )
        given = [p for p, b in zip(placebos, namespace["befores"]) if b is not None]
        self.assertEqual(len(given), sum(p is not None for p in placebos))
        self.assertEqual(len(set(given)), len(given), "a placebo pair was given twice")
        self.assertFalse(
            set(given) & set(five), "a placebo is one of the five examples"
        )
        pool_pairs = set(zip(namespace["pool"].src, namespace["pool"].tgt))
        self.assertLessEqual(set(given), pool_pairs)
        table = pd.read_csv(
            out / "before-5.tsv", sep="\t", dtype=str, keep_default_na=False
        )
        lines, test = namespace["lines"], namespace["test"]
        for row, line in zip(table.itertuples(), test.line):
            if line > 0:
                self.assertEqual(row.before_src, lines.src.iloc[line - 1])
                self.assertEqual(row.before_tgt, lines.tgt.iloc[line - 1])
        self.assertEqual(
            list(table.columns),
            [
                "test_src",
                "before_src",
                "before_tgt",
                "reference",
                "hyp_5",
                "hyp_placebo",
                "hyp_before",
                "pronouns",
            ],
        )
        self.assertEqual(
            set(scores.condition) & {"before", "placebo"}, {"before", "placebo"}
        )
        self.assertTrue(scores.bleu_signature.str.contains("nrefs:1").all())
        self.assertTrue(scores.chrf_signature.str.contains("nc:6").all())
        by_run = scores.set_index(["condition", "context", "draw"])
        self.assertLess(by_run.bleu["none", "0", ""], by_run.bleu["random", "10", "1"])
        self.assertEqual(by_run.bleu["nearest", "20", ""], 100.0)
        nllb = scores[scores.condition == "NLLB"].iloc[0]
        self.assertTrue(0 < nllb.chrf < 100)

        # The summary folds the three draws into one line with their spread
        self.assertIn("random   k = 10  BLEU 100.00", printed)
        self.assertIn("(draws 100.00 to 100.00", printed)

        rerun = json.loads((out / "a12-rerun.json").read_text(encoding="utf-8"))
        self.assertEqual(rerun["draw_seeds"], [1, 2, 3])
        self.assertEqual(rerun["nllb"], "stub-nllb")
        self.assertIn("greedy", rerun["decoding"])

    def test_nllb_covers_only_telugu(self):
        """The stub's languages match the notebook's: of the A12 six, NLLB-200 has Telugu."""
        nllb_cell = next(src for _, src in code_cells(NOTEBOOK) if NLLB_CELL in src)
        self.assertIn(
            'NLLB_CODES = {"Spanish": "spa_Latn", "Telugu": "tel_Telu"}', nllb_cell
        )

    def test_nearest_examples_put_the_most_similar_last(self):
        namespace, _, _ = self.run_notebook()
        pool = namespace["pool"]
        target = pool.src.iloc[7]
        chosen = namespace["nearest_examples"](target, 3)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(chosen[-1], (target, pool.tgt.iloc[7]))

    def test_without_a_prediction_it_stops_before_translating(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook(predicted=False)
        self.assertIn("Write your prediction", str(stopped.exception))


BOM = chr(0xFEFF)  # the byte-order mark a Windows editor may put at a file's start


def write_a12_folder(
    data_dir: Path, rows, newline: str = "\n", bom: bool = False
) -> None:
    """The A12 data folder's layout: line-aligned src.txt (English) and tgt.txt."""
    src, tgt = zip(*rows)
    for name, lines in (("src.txt", src), ("tgt.txt", tgt)):
        text = (BOM if bom else "") + newline.join(lines) + newline
        (data_dir / name).write_bytes(text.encode("utf-8"))


class TestStudentData(NotebookCase):
    def test_reads_the_a12_folder(self):
        namespace, printed, _ = self.run_notebook(
            lambda d: write_a12_folder(d, pairs(80))
        )
        self.assertNotIn("Spanish is not the assignment", printed)
        self.assertIn("NLLB-200 does not cover Pohnpeian", printed)
        self.assertNotIn("NLLB", set(namespace["runs"].condition))
        self.assertEqual(len(namespace["pairs"]), 80)
        self.assertTrue(namespace["test"].src.str.startswith("sentence number").all())
        self.assertTrue(namespace["test"].tgt.str.startswith("lino").all())

    def test_reads_the_real_folders_first_lines_unchanged(self):
        """Lines as the real A12 files have them (Cowork read all twelve, Oct 19): Efik's
        opening three, the third English left untranslated; Telugu script; and Yapese's curly
        apostrophe, which is orthography and must not be normalized away."""
        real = [
            ("Jesus is the Christ.", "Jesus edi Christ."),
            ("This is His Church.", "Emi edi Ufok Abasi Esie"),
            (
                "See title page of the Book of Mormon.",
                "Se title page of the Book of Mormon..",
            ),
            (
                (
                    "Helping others understand their true identity and purpose is one of "
                    "the greatest gifts we can give."
                ),
                (
                    "వారి నిజమైన గుర్తింపును మరియు ఉద్దేశమును గ్రహించడానికి ఇతరులకు సహాయపడుట "
                    "మనము ఇవ్వగల మిక్కిలి గొప్ప బహుమానాలలో ఒకటి."
                ),
            ),
            ("A line with a Yapese word.", "Gidii’en"),
        ]
        namespace, _, _ = self.run_notebook(
            lambda d: write_a12_folder(d, real + pairs(80))
        )
        loaded = list(zip(namespace["pairs"].src, namespace["pairs"].tgt))
        self.assertEqual(len(loaded), 85)
        for pair in real:
            self.assertIn(pair, loaded)

    def test_reads_windows_line_endings_and_a_byte_order_mark(self):
        namespace, _, _ = self.run_notebook(
            lambda d: write_a12_folder(d, pairs(80), newline="\r\n", bom=True)
        )
        loaded = namespace["pairs"]
        self.assertEqual(len(loaded), 80)
        for column in (loaded.src, loaded.tgt):
            self.assertFalse(column.str.contains("\r", regex=False).any())
            self.assertFalse(column.str.contains(BOM, regex=False).any())

    def test_a_unicode_line_separator_stays_inside_its_line(self):
        """str.splitlines would split there, and the two files' line counts would differ."""
        rows = pairs(80)
        rows[3] = (rows[3][0], rows[3][1] + chr(0x2028) + "continued")
        namespace, _, _ = self.run_notebook(lambda d: write_a12_folder(d, rows))
        self.assertEqual(len(namespace["pairs"]), 80)

    def test_a_missing_final_newline_is_not_a_length_mismatch(self):
        def write(data_dir):
            write_a12_folder(data_dir, pairs(80))
            tgt = data_dir / "tgt.txt"
            tgt.write_bytes(tgt.read_bytes().rstrip(b"\n"))

        namespace, _, _ = self.run_notebook(write)
        self.assertEqual(len(namespace["pairs"]), 80)

    def test_skips_blank_lines_and_duplicate_sources(self):
        rows = pairs(80) + [("", "lino blank"), ("only english here", "")]
        rows += pairs(5)
        namespace, _, _ = self.run_notebook(lambda d: write_a12_folder(d, rows))
        self.assertEqual(len(namespace["pairs"]), 80)

    def test_stops_naming_the_files_it_wants(self):
        def write(data_dir):
            (data_dir / "efik.tsv").write_text("a\tb\n", encoding="utf-8")

        with self.assertRaises(SystemExit) as stopped:
            self.run_notebook(write)
        message = str(stopped.exception)
        self.assertIn("expected src.txt and tgt.txt", message)
        self.assertIn("efik.tsv", message)

    def test_stops_when_the_files_differ_in_length(self):
        def write(data_dir):
            write_a12_folder(data_dir, pairs(80))
            with open(data_dir / "tgt.txt", "a", encoding="utf-8") as f:
                f.write("one line too many\n")

        with self.assertRaises(SystemExit) as stopped:
            self.run_notebook(write)
        self.assertIn("src.txt has 80 lines and tgt.txt 81", str(stopped.exception))

    def test_data_dir_without_language_stops(self):
        """Otherwise the prompt asks for Spanish while the data is Pohnpeian."""

        def write(data_dir):
            write_a12_folder(data_dir, pairs(80))

        with self.assertRaises(SystemExit) as stopped:
            self.run_notebook(write, language=None)
        self.assertIn("Set LANGUAGE and DATA_DIR together", str(stopped.exception))


class TestExercises(unittest.TestCase):
    """The checks around the two exercises, run on the cells up to the split alone."""

    def cells_to_split(self, answered: bool = True) -> list[tuple[int, str]]:
        """The settings and loader cells, then the split cell last."""
        cells = notebook_cells(str(self.data_dir), answered=answered)
        split_at = next(i for i, src in cells if "def split_pool_and_test(" in src)
        before = [c for c in cells if self.setup_cell[0] < c[0] <= split_at]
        return [self.setup_cell, *before]

    def rejected(self, answer: str) -> str:
        """Run the split cell with ``answer`` as its body; return the failed check's message."""
        *before, (index, source) = self.cells_to_split()
        wrong = source.replace(
            ANSWERS[
                'raise NotImplementedError("write split_pool_and_test, then run this cell again")'
            ],
            answer,
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells([*before, (index, wrong)])
        return str(stopped.exception)

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.drive = Path(scratch.name)
        self.data_dir = self.drive / "a12-data"
        self.data_dir.mkdir()
        write_a12_folder(self.data_dir, pairs(80))
        self.setup_cell = next(
            c for c in code_cells(NOTEBOOK) if "setup(gpu=True" in c[1]
        )

    def test_as_shipped_it_stops_at_the_first_exercise(self):
        with (
            colab_faked(self.drive),
            self.assertRaises(NotImplementedError) as stopped,
        ):
            run_cells(self.cells_to_split(answered=False))
        self.assertIn("write split_pool_and_test", str(stopped.exception))

    def test_a_split_that_shares_a_sentence_is_rejected(self):
        """One pair off: the right sizes, one sentence in both."""
        message = self.rejected(
            "chosen = pairs.sample(pool_size + test_size, random_state=seed)\n"
            "    start = pool_size - 1\n"
            "    return chosen.iloc[:pool_size], chosen.iloc[start : start + test_size]"
        )
        self.assertIn("in both the pool and the test set", message)

    def test_a_split_that_ignores_the_seed_is_rejected(self):
        """The first rows, not a random draw: sizes and overlap are right, the seed does nothing."""
        message = self.rejected(
            "return pairs.iloc[:pool_size], pairs.iloc[pool_size : pool_size + test_size]"
        )
        self.assertIn("a different seed must give a different split", message)


if __name__ == "__main__":
    unittest.main()
