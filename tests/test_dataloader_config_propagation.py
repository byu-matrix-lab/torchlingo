"""The Config handed to create_dataloaders must reach the datasets it builds.

``create_dataloaders`` accepts a ``Config`` and used it for the loader and
bucketing settings, but constructed its ``NMTDataset`` objects without passing it
on. The datasets therefore fell back to ``get_default_config()`` and every
dataset-level field was discarded in silence.

No exception marked the failure, which is what made it expensive. A length ladder
built to vary ``max_seq_length`` produced two rungs, nominally 5 tokens and 10,
that agreed to three decimal places on epoch time, held memory and validation
loss -- because both had in fact trained at the default 512 cap. The knob had
never been connected.

``max_seq_length`` is the field these tests lead with, since it is measurable, but
it is not the only one. ``NMTDataset.__init__`` resolves ``src_col``, ``tgt_col``,
``src_tok_col``, ``tgt_tok_col``, ``max_length`` and ``eos_idx`` from the config
too. A custom column name is the one likeliest to bite a student: it surfaces as a
missing-column ``ValueError`` pointing at the data file, which sends them to look
at the corpus rather than at the config that was thrown away.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from torchlingo.config import Config, get_default_config
from torchlingo.data_processing.batching import create_dataloaders


def _write_corpus(directory: Path, src_col: str = "src", tgt_col: str = "tgt") -> Path:
    """Write a tiny two-column TSV and return its path.

    Sentences are long enough that a small ``max_seq_length`` truncates them,
    which is what makes the truncation assertions meaningful rather than
    vacuously true.

    Args:
        directory (Path): Directory to write into.
        src_col (str): Name for the source column.
        tgt_col (str): Name for the target column.

    Returns:
        Path: The TSV path.
    """
    path = directory / "corpus.tsv"
    rows = {
        src_col: [
            f"this is source sentence number {i} and it runs on" for i in range(8)
        ],
        tgt_col: [
            f"this is target sentence number {i} and it runs on" for i in range(8)
        ],
    }
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


class TestConfigReachesDatasets(unittest.TestCase):
    """create_dataloaders must forward its Config to every dataset it builds."""

    def test_max_seq_length_reaches_train_dataset(self):
        """The train dataset's max_length tracks the config, not the default."""
        with tempfile.TemporaryDirectory() as tmp:
            corpus = _write_corpus(Path(tmp))
            config = Config(batch_size=2, max_seq_length=5)
            train_loader, _, _, _ = create_dataloaders(
                train_file=corpus, batch_size=2, config=config
            )
            self.assertEqual(train_loader.dataset.max_length, 5)

    def test_max_seq_length_reaches_val_dataset(self):
        """The validation dataset gets the same treatment as the train one.

        Built at a separate call site, so it can regress independently.
        """
        with tempfile.TemporaryDirectory() as tmp:
            corpus = _write_corpus(Path(tmp))
            config = Config(batch_size=2, max_seq_length=5)
            _, val_loader, _, _ = create_dataloaders(
                train_file=corpus, val_file=corpus, batch_size=2, config=config
            )
            self.assertIsNotNone(val_loader)
            self.assertEqual(val_loader.dataset.max_length, 5)

    def test_truncation_actually_happens(self):
        """A short cap must shorten real batches.

        The attribute assertions above would pass if ``max_length`` were stored
        and never consulted. This checks the tensors.
        """
        with tempfile.TemporaryDirectory() as tmp:
            corpus = _write_corpus(Path(tmp))
            config = Config(batch_size=2, max_seq_length=5)
            train_loader, _, _, _ = create_dataloaders(
                train_file=corpus, batch_size=2, config=config
            )
            for src, tgt in train_loader:
                self.assertLessEqual(src.shape[1], 5)
                self.assertLessEqual(tgt.shape[1], 5)

    def test_custom_column_names_reach_the_dataset(self):
        """A renamed column pair is honoured rather than reported as missing.

        Before the fix this raised ValueError naming the default columns, which
        blames the corpus for a discarded config.
        """
        with tempfile.TemporaryDirectory() as tmp:
            corpus = _write_corpus(Path(tmp), src_col="german", tgt_col="english")
            config = Config(batch_size=2, src_col="german", tgt_col="english")
            train_loader, _, _, _ = create_dataloaders(
                train_file=corpus, batch_size=2, config=config
            )
            self.assertEqual(train_loader.dataset.src_col, "german")
            self.assertEqual(train_loader.dataset.tgt_col, "english")

    def test_omitting_config_keeps_the_default(self):
        """Passing no config must not change behaviour.

        The fix forwards ``config`` even when it is None, so the dataset resolves
        the default itself. This pins that the forwarding did not accidentally
        become a new source of truth.
        """
        with tempfile.TemporaryDirectory() as tmp:
            corpus = _write_corpus(Path(tmp))
            train_loader, _, _, _ = create_dataloaders(train_file=corpus, batch_size=2)
            self.assertEqual(
                train_loader.dataset.max_length, get_default_config().max_seq_length
            )


if __name__ == "__main__":
    unittest.main()
