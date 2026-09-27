"""
Tests for model training convergence and optimization behavior.

Verifies that models actually learn and that training hyperparameters
work as expected.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import optim

from torchlingo.config import Config
from torchlingo.data_processing.batching import collate_fn
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.vocab import SimpleVocab
from torchlingo.models import SimpleSeq2SeqLSTM, SimpleTransformer
from torchlingo.training import train_model


class TestTransformerConvergence(unittest.TestCase):
    """Test that Transformer model actually learns."""

    def test_transformer_overfits_small_dataset(self):
        """Verify Transformer can overfit a tiny dataset (proof of learning)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create tiny repeating dataset
            data = pd.DataFrame(
                {
                    "src": ["hello world", "good morning"] * 10,
                    "tgt": ["hola mundo", "buenos dias"] * 10,
                }
            )
            data_file = tmp / "tiny_train.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            # Build vocabularies
            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            cfg = Config(batch_size=4, learning_rate=0.01)
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
            )

            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=64,
                n_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                d_ff=128,
                dropout=0.0,  # No dropout for overfitting test
                config=cfg,
            )

            # Train for multiple epochs
            result = train_model(
                model,
                train_loader=loader,
                num_epochs=20,
                gradient_clip=1.0,
                device=torch.device("cpu"),
                config=cfg,
            )

            # Loss should decrease significantly
            initial_loss = result.train_losses[0]
            final_loss = result.train_losses[-1]

            self.assertLess(final_loss, initial_loss * 0.5)
            self.assertLess(final_loss, 2.0)

    def test_transformer_loss_decreases_monotonically_on_simple_task(self):
        """Test loss decreases on a simple predictable task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create a very simple pattern: copy task
            data = pd.DataFrame(
                {
                    "src": ["a b c", "d e f", "g h i", "j k l"] * 5,
                    "tgt": ["a b c", "d e f", "g h i", "j k l"] * 5,
                }
            )
            data_file = tmp / "copy_task.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            cfg = Config(batch_size=4, learning_rate=0.01)
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
            )

            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dropout=0.0,
                config=cfg,
            )

            result = train_model(model, train_loader=loader, num_epochs=15, config=cfg)

            # Check that loss generally trends downward
            losses = result.train_losses
            # Use moving average to check trend
            window = 3
            early_avg = sum(losses[:window]) / window
            late_avg = sum(losses[-window:]) / window

            self.assertLess(late_avg, early_avg)


class TestLSTMConvergence(unittest.TestCase):
    """Test that LSTM model actually learns."""

    def test_lstm_overfits_small_dataset(self):
        """Verify LSTM can overfit a tiny dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            data = pd.DataFrame(
                {
                    "src": ["cat dog", "bird fish"] * 8,
                    "tgt": ["gato perro", "pajaro pez"] * 8,
                }
            )
            data_file = tmp / "tiny_lstm.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            cfg = Config(batch_size=4, learning_rate=0.01, scheduler_type="none")
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
            )

            model = SimpleSeq2SeqLSTM(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                emb_dim=32,
                hidden_dim=64,
                num_layers=1,
                dropout=0.0,
                config=cfg,
            )

            result = train_model(model, train_loader=loader, num_epochs=15, config=cfg)

            # Loss should decrease
            initial_loss = result.train_losses[0]
            final_loss = result.train_losses[-1]

            self.assertLess(final_loss, initial_loss * 0.5)


class TestOptimizationComponents(unittest.TestCase):
    """Test that optimization components work correctly."""

    def test_different_optimizers_produce_different_results(self):
        """Test Adam vs SGD produce different training dynamics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            data = pd.DataFrame(
                {
                    "src": ["test data"] * 10,
                    "tgt": ["prueba datos"] * 10,
                }
            )
            data_file = tmp / "optim_test.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            cfg = Config(batch_size=2)
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=2, collate_fn=collate_fn
            )

            # Seed immediately before each model, so both start from IDENTICAL weights.
            #
            # Two reasons, and the second is why this test was failing intermittently for a
            # month (#79). Identical initialization is what makes this a controlled
            # comparison of optimizers rather than of starting points. And without a seed the
            # initial weights came from whatever global RNG state the previously-run tests
            # happened to leave, so the result depended on test ORDER: it passed under pytest
            # and failed under `python -m unittest discover`, which is the command CLAUDE.md
            # documents first.
            torch.manual_seed(0)
            model_adam = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )
            optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
            result_adam = train_model(
                model_adam,
                train_loader=loader,
                optimizer=optimizer_adam,
                num_epochs=5,
                config=cfg,
            )

            # The same seed, so this model is weight-for-weight the one above.
            torch.manual_seed(0)
            model_sgd = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )
            optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
            result_sgd = train_model(
                model_sgd,
                train_loader=loader,
                optimizer=optimizer_sgd,
                num_epochs=5,
                config=cfg,
            )

            self.assertEqual(len(result_adam.train_losses), 5)
            self.assertEqual(len(result_sgd.train_losses), 5)

            # Assert what the test is named for: the two optimizers take different paths.
            #
            # The previous assertion was "at least one shows improvement", which is a weaker
            # claim than the name makes and a noisier one to check -- five epochs on ten
            # identical rows is not enough for either optimizer to be reliably downhill, which
            # is what made the failure look random rather than ordered. Because the models now
            # start from identical weights, ANY divergence in the loss curves is attributable
            # to the optimizer and nothing else, which is the real content of the test.
            self.assertNotEqual(
                result_adam.train_losses,
                result_sgd.train_losses,
                "Adam and SGD produced identical loss curves from identical weights, "
                "which means the optimizer argument is not reaching the training loop",
            )

            # And the first step must differ, not merely some later one: an optimizer that was
            # ignored until epoch three would still pass the check above.
            self.assertNotAlmostEqual(
                result_adam.train_losses[1],
                result_sgd.train_losses[1],
                places=6,
            )

    def test_learning_rate_affects_convergence_speed(self):
        """A higher learning rate reaches a lower loss at the same epoch.

        Deliberately not "faster initial descent", which is what this used to claim and
        measure. Descent *rate* between epochs is the wrong quantity: see the comment on the
        assertion.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            data = pd.DataFrame(
                {
                    "src": ["learn fast"] * 12,
                    "tgt": ["aprender rapido"] * 12,
                }
            )
            data_file = tmp / "lr_test.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            cfg = Config(batch_size=4, scheduler_type="none")
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=4, collate_fn=collate_fn
            )

            # Seeded, and with the same seed as the low-rate model below, for the reason
            # given on the optimizer test above: without it the two models start from
            # different weights AND from whatever RNG state earlier tests left behind, so a
            # comparison of learning rates is confounded by both. This test was the second
            # instance of #79 and was hidden behind the first.
            torch.manual_seed(0)
            model_high = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )
            # 0.001, the library default, not 0.01.
            #
            # Measured across three seeds: at 0.01 the high rate reaches a WORSE final
            # loss than the low rate at two of them, because 0.01 overshoots on a
            # 32-dimensional model over twelve identical rows. So the original pair put
            # the "faster" rate outside its stable range and then asserted it was faster.
            # 0.001 against 1e-05 holds at every seed with a wide margin.
            opt_high = optim.Adam(model_high.parameters(), lr=0.001)
            result_high = train_model(
                model_high,
                train_loader=loader,
                optimizer=opt_high,
                num_epochs=3,
                config=cfg,
            )

            # Same seed: weight-for-weight identical to the model above.
            torch.manual_seed(0)
            model_low = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )
            opt_low = optim.Adam(model_low.parameters(), lr=0.00001)
            result_low = train_model(
                model_low,
                train_loader=loader,
                optimizer=opt_low,
                num_epochs=3,
                config=cfg,
            )

            # Assert the loss REACHED, not the change between epochs.
            #
            # The original test compared `losses[0] - losses[1]` across the two rates and
            # called the larger one faster. That measures the wrong quantity, and measuring it
            # is what made the test look flaky. `train_losses[0]` is the *average over epoch
            # one*, not the starting loss, so a high rate that learns quickly WITHIN epoch one
            # reports a lower first value and therefore a smaller subsequent drop. Measured
            # here on other data: a fast rate can be ahead at every epoch while its
            # epoch-to-epoch delta is *smaller*, because it starts each epoch from further
            # along. The old assertion was reading the right data through a formula that
            # inverted it.
            #
            # What "converges faster" actually means is a lower loss at the same epoch.
            self.assertLess(
                result_high.train_losses[0],
                result_low.train_losses[0],
                f"lr 0.001 ended epoch 1 at {result_high.train_losses[0]:.4f} and lr 1e-05 "
                f"at {result_low.train_losses[0]:.4f}; the faster rate should be ahead",
            )
            self.assertLess(
                result_high.train_losses[-1],
                result_low.train_losses[-1],
                "the faster rate should still be ahead at the last epoch",
            )

    def test_gradient_clipping_prevents_nan_loss(self):
        """Test gradient clipping prevents loss from becoming NaN."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            data = pd.DataFrame(
                {
                    "src": ["clip test"] * 8,
                    "tgt": ["prueba recorte"] * 8,
                }
            )
            data_file = tmp / "clip_test.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            cfg = Config(batch_size=2)
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=2, collate_fn=collate_fn
            )

            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )

            # Very high learning rate + gradient clipping
            optimizer = optim.SGD(model.parameters(), lr=1.0)
            result = train_model(
                model,
                train_loader=loader,
                optimizer=optimizer,
                gradient_clip=1.0,
                num_epochs=3,
                config=cfg,
            )

            # All losses should be finite
            for loss in result.train_losses:
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                self.assertFalse(torch.isnan(torch.tensor(loss)))


class TestValidationAndEarlyStopping(unittest.TestCase):
    """Test validation monitoring and early stopping."""

    def test_early_stopping_triggers_on_no_improvement(self):
        """Test that training stops when validation doesn't improve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create train and val data
            train_data = pd.DataFrame(
                {
                    "src": ["train sentence"] * 20,
                    "tgt": ["oracion entrenamiento"] * 20,
                }
            )
            val_data = pd.DataFrame(
                {
                    "src": ["val sentence"] * 5,
                    "tgt": ["oracion validacion"] * 5,
                }
            )

            train_file = tmp / "train.tsv"
            val_file = tmp / "val.tsv"
            train_data.to_csv(train_file, sep="\t", index=False)
            val_data.to_csv(val_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(train_data["src"].tolist())
            tgt_vocab.build_vocab(train_data["tgt"].tolist())

            # Set patience low to trigger early stopping
            cfg = Config(batch_size=4, patience=2)

            train_dataset = NMTDataset(
                train_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            val_dataset = NMTDataset(
                val_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=4, collate_fn=collate_fn
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=4, collate_fn=collate_fn
            )

            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )

            result = train_model(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=100,  # Request many epochs
                config=cfg,
            )

            # Should stop before 100 epochs due to patience
            self.assertLess(len(result.train_losses), 100)
            self.assertGreater(len(result.val_losses), 0)

    def test_best_checkpoint_has_lowest_val_loss(self):
        """Test that best checkpoint corresponds to lowest validation loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            train_data = pd.DataFrame(
                {
                    "src": ["checkpoint test"] * 16,
                    "tgt": ["prueba punto control"] * 16,
                }
            )
            val_data = pd.DataFrame(
                {
                    "src": ["validation test"] * 4,
                    "tgt": ["prueba validacion"] * 4,
                }
            )

            train_file = tmp / "train.tsv"
            val_file = tmp / "val.tsv"
            train_data.to_csv(train_file, sep="\t", index=False)
            val_data.to_csv(val_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(train_data["src"].tolist())
            tgt_vocab.build_vocab(train_data["tgt"].tolist())

            cfg = Config(batch_size=4)

            train_dataset = NMTDataset(
                train_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            val_dataset = NMTDataset(
                val_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=4, collate_fn=collate_fn
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=4, collate_fn=collate_fn
            )

            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )

            save_dir = tmp / "checkpoints"
            result = train_model(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=5,
                save_dir=save_dir,
                config=cfg,
            )

            # Checkpoint should exist
            self.assertIsNotNone(result.best_checkpoint)
            self.assertTrue(result.best_checkpoint.exists())

            # Load checkpoint and verify it's valid
            checkpoint = torch.load(result.best_checkpoint, weights_only=True)
            self.assertIsInstance(checkpoint, dict)


class TestModelCapacity(unittest.TestCase):
    """Test that model size affects learning capacity."""

    def test_larger_model_achieves_lower_loss(self):
        """Test that a larger model can achieve lower loss on the same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create moderately complex data
            data = pd.DataFrame(
                {
                    "src": [
                        f"sentence number {i} with unique content" for i in range(30)
                    ],
                    "tgt": [
                        f"oracion numero {i} con contenido unico" for i in range(30)
                    ],
                }
            )
            data_file = tmp / "capacity_test.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            cfg = Config(batch_size=6, learning_rate=0.001)
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=6, shuffle=True, collate_fn=collate_fn
            )

            # Small model
            model_small = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                d_ff=64,
                config=cfg,
            )

            result_small = train_model(
                model_small, train_loader=loader, num_epochs=10, config=cfg
            )

            # Larger model
            model_large = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=128,
                n_heads=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                d_ff=512,
                config=cfg,
            )

            result_large = train_model(
                model_large, train_loader=loader, num_epochs=10, config=cfg
            )

            # Larger model should achieve equal or lower final loss
            small_final = result_small.train_losses[-1]
            large_final = result_large.train_losses[-1]

            # Allow some variance but larger should be better or equal
            self.assertLessEqual(large_final, small_final * 1.2)


if __name__ == "__main__":
    unittest.main()
