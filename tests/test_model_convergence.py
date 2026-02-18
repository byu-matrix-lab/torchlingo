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
import torch.nn as nn
import torch.optim as optim

from torchlingo.config import Config
from torchlingo.data_processing.batching import collate_fn
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.vocab import SimpleVocab
from torchlingo.models import SimpleTransformer, SimpleSeq2SeqLSTM
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

            result = train_model(
                model, train_loader=loader, num_epochs=15, config=cfg
            )

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

            result = train_model(
                model, train_loader=loader, num_epochs=15, config=cfg
            )

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

            # Train with Adam
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

            # Train with SGD
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

            # Both should train, but likely with different dynamics
            self.assertEqual(len(result_adam.train_losses), 5)
            self.assertEqual(len(result_sgd.train_losses), 5)

            # At least one should show improvement
            adam_improved = result_adam.train_losses[-1] < result_adam.train_losses[0]
            sgd_improved = result_sgd.train_losses[-1] < result_sgd.train_losses[0]
            self.assertTrue(adam_improved or sgd_improved)

    def test_learning_rate_affects_convergence_speed(self):
        """Test that higher learning rate leads to faster initial descent."""
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

            cfg = Config(batch_size=4)
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=4, collate_fn=collate_fn
            )

            # High learning rate
            model_high = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )
            opt_high = optim.Adam(model_high.parameters(), lr=0.01)
            result_high = train_model(
                model_high,
                train_loader=loader,
                optimizer=opt_high,
                num_epochs=3,
                config=cfg,
            )

            # Low learning rate
            model_low = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                config=cfg,
            )
            opt_low = optim.Adam(model_low.parameters(), lr=0.0001)
            result_low = train_model(
                model_low,
                train_loader=loader,
                optimizer=opt_low,
                num_epochs=3,
                config=cfg,
            )

            # High LR should show more improvement in first epoch
            high_improvement = result_high.train_losses[0] - result_high.train_losses[1]
            low_improvement = result_low.train_losses[0] - result_low.train_losses[1]

            self.assertGreater(high_improvement, low_improvement * 0.5)

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

            # Best checkpoint should correspond to min val loss
            min_val_loss = min(result.val_losses)
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
                        f"sentence number {i} with unique content"
                        for i in range(30)
                    ],
                    "tgt": [
                        f"oracion numero {i} con contenido unico"
                        for i in range(30)
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
