"""
Comprehensive integration tests for TorchLingo.

Tests the full pipeline: data loading → preprocessing → tokenization →
model training → inference, with special focus on edge cases and robustness.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from torchlingo.config import Config
from torchlingo.data_processing.batching import collate_fn
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.vocab import SimpleVocab, SentencePieceVocab
from torchlingo.inference import greedy_decode, translate_batch
from torchlingo.models import SimpleTransformer, SimpleSeq2SeqLSTM
from torchlingo.preprocessing.base import load_data, save_data
from torchlingo.preprocessing.sentencepiece import train_sentencepiece
from torchlingo.training import train_model


class TestEndToEndTransformerPipeline(unittest.TestCase):
    """Test complete pipeline from raw data to trained model and inference."""

    def test_full_pipeline_with_simple_vocab(self):
        """Test complete pipeline: data → vocab → dataset → train → inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. Create sample data
            data = pd.DataFrame(
                {
                    "src": [
                        "hello world",
                        "good morning",
                        "how are you",
                        "nice day",
                        "see you later",
                        "thank you",
                        "welcome back",
                        "good night",
                    ],
                    "tgt": [
                        "hola mundo",
                        "buenos dias",
                        "como estas",
                        "buen dia",
                        "hasta luego",
                        "gracias",
                        "bienvenido",
                        "buenas noches",
                    ],
                }
            )
            data_file = tmp / "train.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            # 2. Build vocabularies
            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            # 3. Create dataset
            cfg = Config(
                pad_idx=0,
                sos_idx=1,
                eos_idx=2,
                unk_idx=3,
                batch_size=2,
                max_seq_length=20,
            )
            dataset = NMTDataset(
                data_file,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                config=cfg,
            )

            # Verify dataset
            self.assertEqual(len(dataset), 8)
            src_sample, tgt_sample = dataset[0]
            self.assertIsInstance(src_sample, torch.Tensor)
            self.assertIsInstance(tgt_sample, torch.Tensor)

            # 4. Create DataLoader
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
            )

            # 5. Create model
            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=64,
                n_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                d_ff=128,
                dropout=0.1,
                config=cfg,
            )

            # 6. Train model
            result = train_model(
                model,
                train_loader=loader,
                val_loader=None,
                num_epochs=3,
                gradient_clip=1.0,
                device=torch.device("cpu"),
                config=cfg,
            )

            # Verify training completed
            self.assertEqual(len(result.train_losses), 3)
            self.assertTrue(all(loss < 10.0 for loss in result.train_losses))
            # Training loss should decrease
            self.assertLess(result.train_losses[-1], result.train_losses[0])

            # 7. Test inference
            test_sentences = ["hello world", "good morning"]
            translations = translate_batch(
                model,
                test_sentences,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                decode_strategy="greedy",
                max_len=20,
                device=torch.device("cpu"),
                config=cfg,
            )

            self.assertEqual(len(translations), 2)
            for translation in translations:
                self.assertIsInstance(translation, str)
                self.assertGreater(len(translation), 0)


    def test_full_pipeline_with_sentencepiece(self):
        """Test complete pipeline with SentencePiece tokenization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. Create sample data with more varied text
            data = pd.DataFrame(
                {
                    "src": [
                        "the quick brown fox jumps over lazy dog",
                        "neural machine translation is fascinating",
                        "transformers use attention mechanisms",
                        "subword tokenization helps with rare words",
                        "pytorch makes deep learning easier",
                        "gradient descent optimizes model parameters",
                        "encoder decoder architectures are powerful",
                        "beam search improves translation quality",
                        "multilingual models handle multiple languages",
                        "position embeddings encode sequence order",
                    ],
                    "tgt": [
                        "el rapido zorro marron salta sobre perro perezoso",
                        "traduccion automatica neural es fascinante",
                        "transformadores usan mecanismos de atencion",
                        "tokenizacion de subpalabras ayuda con palabras raras",
                        "pytorch facilita aprendizaje profundo",
                        "descenso gradiente optimiza parametros modelo",
                        "arquitecturas codificador decodificador son poderosas",
                        "busqueda haz mejora calidad traduccion",
                        "modelos multilingues manejan multiples idiomas",
                        "embeddings posicion codifican orden secuencia",
                    ],
                }
            )
            data_file = tmp / "train.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            # 2. Train SentencePiece models
            cfg = Config(
                data_dir=tmp,
                sentencepiece_model_prefix=str(tmp / "sp_model"),
                vocab_size=100,
                pad_idx=0,
                unk_idx=1,
                sos_idx=2,
                eos_idx=3,
                batch_size=2,
            )

            train_sentencepiece(
                [data_file],
                cfg.sentencepiece_model_prefix,
                columns=[cfg.src_col, cfg.tgt_col],
                config=cfg,
            )

            # Verify model was created
            model_path = Path(cfg.sentencepiece_model)
            self.assertTrue(model_path.exists())

            # 3. Create SentencePiece vocabularies
            src_vocab = SentencePieceVocab(cfg.sentencepiece_model, config=cfg)
            tgt_vocab = SentencePieceVocab(cfg.sentencepiece_model, config=cfg)

            # Test encoding/decoding
            test_text = "neural machine translation"
            encoded = src_vocab.encode(test_text)
            decoded = src_vocab.decode(encoded)
            self.assertIsInstance(decoded, str)
            # Verify decoding preserves essential content (words may have subwords)
            self.assertIn("neural", decoded.lower())

            # 4. Create dataset
            dataset = NMTDataset(
                data_file,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                config=cfg,
            )

            self.assertEqual(len(dataset), 10)

            # 5. Create DataLoader
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
            )

            # 6. Create model
            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=64,
                n_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                d_ff=128,
                config=cfg,
            )

            # 7. Train model
            result = train_model(
                model,
                train_loader=loader,
                val_loader=None,
                num_epochs=2,
                gradient_clip=1.0,
                device=torch.device("cpu"),
                config=cfg,
            )

            self.assertEqual(len(result.train_losses), 2)

            # 8. Test inference
            test_sentences = ["transformers use attention", "pytorch makes learning easier"]
            translations = translate_batch(
                model,
                test_sentences,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                decode_strategy="greedy",
                max_len=30,
                config=cfg,
            )

            self.assertEqual(len(translations), 2)
            for translation in translations:
                self.assertIsInstance(translation, str)


class TestLSTMEndToEndPipeline(unittest.TestCase):
    """Test LSTM model with complete pipeline."""

    def test_lstm_full_pipeline(self):
        """Test LSTM from data to inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create data
            data = pd.DataFrame(
                {
                    "src": ["hello", "world", "test", "data", "sample"] * 2,
                    "tgt": ["hola", "mundo", "prueba", "datos", "muestra"] * 2,
                }
            )
            data_file = tmp / "train.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            # Build vocabularies
            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            # Create dataset and loader
            cfg = Config(batch_size=2, max_seq_length=10)
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
            )

            # Create LSTM model
            model = SimpleSeq2SeqLSTM(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                emb_dim=32,
                hidden_dim=64,
                num_layers=1,
                config=cfg,
            )

            # Train
            result = train_model(
                model,
                train_loader=loader,
                num_epochs=2,
                device=torch.device("cpu"),
                config=cfg,
            )

            self.assertEqual(len(result.train_losses), 2)

            # Test inference
            translations = translate_batch(
                model,
                ["hello", "world"],
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                decode_strategy="greedy",
                max_len=10,
                config=cfg,
            )

            self.assertEqual(len(translations), 2)


class TestModelTrainingRobustness(unittest.TestCase):
    """Test model training under various conditions."""

    def test_training_with_validation_and_early_stopping(self):
        """Test training with validation loss monitoring and early stopping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create train and val data
            train_data = pd.DataFrame(
                {
                    "src": ["sentence one", "sentence two", "sentence three"] * 4,
                    "tgt": ["oracion uno", "oracion dos", "oracion tres"] * 4,
                }
            )
            val_data = pd.DataFrame(
                {
                    "src": ["test one", "test two"],
                    "tgt": ["prueba uno", "prueba dos"],
                }
            )

            train_file = tmp / "train.tsv"
            val_file = tmp / "val.tsv"
            train_data.to_csv(train_file, sep="\t", index=False)
            val_data.to_csv(val_file, sep="\t", index=False)

            # Build vocabularies
            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(train_data["src"].tolist())
            tgt_vocab.build_vocab(train_data["tgt"].tolist())

            cfg = Config(batch_size=3, patience=5, val_interval=4)

            # Create datasets
            train_dataset = NMTDataset(
                train_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            val_dataset = NMTDataset(
                val_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=3, shuffle=False, collate_fn=collate_fn
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
            )

            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=32,
                n_heads=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                d_ff=64,
                config=cfg,
            )

            result = train_model(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=3,
                save_dir=tmp,
                device=torch.device("cpu"),
                config=cfg,
            )

            # Verify validation losses were computed
            self.assertGreater(len(result.val_losses), 0)
            # Verify checkpoint was saved
            self.assertIsNotNone(result.best_checkpoint)
            self.assertTrue(result.best_checkpoint.exists())

    def test_training_handles_gradient_explosion(self):
        """Test that gradient clipping prevents gradient explosion."""
        # Create small dataset
        data = pd.DataFrame(
            {
                "src": ["a b c", "d e f", "g h i"],
                "tgt": ["x y z", "u v w", "r s t"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_file = tmp / "data.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            src_vocab = SimpleVocab(min_freq=1)
            tgt_vocab = SimpleVocab(min_freq=1)
            src_vocab.build_vocab(data["src"].tolist())
            tgt_vocab.build_vocab(data["tgt"].tolist())

            cfg = Config(batch_size=3)
            dataset = NMTDataset(
                data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=3, collate_fn=collate_fn
            )

            model = SimpleTransformer(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=16,
                n_heads=2,
                config=cfg,
            )

            # Train with gradient clipping
            result = train_model(
                model,
                train_loader=loader,
                num_epochs=2,
                gradient_clip=1.0,
                device=torch.device("cpu"),
                config=cfg,
            )

            # All losses should be finite
            for loss in result.train_losses:
                self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_training_with_amp(self):
        """Test automatic mixed precision training."""
        data = pd.DataFrame(
            {
                "src": ["test sentence"] * 4,
                "tgt": ["oracion prueba"] * 4,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_file = tmp / "data.tsv"
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

            # Train with AMP
            result = train_model(
                model,
                train_loader=loader,
                num_epochs=1,
                use_amp=True,
                device=torch.device("cpu"),
                config=cfg,
            )

            self.assertEqual(len(result.train_losses), 1)
            self.assertTrue(torch.isfinite(torch.tensor(result.train_losses[0])))


class TestSentencePieceEdgeCases(unittest.TestCase):
    """Test SentencePiece handling of edge cases."""

    def test_sentencepiece_handles_unicode(self):
        """Test SentencePiece with various Unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Data with various Unicode characters
            data = pd.DataFrame(
                {
                    "src": [
                        "hello world",
                        "café résumé",
                        "naïve coöperate",
                        "日本語 中文",
                        "emoji 😀 🎉",
                        "greek α β γ",
                    ],
                    "tgt": [
                        "hola mundo",
                        "coffee summary",
                        "innocent cooperate",
                        "japanese chinese",
                        "emoji happy party",
                        "letters alpha beta gamma",
                    ],
                }
            )
            data_file = tmp / "unicode_data.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            cfg = Config(
                data_dir=tmp,
                sentencepiece_model_prefix=str(tmp / "sp_unicode"),
                vocab_size=100,
                sp_character_coverage=1.0,  # Cover all characters
            )

            # Train SentencePiece
            train_sentencepiece([data_file], cfg.sentencepiece_model_prefix, config=cfg)

            # Test encoding/decoding
            vocab = SentencePieceVocab(cfg.sentencepiece_model, config=cfg)

            for text in data["src"].tolist():
                encoded = vocab.encode(text)
                decoded = vocab.decode(encoded)
                self.assertIsInstance(encoded, list)
                self.assertIsInstance(decoded, str)
                # Should produce some output
                self.assertGreater(len(encoded), 0)

    def test_sentencepiece_handles_empty_and_whitespace(self):
        """Test SentencePiece with empty strings and whitespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            data = pd.DataFrame(
                {
                    "src": [
                        "normal sentence",
                        "   spaces   around   ",
                        "multiple  spaces  inside",
                        "tab\tseparated\twords",
                        "newline\ncharacter",
                    ],
                    "tgt": ["a", "b", "c", "d", "e"],
                }
            )
            data_file = tmp / "whitespace_data.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            cfg = Config(
                data_dir=tmp,
                sentencepiece_model_prefix=str(tmp / "sp_whitespace"),
                vocab_size=50,
            )

            train_sentencepiece([data_file], cfg.sentencepiece_model_prefix, config=cfg)

            vocab = SentencePieceVocab(cfg.sentencepiece_model, config=cfg)

            # Test various edge cases
            test_cases = [
                "normal text",
                "  leading spaces",
                "trailing spaces  ",
                "  both  ",
                "multiple    spaces",
            ]

            for text in test_cases:
                encoded = vocab.encode(text)
                decoded = vocab.decode(encoded)
                # Should handle gracefully
                self.assertIsInstance(encoded, list)
                self.assertIsInstance(decoded, str)

    def test_sentencepiece_handles_very_long_sequences(self):
        """Test SentencePiece with very long text sequences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create data with varying lengths including very long
            long_text = " ".join(["word"] * 500)  # 500 words
            data = pd.DataFrame(
                {
                    "src": [
                        "short",
                        "medium length sentence here",
                        long_text,
                        "another short",
                    ],
                    "tgt": ["a", "b", "c", "d"],
                }
            )
            data_file = tmp / "long_data.tsv"
            data.to_csv(data_file, sep="\t", index=False)

            cfg = Config(
                data_dir=tmp,
                sentencepiece_model_prefix=str(tmp / "sp_long"),
                vocab_size=100,
            )

            train_sentencepiece([data_file], cfg.sentencepiece_model_prefix, config=cfg)

            vocab = SentencePieceVocab(cfg.sentencepiece_model, config=cfg)

            # Test encoding very long text
            encoded_long = vocab.encode(long_text)
            decoded_long = vocab.decode(encoded_long)

            self.assertIsInstance(encoded_long, list)
            self.assertGreater(len(encoded_long), 0)
            self.assertIsInstance(decoded_long, str)


class TestInferenceRobustness(unittest.TestCase):
    """Test inference under various conditions."""

    def test_greedy_decode_batch_processing(self):
        """Test greedy decode handles varying batch sizes."""
        cfg = Config(batch_size=2)

        # Create simple model
        model = SimpleTransformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=32,
            n_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            config=cfg,
        )
        model.eval()

        # Test different batch sizes
        for batch_size in [1, 2, 5, 10]:
            src = torch.randint(4, 50, (batch_size, 10))
            decoded = greedy_decode(model, src, max_len=15, config=cfg)

            self.assertEqual(len(decoded), batch_size)
            for seq in decoded:
                self.assertIsInstance(seq, list)
                self.assertGreater(len(seq), 0)

    def test_translate_batch_handles_varying_lengths(self):
        """Test translate_batch with sentences of different lengths."""
        # Create vocabularies
        src_vocab = SimpleVocab(min_freq=1)
        tgt_vocab = SimpleVocab(min_freq=1)

        texts = [
            "short",
            "a bit longer sentence",
            "this is a much longer sentence with many more words in it",
            "x",
        ]

        src_vocab.build_vocab(texts)
        tgt_vocab.build_vocab(texts)

        cfg = Config(batch_size=2)
        model = SimpleTransformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=32,
            n_heads=4,
            config=cfg,
        )
        model.eval()

        # Translate varying lengths
        translations = translate_batch(
            model,
            texts,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            decode_strategy="greedy",
            max_len=50,
            config=cfg,
        )

        self.assertEqual(len(translations), len(texts))
        for trans in translations:
            self.assertIsInstance(trans, str)


if __name__ == "__main__":
    unittest.main()
