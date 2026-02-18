"""Tests for enhanced multilingual utilities."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from torchlingo.config import Config
from torchlingo.data_processing.vocab import SimpleVocab, SentencePieceVocab
from torchlingo.preprocessing.multilingual_helpers import (
    DEFAULT_LANGUAGE_TAGS,
    add_language_tags_multi,
    create_multilingual_dataset,
    ensure_language_tags_in_vocab,
    get_language_tags_from_vocab,
    save_multilingual_splits,
)
from torchlingo.preprocessing.sentencepiece import train_sentencepiece


class TestAddLanguageTagsMulti(unittest.TestCase):
    """Test add_language_tags_multi function."""

    def test_adds_spanish_tag(self):
        """Test adding Spanish language tag."""
        df = pd.DataFrame({'src': ['hello world'], 'tgt': ['hola mundo']})
        tagged = add_language_tags_multi(df, 'es')

        self.assertEqual(tagged['src'].iloc[0], '<2es> hello world')
        self.assertEqual(tagged['tgt'].iloc[0], 'hola mundo')

    def test_adds_multiple_language_tags(self):
        """Test adding tags for multiple languages."""
        df = pd.DataFrame({'src': ['hello', 'world'], 'tgt': ['x', 'y']})

        for lang, expected_tag in [('es', '<2es>'), ('fr', '<2fr>'), ('de', '<2de>')]:
            tagged = add_language_tags_multi(df, lang)
            self.assertTrue(all(s.startswith(expected_tag) for s in tagged['src']))

    def test_raises_on_unknown_language(self):
        """Test error handling for unknown language codes."""
        df = pd.DataFrame({'src': ['hello'], 'tgt': ['x']})

        with self.assertRaises(ValueError) as ctx:
            add_language_tags_multi(df, 'xyz')

        self.assertIn('Unknown target language', str(ctx.exception))
        self.assertIn('xyz', str(ctx.exception))

    def test_accepts_custom_language_tags(self):
        """Test using custom language tag mapping."""
        df = pd.DataFrame({'src': ['hello'], 'tgt': ['hola']})
        custom_tags = {'es': '[SPANISH]', 'fr': '[FRENCH]'}

        tagged = add_language_tags_multi(df, 'es', language_tags=custom_tags)
        self.assertEqual(tagged['src'].iloc[0], '[SPANISH] hello')

    def test_respects_config_src_col(self):
        """Test that custom src_col is respected."""
        df = pd.DataFrame({'source': ['hello'], 'target': ['hola']})
        cfg = Config(src_col='source', tgt_col='target')

        tagged = add_language_tags_multi(df, 'es', src_col='source', config=cfg)
        self.assertEqual(tagged['source'].iloc[0], '<2es> hello')


class TestCreateMultilingualDataset(unittest.TestCase):
    """Test create_multilingual_dataset function."""

    def test_combines_multiple_language_pairs(self):
        """Test combining data from multiple language pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create sample data files
            es_data = pd.DataFrame(
                {'src': ['hello', 'world'], 'tgt': ['hola', 'mundo']}
            )
            fr_data = pd.DataFrame(
                {'src': ['hello', 'world'], 'tgt': ['bonjour', 'monde']}
            )

            es_file = tmp / 'en_es.tsv'
            fr_file = tmp / 'en_fr.tsv'
            es_data.to_csv(es_file, sep='\t', index=False)
            fr_data.to_csv(fr_file, sep='\t', index=False)

            # Create multilingual dataset
            data_sources = {'es': es_file, 'fr': fr_file}
            result = create_multilingual_dataset(data_sources)

            # Verify results
            self.assertEqual(len(result), 4)  # 2 from ES + 2 from FR
            self.assertIn('src', result.columns)
            self.assertIn('tgt', result.columns)

            # Check that language tags are present
            es_count = sum(s.startswith('<2es>') for s in result['src'])
            fr_count = sum(s.startswith('<2fr>') for s in result['src'])
            self.assertEqual(es_count, 2)
            self.assertEqual(fr_count, 2)

    def test_handles_missing_files_gracefully(self):
        """Test that missing files are skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create only one file
            es_data = pd.DataFrame({'src': ['hello'], 'tgt': ['hola']})
            es_file = tmp / 'en_es.tsv'
            es_data.to_csv(es_file, sep='\t', index=False)

            # Reference missing file
            data_sources = {
                'es': es_file,
                'fr': tmp / 'nonexistent.tsv',  # Missing
            }

            result = create_multilingual_dataset(data_sources)

            # Should still work with available files
            self.assertEqual(len(result), 1)
            self.assertTrue(result['src'].iloc[0].startswith('<2es>'))

    def test_raises_when_no_valid_sources(self):
        """Test error when no valid data sources are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            data_sources = {
                'es': tmp / 'missing1.tsv',
                'fr': tmp / 'missing2.tsv',
            }

            with self.assertRaises(ValueError) as ctx:
                create_multilingual_dataset(data_sources)

            self.assertIn('No valid data sources', str(ctx.exception))

    def test_shuffles_with_seed(self):
        """Test that shuffling is deterministic with seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create data
            data = pd.DataFrame(
                {'src': [f'sentence_{i}' for i in range(10)], 'tgt': ['x'] * 10}
            )
            data_file = tmp / 'data.tsv'
            data.to_csv(data_file, sep='\t', index=False)

            data_sources = {'es': data_file}

            # Create dataset twice with same seed
            result1 = create_multilingual_dataset(data_sources, seed=42)
            result2 = create_multilingual_dataset(data_sources, seed=42)

            # Should be identical
            self.assertTrue(result1['src'].equals(result2['src']))

            # Different seed should give different order
            result3 = create_multilingual_dataset(data_sources, seed=123)
            self.assertFalse(result1['src'].equals(result3['src']))


class TestSaveMultilingualSplits(unittest.TestCase):
    """Test save_multilingual_splits function."""

    def test_saves_train_val_test_splits(self):
        """Test saving complete train/val/test splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create sample split files
            for lang in ['es', 'fr']:
                for split in ['train', 'val', 'test']:
                    data = pd.DataFrame(
                        {
                            'src': [f'{split} sentence'] * 5,
                            'tgt': [f'{split} traduccion'] * 5,
                        }
                    )
                    filepath = tmp / f'en_{lang}_{split}.tsv'
                    data.to_csv(filepath, sep='\t', index=False)

            # Prepare data sources
            data_sources = {
                'es': {
                    'train': tmp / 'en_es_train.tsv',
                    'val': tmp / 'en_es_val.tsv',
                    'test': tmp / 'en_es_test.tsv',
                },
                'fr': {
                    'train': tmp / 'en_fr_train.tsv',
                    'val': tmp / 'en_fr_val.tsv',
                    'test': tmp / 'en_fr_test.tsv',
                },
            }

            output_dir = tmp / 'multilingual'
            save_multilingual_splits(data_sources, output_dir)

            # Verify files were created
            self.assertTrue((output_dir / 'multi_train.tsv').exists())
            self.assertTrue((output_dir / 'multi_val.tsv').exists())
            self.assertTrue((output_dir / 'multi_test.tsv').exists())

            # Verify content
            train_df = pd.read_csv(output_dir / 'multi_train.tsv', sep='\t')
            self.assertEqual(len(train_df), 10)  # 5 from ES + 5 from FR


class TestGetLanguageTagsFromVocab(unittest.TestCase):
    """Test get_language_tags_from_vocab function."""

    def test_extracts_tags_from_simple_vocab(self):
        """Test extracting language tags from SimpleVocab."""
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab([
            '<2es> hello world',
            '<2fr> bonjour monde',
            '<2de> hallo welt',
            'regular sentence',
        ])

        tags = get_language_tags_from_vocab(vocab)

        self.assertEqual(len(tags), 3)
        self.assertIn('<2es>', tags)
        self.assertIn('<2fr>', tags)
        self.assertIn('<2de>', tags)

    def test_returns_empty_when_no_tags(self):
        """Test returns empty list when no language tags present."""
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab(['hello world', 'no tags here'])

        tags = get_language_tags_from_vocab(vocab)

        self.assertEqual(len(tags), 0)


class TestEnsureLanguageTagsInVocab(unittest.TestCase):
    """Test ensure_language_tags_in_vocab function."""

    def test_adds_missing_tags_to_simple_vocab(self):
        """Test adding language tags to SimpleVocab."""
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab(['hello world', 'good morning'])

        # Initially no language tags
        self.assertNotIn('<2es>', vocab.token2idx)
        self.assertNotIn('<2fr>', vocab.token2idx)

        # Add language tags
        ensure_language_tags_in_vocab(vocab, ['es', 'fr', 'de'])

        # Verify tags were added
        self.assertIn('<2es>', vocab.token2idx)
        self.assertIn('<2fr>', vocab.token2idx)
        self.assertIn('<2de>', vocab.token2idx)

        # Verify they have valid indices
        self.assertIsInstance(vocab.token2idx['<2es>'], int)
        self.assertEqual(vocab.idx2token[vocab.token2idx['<2es>']], '<2es>')

    def test_does_not_duplicate_existing_tags(self):
        """Test that existing tags are not duplicated."""
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab(['<2es> hello', 'world'])

        initial_size = len(vocab)
        initial_es_idx = vocab.token2idx['<2es>']

        # Try to add ES again
        ensure_language_tags_in_vocab(vocab, ['es', 'fr'])

        # Size should increase by 1 (only FR added)
        self.assertEqual(len(vocab), initial_size + 1)

        # ES index should be unchanged
        self.assertEqual(vocab.token2idx['<2es>'], initial_es_idx)

        # FR should be added
        self.assertIn('<2fr>', vocab.token2idx)

    def test_handles_sentencepiece_vocab(self):
        """Test that SentencePiece vocab triggers warning (not error)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create minimal training data with more content for SentencePiece
            sentences = [f'hello world sentence number {i}' for i in range(20)]
            data = pd.DataFrame({'src': sentences, 'tgt': ['x'] * 20})
            data_file = tmp / 'train.tsv'
            data.to_csv(data_file, sep='\t', index=False)

            # Train SentencePiece model with smaller vocab
            cfg = Config(
                data_dir=tmp,
                sentencepiece_model_prefix=str(tmp / 'sp'),
                vocab_size=32,  # Smaller vocab size
            )
            train_sentencepiece([data_file], cfg.sentencepiece_model_prefix, config=cfg)

            # Create SentencePiece vocab
            sp_vocab = SentencePieceVocab(cfg.sentencepiece_model, config=cfg)

            # Should not raise error, just print warning
            ensure_language_tags_in_vocab(sp_vocab, ['es', 'fr'])

    def test_accepts_custom_language_tags(self):
        """Test using custom language tags."""
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab(['hello'])

        custom_tags = {'es': '[ES]', 'fr': '[FR]'}
        ensure_language_tags_in_vocab(vocab, ['es', 'fr'], language_tags=custom_tags)

        self.assertIn('[ES]', vocab.token2idx)
        self.assertIn('[FR]', vocab.token2idx)


class TestDefaultLanguageTags(unittest.TestCase):
    """Test DEFAULT_LANGUAGE_TAGS constant."""

    def test_contains_common_languages(self):
        """Test that default tags include common languages."""
        common_langs = ['es', 'fr', 'de', 'zh', 'ja', 'ar', 'ru', 'it', 'pt']

        for lang in common_langs:
            self.assertIn(lang, DEFAULT_LANGUAGE_TAGS)
            self.assertEqual(DEFAULT_LANGUAGE_TAGS[lang], f'<2{lang}>')

    def test_all_tags_follow_pattern(self):
        """Test that all default tags follow the <2XX> pattern."""
        import re

        pattern = re.compile(r'<2[a-z]{2,3}>')

        for tag in DEFAULT_LANGUAGE_TAGS.values():
            self.assertTrue(
                pattern.match(tag),
                f"Tag '{tag}' does not match expected pattern <2XX>"
            )


if __name__ == '__main__':
    unittest.main()
