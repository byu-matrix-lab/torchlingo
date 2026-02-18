"""Enhanced multilingual utilities for multi-language training.

Extends the basic bidirectional EN↔X support to handle multiple target
languages in a single model (e.g., EN→{ES, FR, DE, ...}).
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import Config, get_default_config
from .base import load_data, save_data
from .multilingual import add_language_tags


# Default language tags following mBART/M2M-100 convention
DEFAULT_LANGUAGE_TAGS = {
    'es': '<2es>',  # Spanish
    'fr': '<2fr>',  # French
    'de': '<2de>',  # German
    'zh': '<2zh>',  # Chinese
    'ja': '<2ja>',  # Japanese
    'ar': '<2ar>',  # Arabic
    'ru': '<2ru>',  # Russian
    'pt': '<2pt>',  # Portuguese
    'it': '<2it>',  # Italian
    'nl': '<2nl>',  # Dutch
    'ko': '<2ko>',  # Korean
    'tr': '<2tr>',  # Turkish
    'pl': '<2pl>',  # Polish
    'vi': '<2vi>',  # Vietnamese
    'th': '<2th>',  # Thai
    'hi': '<2hi>',  # Hindi
}


def add_language_tags_multi(
    df: pd.DataFrame,
    target_lang: str,
    language_tags: Optional[Dict[str, str]] = None,
    src_col: Optional[str] = None,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """Prepend language tag based on target language code.

    Enhanced version of add_language_tags() that accepts ISO language codes
    and looks up the appropriate tag from a dictionary.

    Args:
        df (pd.DataFrame): Input DataFrame with source/target columns.
        target_lang (str): ISO 639-1 language code (e.g., 'es', 'fr', 'de').
        language_tags (Dict[str, str], optional): Mapping of language codes to tags.
            Falls back to DEFAULT_LANGUAGE_TAGS if not provided.
        src_col (str, optional): Source column name. Falls back to config.src_col.
        config (Config, optional): Configuration object.

    Returns:
        pd.DataFrame: Copy of df with language tag prepended to src_col.

    Raises:
        ValueError: If target_lang is not in language_tags dictionary.

    Examples:
        >>> df = pd.DataFrame({'src': ['hello world'], 'tgt': ['hola mundo']})
        >>> tagged = add_language_tags_multi(df, 'es')
        >>> tagged['src'].iloc[0]
        '<2es> hello world'

        >>> # Custom language tags
        >>> custom_tags = {'es': '[ES]', 'fr': '[FR]'}
        >>> tagged = add_language_tags_multi(df, 'es', language_tags=custom_tags)
        >>> tagged['src'].iloc[0]
        '[ES] hello world'
    """
    tags = language_tags if language_tags is not None else DEFAULT_LANGUAGE_TAGS

    if target_lang not in tags:
        available = ', '.join(sorted(tags.keys()))
        raise ValueError(
            f"Unknown target language: '{target_lang}'. "
            f"Available languages: {available}"
        )

    tag = tags[target_lang]
    return add_language_tags(df, tag, src_col=src_col, config=config)


def create_multilingual_dataset(
    data_sources: Dict[str, Path],
    language_tags: Optional[Dict[str, str]] = None,
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
    seed: Optional[int] = None,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """Combine multiple language pairs into a single multilingual dataset.

    Loads parallel data for multiple target languages, tags each with the
    appropriate language token, combines them, and shuffles for training.

    Args:
        data_sources (Dict[str, Path]): Mapping of language codes to data files.
            Keys are ISO 639-1 codes (e.g., 'es', 'fr'), values are Paths to
            parallel data files containing src_col and tgt_col.
        language_tags (Dict[str, str], optional): Custom language code to tag mapping.
            Falls back to DEFAULT_LANGUAGE_TAGS.
        src_col (str, optional): Source column name. Falls back to config.src_col.
        tgt_col (str, optional): Target column name. Falls back to config.tgt_col.
        seed (int, optional): Random seed for shuffling. Falls back to config.seed.
        config (Config, optional): Configuration object.

    Returns:
        pd.DataFrame: Combined and shuffled DataFrame with tagged source sentences.

    Examples:
        >>> data_sources = {
        ...     'es': Path('data/en_es.tsv'),
        ...     'fr': Path('data/en_fr.tsv'),
        ...     'de': Path('data/en_de.tsv'),
        ... }
        >>> multilingual_df = create_multilingual_dataset(data_sources)
        >>> multilingual_df.head()
           src                           tgt
        0  <2es> hello world             hola mundo
        1  <2fr> good morning             bonjour
        2  <2de> how are you              wie geht es dir
        ...
    """
    cfg = config if config is not None else get_default_config()
    src_col = src_col if src_col is not None else cfg.src_col
    tgt_col = tgt_col if tgt_col is not None else cfg.tgt_col
    seed = seed if seed is not None else cfg.seed
    tags = language_tags if language_tags is not None else DEFAULT_LANGUAGE_TAGS

    combined_dfs: List[pd.DataFrame] = []

    for lang_code, data_file in data_sources.items():
        if not data_file.exists():
            print(f"Warning: Data file not found: {data_file}. Skipping {lang_code}.")
            continue

        # Load parallel data
        df = load_data(data_file)

        # Verify required columns exist
        if src_col not in df.columns or tgt_col not in df.columns:
            print(
                f"Warning: Missing columns in {data_file}. "
                f"Expected {src_col} and {tgt_col}. Skipping {lang_code}."
            )
            continue

        # Add language tag
        tagged_df = add_language_tags_multi(
            df, lang_code, language_tags=tags, src_col=src_col, config=cfg
        )

        combined_dfs.append(tagged_df)

    if not combined_dfs:
        raise ValueError(
            "No valid data sources found. Check that files exist and contain "
            f"required columns: {src_col}, {tgt_col}"
        )

    # Combine all language pairs
    result = pd.concat(combined_dfs, ignore_index=True)

    # Shuffle for better training dynamics
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(
        f"Created multilingual dataset with {len(result)} examples "
        f"from {len(combined_dfs)} language pairs."
    )

    return result


def save_multilingual_splits(
    data_sources: Dict[str, Dict[str, Path]],
    output_dir: Path,
    language_tags: Optional[Dict[str, str]] = None,
    train_file: Optional[str] = None,
    val_file: Optional[str] = None,
    test_file: Optional[str] = None,
    data_format: Optional[str] = None,
    config: Optional[Config] = None,
) -> None:
    """Create and save multilingual train/val/test splits.

    Args:
        data_sources (Dict[str, Dict[str, Path]]): Nested dict mapping:
            {language_code: {'train': path, 'val': path, 'test': path}}
        output_dir (Path): Directory to save multilingual splits.
        language_tags (Dict[str, str], optional): Custom language tags.
        train_file (str, optional): Output train filename. Defaults to 'multi_train.tsv'.
        val_file (str, optional): Output val filename. Defaults to 'multi_val.tsv'.
        test_file (str, optional): Output test filename. Defaults to 'multi_test.tsv'.
        data_format (str, optional): Output format. Falls back to config.data_format.
        config (Config, optional): Configuration object.

    Side Effects:
        Creates output_dir if it doesn't exist.
        Writes train/val/test files to output_dir.

    Examples:
        >>> data_sources = {
        ...     'es': {
        ...         'train': Path('data/en_es_train.tsv'),
        ...         'val': Path('data/en_es_val.tsv'),
        ...         'test': Path('data/en_es_test.tsv'),
        ...     },
        ...     'fr': {
        ...         'train': Path('data/en_fr_train.tsv'),
        ...         'val': Path('data/en_fr_val.tsv'),
        ...         'test': Path('data/en_fr_test.tsv'),
        ...     },
        ... }
        >>> save_multilingual_splits(data_sources, Path('data/multilingual/'))
        Created multilingual dataset with 5000 examples from 2 language pairs.
        Created multilingual dataset with 500 examples from 2 language pairs.
        Created multilingual dataset with 500 examples from 2 language pairs.
        Saved multilingual splits to data/multilingual/
    """
    cfg = config if config is not None else get_default_config()
    data_format = data_format if data_format is not None else cfg.data_format

    train_file = train_file if train_file is not None else f"multi_train.{data_format}"
    val_file = val_file if val_file is not None else f"multi_val.{data_format}"
    test_file = test_file if test_file is not None else f"multi_test.{data_format}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split, filename in [('train', train_file), ('val', val_file), ('test', test_file)]:
        split_sources = {
            lang: paths[split]
            for lang, paths in data_sources.items()
            if split in paths
        }

        if not split_sources:
            print(f"Warning: No {split} files provided. Skipping.")
            continue

        split_df = create_multilingual_dataset(
            split_sources,
            language_tags=language_tags,
            config=cfg,
        )

        output_path = output_dir / filename
        save_data(split_df, output_path, data_format)

    print(f"Saved multilingual splits to {output_dir}/")


def get_language_tags_from_vocab(vocab) -> List[str]:
    """Extract language tags from vocabulary.

    Scans vocabulary for tokens matching language tag pattern (<2XX>).

    Args:
        vocab: Vocabulary object (SimpleVocab or SentencePieceVocab).

    Returns:
        List[str]: List of language tags found in vocabulary.

    Examples:
        >>> vocab = SimpleVocab()
        >>> vocab.build_vocab(["<2es> hello", "<2fr> bonjour"])
        >>> tags = get_language_tags_from_vocab(vocab)
        >>> sorted(tags)
        ['<2es>', '<2fr>']
    """
    import re

    # Language tag pattern: <2XX> where XX is 2-3 letters
    pattern = re.compile(r'<2[a-z]{2,3}>')

    tags = []

    # Handle different vocab types
    if hasattr(vocab, 'token2idx'):  # SimpleVocab
        for token in vocab.token2idx.keys():
            if pattern.match(token):
                tags.append(token)
    elif hasattr(vocab, 'sp'):  # SentencePieceVocab
        vocab_size = vocab.sp.get_piece_size()
        for i in range(vocab_size):
            piece = vocab.sp.id_to_piece(i)
            if pattern.match(piece):
                tags.append(piece)

    return tags


def ensure_language_tags_in_vocab(
    vocab,
    language_codes: List[str],
    language_tags: Optional[Dict[str, str]] = None,
) -> None:
    """Ensure language tags are registered in vocabulary.

    For SimpleVocab, adds missing language tags to token2idx/idx2token.
    For SentencePieceVocab, prints a warning (tags must be added during training).

    Args:
        vocab: Vocabulary object to modify.
        language_codes (List[str]): List of ISO language codes (e.g., ['es', 'fr']).
        language_tags (Dict[str, str], optional): Custom language tag mapping.

    Side Effects:
        Modifies SimpleVocab in-place by adding missing language tags.

    Examples:
        >>> vocab = SimpleVocab()
        >>> vocab.build_vocab(["hello world"])
        >>> ensure_language_tags_in_vocab(vocab, ['es', 'fr', 'de'])
        Added 3 language tags to vocabulary: <2es>, <2fr>, <2de>
    """
    from ..data_processing.vocab import SimpleVocab, SentencePieceVocab

    tags = language_tags if language_tags is not None else DEFAULT_LANGUAGE_TAGS

    if isinstance(vocab, SimpleVocab):
        added_tags = []
        for code in language_codes:
            tag = tags.get(code, f"<2{code}>")
            if tag not in vocab.token2idx:
                idx = len(vocab.token2idx)
                vocab.token2idx[tag] = idx
                vocab.idx2token[idx] = tag
                added_tags.append(tag)

        if added_tags:
            print(f"Added {len(added_tags)} language tags to vocabulary: {', '.join(added_tags)}")

    elif isinstance(vocab, SentencePieceVocab):
        print(
            "Warning: SentencePieceVocab detected. Language tags must be included "
            "during model training using the user_defined_symbols parameter. "
            "Example: train_sentencepiece(..., user_defined_symbols=['<2es>', '<2fr>'])"
        )
    else:
        print(f"Warning: Unknown vocabulary type: {type(vocab).__name__}")


__all__ = [
    'DEFAULT_LANGUAGE_TAGS',
    'add_language_tags_multi',
    'create_multilingual_dataset',
    'save_multilingual_splits',
    'get_language_tags_from_vocab',
    'ensure_language_tags_in_vocab',
]
