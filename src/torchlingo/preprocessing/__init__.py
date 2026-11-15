"""Files and DataFrames of text: reading, writing, splitting and aligning corpora.

Nothing here makes a tensor. Everything works on sentences as text, before a vocabulary exists:
loading parallel files into a DataFrame, splitting it, checking that its sides line up, tagging
pairs for a multilingual model, and training a SentencePiece model. Vocabularies, datasets and batches are the next step, in
:mod:`torchlingo.data_processing`.
"""

from .alignment import (
    AlignmentReport,
    align_one_to_one,
    anchor_agreement,
    anchors,
    diagnose_alignment,
    gale_church_align,
    length_correlation,
    shuffle_target_side,
)
from .base import (
    load_data,
    parallel_txt_to_dataframe,
    preprocess_base,
    read_lines,
    save_data,
    split_data,
    split_exact,
    write_lines,
    written_without_spaces,
)
from .multilingual import add_language_tags, preprocess_multilingual
from .sentencepiece import (
    apply_sentencepiece,
    preprocess_sentencepiece,
    train_sentencepiece,
)

__all__ = [
    "AlignmentReport",
    "add_language_tags",
    "align_one_to_one",
    "anchor_agreement",
    "anchors",
    "apply_sentencepiece",
    "diagnose_alignment",
    "gale_church_align",
    "length_correlation",
    "load_data",
    "parallel_txt_to_dataframe",
    "preprocess_base",
    "preprocess_multilingual",
    "preprocess_sentencepiece",
    "read_lines",
    "save_data",
    "shuffle_target_side",
    "split_data",
    "split_exact",
    "train_sentencepiece",
    "write_lines",
    "written_without_spaces",
]
