"""Check whether a parallel corpus is actually parallel.

This module exists because TorchLingo shipped a corpus that was not. The two
columns of `data/example.tsv` were independent documents zipped together, so
row *n* of the source side had nothing to do with row *n* of the target side.
It looked completely normal: two columns, no empty cells, plausible sentences
on both sides.

A student who trained on it would have watched the loss refuse to fall and had
no way to tell bad data from a mistake of their own. That is the worst failure
mode in a teaching library, and nothing in the repository would have caught it.

These are the two checks that would have. Both are cheap, need no model, and
rest on properties that hold for translations and fail for unrelated text:

**Sentence lengths correlate.** A translation is about as long as its source.
Not exactly, and the ratio varies by language pair, but strongly enough that
genuinely parallel text scores near 0.97 while unrelated pairs score near zero.

**Names and numbers survive translation.** "Stephen Palumbi" and "2010" appear
in both sides of a real pair. They are not translated, so they can be compared
directly without knowing either language.

Neither is proof. A corpus can pass both and still be subtly misaligned, and a
legitimately noisy corpus can score lower than these numbers without being
broken. They are smoke detectors: cheap, and loud about the failure that
actually happened.

Run **both**, because each has a blind spot the other covers. The anchor check
compares *which* names and numbers the two sides share, so a token that appears
in nearly every row carries no signal: a single speaker's name, repeated
throughout their talks, is shared by every pairing whether that pairing is
right or wrong. Scramble such a corpus and anchor agreement stays at 1.0 while
the length correlation collapses. The reverse also happens, on corpora whose
sentences are mostly the same length.

Example:
    >>> import pandas as pd
    >>> good = pd.DataFrame(
    ...     {"src": ["Stephen spoke in 2010."], "tgt": ["Stephen habló en 2010."]}
    ... )
    >>> report = diagnose_alignment(good)
    >>> report.anchor_agreement
    1.0
"""

import re
from dataclasses import dataclass

import pandas as pd

# Capitalized words of four letters or more, and runs of digits. The length
# floor skips sentence-initial short words like "The", which capitalize by
# grammar rather than because they are names.
_NAMES = re.compile(r"\b[A-Z][a-z]{3,}\b")
_DIGITS = re.compile(r"\d+")

# Scoring every row of a large corpus is wasted work: the estimate is stable
# long before that. Raise it if you want a tighter number.
DEFAULT_SAMPLE = 5000


@dataclass
class AlignmentReport:
    """What the two checks found.

    Attributes:
        length_correlation (float): Pearson correlation between source and
            target sentence lengths in whitespace tokens. Near 0.97 for
            genuinely parallel text, near 0.0 for unrelated pairs.
        anchor_agreement (float): Fraction of scorable rows whose two sides
            share at least one name or number. Around 0.39 on the repaired
            example corpus, 0.014 on the broken one.
        scorable_rows (int): Rows that carried a checkable anchor. If this is
            small, ``anchor_agreement`` means little.
        rows (int): Rows examined.
    """

    length_correlation: float
    anchor_agreement: float
    scorable_rows: int
    rows: int

    def looks_aligned(
        self, min_correlation: float = 0.80, min_agreement: float = 0.25
    ) -> bool:
        """Report whether both checks clear conservative thresholds.

        The defaults sit well below what the repaired corpus achieves (0.97 and
        0.39) and far above what the broken one managed (0.001 and 0.014).
        Anything landing between is worth looking at by hand.

        Args:
            min_correlation (float): Floor for ``length_correlation``.
            min_agreement (float): Floor for ``anchor_agreement``.

        Returns:
            bool: True if both checks pass.
        """
        return (
            self.length_correlation >= min_correlation
            and self.anchor_agreement >= min_agreement
        )


def anchors(text: str) -> set[str]:
    """Extract tokens that should survive translation.

    Args:
        text (str): A sentence.

    Returns:
        set[str]: Capitalized words and digit runs found in it.

    Example:
        >>> sorted(anchors("Stephen spoke in 2010."))
        ['2010', 'Stephen']
    """
    return set(_NAMES.findall(text)) | set(_DIGITS.findall(text))


def length_correlation(
    frame: pd.DataFrame, src_col: str = "src", tgt_col: str = "tgt"
) -> float:
    """Correlate source and target sentence lengths.

    Rows where either side is empty are skipped: they carry no length signal
    and would drag the correlation toward zero for the wrong reason.

    Args:
        frame (pd.DataFrame): Corpus with source and target columns.
        src_col (str): Source column name.
        tgt_col (str): Target column name.

    Returns:
        float: Pearson correlation, or 0.0 if nothing is scorable.
    """
    src_len = frame[src_col].astype(str).str.split().str.len()
    tgt_len = frame[tgt_col].astype(str).str.split().str.len()
    usable = (src_len > 0) & (tgt_len > 0)
    if usable.sum() < 2:
        return 0.0
    correlation = src_len[usable].corr(tgt_len[usable])
    return 0.0 if pd.isna(correlation) else float(correlation)


def anchor_agreement(
    frame: pd.DataFrame,
    sample: int = DEFAULT_SAMPLE,
    src_col: str = "src",
    tgt_col: str = "tgt",
) -> tuple[float, int]:
    """Measure how often a row's two sides share a name or number.

    Scored only on rows whose source side actually contains such a token, since
    a row with nothing checkable is neither evidence for nor against alignment.

    Args:
        frame (pd.DataFrame): Corpus with source and target columns.
        sample (int): Maximum rows to score.
        src_col (str): Source column name.
        tgt_col (str): Target column name.

    Returns:
        tuple[float, int]: The agreement fraction, and how many rows were
            scorable. Read the fraction only if the count is meaningful.
    """
    hits = total = 0
    sources = frame[src_col].astype(str).head(sample)
    targets = frame[tgt_col].astype(str).head(sample)
    for src, tgt in zip(sources, targets):
        source_anchors = anchors(src)
        if not source_anchors:
            continue
        total += 1
        hits += bool(source_anchors & anchors(tgt))
    return (hits / total if total else 0.0), total


def diagnose_alignment(
    frame: pd.DataFrame,
    sample: int = DEFAULT_SAMPLE,
    src_col: str = "src",
    tgt_col: str = "tgt",
) -> AlignmentReport:
    """Run both alignment checks on a parallel corpus.

    Args:
        frame (pd.DataFrame): Corpus with source and target columns.
        sample (int): Maximum rows to score for anchor agreement.
        src_col (str): Source column name.
        tgt_col (str): Target column name.

    Returns:
        AlignmentReport: Both measurements, plus how much was scorable.

    Example:
        >>> import pandas as pd
        >>> frame = pd.DataFrame(
        ...     {"src": ["Maria arrived in 1999."], "tgt": ["Maria llegó en 1999."]}
        ... )
        >>> diagnose_alignment(frame).looks_aligned()
        True
    """
    agreement, scorable = anchor_agreement(frame, sample, src_col, tgt_col)
    return AlignmentReport(
        length_correlation=round(length_correlation(frame, src_col, tgt_col), 4),
        anchor_agreement=round(agreement, 4),
        scorable_rows=scorable,
        rows=len(frame),
    )


def shuffle_target_side(frame: pd.DataFrame, tgt_col: str = "tgt") -> pd.DataFrame:
    """Return a copy whose target side no longer matches its source side.

    This reconstructs the failure: the same two columns, each individually
    intact, paired up wrongly. It is how the checks above can be demonstrated
    on any corpus rather than described in the abstract, and it is what the
    broken `data/example.tsv` amounted to.

    Args:
        frame (pd.DataFrame): An aligned corpus.
        tgt_col (str): Target column to rotate.

    Returns:
        pd.DataFrame: A copy with the target column shifted by one row, so no
            row keeps its own translation.

    Example:
        >>> import pandas as pd
        >>> frame = pd.DataFrame({"src": ["a", "b"], "tgt": ["x", "y"]})
        >>> list(shuffle_target_side(frame)["tgt"])
        ['y', 'x']
    """
    broken = frame.copy()
    rotated = list(broken[tgt_col])
    broken[tgt_col] = rotated[1:] + rotated[:1]
    return broken
