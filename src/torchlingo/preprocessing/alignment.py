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

import math
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
            bool: True if both checks pass. A frame with fewer than two usable
                rows has no measurable length correlation and will not pass
                whatever its content, so this is a corpus-level check rather
                than a per-row one.
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
        Two rows at minimum: a correlation needs something to correlate, and a
        single pair has no length variation to measure.

        >>> import pandas as pd
        >>> frame = pd.DataFrame(
        ...     {
        ...         "src": ["Maria arrived in 1999.", "She spoke briefly."],
        ...         "tgt": ["Maria llegó en 1999.", "Ella habló brevemente."],
        ...     }
        ... )
        >>> diagnose_alignment(frame).anchor_agreement
        1.0
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


# --- Repairing a nearly-aligned corpus -------------------------------------
#
# The checks above tell you a corpus is misaligned. They do not fix it. When
# the misalignment is *slight* -- two transcripts of the same talk that were
# segmented into a slightly different number of sentences -- the pairing can
# often be recovered, and that is what the rest of this module does.
#
# The method is Gale and Church (1993), and the insight behind it is the same
# one behind `length_correlation` above: a translation is about as long as its
# source. If sentence 4 on one side is twice as long as sentence 4 on the
# other, but matches sentence 5 well, the two sides have probably slipped by
# one sentence. Turn that into a cost and the best global alignment is a
# shortest-path problem, solvable by dynamic programming.
#
# Students who have seen edit distance already know this algorithm's shape:
# a table, a small set of allowed moves, and a cost per move.

# Prior costs for each alignment pattern, in units of -100*log(probability),
# taken from Gale and Church's measurements on the Hansards. One-to-one is by
# far the most common, so it is free; anything else has to earn its place.
BEAD_COSTS = {
    (1, 1): 0,
    (1, 0): 450,
    (0, 1): 450,
    (2, 1): 440,
    (1, 2): 440,
    (2, 2): 600,
}


def _length_cost(
    src_len: int, tgt_len: int, mean_ratio: float, variance: float
) -> float:
    """Cost of believing a source span of one length matches a target span.

    Models the target length as normally distributed around
    ``src_len * mean_ratio``. A pairing whose lengths disagree badly sits far
    out in the tail and costs a lot.

    Args:
        src_len (int): Characters on the source side of this bead.
        tgt_len (int): Characters on the target side.
        mean_ratio (float): Expected target-to-source character ratio.
        variance (float): Variance of that ratio, per source character.

    Returns:
        float: Cost in units of -100*log(probability). Never infinite, so a
            single implausible bead cannot make the whole search fail.
    """
    if src_len == 0 and tgt_len == 0:
        return 0.0
    expected = src_len * mean_ratio
    spread = math.sqrt(max(src_len, 1) * variance)
    deviation = abs(tgt_len - expected) / spread
    # Two-sided normal tail: P(|Z| > deviation) == erfc(deviation / sqrt(2)).
    tail = math.erfc(deviation / math.sqrt(2))
    return -100.0 * math.log(max(tail, 1e-12))


def _ratio_statistics(source: list[str], target: list[str]) -> tuple[float, float]:
    """Estimate the character-length ratio between the two sides.

    Estimated from the talk being aligned rather than hard-coded, so the same
    code works for language pairs whose typical length ratio is not 1.0.

    Args:
        source (list[str]): Source sentences.
        target (list[str]): Target sentences.

    Returns:
        tuple[float, float]: Mean ratio and its variance, both guarded away
            from zero so the cost function stays finite on degenerate input.
    """
    src_total = sum(len(s) for s in source)
    tgt_total = sum(len(t) for t in target)
    if src_total == 0 or tgt_total == 0:
        return 1.0, 6.8
    mean_ratio = tgt_total / src_total
    # Gale and Church's measured variance, scaled by this pair's ratio. Their
    # 6.8 is for English-French character counts; scaling keeps it sensible
    # when the sides have systematically different lengths.
    return mean_ratio, max(6.8 * mean_ratio, 1e-6)


def gale_church_align(
    source: list[str], target: list[str]
) -> list[tuple[list[int], list[int]]]:
    """Align two lists of sentences by length, allowing for slight drift.

    Finds the lowest-cost way to walk both sides at once, where each step is a
    "bead" pairing some sentences on the left with some on the right. Only the
    patterns in :data:`BEAD_COSTS` are allowed, which covers the ways
    segmentation usually differs: a sentence split in two, a sentence dropped,
    or two sentences merged.

    Args:
        source (list[str]): Source sentences in order.
        target (list[str]): Target sentences in order.

    Returns:
        list[tuple[list[int], list[int]]]: One entry per bead, holding the
            source indices and target indices it pairs. A 1-0 bead has an empty
            target list, and vice versa.

    Example:
        >>> src = ["Hello there.", "How are you?", "Fine."]
        >>> tgt = ["Hola.", "How are you?", "Bien."]
        >>> beads = gale_church_align(src, tgt)
        >>> len(beads)
        3
        >>> beads[0] == ([0], [0])
        True
    """
    n, m = len(source), len(target)
    mean_ratio, variance = _ratio_statistics(source, target)
    src_lengths = [len(s) for s in source]
    tgt_lengths = [len(t) for t in target]

    # cost[i][j] is the cheapest alignment of the first i source sentences
    # against the first j target sentences. back[i][j] records the bead that
    # achieved it, so the path can be walked out at the end.
    infinity = float("inf")
    cost = [[infinity] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if cost[i][j] == infinity:
                continue
            for (take_src, take_tgt), prior in BEAD_COSTS.items():
                next_i, next_j = i + take_src, j + take_tgt
                if next_i > n or next_j > m:
                    continue
                span_src = sum(src_lengths[i:next_i])
                span_tgt = sum(tgt_lengths[j:next_j])
                candidate = (
                    cost[i][j]
                    + prior
                    + _length_cost(span_src, span_tgt, mean_ratio, variance)
                )
                if candidate < cost[next_i][next_j]:
                    cost[next_i][next_j] = candidate
                    back[next_i][next_j] = (take_src, take_tgt)

    beads: list[tuple[list[int], list[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            # No path reached this cell, which can only happen if the allowed
            # bead patterns cannot span the input. Give up rather than return
            # a partial alignment that looks complete.
            return []
        take_src, take_tgt = step
        beads.append((list(range(i - take_src, i)), list(range(j - take_tgt, j))))
        i, j = i - take_src, j - take_tgt
    beads.reverse()
    return beads


def align_one_to_one(source: list[str], target: list[str]) -> list[tuple[str, str]]:
    """Align two sentence lists and keep only the confident pairings.

    Wraps :func:`gale_church_align` and returns just the one-to-one beads.
    Merged and dropped sentences are discarded rather than concatenated: this
    corpus got into trouble by guessing at alignment, and a teaching corpus is
    better small and correct than large and uncertain.

    Args:
        source (list[str]): Source sentences in order.
        target (list[str]): Target sentences in order.

    Returns:
        list[tuple[str, str]]: Confidently paired sentences.

    Example:
        >>> pairs = align_one_to_one(["A short line."], ["Una linea corta."])
        >>> pairs == [("A short line.", "Una linea corta.")]
        True
    """
    return [
        (source[src_idx[0]], target[tgt_idx[0]])
        for src_idx, tgt_idx in gale_church_align(source, target)
        if len(src_idx) == 1 and len(tgt_idx) == 1
    ]
