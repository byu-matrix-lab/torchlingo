"""Repair a TED-style parallel corpus whose two columns drifted out of alignment.

`data/example.tsv` shipped as a two-column TSV that looks parallel and is not.
Each column is an independent document -- a concatenation of per-talk records --
and the two were simply zipped together. Because the Spanish side has fewer
talks (735) than the English side (751), the columns desynchronize almost
immediately and never recover:

    row      0   "Stephen Palumbi: Following the mercury trail"  <->  (correct)
    row  11000   "So, the bacteria grows hair on the crab."      <->  "Lo llevo a terapia."

This is not a fixable offset -- the drift is progressive, so no single shift
realigns it. It *is* fixable by de-interleaving, because each column carries its
own talk boundaries.

**How the repair works**

1. Split the file into two independent streams: the `src` column and the `tgt`
   column, each read as its own document.
2. Segment each stream into per-talk records using its URL lines as boundaries.
   Every record is ``url, description, tags, views, title`` followed by the
   transcript.
3. Match talks across streams by their slug, which appears in both URLs.
4. Emit pairs only from talks whose two transcripts have the **same number of
   lines**. Equal counts mean both sides kept TED's own sentence segmentation,
   so a line-for-line pairing is sound. Talks whose counts differ would need a
   real sentence aligner and are skipped rather than guessed at.

Run:
    python scripts/realign_corpus.py --input data/example.tsv --output data/example.tsv

The input is read fully before the output is written, so in-place repair is safe.
Pass ``--report`` to also write alignment statistics as JSON.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SRC_URL = re.compile(r"^http://www\.ted\.com/talks/([a-z0-9_]+)\.html$")
TGT_URL = re.compile(r"^http://www\.ted\.com/talks/lang/[a-z]+/([a-z0-9_]+)\.html$")

# Every talk record opens with these five fields before the transcript begins.
HEADER_URL, HEADER_DESC, HEADER_TAGS, HEADER_VIEWS, HEADER_TITLE = range(5)
HEADER_LEN = 5

# A whole line that is nothing but a bracketed stage direction: (Laughter),
# (Applause), [Music]. Not a translation pair.
NON_SPEECH = re.compile(r"[\(\[][^)\]]{0,40}[\)\]]")


def split_talks(lines: list[str], pattern: re.Pattern) -> dict[str, list[str]]:
    """Segment one stream into per-talk records keyed by talk slug.

    Args:
        lines (list[str]): Every line of one column, in file order.
        pattern (re.Pattern): URL pattern whose first group is the talk slug.

    Returns:
        dict[str, list[str]]: Talk slug to that talk's lines, the URL first.
    """
    starts = [i for i, line in enumerate(lines) if pattern.match(line)]
    talks: dict[str, list[str]] = {}
    for begin, end in zip(starts, starts[1:] + [len(lines)]):
        slug = pattern.match(lines[begin]).group(1)
        talks[slug] = lines[begin:end]
    return talks


def is_non_speech(line: str) -> bool:
    """Report whether a line is a transcript stage direction rather than speech.

    TED transcripts mark events like ``(Laughter)`` and ``(Applause)``. They are
    not translations of each other and are useless as training pairs.

    In practice this removes almost nothing from this corpus, because such
    markers appear asymmetrically between languages -- a translator drops the
    ``(Laughter)`` the English transcript kept -- which makes the talk's line
    counts disagree, and :func:`pair_talk` already discards those talks
    entirely. The check stays because relying on that side effect would be
    fragile, and because it documents the intent.

    Args:
        line (str): One transcript line.

    Returns:
        bool: True if the whole line is a single bracketed stage direction.
    """
    return bool(NON_SPEECH.fullmatch(line.strip()))


def pair_talk(
    slug: str, src_record: list[str], tgt_record: list[str]
) -> list[tuple[str, str, str, str]]:
    """Pair one talk's lines, or return nothing if the transcripts disagree.

    The title and description are genuine translations and are kept, but tagged
    so they can be excluded: a title is not a spoken sentence and belongs to a
    different register. Tags and view counts are byte-identical across languages
    -- metadata, not translations -- and training on them would teach the model
    to copy.

    Args:
        slug (str): Talk identifier, carried onto every row so that a held-out
            split can be taken by talk rather than by sentence.
        src_record (list[str]): Source-side lines for one talk, URL first.
        tgt_record (list[str]): Target-side lines for the same talk.

    Returns:
        list[tuple[str, str, str, str]]: ``(talk, kind, src, tgt)`` rows, empty
            when the transcript line counts differ.
    """
    if len(src_record) != len(tgt_record):
        return []
    if len(src_record) <= HEADER_LEN:
        return []

    rows = [
        (slug, "description", src_record[HEADER_DESC], tgt_record[HEADER_DESC]),
        (slug, "title", src_record[HEADER_TITLE], tgt_record[HEADER_TITLE]),
    ]
    rows.extend(
        (slug, "transcript", src, tgt)
        for src, tgt in zip(src_record[HEADER_LEN:], tgt_record[HEADER_LEN:])
        if not (is_non_speech(src) or is_non_speech(tgt))
    )
    return rows


def realign(input_path: Path) -> tuple[pd.DataFrame, dict]:
    """De-interleave and realign a drifted parallel corpus.

    Args:
        input_path (Path): The two-column TSV to repair.

    Returns:
        tuple: The realigned DataFrame with `src`/`tgt` columns, and a dict of
            statistics describing what was recovered and what was dropped.
    """
    frame = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    src_talks = split_talks(frame["src"].tolist(), SRC_URL)
    tgt_talks = split_talks(frame["tgt"].tolist(), TGT_URL)

    shared = [slug for slug in src_talks if slug in tgt_talks]
    aligned: list[tuple[str, str, str, str]] = []
    kept_talks = 0
    transcript_lines = 0
    for slug in shared:
        rows = pair_talk(slug, src_talks[slug], tgt_talks[slug])
        if rows:
            aligned.extend(rows)
            kept_talks += 1
            transcript_lines += len(src_talks[slug]) - HEADER_LEN

    # Blank lines carry no signal and pandas would read them back as NaN.
    aligned = [(talk, kind, s.strip(), t.strip()) for talk, kind, s, t in aligned]
    aligned = [row for row in aligned if row[2] and row[3]]

    frame_out = pd.DataFrame(aligned, columns=["talk", "kind", "src", "tgt"])
    # src and tgt first: several loaders and every doc example read positionally
    # or by name from the front, and the metadata is additive.
    frame_out = frame_out[["src", "tgt", "talk", "kind"]]

    kept_transcript = int((frame_out["kind"] == "transcript").sum())
    stats = {
        "input_rows": len(frame),
        "src_talks": len(src_talks),
        "tgt_talks": len(tgt_talks),
        "shared_talks": len(shared),
        "talks_kept": kept_talks,
        "talks_dropped_line_count_mismatch": len(shared) - kept_talks,
        "talks_dropped_not_in_both_streams": len(src_talks) - len(shared),
        "non_speech_lines_removed": transcript_lines - kept_transcript,
        "output_rows": len(frame_out),
    }
    return frame_out, stats


def anchor_agreement(frame: pd.DataFrame, sample: int = 5000) -> float:
    """Estimate alignment quality by checking shared proper nouns and numbers.

    Names and numbers survive translation, so a genuinely parallel row usually
    shares at least one with its partner. This is the same probe that exposed
    the original corpus as misaligned.

    Args:
        frame (pd.DataFrame): Corpus with `src` and `tgt` columns.
        sample (int, optional): Maximum rows to score.

    Returns:
        float: Fraction of scorable rows sharing an anchor token.
    """
    names = re.compile(r"\b[A-Z][a-z]{3,}\b")
    digits = re.compile(r"\d+")

    def anchors(text: str) -> set[str]:
        return set(names.findall(text)) | set(digits.findall(text))

    hits = total = 0
    for src, tgt in zip(frame["src"].head(sample), frame["tgt"].head(sample)):
        source_anchors = anchors(src)
        if not source_anchors:
            continue
        total += 1
        hits += bool(source_anchors & anchors(tgt))
    return hits / max(total, 1)


def main() -> None:
    """Parse arguments, realign the corpus, and report what changed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/example.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/example.tsv"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    before = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False)
    frame, stats = realign(args.input)
    stats["anchor_agreement_before"] = round(anchor_agreement(before), 4)
    stats["anchor_agreement_after"] = round(anchor_agreement(frame), 4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, sep="\t", index=False)

    width = max(len(key) for key in stats)
    for key, value in stats.items():
        print(f"{key:{width}s}  {value}")
    print(f"\nwrote {len(frame)} aligned pairs to {args.output}")

    if args.report:
        args.report.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote report to {args.report}")


if __name__ == "__main__":
    main()
