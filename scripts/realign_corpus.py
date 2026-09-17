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
import sys
from pathlib import Path

import pandas as pd

# The probes that exposed this corpus as misaligned, and the aligner that
# repairs the talks they would otherwise force us to discard, both live in the
# library: students can run them on their own data, and the tests assert on the
# same implementation this script reports.
from torchlingo.preprocessing.alignment import (
    align_one_to_one,
    anchor_agreement,
    diagnose_alignment,
)

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


def recover_talk(
    slug: str, src_record: list[str], tgt_record: list[str]
) -> list[tuple[str, str, str, str]]:
    """Rescue a talk whose two transcripts were segmented differently.

    :func:`pair_talk` discards these outright, because pairing them by position
    would be guessing and guessing is how this corpus broke in the first place.
    A length-based aligner is not a guess: it finds the cheapest way to walk
    both sides at once, and only its confident one-to-one beads are kept.

    The title and description still pair by position. They are single lines in
    a fixed slot, not a segmented transcript, so there is nothing to drift.

    Args:
        slug (str): Talk identifier, carried onto every row.
        src_record (list[str]): Source-side lines for one talk, URL first.
        tgt_record (list[str]): Target-side lines for the same talk.

    Returns:
        list[tuple[str, str, str, str]]: ``(talk, kind, src, tgt)`` rows.
    """
    if len(src_record) <= HEADER_LEN or len(tgt_record) <= HEADER_LEN:
        return []

    rows = [
        (slug, "description", src_record[HEADER_DESC], tgt_record[HEADER_DESC]),
        (slug, "title", src_record[HEADER_TITLE], tgt_record[HEADER_TITLE]),
    ]
    source = [line.strip() for line in src_record[HEADER_LEN:]]
    target = [line.strip() for line in tgt_record[HEADER_LEN:]]
    rows.extend(
        (slug, "transcript", src, tgt)
        for src, tgt in align_one_to_one(source, target)
        if not (is_non_speech(src) or is_non_speech(tgt))
    )
    return rows


def realign(input_path: Path, recover: bool = True) -> tuple[pd.DataFrame, dict]:
    """De-interleave and realign a drifted parallel corpus.

    Args:
        input_path (Path): The two-column TSV to repair.
        recover (bool): Also rescue talks whose transcripts were segmented
            differently, using the length-based aligner. When False, those
            talks are dropped, which is what this script did originally.

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
    kept_transcript_lines = 0
    recovered_transcript_lines = 0
    recovered_talks = 0
    recovered_rows = 0
    recovered_slugs: list[str] = []
    for slug in shared:
        rows = pair_talk(slug, src_talks[slug], tgt_talks[slug])
        if rows:
            aligned.extend(rows)
            kept_talks += 1
            kept_transcript_lines += len(src_talks[slug]) - HEADER_LEN
        elif recover:
            rows = recover_talk(slug, src_talks[slug], tgt_talks[slug])
            if rows:
                aligned.extend(rows)
                recovered_talks += 1
                recovered_rows += len(rows)
                recovered_slugs.append(slug)
                recovered_transcript_lines += len(src_talks[slug]) - HEADER_LEN

    # Blank lines carry no signal and pandas would read them back as NaN.
    aligned = [(talk, kind, s.strip(), t.strip()) for talk, kind, s, t in aligned]
    aligned = [row for row in aligned if row[2] and row[3]]

    frame_out = pd.DataFrame(aligned, columns=["talk", "kind", "src", "tgt"])
    # src and tgt first: several loaders and every doc example read positionally
    # or by name from the front, and the metadata is additive.
    frame_out = frame_out[["src", "tgt", "talk", "kind"]]

    # What positional pairing would have produced on the same talks. This is
    # the control: without it, "the aligner works" rests on the recovered rows
    # scoring well, which they might have done anyway. It is computed here
    # rather than typed into the docs so it cannot drift.
    naive_rows: list[tuple[str, str]] = []
    for slug in recovered_slugs:
        source = [line.strip() for line in src_talks[slug][HEADER_LEN:]]
        target = [line.strip() for line in tgt_talks[slug][HEADER_LEN:]]
        naive_rows.extend(
            (src, tgt)
            for src, tgt in zip(source, target)
            if src and tgt and not (is_non_speech(src) or is_non_speech(tgt))
        )

    # Split the accounting by cause. A talk kept by `pair_talk` loses lines
    # only to the non-speech filter, so that difference really is non-speech.
    # A talk rescued by the aligner also loses every bead that was not a
    # confident one-to-one, and folding the two together would report 222
    # "non-speech" lines in a corpus that has one.
    is_transcript = frame_out["kind"] == "transcript"
    was_recovered = frame_out["talk"].isin(set(recovered_slugs))
    kept_transcript = int((is_transcript & ~was_recovered).sum())
    recovered_transcript = int((is_transcript & was_recovered).sum())

    stats = {
        "input_rows": len(frame),
        "src_talks": len(src_talks),
        "tgt_talks": len(tgt_talks),
        "shared_talks": len(shared),
        "talks_kept": kept_talks,
        "talks_recovered_by_aligner": recovered_talks,
        "rows_recovered_by_aligner": recovered_rows,
        # Transcript rows only. The total above also counts each recovered
        # talk's title and description, which pair by position and are not the
        # aligner's work, so comparing that total against a transcript-only
        # baseline would flatter it.
        "transcript_rows_recovered_by_aligner": recovered_transcript,
        "talks_dropped_line_count_mismatch": len(shared)
        - kept_talks
        - recovered_talks,
        "talks_dropped_not_in_both_streams": len(src_talks) - len(shared),
        "recovered_slugs": recovered_slugs,
        "non_speech_lines_removed": kept_transcript_lines - kept_transcript,
        "lines_the_aligner_would_not_pair": (
            recovered_transcript_lines - recovered_transcript
        ),
        "output_rows": len(frame_out),
    }

    if naive_rows:
        naive = diagnose_alignment(
            pd.DataFrame(naive_rows, columns=["src", "tgt"]), sample=len(naive_rows)
        )
        stats["naive_pairing"] = {
            "rows": len(naive_rows),
            "length_correlation": naive.length_correlation,
            "anchor_agreement": naive.anchor_agreement,
            "passes": naive.looks_aligned(),
        }
    return frame_out, stats


def main() -> None:
    """Parse arguments, realign the corpus, and report what changed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("data/example.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/example.tsv"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--no-recover",
        action="store_true",
        help="Drop talks whose transcripts were segmented differently instead "
        "of realigning them. Reproduces this script's original behaviour.",
    )
    args = parser.parse_args()

    before = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False)
    frame, stats = realign(args.input, recover=not args.no_recover)
    stats["anchor_agreement_before"] = round(anchor_agreement(before)[0], 4)
    stats["anchor_agreement_after"] = round(anchor_agreement(frame)[0], 4)

    # The gate. Rows the aligner produced are the ones most likely to be wrong,
    # so they are checked *on their own* rather than diluted into 73k rows that
    # would mask a bad batch. Refusing to write is the whole point: this corpus
    # shipped misaligned once because nothing stood between a plausible-looking
    # file and `data/`.
    recovered = frame[frame["talk"].isin(stats.get("recovered_slugs", []))]
    if len(recovered):
        report = diagnose_alignment(recovered, sample=len(recovered))
        stats["recovered_length_correlation"] = report.length_correlation
        stats["recovered_anchor_agreement"] = report.anchor_agreement
        stats["recovered_passes"] = report.looks_aligned()
        if not report.looks_aligned():
            print(
                f"REFUSING TO WRITE: recovered rows fail the alignment check "
                f"({report}). Re-run with --no-recover to drop them instead.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, sep="\t", index=False)

    printable = {k: v for k, v in stats.items() if k != "recovered_slugs"}
    width = max(len(key) for key in printable)
    for key, value in printable.items():
        print(f"{key:{width}s}  {value}")
    print(f"\nwrote {len(frame)} aligned pairs to {args.output}")

    if args.report:
        args.report.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote report to {args.report}")

        naive = stats.get("naive_pairing")
        if naive:
            table = f"""<!-- Generated by scripts/realign_corpus.py. Do not edit by hand. -->

Of {stats["shared_talks"]} talks present in both languages,
{stats["talks_kept"]} had transcripts segmented identically and paired by
position. The other {stats["talks_recovered_by_aligner"]} did not, and were
realigned by length. Both ways of handling them, measured on the same talks:

| | rows | length correlation | name/number agreement | passes the check |
|---|---|---|---|---|
| pair by position | {naive["rows"]:,} | {naive["length_correlation"]} | {naive["anchor_agreement"]:.1%} | {"yes" if naive["passes"] else "**no**"} |
| align by length | {stats["transcript_rows_recovered_by_aligner"]:,} | {stats["recovered_length_correlation"]} | {stats["recovered_anchor_agreement"]:.1%} | {"**yes**" if stats["recovered_passes"] else "**no**"} |

Transcript rows only, so the two columns are comparable; the aligner also
recovers each talk's title and description, which pair by position.

Pairing by position yields **more** rows and fails the check. Aligning by length
gives up {naive["rows"] - stats["transcript_rows_recovered_by_aligner"]:,} of them
and passes. That is the trade this corpus has presented from the beginning:
fewer correct pairs beat more uncertain ones.
Reproduce with `python scripts/realign_corpus.py`.
"""
            markdown = args.report.with_suffix(".md")
            markdown.write_text(table)
            print(f"wrote table to {markdown}")


if __name__ == "__main__":
    main()
