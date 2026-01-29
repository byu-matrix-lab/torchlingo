#!/usr/bin/env python3
"""Generate small SentencePiece models for tests/CI.

This script reads a parallel TSV (default: `data/example.tsv`) with columns
`src` and `tgt`, writes short corpora, and trains two SentencePiece models
(`sp_model_src.model` and `sp_model_tgt.model`) into the same `data/` folder.

Usage:
  python scripts/generate_sentencepiece_models.py --example data/example.tsv --n 2000
"""
from pathlib import Path
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=str, default="data/example.tsv")
    parser.add_argument("--n", type=int, default=2000, help="max sentences per side")
    parser.add_argument("--vocab-size", type=int, default=256)
    args = parser.parse_args()

    try:
        import pandas as pd
        import sentencepiece as spm
    except Exception as e:
        print("Missing runtime dependency:", e)
        sys.exit(0)

    data_dir = Path(args.example).resolve().parent
    example_path = Path(args.example)
    if not example_path.exists():
        print(f"{example_path} not found; skipping model generation.")
        return

    df = pd.read_csv(example_path, sep="\t")
    n = min(args.n, len(df))
    src_lines = df["src"].astype(str).head(n).tolist()
    tgt_lines = df["tgt"].astype(str).head(n).tolist()

    src_corpus = data_dir / "sp_src_corpus.txt"
    tgt_corpus = data_dir / "sp_tgt_corpus.txt"
    src_corpus.write_text("\n".join(src_lines), encoding="utf-8")
    tgt_corpus.write_text("\n".join(tgt_lines), encoding="utf-8")

    sp_model_src_prefix = data_dir / "sp_model_src"
    sp_model_tgt_prefix = data_dir / "sp_model_tgt"

    print("Training source SentencePiece model...")
    spm.SentencePieceTrainer.train(
        input=str(src_corpus),
        model_prefix=str(sp_model_src_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<sos>",
        eos_piece="<eos>",
    )

    print("Training target SentencePiece model...")
    spm.SentencePieceTrainer.train(
        input=str(tgt_corpus),
        model_prefix=str(sp_model_tgt_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<sos>",
        eos_piece="<eos>",
    )

    print("SentencePiece models generated:", sp_model_src_prefix.with_suffix('.model'), sp_model_tgt_prefix.with_suffix('.model'))

if __name__ == "__main__":
    main()
