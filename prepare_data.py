#!/usr/bin/env python3
"""
prepare_data.py — Convert MultiClinAI NER data to CoNLL format.

Source layout (inside MultiClinNER/):
  MultiClinNER-{lang}/MultiClinNER-{lang}-train/MultiClinNER-{lang}-train-{entity}/
    txt/  — one .txt file per clinical case
    tsv/  — one master .tsv file: filename TAB label TAB start TAB end TAB text

Output:
  data/raw/en/train.conll  dev.conll  test.conll
  data/raw/es/train.conll  dev.conll  test.conll

Strategy:
  - Process each entity type folder (disease/symptom/procedure) independently.
  - Tokenize text; assign BIO tags from character-span annotations in the TSV.
  - Sentences are delimited by blank lines within each document.
  - Since the dataset has no labeled dev/test splits, the annotated training
    documents are shuffled and split 70 / 15 / 15.
  - The blind test set (test_batch1 / test_batch2) has no gold labels; it is
    NOT written to data/raw because it cannot be used for evaluation.

Usage (run from the project root):
  python prepare_data.py
  python prepare_data.py --src <path-to-MultiClinAI-folder>
  python prepare_data.py --langs en es --seed 42
"""

import argparse
import os
import re
import random
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTITY_MAP = {
    "DISEASE":   "DIS",
    "SYMPTOM":   "SYM",
    "PROCEDURE": "PRO",
}

ENTITY_TYPES = list(ENTITY_MAP.keys())   # ["DISEASE", "SYMPTOM", "PROCEDURE"]

ENTITY_FOLDER_SUFFIX = {
    "DISEASE":   "disease",
    "SYMPTOM":   "symptom",
    "PROCEDURE": "procedure",
}

SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / dev / test


# ---------------------------------------------------------------------------
# TSV annotation loader
# ---------------------------------------------------------------------------

def load_tsv_annotations(tsv_path: Path) -> dict:
    """
    Parse the master TSV file and return:
      {doc_id: [(start, end, entity_label), ...]}
    where doc_id is the bare filename stem (no extension).
    """
    annotations = defaultdict(list)
    with open(tsv_path, encoding="utf-8") as f:
        header = f.readline()   # skip header row
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            doc_id  = parts[0]
            label   = parts[1]
            try:
                start = int(parts[2])
                end   = int(parts[3])
            except ValueError:
                continue
            annotations[doc_id].append((start, end, label))
    return dict(annotations)


# ---------------------------------------------------------------------------
# Tokenisation & BIO tagging
# ---------------------------------------------------------------------------

def split_sentences(text: str):
    """
    Split document text into sentences (lists of (token, start, end) tuples).
    Sentence boundary: one or more blank lines in the source text.
    Within a non-blank paragraph, every whitespace-separated token is kept.
    """
    sentences = []
    current = []

    # Walk the text token-by-token; treat every newline as a potential boundary.
    pos = 0
    n   = len(text)

    while pos < n:
        # Skip whitespace and count newlines
        newline_count = 0
        while pos < n and text[pos] in " \t\r\n":
            if text[pos] == "\n":
                newline_count += 1
            pos += 1

        if newline_count >= 2 and current:
            # Blank line → sentence boundary
            sentences.append(current)
            current = []

        if pos >= n:
            break

        # Read next non-whitespace token
        start = pos
        while pos < n and text[pos] not in " \t\r\n":
            pos += 1
        end = pos
        token = text[start:end]
        if token:
            current.append((token, start, end))

    if current:
        sentences.append(current)

    return sentences


def assign_bio_tags(sentences, annotations):
    """
    Given sentences = [[(token, start, end), ...], ...]
    and annotations = [(start, end, label), ...]
    return [[(token, bio_tag), ...], ...].

    A token is tagged B-X if its span [tok_start, tok_end) overlaps the
    annotation span [ann_start, ann_end) and no adjacent token to its left
    already belongs to the same span.
    """
    # Build a set of (start, end, label) sorted by start position.
    sorted_anns = sorted(annotations, key=lambda x: x[0])

    tagged_sentences = []
    for sent in sentences:
        tags = ["O"] * len(sent)
        for ann_start, ann_end, label in sorted_anns:
            ner_label = ENTITY_MAP.get(label, label)
            first_in_span = True
            for i, (tok, tok_start, tok_end) in enumerate(sent):
                # Overlap: token starts inside annotation OR annotation starts inside token
                overlaps = (tok_start < ann_end) and (tok_end > ann_start)
                if overlaps:
                    if first_in_span:
                        tags[i] = f"B-{ner_label}"
                        first_in_span = False
                    else:
                        # Only override if currently O or a lower-priority label;
                        # keep existing B- if two spans overlap (take the first).
                        if tags[i] == "O":
                            tags[i] = f"I-{ner_label}"
        tagged_sentences.append([(tok, tag) for (tok, _, _), tag in zip(sent, tags)])

    return tagged_sentences


# ---------------------------------------------------------------------------
# Per-entity-type document processor
# ---------------------------------------------------------------------------

def process_entity_folder(lang: str, entity_type: str, ner_base: Path) -> list:
    """
    Process one entity-type subfolder for the given language (train split only).
    Returns a list of documents; each document is a list of tagged sentences:
      [ [(token, bio_tag), ...], ... ]
    """
    suffix = ENTITY_FOLDER_SUFFIX[entity_type]
    folder = ner_base / f"MultiClinNER-{lang}-train" / f"MultiClinNER-{lang}-train-{suffix}"

    txt_dir = folder / "txt"
    tsv_dir = folder / "tsv"

    if not txt_dir.exists():
        print(f"  [WARN] txt dir not found: {txt_dir}")
        return []

    # Load TSV annotations (may be absent for test splits)
    annotations = {}
    if tsv_dir.exists():
        tsv_files = list(tsv_dir.glob("*.tsv"))
        if tsv_files:
            annotations = load_tsv_annotations(tsv_files[0])
            print(f"  Loaded {sum(len(v) for v in annotations.values())} annotations "
                  f"from {tsv_files[0].name}")

    documents = []
    txt_files = sorted(txt_dir.glob("*.txt"))
    for txt_file in txt_files:
        doc_id  = txt_file.stem
        text    = txt_file.read_text(encoding="utf-8")
        doc_ann = annotations.get(doc_id, [])

        sentences = split_sentences(text)
        if not sentences:
            continue

        tagged = assign_bio_tags(sentences, doc_ann)
        # Drop sentences that are entirely empty
        tagged = [s for s in tagged if s]
        if tagged:
            documents.append(tagged)

    print(f"  {lang.upper()} {entity_type}: {len(documents)} docs, "
          f"{sum(len(d) for d in documents)} sentences")
    return documents


# ---------------------------------------------------------------------------
# CoNLL writer
# ---------------------------------------------------------------------------

def write_conll(documents: list, out_path: Path):
    """Write a list of documents (list of sentences) to a CoNLL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in documents:
            for sentence in doc:
                for token, tag in sentence:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")   # blank line = sentence boundary
    n_sent = sum(len(d) for d in documents)
    n_tok  = sum(len(s) for d in documents for s in d)
    print(f"  Wrote {len(documents)} docs / {n_sent} sentences / {n_tok} tokens → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert MultiClinAI data to CoNLL format.")
    parser.add_argument(
        "--src",
        default="MultiClinAI-training+NER_test_bg+CORPUS_test_bg_v1.3_260330",
        help="Path to the root of the extracted MultiClinAI folder.",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        help="Output directory (default: data/raw).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en", "es"],
        help="Languages to process (default: en es).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/dev/test split (default: 42).",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=list(SPLIT_RATIOS),
        metavar=("TRAIN", "DEV", "TEST"),
        help="Train/dev/test split ratios (must sum to 1.0, default: 0.70 0.15 0.15).",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    rng = random.Random(args.seed)

    # Validate split ratios
    total = sum(args.split)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    train_ratio, dev_ratio, _ = args.split

    ner_root = src_root / "MultiClinNER"
    if not ner_root.exists():
        raise FileNotFoundError(f"MultiClinNER folder not found at: {ner_root}")

    for lang in args.langs:
        print(f"\n{'='*60}")
        print(f"Processing language: {lang.upper()}")
        print(f"{'='*60}")

        ner_base = ner_root / f"MultiClinNER-{lang}"
        if not ner_base.exists():
            print(f"  [SKIP] Language folder not found: {ner_base}")
            continue

        # Collect all documents across entity types
        all_documents = []
        for entity_type in ENTITY_TYPES:
            docs = process_entity_folder(lang, entity_type, ner_base)
            all_documents.extend(docs)

        if not all_documents:
            print(f"  [WARN] No documents collected for {lang}")
            continue

        print(f"\n  Total: {len(all_documents)} annotated document instances")

        # Shuffle and split
        rng.shuffle(all_documents)
        n = len(all_documents)
        n_train = int(n * train_ratio)
        n_dev   = int(n * dev_ratio)

        train_docs = all_documents[:n_train]
        dev_docs   = all_documents[n_train : n_train + n_dev]
        test_docs  = all_documents[n_train + n_dev :]

        print(f"\n  Split: train={len(train_docs)} | dev={len(dev_docs)} | test={len(test_docs)}")

        # Write CoNLL files
        print()
        write_conll(train_docs, out_root / lang / "train.conll")
        write_conll(dev_docs,   out_root / lang / "dev.conll")
        write_conll(test_docs,  out_root / lang / "test.conll")

    print(f"\n{'='*60}")
    print("Done. Verify with:")
    print("  head -20 data/raw/en/train.conll")
    print("  head -20 data/raw/es/train.conll")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
