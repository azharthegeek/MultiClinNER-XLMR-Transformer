from collections import Counter

LABEL_LIST = ["O", "B-DIS", "I-DIS", "B-SYM", "I-SYM", "B-PRO", "I-PRO"]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label  = {i: label for i, label in enumerate(LABEL_LIST)}


def load_conll(filepath):
    """Parse a CoNLL-format file into a list of sentence dicts.

    Each sentence dict has keys:
      - "tokens": list of word strings
      - "ner_tags": list of BIO tag strings (same length as tokens)
    """
    sentences, tokens, tags = [], [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                parts = line.split("\t")
                tokens.append(parts[0])
                tags.append(parts[-1])
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences


def entity_counts(sentences):
    """Return per-entity-type counts (B- tags only) across a sentence list."""
    counts = Counter()
    for s in sentences:
        for tag in s["ner_tags"]:
            if tag.startswith("B-"):
                counts[tag[2:]] += 1
    return dict(counts)
