from functools import lru_cache, partial
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from dataset import load_conll, label2id

MAX_LENGTH = 256


@lru_cache(maxsize=4)
def get_tokenizer(model_name="xlm-roberta-base"):
    """Return a cached AutoTokenizer for the given model name.

    Results are cached so switching between languages or runs doesn't reload
    the same tokenizer from disk multiple times.
    """
    return AutoTokenizer.from_pretrained(model_name)


# Backward-compatible alias — existing code that does `from utils import tokenizer`
# continues to work without changes.
tokenizer = get_tokenizer("xlm-roberta-base")


def tokenize_and_align_labels(examples, model_name="xlm-roberta-base"):
    """Tokenize word-level sequences and align BIO labels to subword tokens.

    Uses the first-subword strategy: the original label is assigned to the first
    subword of each word; continuation subwords and special tokens get -100
    (ignored by the cross-entropy loss).

    Args:
        examples:   Batch dict with "tokens" and "ner_tags" keys.
        model_name: HuggingFace model ID — selects the correct tokenizer.
                    Must match the model used for training.
    """
    tok = get_tokenizer(model_name)
    tokenized = tok(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    all_labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:                   # [CLS] or [SEP]
                label_ids.append(-100)
            elif word_idx != prev_word_idx:        # first subword of this word
                label_ids.append(label2id[label_seq[word_idx]])
            else:                                  # continuation subword
                label_ids.append(-100)
            prev_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def build_hf_dataset(train_file, dev_file, test_file, model_name="xlm-roberta-base"):
    """Load CoNLL files and return a tokenized HuggingFace DatasetDict.

    The tokenized output is model-specific: switching to a different tokenizer
    (e.g. allenai/biomed_roberta_base) requires re-running this function and
    saving to a separate path so the two datasets don't overwrite each other.
    """
    splits = {
        "train":      Dataset.from_list(load_conll(train_file)),
        "validation": Dataset.from_list(load_conll(dev_file)),
        "test":       Dataset.from_list(load_conll(test_file)),
    }
    ds = DatasetDict(splits)
    fn = partial(tokenize_and_align_labels, model_name=model_name)
    return ds.map(fn, batched=True, remove_columns=["tokens", "ner_tags"])


def build_joint_dataset(en_ds, es_ds):
    """Concatenate EN and ES training sets for joint multilingual training."""
    joint_train = concatenate_datasets([en_ds["train"], es_ds["train"]])
    return DatasetDict({
        "train":      joint_train,
        "validation": en_ds["validation"],
        "test":       en_ds["test"],
    })


def get_data_paths(lang):
    """Return (train, dev, test) CoNLL file paths for a given language."""
    root = Path(__file__).resolve().parent.parent
    raw  = root / "data" / "raw" / lang
    return (
        str(raw / "train.conll"),
        str(raw / "dev.conll"),
        str(raw / "test.conll"),
    )


def predict(text, model, device=None, model_name="xlm-roberta-base", lang="en", apply_negation=True):
    """Run NER inference on a raw text string.

    Args:
        text:             Plain-text sentence (English or Spanish).
        model:            A loaded AutoModelForTokenClassification model.
        device:           torch.device — auto-detected if None.
        model_name:       HuggingFace model ID used to select the matching tokenizer.
        lang:             Language code ("en" or "es") for NegEx cue lookup.
        apply_negation:   If True, apply rule-based NegEx post-processing to
                          relabel entities in negated contexts back to "O".

    Returns:
        List of dicts: [{"word": str, "label": str}, ...]
        One entry per input word; subword continuations are merged back.
        Negated entities are relabelled to "O" when apply_negation=True.
    """
    import torch
    from dataset import id2label

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tok = get_tokenizer(model_name)
    words = text.strip().split()
    encoding = tok(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids()

    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in encoding.items()}).logits
    pred_ids = logits.argmax(dim=-1).squeeze().tolist()

    results, seen = [], set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        results.append({"word": words[word_idx], "label": id2label[pred_ids[token_idx]]})

    if apply_negation:
        from negation import apply_negex
        results, _ = apply_negex(results, words, lang=lang)

    return results


def predict_and_print(text, model, device=None, model_name="xlm-roberta-base", lang="en", apply_negation=True):
    """Print a colour-coded NER annotation for a sentence.

    Entity colours (ANSI): DIS=red, SYM=yellow, PRO=cyan, O=white
    """
    COLORS = {"DIS": "\033[91m", "SYM": "\033[93m", "PRO": "\033[96m", "O": "\033[0m"}
    RESET  = "\033[0m"

    results = predict(text, model, device, model_name=model_name, lang=lang, apply_negation=apply_negation)
    print("\nInput:", text)
    print("─" * 60)
    print(f"{'Word':<25} {'Label'}")
    print("─" * 60)
    for r in results:
        entity_type = r["label"].split("-")[-1] if r["label"] != "O" else "O"
        color = COLORS.get(entity_type, "")
        print(f"{color}{r['word']:<25} {r['label']}{RESET}")
    print("─" * 60)

    # Summarise extracted spans
    spans, cur_span, cur_type = [], [], None
    for r in results:
        if r["label"].startswith("B-"):
            if cur_span:
                spans.append((" ".join(cur_span), cur_type))
            cur_span = [r["word"]]
            cur_type = r["label"][2:]
        elif r["label"].startswith("I-") and cur_span:
            cur_span.append(r["word"])
        else:
            if cur_span:
                spans.append((" ".join(cur_span), cur_type))
            cur_span, cur_type = [], None
    if cur_span:
        spans.append((" ".join(cur_span), cur_type))

    if spans:
        print("\nExtracted entities:")
        for span, etype in spans:
            print(f"  [{etype}]  {span}")
    else:
        print("\nNo entities found.")
    return results
