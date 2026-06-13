"""Evaluate a trained XLM-R NER model on English or Spanish test sets.

Usage:
    python src/evaluate.py --lang en
    python src/evaluate.py --lang es
    python src/evaluate.py --lang en --model outputs/checkpoints/best_model
    python src/evaluate.py --lang en --split validation
    python src/evaluate.py --lang en --ensemble outputs/checkpoints/xlmr_best outputs/checkpoints/mbert_best
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score
from transformers import AutoModelForTokenClassification

from dataset import id2label, load_conll, label2id, LABEL_LIST
from utils import get_tokenizer, MAX_LENGTH


def _tokenize_conll(conll_path, tokenizer):
    """Load a CoNLL file and return tokenized examples with aligned labels.

    Always re-tokenizes from the raw file — never uses cached Arrow datasets —
    so the token IDs are guaranteed to match the supplied tokenizer.
    """
    sentences = load_conll(conll_path)
    all_input_ids, all_attention_masks, all_labels, all_words = [], [], [], []

    for s in sentences:
        tokens = s["tokens"]
        tags   = s["ner_tags"]
        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors=None,
        )
        word_ids = enc.word_ids()
        label_ids, prev = [], None
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != prev:
                label_ids.append(label2id[tags[w]])
            else:
                label_ids.append(-100)
            prev = w

        all_input_ids.append(enc["input_ids"])
        all_attention_masks.append(enc["attention_mask"])
        all_labels.append(label_ids)
        all_words.append(tokens)

    return all_input_ids, all_attention_masks, all_labels, all_words


def _pad_batch(seqs, pad_value):
    """Right-pad a list of variable-length lists to the same length."""
    max_len = max(len(s) for s in seqs)
    return [[*s, *[pad_value] * (max_len - len(s))] for s in seqs]


def _validate_token_ids(all_input_ids, model_vocab_size, tokenizer_vocab_size):
    """Raise a clear error if any token ID is outside the model's embedding table."""
    max_seen = max(max(ids) for ids in all_input_ids)
    if max_seen >= model_vocab_size:
        raise ValueError(
            f"\n{'='*60}\n"
            f"VOCAB MISMATCH DETECTED\n"
            f"  Max token ID in data : {max_seen}\n"
            f"  Model vocab_size     : {model_vocab_size}\n"
            f"  Tokenizer vocab_size : {tokenizer_vocab_size}\n"
            f"\nCAUSE: The checkpoint was trained with a different model/tokenizer.\n"
            f"The tokenizer produces IDs up to {tokenizer_vocab_size - 1} but the\n"
            f"model's embedding table only has {model_vocab_size} rows.\n"
            f"\nFIX: Check which base model the checkpoint actually came from by running:\n"
            f"  import json; print(json.load(open('outputs/checkpoints/optionA/config.json')))\n"
            f"{'='*60}"
        )


def _run_inference(model, tokenizer, all_input_ids, all_attention_masks, batch_size=32, device=None):
    """Run batched inference and return all logits as a list of numpy arrays."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    pad_id = tokenizer.pad_token_id

    all_preds = []
    n = len(all_input_ids)
    for start in range(0, n, batch_size):
        batch_ids   = all_input_ids[start : start + batch_size]
        batch_masks = all_attention_masks[start : start + batch_size]

        padded_ids   = torch.tensor(_pad_batch(batch_ids,   pad_id), dtype=torch.long).to(device)
        padded_masks = torch.tensor(_pad_batch(batch_masks, 0),      dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_ids=padded_ids, attention_mask=padded_masks).logits
        all_preds.append(logits.cpu().numpy())

    return all_preds


def _load_checkpoint_tokenizer(model_dir, model_name=None):
    """Load the tokenizer for a checkpoint — prefers the saved tokenizer_config.json."""
    from transformers import AutoTokenizer as _AT
    model_dir = Path(model_dir)
    if model_name:
        return get_tokenizer(model_name), model_name

    # Best option: tokenizer saved alongside the checkpoint
    if (model_dir / "tokenizer_config.json").exists():
        tok = _AT.from_pretrained(str(model_dir))
        name = getattr(tok, "name_or_path", str(model_dir))
        return tok, name

    # Fall back: read config.json
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        name = cfg.get("_name_or_path", "")
        # Reject local paths — they won't resolve to a downloadable tokenizer
        if name and not name.startswith("/") and not name.startswith("."):
            return get_tokenizer(name), name
        # Use model_type as last resort
        mtype = cfg.get("model_type", "xlm-roberta")
        defaults = {
            "xlm-roberta": "xlm-roberta-base",
            "roberta":     "roberta-base",
            "bert":        "bert-base-cased",
        }
        name = defaults.get(mtype, "xlm-roberta-base")
        return get_tokenizer(name), name

    return get_tokenizer("xlm-roberta-base"), "xlm-roberta-base"


def evaluate(model_path, lang, split="test", model_name=None, device=None):
    """Evaluate a model on EN or ES test/validation data.

    Tokenizes directly from the raw CoNLL files so token IDs always match the
    supplied model's tokenizer — never relies on cached Arrow datasets.

    Returns (true_labels, true_preds) — lists of tag sequences.
    Saves seqeval JSON to outputs/results/{lang}_{split}_results.json.
    """
    model_dir = Path(model_path) if Path(model_path).is_absolute() else REPO_ROOT / model_path
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")

    tokenizer, detected_name = _load_checkpoint_tokenizer(model_dir, model_name)
    print(f"Tokenizer : {detected_name}  (vocab_size={tokenizer.vocab_size})")

    # Locate raw CoNLL file.
    conll_file = REPO_ROOT / "data" / "raw" / lang / f"{split}.conll"
    if not conll_file.exists():
        raise FileNotFoundError(f"Raw CoNLL not found: {conll_file}")

    print(f"Tokenizing {conll_file} ...")
    all_input_ids, all_attention_masks, all_labels, _ = _tokenize_conll(conll_file, tokenizer)

    print(f"Loading model from {model_dir} ...")
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    print(f"Model     : vocab_size={model.config.vocab_size}  num_labels={model.config.num_labels}")

    # Validate BEFORE touching the GPU — catches tokenizer/model mismatches early.
    _validate_token_ids(all_input_ids, model.config.vocab_size, tokenizer.vocab_size)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print("Running inference ...")
    all_preds = _run_inference(model, tokenizer, all_input_ids, all_attention_masks, device=device)

    # Decode predictions back to BIO tags, skipping special tokens (-100).
    true_labels, true_preds = [], []
    pred_ptr = 0
    for i, label_seq in enumerate(all_labels):
        # Find which batch this sequence belongs to, and its offset within the batch.
        batch_idx = i // 32
        offset    = i % 32
        logits_batch = all_preds[batch_idx]          # (batch, seq_len, num_labels)
        pred_ids = np.argmax(logits_batch[offset], axis=-1)

        gold, pred = [], []
        for p, l in zip(pred_ids, label_seq):
            if l != -100:
                gold.append(id2label[l])
                pred.append(id2label[int(p)])
        true_labels.append(gold)
        true_preds.append(pred)

    report_dict = classification_report(true_labels, true_preds, output_dict=True)
    report_str  = classification_report(true_labels, true_preds)

    print(f"\n=== {lang.upper()} {split.capitalize()} Results ===")
    print(report_str)
    print(f"Overall F1 : {f1_score(true_labels, true_preds):.4f}")

    results_dir = REPO_ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{lang}_{split}_results.json"

    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if hasattr(obj, "item"):  # numpy scalars
            return obj.item()
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_python(report_dict), f, indent=2)
    print(f"Results saved → {out_path}")

    return true_labels, true_preds


def evaluate_ensemble(model_paths, lang, split="test", model_name="xlm-roberta-base"):
    """Run ensemble inference (logit averaging) and print/save seqeval results."""
    tokenizer = get_tokenizer(model_name)
    conll_file = REPO_ROOT / "data" / "raw" / lang / f"{split}.conll"
    all_input_ids, all_attention_masks, all_labels, _ = _tokenize_conll(conll_file, tokenizer)

    all_avg_logits = None
    for path in model_paths:
        model    = AutoModelForTokenClassification.from_pretrained(str(path))
        batches  = _run_inference(model, tokenizer, all_input_ids, all_attention_masks)
        combined = np.concatenate(batches, axis=0)   # (N, seq_len, num_labels)
        all_avg_logits = combined if all_avg_logits is None else all_avg_logits + combined
    all_avg_logits /= len(model_paths)

    true_labels, true_preds = [], []
    for i, label_seq in enumerate(all_labels):
        pred_ids = np.argmax(all_avg_logits[i], axis=-1)
        gold, pred = [], []
        for p, l in zip(pred_ids, label_seq):
            if l != -100:
                gold.append(id2label[l])
                pred.append(id2label[int(p)])
        true_labels.append(gold)
        true_preds.append(pred)

    report_dict = classification_report(true_labels, true_preds, output_dict=True)
    print(f"\n=== ENSEMBLE {lang.upper()} {split.capitalize()} Results ===")
    print(classification_report(true_labels, true_preds))
    print(f"Overall F1 : {f1_score(true_labels, true_preds):.4f}")

    results_dir = REPO_ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{lang}_{split}_ensemble_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Results saved → {out_path}")
    return true_labels, true_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XLM-R clinical NER model")
    parser.add_argument(
        "--model",
        default="outputs/checkpoints/best_model",
        help="Path to saved model checkpoint (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=["en", "es"],
        help="Language of the test set to evaluate on",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        dest="model_name",
        help="HuggingFace model ID used at training time (selects tokenizer). "
             "Auto-detected from model config.json if not specified.",
    )
    parser.add_argument(
        "--ensemble",
        nargs="+",
        default=None,
        metavar="MODEL_PATH",
        help="Two or more checkpoint paths for logit-averaging ensemble.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force inference device: 'cpu' or 'cuda' (auto-detected if omitted). "
             "Use --device cpu to bypass CUDA issues for debugging.",
    )
    args = parser.parse_args()
    _device = torch.device(args.device) if args.device else None
    if args.ensemble:
        evaluate_ensemble(
            args.ensemble, args.lang, args.split,
            model_name=args.model_name or "xlm-roberta-base",
        )
    else:
        evaluate(args.model, args.lang, args.split, model_name=args.model_name, device=_device)
