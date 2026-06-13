"""Train XLM-R (or biomedical variants) for multilingual clinical NER.

Usage:
    python src/train.py                          # joint EN+ES, default hyperparams
    python src/train.py --lang en --epochs 8
    python src/train.py --lang en --model-name allenai/biomed_roberta_base --reprocess
    python src/train.py --lang es --model-name PlanTL-GOB-ES/roberta-base-biomedical-clinical-es --reprocess
"""

import argparse
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from transformers import DataCollatorForTokenClassification, EarlyStoppingCallback, TrainingArguments

from dataset import id2label
from model import compute_class_weights, compute_metrics, load_model, WeightedLossTrainer
from utils import get_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_training_args(output_dir, epochs, lr, fp16, label_smoothing=0.1, batch_size=4):
    # Each CLI run gets its own timestamped subdirectory so TensorBoard can
    # overlay and compare multiple runs without them overwriting each other.
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_dir = str(REPO_ROOT / "outputs" / "logs" / run_tag)
    # Keep effective batch at 32 regardless of per-device batch size.
    grad_accum = max(1, 32 // batch_size)
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=max(8, batch_size * 2),
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        # Cosine decay smoothly anneals the LR to near-zero, giving the model
        # more stable convergence than linear decay in the later epochs.
        lr_scheduler_type="cosine",
        # Label smoothing reduces overconfidence on noisy clinical boundary labels.
        label_smoothing_factor=label_smoothing,
        optim="adamw_torch",
        gradient_checkpointing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        greater_is_better=True,
        fp16=fp16,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        logging_steps=50,
        logging_dir=logging_dir,
        report_to="tensorboard",
        seed=42,
    )


def main(args):
    # Model-specific processed dataset path avoids tokenizer mismatch when
    # switching between xlm-roberta-base and biomedical variants whose
    # SentencePiece vocabularies differ.
    if args.model_name == "xlm-roberta-base":
        processed_dir = REPO_ROOT / "data" / "processed" / args.lang
    else:
        safe_name = args.model_name.replace("/", "_")
        processed_dir = REPO_ROOT / "data" / "processed" / f"{args.lang}_{safe_name}"

    # Re-tokenize when explicitly requested or when processed data doesn't exist.
    if args.reprocess or not processed_dir.exists():
        from utils import build_hf_dataset, build_joint_dataset, get_data_paths
        print(f"Tokenizing dataset with {args.model_name} ...")
        if args.lang == "joint":
            en_ds = build_hf_dataset(*get_data_paths("en"), model_name=args.model_name)
            es_ds = build_hf_dataset(*get_data_paths("es"), model_name=args.model_name)
            ds_built = build_joint_dataset(en_ds, es_ds)
        else:
            ds_built = build_hf_dataset(*get_data_paths(args.lang), model_name=args.model_name)
        ds_built.save_to_disk(str(processed_dir))
        print(f"Saved processed dataset to {processed_dir}")

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_dir}.\n"
            "Run with --reprocess to build it, or run notebooks/02_preprocessing.ipynb first."
        )

    tokenizer = get_tokenizer(args.model_name)
    ds = load_from_disk(str(processed_dir))
    model = load_model(args.model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    class_weights = compute_class_weights(ds["train"], max_ratio=args.weight_cap)
    print("Class weights:", {id2label[i]: f"{w:.3f}" for i, w in enumerate(class_weights.tolist())})

    checkpoint_dir = REPO_ROOT / "outputs" / "checkpoints"
    training_args = build_training_args(
        output_dir=checkpoint_dir,
        epochs=args.epochs,
        lr=args.lr,
        fp16=args.fp16,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        llrd_decay=args.llrd_decay,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    print(f"Training on '{args.lang}' split ({len(ds['train'])} examples) ...")
    trainer.train()

    best_model_dir = checkpoint_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    print(f"Best model saved to {best_model_dir}")
    return trainer


if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(description="Fine-tune XLM-R (or biomedical variants) for clinical NER")
    parser.add_argument(
        "--lang",
        default="joint",
        choices=["en", "es", "joint"],
        help="Which processed dataset to train on (default: joint EN+ES)",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument(
        "--weight-cap",
        type=float,
        default=5.0,
        dest="weight_cap",
        help="Max ratio between entity and O class weights (default: 5.0). "
             "Run 02 showed 5× is the optimal value: enough signal for rare "
             "entity classes without collapsing precision.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        dest="focal_gamma",
        help="Focal loss exponent γ (default: 0.0 = standard weighted CE). "
             "γ=2 degraded performance at this dataset scale (~20K sentences); "
             "disable unless experimenting with a significantly larger dataset.",
    )
    parser.add_argument(
        "--llrd-decay",
        type=float,
        default=0.95,
        dest="llrd_decay",
        help="Layer-wise LR decay rate (default: 0.95). Each lower encoder layer "
             "receives base_lr × decay^(L-i). None = uniform LR for all layers.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        dest="label_smoothing",
        help="Label smoothing factor (default: 0.1). Reduces overconfidence on "
             "noisy clinical entity boundary labels by softening the target distribution.",
    )
    parser.add_argument(
        "--model-name",
        default="xlm-roberta-base",
        dest="model_name",
        help="HuggingFace model ID (default: xlm-roberta-base). Examples: "
             "'allenai/biomed_roberta_base' (EN biomedical, PubMed+PMC), "
             "'PlanTL-GOB-ES/roberta-base-biomedical-clinical-es' (ES clinical EHR), "
             "'bert-base-multilingual-cased' (mBERT baseline).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        dest="batch_size",
        help="Per-device train batch size (default: 4 for ~4GB GPU). "
             "Gradient accumulation auto-scales to keep effective batch=32.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=False,
        help="Force re-tokenization of raw CoNLL files with the selected model's "
             "tokenizer and save to a model-specific path under data/processed/. "
             "Required on first use of any non-xlm-roberta-base model.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=torch.cuda.is_available(),
        help="Use mixed-precision training (auto-enabled when CUDA is available)",
    )
    args = parser.parse_args()
    main(args)
