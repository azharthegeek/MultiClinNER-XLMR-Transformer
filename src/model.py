import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from transformers import AutoModelForTokenClassification, Trainer

from dataset import LABEL_LIST, label2id, id2label


def load_model(model_name="xlm-roberta-base"):
    """Load XLM-R (or any HuggingFace model) with a token classification head.

    Pass model_name="bert-base-multilingual-cased" to run the mBERT baseline.
    """
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )


def compute_class_weights(train_dataset, num_labels=None, ignore_index=-100, max_ratio=5.0):
    """Compute inverse-frequency class weights capped at max_ratio × the minimum weight.

    In clinical NER, O tokens make up ~90-95% of all tokens, so pure inverse-
    frequency weighting produces ratios of 60-70×, which causes the model to
    over-predict entities everywhere (high recall, collapsed precision, low F1).
    Capping at max_ratio (default 5×, validated by Run 02) preserves enough
    signal for rare entity classes while keeping precision and recall balanced.
    """
    if num_labels is None:
        num_labels = len(LABEL_LIST)
    all_seqs = train_dataset["labels"]
    flat = [lbl for seq in all_seqs for lbl in seq if lbl != ignore_index]
    counts = torch.tensor(
        np.bincount(flat, minlength=num_labels), dtype=torch.float
    )
    total = counts.sum()
    weights = total / (num_labels * counts.clamp(min=1))
    # Without capping, entity weights are ~60× the O weight, forcing the model
    # to predict entities everywhere to minimise loss — killing precision.
    min_w = weights.min()
    weights = weights.clamp(max=min_w * max_ratio)
    return weights


class WeightedLossTrainer(Trainer):
    """HuggingFace Trainer with per-class weighted cross-entropy or focal loss,
    and optional layer-wise learning rate decay (LLRD).

    Args:
        class_weights: Per-class loss weights (from compute_class_weights).
        focal_gamma: Focal loss exponent γ. 0.0 = standard weighted CE (default).
            γ=2.0 is the standard starting point; higher values focus more on
            hard examples. Focal loss combined with class weights acts as both
            an imbalance corrector (alpha) and a hard-example focuser (gamma).
        llrd_decay: Layer-wise LR decay rate (e.g. 0.9). None = uniform LR
            for all layers. When set, the classification head uses base_lr,
            each encoder layer uses base_lr × decay^(L-i), and embeddings use
            base_lr × decay^(L+1).
    """

    def __init__(self, class_weights=None, focal_gamma=0.0, llrd_decay=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.llrd_decay = llrd_decay

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Read labels without popping — prediction_step reads inputs["labels"]
        # again after this call to extract gold labels for compute_metrics.
        labels = inputs.get("labels")
        # Forward pass without labels so the model doesn't compute its own
        # unweighted loss internally.
        forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**forward_inputs)
        logits = outputs.logits

        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )

        flat_logits = logits.view(-1, model.config.num_labels)
        flat_labels = labels.view(-1)
        valid_mask  = flat_labels != -100
        # Clamp so -100 indices don't cause out-of-bounds during gather/lookup.
        safe_labels = flat_labels.clamp(min=0)

        if self.focal_gamma > 0:
            # Focal loss: FL = -(1 - p_t)^γ · log(p_t)
            # Combined with per-class alpha weights (same role as the `weight`
            # arg in standard CE). Using log-softmax + gather avoids the
            # numerical instability of computing softmax then log separately.
            log_probs = F.log_softmax(flat_logits, dim=-1)
            log_pt    = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
            pt        = log_pt.exp()
            focal     = (1 - pt) ** self.focal_gamma * (-log_pt)
            if weight is not None:
                focal = focal * weight[safe_labels]
            loss = focal[valid_mask].mean()
        else:
            loss = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)(
                flat_logits, flat_labels
            )

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Build per-layer parameter groups for LLRD when llrd_decay is set."""
        if not self.llrd_decay:
            return super().create_optimizer()

        model   = self.model
        base_lr = self.args.learning_rate
        decay   = self.llrd_decay
        wd      = self.args.weight_decay
        no_decay = {"bias", "LayerNorm.weight"}
        L = model.config.num_hidden_layers   # 12 for xlm-roberta-base

        def _group(params, lr, wd_val):
            return {"params": params, "lr": lr, "weight_decay": wd_val}

        groups = []
        all_named = list(model.named_parameters())

        # Classification head — trained with the full base learning rate.
        for name, param in [(n, p) for n, p in all_named if "classifier" in n]:
            wd_val = 0.0 if any(k in name for k in no_decay) else wd
            groups.append(_group([param], base_lr, wd_val))

        # Encoder layers — LR decays exponentially toward layer 0.
        # Layer L-1 (top) gets base_lr × decay^1; layer 0 gets base_lr × decay^L.
        for i in range(L - 1, -1, -1):
            lr_i = base_lr * (decay ** (L - i))
            for name, param in [(n, p) for n, p in all_named if f"encoder.layer.{i}." in n]:
                wd_val = 0.0 if any(k in name for k in no_decay) else wd
                groups.append(_group([param], lr_i, wd_val))

        # Embeddings — lowest LR to preserve pretrained multilingual representations.
        embed_lr = base_lr * (decay ** (L + 1))
        for name, param in [(n, p) for n, p in all_named if "embeddings" in n]:
            wd_val = 0.0 if any(k in name for k in no_decay) else wd
            groups.append(_group([param], embed_lr, wd_val))

        print(f"LLRD: classifier lr={base_lr:.2e}, "
              f"top encoder lr={base_lr * decay:.2e}, "
              f"embeddings lr={embed_lr:.2e}")

        self.optimizer = AdamW(groups)
        return self.optimizer


def compute_metrics(eval_preds):
    """Compute entity-level precision, recall, and F1 using seqeval.

    seqeval evaluates full entity spans: a prediction is correct only when
    both the span boundary and entity type match the gold annotation.
    Per-entity F1 (DIS_f1, SYM_f1, PRO_f1) is included so TensorBoard
    shows the clinical breakdown across epochs.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)

    true_labels = [
        [id2label[l] for l in label_seq if l != -100]
        for label_seq in labels
    ]
    true_preds = [
        [id2label[p] for (p, l) in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_preds, output_dict=True, zero_division=0)
    return {
        "precision":  precision_score(true_labels, true_preds, zero_division=0),
        "recall":     recall_score(true_labels, true_preds, zero_division=0),
        "overall_f1": f1_score(true_labels, true_preds, zero_division=0),
        "DIS_f1":     report.get("DIS", {}).get("f1-score", 0.0),
        "SYM_f1":     report.get("SYM", {}).get("f1-score", 0.0),
        "PRO_f1":     report.get("PRO", {}).get("f1-score", 0.0),
    }
