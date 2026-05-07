"""
finetune_esm_membrane.py
=========================
Fine-tunes ESM-2 (small / 8M or 35M) on the protein–membrane interaction
dataset produced by build_esm_membrane_dataset.py.

Supports two tasks:
  --task binary       → membrane vs non-membrane (2-class)
  --task multiclass   → Soluble / TM / Peripheral / Lipid-anchored (4-class)

Usage:
  pip install transformers datasets torch scikit-learn
  python finetune_esm_membrane.py \
    --data_dir esm_membrane_dataset \
    --task binary \
    --model facebook/esm2_t6_8M_UR50D \
    --output_dir ./esm_membrane_finetuned \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-4

ESM-2 model sizes:
  facebook/esm2_t6_8M_UR50D     →   8M params  (fastest)
  facebook/esm2_t12_35M_UR50D   →  35M params
  facebook/esm2_t30_150M_UR50D  → 150M params
  facebook/esm2_t33_650M_UR50D  → 650M params  (best quality)
"""

import os, argparse, json, logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, classification_report
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Label maps ────────────────────────────────────────────────────────────────
BINARY_LABELS = {0: "Non-membrane", 1: "Membrane"}
MULTI_LABELS  = {0: "Soluble", 1: "Transmembrane",
                 2: "Peripheral", 3: "Lipid-anchored"}

# ── Dataset class ─────────────────────────────────────────────────────────────
class MembraneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, task: str):
        self.sequences = df["sequence"].tolist()
        if task == "binary":
            self.labels = df["membrane_binary"].tolist()
        else:
            self.labels = df["membrane_type"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Weighted loss trainer (handles class imbalance) ───────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Metrics ───────────────────────────────────────────────────────────────────
def build_compute_metrics(task: str, num_labels: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        f1  = f1_score(labels, preds, average="macro", zero_division=0)

        metrics = {"accuracy": acc, "mcc": mcc, "f1_macro": f1}

        if task == "binary":
            try:
                auc = roc_auc_score(labels, probs[:, 1])
                metrics["roc_auc"] = auc
            except Exception:
                pass

        return metrics
    return compute_metrics

# ── Class weights ─────────────────────────────────────────────────────────────
def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    total  = counts.sum()
    weights = total / (num_classes * counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32)
    log.info(f"Class weights: {weights.tolist()}")
    return weights

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="esm_membrane_dataset")
    parser.add_argument("--task",        default="binary",
                        choices=["binary", "multiclass"])
    parser.add_argument("--model",       default="facebook/esm2_t6_8M_UR50D",
                        help="HuggingFace model name or local path")
    parser.add_argument("--output_dir",  default="esm_membrane_finetuned")
    parser.add_argument("--max_length",  type=int, default=512)
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--warmup_ratio",type=float, default=0.1)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no_cuda",     action="store_true")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_col  = "membrane_binary" if args.task == "binary" else "membrane_type"
    label_map  = BINARY_LABELS if args.task == "binary" else MULTI_LABELS
    num_labels = len(label_map)

    # ── Load data ─────────────────────────────────────────────────────────────
    csv_path = data_dir / ("esm_dataset_binary.csv"
                           if args.task == "binary"
                           else "esm_dataset_multiclass.csv")
    log.info(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sequence", label_col])
    df[label_col] = df[label_col].astype(int)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)

    log.info(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
    log.info(f"Label distribution (train):\n"
             f"{df_train[label_col].value_counts().to_dict()}")

    # ── Tokenizer & model ─────────────────────────────────────────────────────
    log.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label={str(k): v for k, v in label_map.items()},
        label2id={v: str(k) for k, v in label_map.items()},
    )
    # Freeze the ESM encoder body — only train the classification head
    # (Remove the next 3 lines to fully fine-tune all parameters)
    for name, param in model.esm.named_parameters():
        if "contact_head" not in name:
            param.requires_grad = False
    log.info("ESM encoder frozen; only classifier head will be trained.")
    log.info("  To full-finetune, remove the freeze block in the script.")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = MembraneDataset(df_train, tokenizer, args.max_length, args.task)
    val_ds   = MembraneDataset(df_val,   tokenizer, args.max_length, args.task)
    test_ds  = MembraneDataset(df_test,  tokenizer, args.max_length, args.task)

    # ── Class weights ─────────────────────────────────────────────────────────
    train_labels    = df_train[label_col].tolist()
    class_weights   = compute_class_weights(train_labels, num_labels)

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate               = args.lr,
        warmup_ratio                = args.warmup_ratio,
        weight_decay                = args.weight_decay,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        greater_is_better           = True,
        logging_steps               = 50,
        seed                        = args.seed,
        no_cuda                     = args.no_cuda,
        report_to                   = "none",
        save_total_limit            = 2,
        dataloader_num_workers      = 2,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        class_weights   = class_weights,
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = build_compute_metrics(args.task, num_labels),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training ...")
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    # ── Evaluate on test set ──────────────────────────────────────────────────
    log.info("Evaluating on test set ...")
    predictions = trainer.predict(test_ds)
    preds  = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    report = classification_report(
        labels, preds,
        target_names=[label_map[i] for i in sorted(label_map)],
        digits=4
    )
    mcc = matthews_corrcoef(labels, preds)
    log.info(f"\nTest Classification Report:\n{report}")
    log.info(f"Test MCC: {mcc:.4f}")

    # Save results
    results = {
        "task":         args.task,
        "model":        args.model,
        "test_mcc":     mcc,
        "test_report":  report,
        "train_samples": len(df_train),
        "val_samples":  len(df_val),
        "test_samples": len(df_test),
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n✅ Fine-tuning complete! Model saved to {output_dir / 'best_model'}")
    log.info("Load it later with:")
    log.info(f"  from transformers import EsmForSequenceClassification, AutoTokenizer")
    log.info(f"  model = EsmForSequenceClassification.from_pretrained('{output_dir}/best_model')")
    log.info(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}/best_model')")

if __name__ == "__main__":
    main()
