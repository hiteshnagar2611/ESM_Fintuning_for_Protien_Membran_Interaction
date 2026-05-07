# ESM Membrane Interaction Dataset Builder

## Overview

Two scripts that build a unified, ESM-ready protein–membrane interaction dataset
from five curated sources and then fine-tune ESM-2 on it.

---

## Files

| File | Purpose |
|------|---------|
| `build_esm_membrane_dataset.py` | Downloads & merges all sources → clean CSVs |
| `finetune_esm_membrane.py` | Fine-tunes ESM-2 on the built dataset |

---

## Step 1 — Install dependencies

```bash
pip install requests pandas biopython tqdm datasets scikit-learn \
            torch transformers
```

---

## Step 2 — Build the dataset

```bash
python build_esm_membrane_dataset.py
```

This will create `esm_membrane_dataset/` containing:

```
esm_membrane_dataset/
├── all_sources_raw.csv          # merged before deduplication
├── esm_dataset_binary.csv       # binary task (membrane=1 / non-membrane=0)
├── esm_dataset_multiclass.csv   # 4-class (Soluble/TM/Peripheral/Lipid)
├── split_train.csv              # 80% stratified train split
├── split_val.csv                # 10% val split
├── split_test.csv               # 10% test split
├── all_sequences.fasta          # FASTA with labels in headers
├── esm_dataset_hf/              # HuggingFace DatasetDict (train/val/test)
└── dataset_stats.txt            # full statistics report
```

### Data sources pulled

| Source | Labels | URL |
|--------|--------|-----|
| DeepLoc 2.0 | Binary membrane/non-membrane | healthtech.dtu.dk |
| DeepLoc 2.1 | TM / Peripheral / Lipid / Soluble | healthtech.dtu.dk |
| UniProt REST | KW-0812 (TM), KW-0472 (Membrane), KW-0560 (Lipid), soluble controls | rest.uniprot.org |
| OPM | Integral / Peripheral (structure-based) | opm.phar.umich.edu |
| mpstruc | Membrane proteins of known 3D structure | blanco.biomol.uci.edu |

### Column schema

| Column | Type | Description |
|--------|------|-------------|
| `seq_hash` | str | MD5 of sequence (unique ID) |
| `sequence` | str | Amino acid sequence (50–1022 residues) |
| `membrane_binary` | int | 0 = non-membrane, 1 = membrane |
| `membrane_type` | int | 0=Soluble, 1=TM, 2=Peripheral, 3=Lipid |
| `source` | str | Origin database |
| `split` | str | train / val / test |

---

## Step 3 — Fine-tune ESM-2

### Binary task (membrane vs non-membrane)

```bash
python finetune_esm_membrane.py \
  --data_dir esm_membrane_dataset \
  --task binary \
  --model facebook/esm2_t6_8M_UR50D \
  --output_dir ./esm_binary_finetuned \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-4
```

### Multi-class task (TM / Peripheral / Lipid / Soluble)

```bash
python finetune_esm_membrane.py \
  --data_dir esm_membrane_dataset \
  --task multiclass \
  --model facebook/esm2_t12_35M_UR50D \
  --output_dir ./esm_multiclass_finetuned \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4
```

### ESM-2 model sizes

| Model | Params | Notes |
|-------|--------|-------|
| `facebook/esm2_t6_8M_UR50D` | 8M | Fastest, good for prototyping |
| `facebook/esm2_t12_35M_UR50D` | 35M | Good balance |
| `facebook/esm2_t30_150M_UR50D` | 150M | Strong performance |
| `facebook/esm2_t33_650M_UR50D` | 650M | Best quality, needs GPU |

---

## Step 4 — Load and use the fine-tuned model

```python
from transformers import EsmForSequenceClassification, AutoTokenizer
import torch

model_path = "./esm_binary_finetuned/best_model"
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = EsmForSequenceClassification.from_pretrained(model_path)
model.eval()

sequence = "MKTIIALSYIFCLVFA..."   # your protein sequence
inputs   = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    logits = model(**inputs).logits
    pred   = logits.argmax(-1).item()
    prob   = torch.softmax(logits, dim=-1)[0, pred].item()

label_map = {0: "Non-membrane", 1: "Membrane"}
print(f"Prediction: {label_map[pred]}  (confidence: {prob:.3f})")
```

---

## Tips

- **Class imbalance**: The trainer uses `WeightedCrossEntropyLoss` automatically.
- **Frozen encoder**: By default only the classification head is trained (fast, low memory). Remove the freeze block in `finetune_esm_membrane.py` for full fine-tuning.
- **Max sequence length**: ESM-2 supports up to 1022 residues. Sequences longer than `--max_length` are truncated.
- **Homology partitioning**: DeepLoc sources use ≤30% sequence similarity between partitions to prevent data leakage. For stricter splits, run `cd-hit` on the merged dataset before training.
