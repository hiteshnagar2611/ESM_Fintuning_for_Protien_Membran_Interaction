"""
build_esm_membrane_dataset.py
==============================
Builds a unified, ESM-ready protein–membrane interaction dataset
by pulling from multiple curated sources:

  1. DeepLoc 2.1  – Swiss-Prot membrane type labels (TM / Peripheral / Lipid / Soluble)
  2. DeepLoc 2.0  – Swiss-Prot binary membrane/non-membrane labels
  3. UniProt REST  – Live Swiss-Prot membrane keyword query (KW-0812)
  4. OPM          – Orientations of Proteins in Membranes (structural labels)
  5. mpstruc       – Membrane Proteins of Known 3D Structure

Output files (in ./esm_membrane_dataset/):
  - all_sources_raw.csv          Raw merged data before deduplication
  - esm_dataset_binary.csv       Binary task   (membrane=1, non-membrane=0)
  - esm_dataset_multiclass.csv   Multi-class   (0=Soluble, 1=Transmembrane,
                                                2=Peripheral, 3=Lipid-anchored)
  - esm_dataset_hf/              HuggingFace DatasetDict (train/val/test splits)
  - dataset_stats.txt            Summary statistics

Usage:
  pip install requests pandas biopython tqdm datasets
  python build_esm_membrane_dataset.py

Author: auto-generated for ESM fine-tuning
"""

import os, re, time, json, logging, hashlib
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("esm_membrane_dataset")
OUT_DIR.mkdir(exist_ok=True)

# ── Amino-acid alphabet (standard 20 + ambiguous B/Z/X/U/O) ──────────────────
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBZXUO")

def is_valid_sequence(seq: str, min_len=50, max_len=1022) -> bool:
    """ESM-2 tokenizer supports up to 1022 residues (1024 - 2 special tokens)."""
    if not seq or not isinstance(seq, str):
        return False
    seq = seq.upper().strip()
    if not (min_len <= len(seq) <= max_len):
        return False
    return all(c in VALID_AA for c in seq)

def clean_seq(seq: str) -> str:
    return seq.upper().strip().replace(" ", "").replace("\n", "")

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: DeepLoc 2.0 (binary membrane label)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_deeploc20() -> pd.DataFrame:
    """
    Downloads the DeepLoc 2.0 Swiss-Prot Training/Validation CSV.
    Columns of interest: Sequence, Membrane (0/1)
    URL: https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/
         Swissprot_Train_Validation_dataset.csv
    """
    url = ("https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/"
           "Swissprot_Train_Validation_dataset.csv")
    log.info("Fetching DeepLoc 2.0 ...")
    try:
        r = requests.get(url, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Referer": "https://services.healthtech.dtu.dk/"})
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        # Normalise columns
        df = df.rename(columns={"Sequence": "sequence", "Membrane": "membrane_binary"})
        df["sequence"] = df["sequence"].apply(clean_seq)
        df = df[df["sequence"].apply(is_valid_sequence)].copy()
        df["source"] = "DeepLoc2.0"
        df["label_type"] = "binary"
        # membrane_type: TM unknown for this source → set to NaN
        df["membrane_type"] = np.nan
        log.info(f"  DeepLoc 2.0: {len(df)} sequences "
                 f"({df['membrane_binary'].sum()} membrane)")
        return df[["sequence", "membrane_binary", "membrane_type", "source"]]
    except Exception as e:
        log.warning(f"  DeepLoc 2.0 failed: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: DeepLoc 2.1 (multi-class membrane type)
# ─────────────────────────────────────────────────────────────────────────────
DEEPLOC21_TYPE_MAP = {
    "Transmembrane": 1,
    "Peripheral":    2,
    "Lipid-anchored": 3,
    "Soluble":       0,
}

def fetch_deeploc21() -> pd.DataFrame:
    """
    Downloads the DeepLoc 2.1 Swiss-Prot membrane-type dataset.
    URL: https://services.healthtech.dtu.dk/services/DeepLoc-2.1/data/
         Swissprot_Train_Validation_dataset_membrane.csv
    Columns: Sequence, Type (Transmembrane/Peripheral/Lipid-anchored/Soluble)
    """
    url = ("https://services.healthtech.dtu.dk/services/DeepLoc-2.1/data/"
           "Swissprot_Train_Validation_dataset_membrane.csv")
    log.info("Fetching DeepLoc 2.1 ...")
    try:
        r = requests.get(url, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Referer": "https://services.healthtech.dtu.dk/"})
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        df = df.rename(columns={"Sequence": "sequence", "Type": "membrane_type_str"})
        df["sequence"] = df["sequence"].apply(clean_seq)
        df = df[df["sequence"].apply(is_valid_sequence)].copy()
        df["membrane_type"] = df["membrane_type_str"].map(DEEPLOC21_TYPE_MAP)
        df["membrane_binary"] = (df["membrane_type"] > 0).astype(int)
        df["source"] = "DeepLoc2.1"
        log.info(f"  DeepLoc 2.1: {len(df)} sequences, "
                 f"type distribution:\n{df['membrane_type_str'].value_counts().to_dict()}")
        return df[["sequence", "membrane_binary", "membrane_type", "source"]]
    except Exception as e:
        log.warning(f"  DeepLoc 2.1 failed: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: UniProt REST API
# ─────────────────────────────────────────────────────────────────────────────
UNIPROT_MEMBRANE_KEYWORDS = {
    "KW-0812": ("Transmembrane", 1),   # Transmembrane
    "KW-0472": ("Membrane",       1),  # Membrane
    "KW-0560": ("Lipid-anchor",   3),  # Lipoprotein / lipid anchor
}
UNIPROT_SOLUBLE_KEYWORDS = {
    "KW-0963": ("Cytoplasm",      0),
    "KW-0539": ("Nucleus",        0),
    "KW-0964": ("Secreted",       0),
}

def _uniprot_fetch_batch(keyword_kw: str, size: int = 500,
                         cursor: str = None) -> dict:
    """Single paginated call to UniProt REST API."""
    params = {
        "query":  f"keyword:{keyword_kw} AND reviewed:true",
        "format": "json",
        "fields": "accession,sequence,length,keyword",
        "size":   size,
    }
    if cursor:
        params["cursor"] = cursor
    r = requests.get("https://rest.uniprot.org/uniprotkb/search",
                     params=params, timeout=60)
    r.raise_for_status()
    return r.json(), r.headers.get("x-next-cursor")

def fetch_uniprot(max_per_keyword: int = 2000) -> pd.DataFrame:
    """
    Streams UniProt REST API for membrane and soluble proteins.
    Keeps ≤ max_per_keyword entries per keyword to balance classes.
    """
    log.info("Fetching UniProt (membrane + soluble) ...")
    all_rows = []

    kw_dict = {**UNIPROT_MEMBRANE_KEYWORDS, **UNIPROT_SOLUBLE_KEYWORDS}
    for kw, (label_str, mem_bin) in kw_dict.items():
        log.info(f"  UniProt {kw} ({label_str}) ...")
        mem_type = 1 if mem_bin == 1 and "Transmembrane" in label_str else \
                   3 if mem_bin == 1 else 0
        collected = 0
        cursor = None
        try:
            while collected < max_per_keyword:
                data, cursor = _uniprot_fetch_batch(kw, size=500, cursor=cursor)
                results = data.get("results", [])
                if not results:
                    break
                for entry in results:
                    seq = entry.get("sequence", {}).get("value", "")
                    acc = entry.get("primaryAccession", "")
                    seq = clean_seq(seq)
                    if is_valid_sequence(seq):
                        all_rows.append({
                            "accession":     acc,
                            "sequence":      seq,
                            "membrane_binary": mem_bin,
                            "membrane_type": mem_type,
                            "source":        f"UniProt_{kw}",
                        })
                        collected += 1
                if not cursor or collected >= max_per_keyword:
                    break
                time.sleep(0.3)   # be polite to the API
        except Exception as e:
            log.warning(f"  UniProt {kw} failed: {e}")

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    log.info(f"  UniProt total: {len(df)} sequences "
             f"({df['membrane_binary'].sum()} membrane)")
    return df[["sequence", "membrane_binary", "membrane_type", "source"]]

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4: OPM (Orientations of Proteins in Membranes)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_opm(max_entries: int = 1000) -> pd.DataFrame:
    """
    Fetches membrane protein sequences from OPM via its REST API.
    OPM classifies by: Integral (type=1), Peripheral (type=2),
    Monotopic/Peripheral associated (type=3).
    We query the list endpoint and then retrieve FASTA from PDB.
    """
    log.info("Fetching OPM protein list ...")
    rows = []
    try:
        # OPM list endpoint
        r = requests.get(
            "https://opm-backend.alphafold.ebi.ac.uk/opm/proteins",
            params={"limit": max_entries, "type": "all"},
            timeout=30)
        r.raise_for_status()
        data = r.json()
        proteins = data.get("objects", data if isinstance(data, list) else [])
        log.info(f"  OPM returned {len(proteins)} entries")
        for prot in tqdm(proteins[:max_entries], desc="  OPM"):
            pdb_id  = prot.get("pdbid", "")
            subtype = prot.get("subtype", {})
            type_name = subtype.get("name", "") if isinstance(subtype, dict) else ""
            mem_bin  = 1
            mem_type = 2 if "peripheral" in type_name.lower() else 1
            # Fetch FASTA from RCSB
            try:
                fa_r = requests.get(
                    f"https://www.rcsb.org/fasta/entry/{pdb_id}/display",
                    timeout=15)
                if fa_r.status_code == 200:
                    lines = fa_r.text.strip().split("\n")
                    seq = "".join(l for l in lines if not l.startswith(">"))
                    seq = clean_seq(seq)
                    if is_valid_sequence(seq):
                        rows.append({
                            "sequence": seq,
                            "membrane_binary": mem_bin,
                            "membrane_type":  mem_type,
                            "source": "OPM",
                        })
                time.sleep(0.2)
            except Exception:
                pass
    except Exception as e:
        log.warning(f"  OPM failed: {e}")

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    log.info(f"  OPM: {len(df)} valid sequences")
    return df[["sequence", "membrane_binary", "membrane_type", "source"]]

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 5: mpstruc (Membrane Proteins of Known 3D Structure)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_mpstruc(max_entries: int = 500) -> pd.DataFrame:
    """
    Fetches PDB IDs from mpstruc JSON endpoint and retrieves sequences from RCSB.
    All mpstruc proteins are membrane proteins (membrane_binary=1).
    """
    log.info("Fetching mpstruc ...")
    rows = []
    try:
        r = requests.get("https://blanco.biomol.uci.edu/mpstruc/listAll/list",
                         timeout=30)
        r.raise_for_status()
        data = r.json()
        # mpstruc returns a list of protein entries
        entries = data if isinstance(data, list) else data.get("proteins", [])
        log.info(f"  mpstruc: {len(entries)} entries found")
        pdb_ids = []
        for entry in entries:
            pdb = (entry.get("pdbCode") or entry.get("pdb_id") or
                   entry.get("PDB_code") or "").upper()
            if pdb and len(pdb) == 4:
                pdb_ids.append(pdb)
        pdb_ids = list(set(pdb_ids))[:max_entries]
        log.info(f"  mpstruc: fetching FASTA for {len(pdb_ids)} unique PDB IDs")
        for pdb_id in tqdm(pdb_ids, desc="  mpstruc"):
            try:
                fa_r = requests.get(
                    f"https://www.rcsb.org/fasta/entry/{pdb_id}/display",
                    timeout=15)
                if fa_r.status_code == 200:
                    lines = fa_r.text.strip().split("\n")
                    seq = "".join(l for l in lines if not l.startswith(">"))
                    seq = clean_seq(seq)
                    if is_valid_sequence(seq):
                        rows.append({
                            "sequence": seq,
                            "membrane_binary": 1,
                            "membrane_type":  1,  # assume TM unless known otherwise
                            "source": "mpstruc",
                        })
                time.sleep(0.15)
            except Exception:
                pass
    except Exception as e:
        log.warning(f"  mpstruc failed: {e}")

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    log.info(f"  mpstruc: {len(df)} valid sequences")
    return df[["sequence", "membrane_binary", "membrane_type", "source"]]

# ─────────────────────────────────────────────────────────────────────────────
# MERGING, DEDUPLICATION & SPLITS
# ─────────────────────────────────────────────────────────────────────────────
def sequence_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()

def merge_and_deduplicate(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate all source DataFrames, deduplicate by exact sequence hash,
    and resolve label conflicts (membrane label wins; more specific type wins).
    """
    combined = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)
    log.info(f"Combined (pre-dedup): {len(combined)} rows")

    combined["seq_hash"] = combined["sequence"].apply(sequence_hash)

    # For duplicates: keep the row with highest membrane_binary (membrane wins),
    # and pick the most specific membrane_type available.
    deduped = combined.sort_values(
        ["membrane_binary", "membrane_type"], ascending=[False, False]
    ).drop_duplicates(subset=["seq_hash"], keep="first").reset_index(drop=True)
    log.info(f"After deduplication: {len(deduped)} unique sequences")
    return deduped

def make_splits(df: pd.DataFrame,
                train_frac=0.8, val_frac=0.1,
                seed=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split on membrane_binary."""
    from sklearn.model_selection import train_test_split

    # First split off test
    train_val, test = train_test_split(
        df, test_size=1-train_frac-val_frac,
        stratify=df["membrane_binary"], random_state=seed)
    # Then split train/val
    val_ratio = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val, test_size=val_ratio,
        stratify=train_val["membrane_binary"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame, path: Path, label: str):
    df.to_csv(path, index=False)
    log.info(f"Saved {label}: {path}  ({len(df)} rows)")

def save_huggingface(train, val, test, out_dir: Path):
    """Save as HuggingFace DatasetDict (requires `datasets` library)."""
    try:
        from datasets import Dataset, DatasetDict
        ds = DatasetDict({
            "train": Dataset.from_pandas(train),
            "validation": Dataset.from_pandas(val),
            "test": Dataset.from_pandas(test),
        })
        ds.save_to_disk(str(out_dir))
        log.info(f"HuggingFace DatasetDict saved to {out_dir}")
    except ImportError:
        log.warning("datasets library not found – skipping HuggingFace save. "
                    "Install with: pip install datasets")

def save_fasta(df: pd.DataFrame, path: Path, id_col: str = "seq_hash"):
    """Save as FASTA for use with ESM CLI tools."""
    with open(path, "w") as f:
        for _, row in df.iterrows():
            label = int(row.get("membrane_binary", 0))
            mtype = int(row.get("membrane_type", 0)) if pd.notna(row.get("membrane_type")) else -1
            f.write(f">{row[id_col]}|membrane={label}|type={mtype}\n{row['sequence']}\n")
    log.info(f"FASTA saved: {path}  ({len(df)} sequences)")

# ─────────────────────────────────────────────────────────────────────────────
# STATS REPORT
# ─────────────────────────────────────────────────────────────────────────────
MEMBRANE_TYPE_NAMES = {0: "Soluble", 1: "Transmembrane",
                       2: "Peripheral", 3: "Lipid-anchored"}

def write_stats(df_all, train, val, test, path: Path):
    lines = []
    lines.append("=" * 60)
    lines.append("ESM Membrane Interaction Dataset — Statistics")
    lines.append("=" * 60)
    lines.append(f"\nTotal unique sequences : {len(df_all)}")
    lines.append(f"  Membrane (label=1)   : {df_all['membrane_binary'].sum()}")
    lines.append(f"  Non-membrane (label=0): {(df_all['membrane_binary']==0).sum()}")
    lines.append(f"\nSequence length stats:")
    lens = df_all["sequence"].str.len()
    lines.append(f"  min={lens.min()}, median={lens.median():.0f}, "
                 f"mean={lens.mean():.0f}, max={lens.max()}")
    lines.append(f"\nSource breakdown:")
    for src, cnt in df_all["source"].value_counts().items():
        lines.append(f"  {src:20s}: {cnt}")
    lines.append(f"\nMulti-class label distribution:")
    for code, name in MEMBRANE_TYPE_NAMES.items():
        cnt = (df_all["membrane_type"] == code).sum()
        lines.append(f"  {code} ({name:15s}): {cnt}")
    lines.append(f"\nTrain / Val / Test split:")
    lines.append(f"  Train : {len(train)} ({len(train)/len(df_all)*100:.1f}%)")
    lines.append(f"  Val   : {len(val)}  ({len(val)/len(df_all)*100:.1f}%)")
    lines.append(f"  Test  : {len(test)} ({len(test)/len(df_all)*100:.1f}%)")
    lines.append("\n" + "=" * 60)
    lines.append("Column schema (esm_dataset_binary.csv / multiclass.csv):")
    lines.append("  sequence        : amino acid sequence (str)")
    lines.append("  membrane_binary : 0=non-membrane, 1=membrane")
    lines.append("  membrane_type   : 0=Soluble, 1=TM, 2=Peripheral, 3=Lipid")
    lines.append("  source          : origin database")
    lines.append("  seq_hash        : MD5 of sequence (unique ID)")
    lines.append("  split           : train / val / test")
    lines.append("=" * 60)

    report = "\n".join(lines)
    path.write_text(report)
    print("\n" + report)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("Starting ESM membrane dataset build ...")

    # ── 1. Fetch all sources ────────────────────────────────────────────────
    dfs = []
    dfs.append(fetch_deeploc20())
    dfs.append(fetch_deeploc21())
    dfs.append(fetch_uniprot(max_per_keyword=2000))
    dfs.append(fetch_opm(max_entries=1000))
    dfs.append(fetch_mpstruc(max_entries=500))

    # ── 2. Merge & deduplicate ──────────────────────────────────────────────
    df_all = merge_and_deduplicate(dfs)

    # Fill missing membrane_type for binary-only sources:
    # If membrane_binary=1 and type unknown → set type=1 (TM as default)
    # If membrane_binary=0 → type=0 (Soluble)
    mask_mem  = df_all["membrane_binary"] == 1
    mask_sol  = df_all["membrane_binary"] == 0
    mask_null = df_all["membrane_type"].isna()
    df_all.loc[mask_mem & mask_null, "membrane_type"] = 1
    df_all.loc[mask_sol & mask_null, "membrane_type"] = 0
    df_all["membrane_type"] = df_all["membrane_type"].astype(int)

    # Save raw merged
    save_csv(df_all, OUT_DIR / "all_sources_raw.csv", "all_sources_raw")

    # ── 3. Make splits ──────────────────────────────────────────────────────
    try:
        train, val, test = make_splits(df_all)
    except ImportError:
        log.warning("scikit-learn not found – using random split instead.")
        df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df_all)
        train = df_all.iloc[:int(n*0.8)]
        val   = df_all.iloc[int(n*0.8):int(n*0.9)]
        test  = df_all.iloc[int(n*0.9):]

    # Add split column to individual dfs and re-concatenate into df_all
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    df_all = pd.concat([train, val, test], ignore_index=True)

    # ── 4. Save binary dataset ──────────────────────────────────────────────
    binary_cols = ["seq_hash", "sequence", "membrane_binary", "source", "split"]
    save_csv(df_all[binary_cols], OUT_DIR / "esm_dataset_binary.csv", "binary")

    # ── 5. Save multi-class dataset ─────────────────────────────────────────
    multi_cols = ["seq_hash", "sequence", "membrane_binary",
                  "membrane_type", "source", "split"]
    save_csv(df_all[multi_cols], OUT_DIR / "esm_dataset_multiclass.csv", "multiclass")

    # ── 6. Save split CSVs (handy for direct training) ──────────────────────
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        save_csv(split_df[multi_cols],
                 OUT_DIR / f"split_{split_name}.csv",
                 split_name)

    # ── 7. FASTA files ──────────────────────────────────────────────────────
    save_fasta(df_all, OUT_DIR / "all_sequences.fasta")

    # ── 8. HuggingFace DatasetDict ──────────────────────────────────────────
    save_huggingface(
        train[multi_cols], val[multi_cols], test[multi_cols],
        OUT_DIR / "esm_dataset_hf"
    )

    # ── 9. Stats report ─────────────────────────────────────────────────────
    write_stats(df_all, train, val, test, OUT_DIR / "dataset_stats.txt")

    log.info(f"\n✅  Done! All files written to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
