"""Abt-Buy dataset loader from HuggingFace."""

from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "matchbench/Abt-Buy"
CACHE_DIR = Path("data/raw") # Stores first time downloads from HuggingFace, then used for local loading on subsequent calls

_HF_FILES = {
  "source": "tableA.csv",
  "target": "tableB.csv",
  "train_pairs": "train.csv",
  "valid_pairs": "valid.csv",
  "test_pairs": "test.csv",
}


def load_abt_buy(cache_dir: Path = CACHE_DIR) -> dict[str, pd.DataFrame]:
  """
  Load the Abt-Buy dataset from HuggingFace and return as DataFrames.

  Downloads CSVs on first call and caches locally. Returns a dict with keys:
    source_df: Abt.com products (id, name, description, price)
    target_df: Buy.com products (id, name, description, price)
    train_pairs_df: Training pairs with labels
    valid_pairs_df: Validation pairs with labels
    test_pairs_df: Test pairs with labels
  """
  cache_dir = Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)

  # Create dict mapping logical keys to local cache paths
  local_paths = {key: cache_dir / fname for key, fname in _HF_FILES.items()}

  # Download files from HuggingFace if not already cached locally
  if not all(p.exists() for p in local_paths.values()):
    _download_to_cache(local_paths)

  source_df = pd.read_csv(local_paths["source"])
  target_df = pd.read_csv(local_paths["target"])

  for df in (source_df, target_df):
    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["price"] = df["price"].astype(str)

  # Create train/valid/test pairs DataFrames with correct dtypes
  train_pairs_df = _read_pairs(local_paths["train_pairs"])
  valid_pairs_df = _read_pairs(local_paths["valid_pairs"])
  test_pairs_df = _read_pairs(local_paths["test_pairs"])

  return {
    "source_df": source_df,
    "target_df": target_df,
    "train_pairs_df": train_pairs_df,
    "valid_pairs_df": valid_pairs_df,
    "test_pairs_df": test_pairs_df,
  }


def _download_to_cache(local_paths: dict[str, Path]) -> None:
  """Download CSV files from HuggingFace and copy to local cache."""
  import shutil

  # Download each of the 5 files from HuggingFace and copy to local cache directory
  for key, fname in _HF_FILES.items():
    hf_path = hf_hub_download(REPO_ID, fname, repo_type="dataset")
    shutil.copy2(hf_path, local_paths[key])


def _read_pairs(path: Path) -> pd.DataFrame:
  """Read a pairs CSV with correct dtypes."""
  df = pd.read_csv(path)
  df["ltable_id"] = df["ltable_id"].astype(str)
  df["rtable_id"] = df["rtable_id"].astype(str)
  df["label"] = df["label"].astype(int)

  return df
