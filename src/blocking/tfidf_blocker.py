"""TF-IDF cosine similarity blocking for entity resolution."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_candidate_pairs(
  source_df: pd.DataFrame,
  target_df: pd.DataFrame,
  threshold: float = 0.2,
) -> pd.DataFrame:
  """
  Generate candidate pairs using TF-IDF cosine similarity on product names.

  Fits a character-level TF-IDF vectorizer on the combined vocabulary of both
  tables, then computes cross-table cosine similarity. Pairs with similarity
  >= threshold are returned.

  Args:
    source_df: Source table with 'id' and 'name' columns.
    target_df: Target table with 'id' and 'name' columns.
    threshold: Minimum cosine similarity to include a pair.

  Returns:
    DataFrame with columns: source_id, target_id, similarity_score.
  """
  source_names = source_df["name"].fillna("").tolist()
  target_names = target_df["name"].fillna("").tolist()

  vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 3),
    lowercase=True,
  )

  all_names = source_names + target_names
  vectorizer.fit(all_names)

  # Vectorize product names
  source_tfidf = vectorizer.transform(source_names)
  target_tfidf = vectorizer.transform(target_names)

  sim_matrix = cosine_similarity(source_tfidf, target_tfidf)

  source_ids = source_df["id"].values
  target_ids = target_df["id"].values

  rows, cols = np.where(sim_matrix >= threshold)

  # Extract candidate pairs and their similarity scores that meet the threshold
  pairs = pd.DataFrame({
    "source_id": source_ids[rows],
    "target_id": target_ids[cols],
    "similarity_score": sim_matrix[rows, cols],
  })

  return pairs


def save_candidate_pairs(pairs_df: pd.DataFrame, path: str | Path) -> None:
  """Save candidate pairs DataFrame to CSV."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  pairs_df.to_csv(path, index=False)


def load_candidate_pairs(path: str | Path) -> pd.DataFrame:
  """Load candidate pairs DataFrame from CSV."""
  df = pd.read_csv(path)
  df["source_id"] = df["source_id"].astype(str)
  df["target_id"] = df["target_id"].astype(str)

  return df
