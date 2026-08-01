"""Tests for TF-IDF blocking module."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from src.blocking import build_candidate_pairs, load_candidate_pairs, save_candidate_pairs
from src.data import load_abt_buy


@pytest.fixture
def synthetic_data():
  source = pd.DataFrame({
    "id": ["s1", "s2", "s3"],
    "name": ["sony turntable pslx350h", "apple iphone 12 pro", "samsung galaxy s21"],
  })
  target = pd.DataFrame({
    "id": ["t1", "t2", "t3"],
    "name": ["sony turntable pslx350", "apple iphone 12", "lg monitor 27inch"],
  })
  return source, target


def test_blocking_produces_pairs(synthetic_data):
  """Test that tf-idf blocking produces candidate pairs with expected columns."""
  source, target = synthetic_data
  pairs = build_candidate_pairs(source, target, threshold=0.2)
  assert isinstance(pairs, pd.DataFrame)
  assert list(pairs.columns) == ["source_id", "target_id", "similarity_score"]
  assert len(pairs) > 0


def test_threshold_filtering(synthetic_data):
  """Test that the blocking threshold filters candidate pairs correctly."""
  source, target = synthetic_data
  pairs = build_candidate_pairs(source, target, threshold=0.5)
  assert (pairs["similarity_score"] >= 0.5).all()


def test_similarity_score_range(synthetic_data):
  """Test that the similarity scores are within 0.0 and 1.0."""
  source, target = synthetic_data
  pairs = build_candidate_pairs(source, target, threshold=0.1)
  assert (pairs["similarity_score"] >= 0.0).all()
  assert (pairs["similarity_score"] <= 1.0).all()


def test_save_and_load_roundtrip(synthetic_data):
  """Test that saving and loading candidate pairs preserves the data."""
  source, target = synthetic_data
  pairs = build_candidate_pairs(source, target, threshold=0.2)

  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "pairs.csv"
    save_candidate_pairs(pairs, path)
    loaded = load_candidate_pairs(path)

  assert len(loaded) == len(pairs)
  assert list(loaded.columns) == list(pairs.columns)


def test_blocking_recall_on_real_data():
  """Blocking recall must be >= 95% on train split ground truth."""
  data = load_abt_buy()
  pairs = build_candidate_pairs(data["source_df"], data["target_df"])

  blocked_set = set(zip(
    pairs["source_id"].astype(str),
    pairs["target_id"].astype(str),
  ))

  train_matches = data["train_pairs_df"][data["train_pairs_df"]["label"] == 1]
  found = sum(
    1 for _, row in train_matches.iterrows()
    if (str(row["ltable_id"]), str(row["rtable_id"])) in blocked_set
)
  recall = found / len(train_matches)
  assert recall >= 0.95, f"Blocking recall {recall:.4f} is below 0.95"
