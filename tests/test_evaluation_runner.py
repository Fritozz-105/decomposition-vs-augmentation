"""Tests for evaluation runner and results output."""

import json
import pandas as pd
import numpy as np
import pytest
from src.evaluation.runner import prepare_test_candidates, _aggregate_metrics
from src.evaluation.results import save_results_json, generate_results_markdown


@pytest.fixture
def test_pairs_df():
  return pd.DataFrame({
    "ltable_id": [1, 2, 3, 4],
    "rtable_id": [10, 20, 30, 40],
    "label": [1, 0, 1, 0],
  })


@pytest.fixture
def sample_metrics():
  return [
    {"precision": 0.8, "recall": 0.6, "f1": 0.69, "total_runtime": 10.0,
     "runtime_per_pair": 0.01, "total_tokens": 1000, "tokens_per_pair": 1.0, "parse_errors": 2},
    {"precision": 0.9, "recall": 0.7, "f1": 0.79, "total_runtime": 12.0,
     "runtime_per_pair": 0.012, "total_tokens": 1200, "tokens_per_pair": 1.2, "parse_errors": 1},
    {"precision": 0.85, "recall": 0.65, "f1": 0.74, "total_runtime": 11.0,
     "runtime_per_pair": 0.011, "total_tokens": 1100, "tokens_per_pair": 1.1, "parse_errors": 3},
  ]


def test_prepare_test_candidates_columns(test_pairs_df):
  """Output has source_id and target_id but not ltable_id, rtable_id, or label."""
  result = prepare_test_candidates(test_pairs_df)
  assert "source_id" in result.columns
  assert "target_id" in result.columns
  assert "ltable_id" not in result.columns
  assert "rtable_id" not in result.columns
  assert "label" not in result.columns


def test_prepare_test_candidates_preserves_rows(test_pairs_df):
  """Row count is preserved after conversion."""
  result = prepare_test_candidates(test_pairs_df)
  assert len(result) == len(test_pairs_df)


def test_prepare_test_candidates_string_ids(test_pairs_df):
  """IDs are converted to strings."""
  result = prepare_test_candidates(test_pairs_df)
  assert all(isinstance(v, str) for v in result["source_id"])
  assert all(isinstance(v, str) for v in result["target_id"])
  assert result.iloc[0]["source_id"] == "1"


def test_aggregate_metrics_mean_std(sample_metrics):
  """Mean and std are computed correctly for all tracked metrics."""
  result = _aggregate_metrics(sample_metrics)
  assert "mean" in result
  assert "std" in result
  assert result["mean"]["precision"] == pytest.approx(np.mean([0.8, 0.9, 0.85]))
  assert result["std"]["precision"] == pytest.approx(np.std([0.8, 0.9, 0.85]))
  assert result["mean"]["total_tokens"] == pytest.approx(np.mean([1000, 1200, 1100]))
