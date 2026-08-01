"""Tests for src/evaluation/metrics.py."""

import pandas as pd

from src.evaluation.metrics import build_ground_truth_set, compute_er_metrics


def _make_pairs_df(rows):
  """Helper to create a pairs DataFrame."""
  return pd.DataFrame(rows, columns=["ltable_id", "rtable_id", "label"])


def test_build_ground_truth_set_only_label_1():
  """Only label=1 pairs are included; label=0 pairs are excluded."""
  pairs = _make_pairs_df([("1", "A", 1), ("2", "B", 0), ("3", "C", 1), ("4", "D", 0)])
  gt = build_ground_truth_set(pairs)
  assert gt == {("1", "A"), ("3", "C")}


def test_perfect_predictions():
  """All GT matches predicted correctly and no false positives yields P=R=F1=1.0."""
  gt = {("1", "A"), ("2", "B")}
  preds = [
    {"source_id": "1", "target_id": "A", "verdict": "MATCH", "confidence": 1.0, "parse_error": False},
    {"source_id": "2", "target_id": "B", "verdict": "MATCH", "confidence": 1.0, "parse_error": False},
    {"source_id": "3", "target_id": "C", "verdict": "NO MATCH", "confidence": 0.1, "parse_error": False},
  ]
  m = compute_er_metrics(preds, gt)
  assert m["tp"] == 2
  assert m["fp"] == 0
  assert m["fn"] == 0
  assert m["precision"] == 1.0
  assert m["recall"] == 1.0
  assert m["f1"] == 1.0


def test_all_false_negatives():
  """Predicting no matches when GT matches exist yields P=R=F1=0.0."""
  gt = {("1", "A"), ("2", "B")}
  preds = [
    {"source_id": "1", "target_id": "A", "verdict": "NO MATCH", "confidence": 0.1, "parse_error": False},
    {"source_id": "2", "target_id": "B", "verdict": "NO MATCH", "confidence": 0.1, "parse_error": False},
  ]
  m = compute_er_metrics(preds, gt)
  assert m["tp"] == 0
  assert m["fn"] == 2
  assert m["precision"] == 0.0
  assert m["recall"] == 0.0
  assert m["f1"] == 0.0


def test_mixed_predictions():
  """One correct match, one missed match, one false positive yields P=R=0.5."""
  gt = {("1", "A"), ("2", "B")}
  preds = [
    {"source_id": "1", "target_id": "A", "verdict": "MATCH", "confidence": 0.9, "parse_error": False},
    {"source_id": "2", "target_id": "B", "verdict": "NO MATCH", "confidence": 0.2, "parse_error": False},
    {"source_id": "3", "target_id": "C", "verdict": "MATCH", "confidence": 0.7, "parse_error": False},
  ]
  m = compute_er_metrics(preds, gt)
  assert m["tp"] == 1
  assert m["fp"] == 1
  assert m["fn"] == 1
  assert m["precision"] == 0.5
  assert m["recall"] == 0.5


def test_token_and_runtime_tracking():
  """Token and runtime totals are passed through and per-pair averages computed correctly."""
  gt = set()
  preds = [
    {"source_id": "1", "target_id": "A", "verdict": "NO MATCH", "confidence": 0.1, "parse_error": False},
    {"source_id": "2", "target_id": "B", "verdict": "NO MATCH", "confidence": 0.1, "parse_error": False},
  ]
  m = compute_er_metrics(preds, gt, total_tokens=600, elapsed_seconds=10.0)
  assert m["total_tokens"] == 600
  assert m["tokens_per_pair"] == 300.0
  assert m["total_runtime"] == 10.0
  assert m["runtime_per_pair"] == 5.0


def test_parse_error_count():
  """Parse errors are counted separately so we can track LLM response quality."""
  gt = set()
  preds = [
    {"source_id": "1", "target_id": "A", "verdict": "NO MATCH", "confidence": 0.0, "parse_error": True},
    {"source_id": "2", "target_id": "B", "verdict": "NO MATCH", "confidence": 0.0, "parse_error": False},
    {"source_id": "3", "target_id": "C", "verdict": "NO MATCH", "confidence": 0.0, "parse_error": True},
  ]
  m = compute_er_metrics(preds, gt)
  assert m["parse_errors"] == 2


def test_empty_predictions():
  """Empty predictions list returns zero metrics without division errors."""
  gt = {("1", "A")}
  m = compute_er_metrics([], gt)
  assert m["total_pairs"] == 0
  assert m["precision"] == 0.0
  assert m["recall"] == 0.0
  assert m["f1"] == 0.0
  assert m["tokens_per_pair"] == 0.0


def test_string_verdict_match():
  """String verdict 'MATCH' is treated as a positive prediction."""
  gt = {("1", "A")}
  preds = [
    {"source_id": "1", "target_id": "A", "verdict": "MATCH", "confidence": 0.9, "parse_error": False},
    {"source_id": "2", "target_id": "B", "verdict": "NO MATCH", "confidence": 0.8, "parse_error": False},
  ]
  m = compute_er_metrics(preds, gt)
  assert m["tp"] == 1
  assert m["fp"] == 0
  assert m["fn"] == 0
  assert m["predicted_matches"] == 1


def test_string_verdict_no_match_not_counted():
  """String verdict 'NO MATCH' is NOT treated as a positive prediction."""
  gt = {("1", "A")}
  preds = [
    {"source_id": "1", "target_id": "A", "verdict": "NO MATCH", "confidence": 0.8, "parse_error": False},
  ]
  m = compute_er_metrics(preds, gt)
  assert m["predicted_matches"] == 0
  assert m["fn"] == 1
