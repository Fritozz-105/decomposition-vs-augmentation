"""Tests for src/pipelines/magellan/features.py and pipeline.py."""

import numpy as np
import pandas as pd
import pytest

from src.pipelines.magellan.features import (
  FEATURE_NAMES,
  abs_norm,
  compute_features_for_pairs,
  compute_pair_features,
  cosine_tfidf,
  cosine_tokens,
  jaccard_qgrams,
  jaccard_tokens,
  lev_dist,
  lev_sim,
  monge_elkan,
  needleman_wunsch,
  parse_price,
  price_exact_match,
  smith_waterman,
)
from src.pipelines.magellan.pipeline import run_magellan_pipeline


# Fixtures
def _source_df():
  return pd.DataFrame({
    "id": ["s1", "s2", "s3"],
    "name": ["Sony DVD Player 100", "Apple iPad 4th Gen", "Samsung TV 50 inch"],
    "description": ["Region free dvd player", "Apple tablet device", "4K smart television"],
    "price": ["29.99", "499.00", "799.00"],
  })


def _target_df():
  return pd.DataFrame({
    "id": ["t1", "t2", "t3"],
    "name": ["Sony DVD Player", "iPad 4th Generation", "LG TV 55 inch"],
    "description": ["DVD player multi region", "Apple tablet", "OLED television"],
    "price": ["31.00", "479.00", "999.00"],
  })


def _candidate_pairs():
  return pd.DataFrame({
    "source_id": ["s1", "s2", "s3"],
    "target_id": ["t1", "t2", "t3"],
    "similarity_score": [0.8, 0.7, 0.4],
  })


def _train_pairs_df():
  # s1-t1 is a match, s3-t3 is not
  return pd.DataFrame({
    "ltable_id": ["s1", "s3"],
    "rtable_id": ["t1", "t3"],
    "label": [1, 0],
  })


def _valid_pairs_df():
  # s2-t2 is a match
  return pd.DataFrame({
    "ltable_id": ["s2"],
    "rtable_id": ["t2"],
    "label": [1],
  })


# jaccard_tokens
def test_jaccard_tokens_identical():
  assert jaccard_tokens("Sony DVD Player", "Sony DVD Player") == pytest.approx(1.0)


def test_jaccard_tokens_disjoint():
  assert jaccard_tokens("uf gators", "hivemind") == pytest.approx(0.0)


def test_jaccard_tokens_partial():
  score = jaccard_tokens("Sony DVD Player", "Sony Player Pro")
  assert 0 < score < 1


def test_jaccard_tokens_empty_strings():
  assert jaccard_tokens("", "") == pytest.approx(1.0)


def test_jaccard_tokens_one_empty():
  assert jaccard_tokens("", "apple") == pytest.approx(0.0)


# jaccard_qgrams (new)
def test_jaccard_qgrams_identical():
  assert jaccard_qgrams("hello", "hello") == pytest.approx(1.0)


def test_jaccard_qgrams_disjoint():
  assert jaccard_qgrams("abc", "xyz") == pytest.approx(0.0)


def test_jaccard_qgrams_partial():
  score = jaccard_qgrams("hello", "helly")
  assert 0 < score < 1


# cosine_tokens
def test_cosine_tokens_identical():
  assert cosine_tokens("Sony DVD Player", "Sony DVD Player") == pytest.approx(1.0)


def test_cosine_tokens_disjoint():
  assert cosine_tokens("uf gators", "hivemind") == pytest.approx(0.0)


def test_cosine_tokens_empty():
  assert cosine_tokens("", "hello") == pytest.approx(0.0)


# cosine_tfidf
def test_cosine_tfidf_identical():
  """TF-IDF cosine of identical strings is 1.0."""
  assert cosine_tfidf("Sony DVD Player review", "Sony DVD Player review") == pytest.approx(1.0)


def test_cosine_tfidf_disjoint():
  """TF-IDF cosine of completely different strings is 0.0."""
  assert cosine_tfidf("apple banana cherry", "xylophone zebra") == pytest.approx(0.0)


def test_cosine_tfidf_empty():
  """TF-IDF cosine returns 0.0 when either string is empty."""
  assert cosine_tfidf("", "hello world") == pytest.approx(0.0)
  assert cosine_tfidf("hello world", "") == pytest.approx(0.0)


def test_cosine_tfidf_partial_overlap():
  """TF-IDF cosine of partially overlapping strings is between 0 and 1."""
  score = cosine_tfidf("bluetooth wireless headset noise cancelling", "bluetooth wireless earbuds noise cancelling")
  assert 0.0 < score < 1.0


# lev_dist / lev_sim
def test_lev_dist_identical():
  assert lev_dist("abc", "abc") == 0


def test_lev_dist_one_edit():
  assert lev_dist("abc", "abd") == 1


def test_lev_sim_identical():
  assert lev_sim("hello", "hello") == pytest.approx(1.0)


def test_lev_sim_different():
  assert lev_sim("abc", "xyz") < 0.5


def test_lev_sim_case_insensitive():
  assert lev_sim("Hello", "hello") == pytest.approx(1.0)


# monge_elkan
def test_monge_elkan_identical():
  assert monge_elkan("Sony DVD", "Sony DVD") == pytest.approx(1.0)


def test_monge_elkan_partial():
  score = monge_elkan("Sony DVD Player", "Sony Player")
  assert 0 < score <= 1.0


# needleman_wunsch / smith_waterman
def test_needleman_wunsch_identical():
  score = needleman_wunsch("abc", "abc")
  assert score == pytest.approx(3.0)


def test_needleman_wunsch_different():
  score = needleman_wunsch("abc", "xyz")
  assert score == pytest.approx(0.0)


def test_smith_waterman_identical():
  score = smith_waterman("abc", "abc")
  assert score == pytest.approx(3.0)


def test_smith_waterman_partial():
  score = smith_waterman("xabcy", "abc")
  assert score == pytest.approx(3.0)


# parse_price
def test_parse_price_basic():
  assert parse_price("29.99") == pytest.approx(29.99)


def test_parse_price_dollar_sign():
  assert parse_price("$199.00") == pytest.approx(199.00)


def test_parse_price_comma():
  assert parse_price("1,299.99") == pytest.approx(1299.99)


def test_parse_price_invalid():
  assert parse_price("N/A") is None
  assert parse_price("") is None


# price_exact_match / abs_norm
def test_price_exact_match_equal():
  assert price_exact_match("100", "100") == pytest.approx(1.0)


def test_price_exact_match_different():
  assert price_exact_match("50", "100") == pytest.approx(0.0)


def test_abs_norm_equal():
  assert abs_norm("100", "100") == pytest.approx(1.0)


def test_abs_norm_different():
  assert abs_norm("50", "100") == pytest.approx(0.5)


def test_abs_norm_unparseable():
  assert abs_norm("N/A", "100") == pytest.approx(0.5)


def test_abs_norm_both_zero():
  assert abs_norm("0", "0") == pytest.approx(1.0)


# compute_pair_features
def test_compute_pair_features_shape():
  """compute_pair_features returns a list with len == FEATURE_NAMES (14)."""
  src = {"name": "Sony DVD", "description": "dvd player", "price": "29.99"}
  tgt = {"name": "Sony DVD Player", "description": "dvd", "price": "31.00"}
  features = compute_pair_features(src, tgt)
  assert len(features) == len(FEATURE_NAMES)
  assert len(features) == 14


def test_compute_pair_features_range():
  """Bounded features are in [0,1]; unbounded features are numeric."""
  src = {"name": "Apple iPad 4", "description": "tablet", "price": "499.00"}
  tgt = {"name": "iPad 4th Gen", "description": "Apple tablet", "price": "479.00"}
  features = compute_pair_features(src, tgt)
  # Unbounded: lev_dist(4), nw(6), sw(7), price_lev_dist(12)
  # NW can be negative (gap_cost=1), SW >= 0, lev_dist >= 0
  unbounded = {4, 6, 7, 12}
  for i, val in enumerate(features):
    if i not in unbounded:
      assert 0.0 <= val <= 1.0, f"Feature {FEATURE_NAMES[i]} out of [0,1]: {val}"
    else:
      assert isinstance(val, (int, float)), f"Feature {FEATURE_NAMES[i]} not numeric: {val}"


# compute_features_for_pairs
def test_compute_features_for_pairs_shape():
  src_lookup = {
    "s1": {"name": "Sony DVD", "description": "dvd", "price": "30"},
    "s2": {"name": "Apple iPad", "description": "tablet", "price": "500"},
  }
  tgt_lookup = {
    "t1": {"name": "Sony DVD Player", "description": "dvd player", "price": "32"},
    "t2": {"name": "iPad 4th", "description": "Apple tablet", "price": "480"},
  }
  pairs = pd.DataFrame({"source_id": ["s1", "s2"], "target_id": ["t1", "t2"]})
  X, ids = compute_features_for_pairs(pairs, src_lookup, tgt_lookup)
  assert X.shape == (2, len(FEATURE_NAMES))
  assert ids == [("s1", "t1"), ("s2", "t2")]


def test_compute_features_for_pairs_missing_id():
  src_lookup = {"s1": {"name": "Sony", "description": "d", "price": "10"}}
  tgt_lookup = {"t1": {"name": "Sony Player", "description": "d", "price": "10"}}
  pairs = pd.DataFrame({
    "source_id": ["s1", "s_MISSING"],
    "target_id": ["t1", "t1"],
  })
  X, ids = compute_features_for_pairs(pairs, src_lookup, tgt_lookup)
  assert X.shape[0] == 1
  assert ids == [("s1", "t1")]


def test_compute_features_for_pairs_empty():
  X, ids = compute_features_for_pairs(
    pd.DataFrame({"source_id": [], "target_id": []}), {}, {},
  )
  assert X.shape[0] == 0
  assert ids == []


# run_magellan_pipeline tests
def test_pipeline_returns_predictions():
  preds, tokens, elapsed = run_magellan_pipeline(
    candidate_pairs=_candidate_pairs(),
    source_df=_source_df(),
    target_df=_target_df(),
    train_pairs_df=_train_pairs_df(),
    valid_pairs_df=_valid_pairs_df(),

  )
  assert isinstance(preds, list)
  assert tokens == 0
  assert elapsed > 0


def test_pipeline_prediction_count():
  preds, _, _ = run_magellan_pipeline(
    candidate_pairs=_candidate_pairs(),
    source_df=_source_df(),
    target_df=_target_df(),
    train_pairs_df=_train_pairs_df(),
    valid_pairs_df=_valid_pairs_df(),

  )
  assert len(preds) == len(_candidate_pairs())


def test_pipeline_max_pairs():
  preds, _, _ = run_magellan_pipeline(
    candidate_pairs=_candidate_pairs(),
    source_df=_source_df(),
    target_df=_target_df(),
    train_pairs_df=_train_pairs_df(),
    valid_pairs_df=_valid_pairs_df(),

    max_pairs=1,
  )
  assert len(preds) == 1


def test_prediction_format():
  """Checks that predictions have required fields and valid values."""
  preds, _, _ = run_magellan_pipeline(
    candidate_pairs=_candidate_pairs(),
    source_df=_source_df(),
    target_df=_target_df(),
    train_pairs_df=_train_pairs_df(),
    valid_pairs_df=_valid_pairs_df(),

  )
  for pred in preds:
    assert "source_id" in pred
    assert "target_id" in pred
    assert pred["verdict"] in ("MATCH", "NO MATCH"), f"Bad verdict: {pred['verdict']}"
    assert isinstance(pred["confidence"], float)
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["parse_error"] is False


def test_prediction_ids_match_candidates():
  """Checks for source and target id set equality between predictions and candidate pairs."""
  preds, _, _ = run_magellan_pipeline(
    candidate_pairs=_candidate_pairs(),
    source_df=_source_df(),
    target_df=_target_df(),
    train_pairs_df=_train_pairs_df(),
    valid_pairs_df=_valid_pairs_df(),

  )
  cand_set = {
    (str(r["source_id"]), str(r["target_id"]))
    for _, r in _candidate_pairs().iterrows()
  }
  pred_set = {(p["source_id"], p["target_id"]) for p in preds}
  assert pred_set == cand_set


# find_best_threshold / load_best_params
from src.pipelines.magellan.tuning import find_best_threshold, load_best_params


def test_find_best_threshold_perfect_separation():
  """When probabilities perfectly separate classes, threshold yields F1=1.0."""
  y_true = np.array([1, 1, 0, 0])
  y_proba = np.array([0.9, 0.8, 0.1, 0.2])
  thresh, f1 = find_best_threshold(y_true, y_proba)
  assert f1 == pytest.approx(1.0)
  assert 0.2 < thresh <= 0.8


def test_find_best_threshold_returns_tuple():
  """find_best_threshold returns (threshold, f1) as floats."""
  y_true = np.array([1, 0, 0])
  y_proba = np.array([0.6, 0.3, 0.1])
  result = find_best_threshold(y_true, y_proba)
  assert isinstance(result, tuple)
  assert len(result) == 2
  assert isinstance(result[0], float)
  assert isinstance(result[1], float)


def test_find_best_threshold_all_negative():
  """When all labels are negative, best F1 is 0.0 and threshold defaults to 0.5."""
  y_true = np.array([0, 0, 0])
  y_proba = np.array([0.3, 0.5, 0.7])
  thresh, f1 = find_best_threshold(y_true, y_proba)
  assert f1 == pytest.approx(0.0)
  assert thresh == pytest.approx(0.5)


def test_load_best_params_missing_file():
  """Returns (None, 0.5) when JSON file does not exist."""
  from pathlib import Path
  params, thresh = load_best_params(Path("nonexistent_file_12345.json"))
  assert params is None
  assert thresh == 0.5
