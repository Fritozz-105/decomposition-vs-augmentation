"""Tests for src/pipelines/single_llm/pipeline.py."""

import json
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from src.pipelines.single_llm.pipeline import (
  _load_cache,
  _save_cache,
  run_single_llm_pipeline,
)
from src.utils import build_product_lookup


def _mock_source_df():
  return pd.DataFrame({
    "id": ["s1", "s2"],
    "name": ["Sony DVD Player", "Apple iPad"],
    "description": ["A dvd player", "A tablet"],
    "price": ["29.99", "499.00"],
  })


def _mock_target_df():
  return pd.DataFrame({
    "id": ["t1", "t2"],
    "name": ["Sony DVD", "iPad Air"],
    "description": ["DVD player", "Apple tablet"],
    "price": ["31.00", "479.00"],
  })


def _mock_candidates():
  return pd.DataFrame({
    "source_id": ["s1", "s2"],
    "target_id": ["t1", "t2"],
  })


def _make_mock_response(text, tokens=100):
  """Create a mock OpenAI chat completion response."""
  msg = MagicMock()
  msg.content = text
  choice = MagicMock()
  choice.message = msg
  usage = MagicMock()
  usage.total_tokens = tokens
  resp = MagicMock()
  resp.choices = [choice]
  resp.usage = usage
  return resp


def test_build_lookup():
  """Maps product IDs to their fields for O(1) access during pipeline iteration."""
  df = _mock_source_df()
  lookup = build_product_lookup(df)
  assert "s1" in lookup
  assert lookup["s1"]["name"] == "Sony DVD Player"
  assert lookup["s2"]["price"] == "499.00"


def test_cache_miss_returns_none(tmp_path):
  """Returns None for uncached pairs so the pipeline knows to call the API."""
  assert _load_cache(tmp_path, "x", "y") is None


def test_cache_roundtrip(tmp_path):
  """Saved predictions can be loaded back with identical data and token count."""
  prediction = {"source_id": "s1", "target_id": "t1", "verdict": True, "confidence": 0.9}
  _save_cache(tmp_path, "s1", "t1", prediction, 150)
  loaded = _load_cache(tmp_path, "s1", "t1")
  assert loaded is not None
  assert loaded["prediction"]["verdict"] is True
  assert loaded["tokens"] == 150


def test_pipeline_returns_predictions(tmp_path):
  """Processes all candidate pairs and returns parsed predictions with correct token totals."""
  client = MagicMock()
  match_resp = '{"verdict": "MATCH", "confidence": 0.9, "reasoning": "same"}'
  no_match_resp = '{"verdict": "NO MATCH", "confidence": 0.3, "reasoning": "different"}'
  client.chat.completions.create.side_effect = [
    _make_mock_response(match_resp, 100),
    _make_mock_response(no_match_resp, 120),
  ]

  preds, tokens, elapsed = run_single_llm_pipeline(
    client, _mock_candidates(), _mock_source_df(), _mock_target_df(),
    cache_dir=tmp_path,
  )
  assert len(preds) == 2
  assert preds[0]["verdict"] == "MATCH"
  assert preds[1]["verdict"] == "NO MATCH"
  assert tokens == 220


def test_pipeline_max_pairs(tmp_path):
  """Respects max_pairs limit to control API costs during development."""
  client = MagicMock()
  resp = '{"verdict": "NO MATCH", "confidence": 0.1, "reasoning": "no"}'
  client.chat.completions.create.return_value = _make_mock_response(resp)

  preds, _, _ = run_single_llm_pipeline(
    client, _mock_candidates(), _mock_source_df(), _mock_target_df(),
    max_pairs=1, cache_dir=tmp_path,
  )
  assert len(preds) == 1


def test_pipeline_uses_cache(tmp_path):
  """Second run serves all results from disk cache with zero API calls, saving tokens."""
  client = MagicMock()
  resp = '{"verdict": "MATCH", "confidence": 0.8, "reasoning": "cached"}'
  client.chat.completions.create.return_value = _make_mock_response(resp, 100)

  # First run, calls API
  run_single_llm_pipeline(
    client, _mock_candidates(), _mock_source_df(), _mock_target_df(),
    cache_dir=tmp_path,
  )
  assert client.chat.completions.create.call_count == 2

  # Second run, uses cache, no new API calls
  client.chat.completions.create.reset_mock()
  preds, _, _ = run_single_llm_pipeline(
    client, _mock_candidates(), _mock_source_df(), _mock_target_df(),
    cache_dir=tmp_path,
  )
  assert client.chat.completions.create.call_count == 0
  assert len(preds) == 2
