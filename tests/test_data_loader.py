"""Tests for Abt-Buy dataset loader."""

import pandas as pd
from src.data import load_abt_buy

# Load once for all tests in this module
_data = load_abt_buy()


def test_load_returns_all_dataframes():
  """Verifies that the loader returns all expected DataFrames."""
  expected_keys = {"source_df", "target_df", "train_pairs_df", "valid_pairs_df", "test_pairs_df"}
  assert set(_data.keys()) == expected_keys


def test_source_df_columns():
  """Verifies the columns of the source DataFrame."""
  assert list(_data["source_df"].columns) == ["id", "name", "description", "price"]


def test_target_df_columns():
  """Verifies the columns of the target DataFrame."""
  assert list(_data["target_df"].columns) == ["id", "name", "description", "price"]


def test_pairs_columns():
  """Verifies the columns of the pairs DataFrames."""
  for key in ["train_pairs_df", "valid_pairs_df", "test_pairs_df"]:
    assert list(_data[key].columns) == ["ltable_id", "rtable_id", "label"]


def test_source_row_count():
  """Verifies the row count of the source DataFrame."""
  assert len(_data["source_df"]) == 1081


def test_target_row_count():
  """Verifies the row count of the target DataFrame."""
  assert len(_data["target_df"]) == 1092


def test_train_pairs_row_count():
  """Verifies the row count of the training pairs DataFrame."""
  assert len(_data["train_pairs_df"]) == 5743


def test_no_null_names():
  """Verifies that there are no null values in the 'name' column of both DataFrames."""
  assert _data["source_df"]["name"].notna().all()
  assert _data["target_df"]["name"].notna().all()
