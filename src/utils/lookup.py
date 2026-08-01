"""Shared product lookup and label builders for entity resolution pipelines."""

import pandas as pd


def build_product_lookup(df: pd.DataFrame) -> dict[str, dict]:
  """Build a dict mapping product ID to its fields (name, description, price)."""
  return {
    str(row["id"]): {
      "name": str(row["name"]),
      "description": str(row["description"]),
      "price": str(row["price"]),
    }
    for _, row in df.iterrows()
  }


def build_label_series(pairs_df: pd.DataFrame) -> dict[tuple[str, str], int]:
  """Build a dict mapping (ltable_id, rtable_id) -> label."""
  return {
    (str(row["ltable_id"]), str(row["rtable_id"])): int(row["label"])
    for _, row in pairs_df.iterrows()
  }
