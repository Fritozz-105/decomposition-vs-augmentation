"""Shared evaluation metrics for entity resolution pipelines."""

import pandas as pd


def build_ground_truth_set(pairs_df: pd.DataFrame) -> set[tuple[str, str]]:
  """
  Build a set of ground truth match pairs from a labeled pairs DataFrame.

  Args:
    pairs_df: DataFrame with 'ltable_id', 'rtable_id', and 'label' columns.

  Returns:
    Set of (ltable_id, rtable_id) tuples where label == 1.
  """
  matches = pairs_df[pairs_df["label"] == 1]
  return {
    (str(row["ltable_id"]), str(row["rtable_id"]))
    for _, row in matches.iterrows()
  }


def compute_er_metrics(
  predictions: list[dict],
  ground_truth: set[tuple[str, str]],
  total_tokens: int = 0,
  elapsed_seconds: float = 0.0,
) -> dict:
  """
  Compute precision, recall, F1, token cost, and runtime metrics.

  Args:
    predictions: List of dicts with source_id, target_id, verdict, confidence, parse_error.
    ground_truth: Set of (source_id, target_id) true match tuples.
    total_tokens: Total tokens consumed by the pipeline.
    elapsed_seconds: Total wall-clock time in seconds.

  Returns:
    Dict with all computed metrics.
  """
  tp = fp = fn = 0
  parse_errors = 0
  predicted_matches = 0

  predicted_match_set: set[tuple[str, str]] = set()

  for pred in predictions:
    pair = (str(pred["source_id"]), str(pred["target_id"]))
    if pred.get("parse_error", False):
      parse_errors += 1

    verdict = pred["verdict"]
    is_match = isinstance(verdict, str) and verdict.upper() == "MATCH"
    if is_match:
      predicted_matches += 1
      predicted_match_set.add(pair)
      if pair in ground_truth:
        tp += 1
      else:
        fp += 1

  # Count false negatives: ground truth pairs in candidate set but not predicted as match
  candidate_pairs = {(str(p["source_id"]), str(p["target_id"])) for p in predictions}
  total_in_candidates = ground_truth & candidate_pairs
  fn = len(total_in_candidates) - tp

  precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

  total_pairs = len(predictions)
  tokens_per_pair = total_tokens / total_pairs if total_pairs > 0 else 0.0
  runtime_per_pair = elapsed_seconds / total_pairs if total_pairs > 0 else 0.0

  return {
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "total_tokens": total_tokens,
    "tokens_per_pair": tokens_per_pair,
    "total_runtime": elapsed_seconds,
    "runtime_per_pair": runtime_per_pair,
    "total_pairs": total_pairs,
    "predicted_matches": predicted_matches,
    "parse_errors": parse_errors,
    "gt_in_candidates": len(total_in_candidates),
  }


def print_metrics(metrics: dict, pipeline_name: str = "Pipeline") -> None:
  """Print formatted evaluation metrics to console."""
  print(f"\n{'-' * 50}")
  print(f"  {pipeline_name} - Evaluation Results")
  print(f"{'.' * 50}")
  print(f"  Pairs evaluated:     {metrics['total_pairs']}")
  print(f"  Predicted matches:   {metrics['predicted_matches']}")
  print(f"  GT matches in set:   {metrics['gt_in_candidates']}")
  print(f"  Parse errors:        {metrics['parse_errors']}")
  print(f"{'.' * 50}")
  print(f"  True Positives:      {metrics['tp']}")
  print(f"  False Positives:     {metrics['fp']}")
  print(f"  False Negatives:     {metrics['fn']}")
  print(f"{'.' * 50}")
  print(f"  Precision:           {metrics['precision']:.4f}")
  print(f"  Recall:              {metrics['recall']:.4f}")
  print(f"  F1 Score:            {metrics['f1']:.4f}")
  print(f"{'.' * 50}")
  print(f"  Total tokens:        {metrics['total_tokens']}")
  print(f"  Tokens/pair:         {metrics['tokens_per_pair']:.1f}")
  print(f"  Total runtime:       {metrics['total_runtime']:.1f}s")
  print(f"  Runtime/pair:        {metrics['runtime_per_pair']:.3f}s")
  print(f"{'-' * 50}\n")
