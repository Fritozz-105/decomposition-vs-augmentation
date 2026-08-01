"""Evaluation runner for all three ER pipelines."""

from datetime import date
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from src.evaluation.metrics import build_ground_truth_set, compute_er_metrics
from src.pipelines.magellan import run_magellan_pipeline
from src.pipelines.single_llm import run_single_llm_pipeline
from src.pipelines.multi_agent import run_multi_agent_pipeline
from src.utils import create_openai_client


MAX_RUNS = 5
MAGELLAN_SEEDS = [2, 31, 76, 90, 101]  # Predefined seeds for reproducibility


class EvalData(TypedDict):
  """Expected shape of the data dict passed to evaluation functions."""
  source_df: pd.DataFrame
  target_df: pd.DataFrame
  train_pairs_df: pd.DataFrame
  valid_pairs_df: pd.DataFrame
  test_pairs_df: pd.DataFrame


def prepare_test_candidates(test_pairs_df: pd.DataFrame) -> pd.DataFrame:
  """Convert test pairs to candidate format (source_id, target_id as strings)."""
  return pd.DataFrame({
    "source_id": test_pairs_df["ltable_id"].astype(str),
    "target_id": test_pairs_df["rtable_id"].astype(str),
  })


def _aggregate_metrics(runs: list[dict]) -> dict:
  """Compute mean and std across multiple metrics dicts."""
  keys = ["precision", "recall", "f1", "total_runtime", "runtime_per_pair",
          "total_tokens", "tokens_per_pair", "parse_errors"]
  mean = {k: float(np.mean([r[k] for r in runs])) for k in keys}
  std = {k: float(np.std([r[k] for r in runs])) for k in keys}
  return {"mean": mean, "std": std}


def _run_magellan_n_times(data: EvalData, candidates, ground_truth, num_runs):
  """Run Magellan pipeline num_runs times with different seeds."""
  runs = []
  seeds = MAGELLAN_SEEDS[:num_runs]
  for i, seed in enumerate(seeds):
    predictions, tokens, elapsed = run_magellan_pipeline(
      candidate_pairs=candidates,
      source_df=data["source_df"],
      target_df=data["target_df"],
      train_pairs_df=data["train_pairs_df"],
      valid_pairs_df=data["valid_pairs_df"],
      random_state=seed,
    )
    metrics = compute_er_metrics(predictions, ground_truth, tokens, elapsed)
    runs.append(metrics)
    print(f"- Magellan run {i + 1}/{num_runs} done: F1={metrics['f1']:.4f} in {elapsed:.1f}s")
  return runs


def _run_llm_pipeline_n_times(name, pipeline_fn, pipeline_kwargs, ground_truth, num_runs, base_cache_dir: Path):
  """Run an LLM pipeline num_runs times with isolated cache per run."""
  runs = []
  for i in range(num_runs):
    run_cache_dir = base_cache_dir / f"eval_run_{i}"
    pipeline_kwargs["cache_dir"] = run_cache_dir
    predictions, tokens, elapsed = pipeline_fn(**pipeline_kwargs)
    metrics = compute_er_metrics(predictions, ground_truth, tokens, elapsed)
    runs.append(metrics)
    print(f"- {name} run {i + 1}/{num_runs} done: F1={metrics['f1']:.4f} in {elapsed:.1f}s")
  return runs


def run_full_evaluation(data: EvalData, num_runs: int = 5) -> dict:
  """Orchestrate full evaluation of all 3 pipelines on test-split pairs."""
  if num_runs < 1 or num_runs > MAX_RUNS:
    raise ValueError(f"num_runs must be between 1 and {MAX_RUNS}, got {num_runs}")

  candidates = prepare_test_candidates(data["test_pairs_df"])
  ground_truth = build_ground_truth_set(data["test_pairs_df"])
  client = create_openai_client()

  print(f"\nEvaluating {len(candidates)} test pairs, {num_runs} runs each\n")

  print("Running Magellan...")
  mag_runs = _run_magellan_n_times(data, candidates, ground_truth, num_runs)

  print("\nRunning Single LLM...")
  sllm_kwargs = {
    "client": client, "candidate_pairs": candidates,
    "source_df": data["source_df"], "target_df": data["target_df"],
  }
  sllm_runs = _run_llm_pipeline_n_times(
    "Single LLM", run_single_llm_pipeline, sllm_kwargs,
    ground_truth, num_runs, base_cache_dir=Path("data/cache/single_llm"),
  )

  print("\nRunning Multi-Agent...")
  multia_kwargs = {
    "candidate_pairs": candidates,
    "source_df": data["source_df"], "target_df": data["target_df"],
  }
  multia_runs = _run_llm_pipeline_n_times(
    "Multi-Agent", run_multi_agent_pipeline, multia_kwargs,
    ground_truth, num_runs, base_cache_dir=Path("data/cache/multi_agent"),
  )

  return {
    "evaluation_date": str(date.today()),
    "num_runs": num_runs,
    "num_test_pairs": len(candidates),
    "ground_truth_matches": len(ground_truth),
    "pipelines": {
      "magellan": {"runs": mag_runs, **_aggregate_metrics(mag_runs)},
      "single_llm": {"runs": sllm_runs, **_aggregate_metrics(sllm_runs)},
      "multi_agent": {"runs": multia_runs, **_aggregate_metrics(multia_runs)},
    },
  }
