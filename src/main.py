"""Entry point for entity resolution pipelines."""

import argparse
import logging
from pathlib import Path
import pandas as pd
from src.blocking import build_candidate_pairs, save_candidate_pairs
from src.data import load_abt_buy
from src.evaluation import (
  build_ground_truth_set, compute_er_metrics, print_metrics,
  prepare_test_candidates, run_full_evaluation, save_results_json, generate_results_markdown,
)
from src.pipelines.single_llm import run_single_llm_pipeline
from src.pipelines.multi_agent import run_multi_agent_pipeline
from src.pipelines.magellan import run_magellan_pipeline, tune_rf_params
from src.utils import create_openai_client


def run_blocking() -> None:
  """Load data, run TF-IDF blocking, save candidate pairs."""
  candidate_pairs_path = Path("data/candidate_pairs.csv")
  print("Loading Abt-Buy dataset...")
  data = load_abt_buy()

  source_df = data["source_df"]
  target_df = data["target_df"]
  print(f"Source records: {len(source_df)}")
  print(f"Target records: {len(target_df)}")

  all_pairs = pd.concat([
    data["train_pairs_df"],
    data["valid_pairs_df"],
    data["test_pairs_df"],
  ])
  total_matches = all_pairs["label"].sum()
  print(f"Ground truth matches: {total_matches}")

  print("\nRunning TF-IDF blocking (threshold=0.2)...")
  candidates = build_candidate_pairs(source_df, target_df, threshold=0.2)
  print(f"Candidate pairs generated: {len(candidates)}")
  print(f"Similarity range: [{candidates['similarity_score'].min():.4f}, {candidates['similarity_score'].max():.4f}]")
  print(f"Mean similarity: {candidates['similarity_score'].mean():.4f}")

  blocked_set = set(zip(
    candidates["source_id"].astype(str),
    candidates["target_id"].astype(str),
  ))
  true_matches = all_pairs[all_pairs["label"] == 1]
  found = sum(
    1 for _, row in true_matches.iterrows()
    if (str(row["ltable_id"]), str(row["rtable_id"])) in blocked_set
  )
  recall = found / len(true_matches)
  print(f"\nBlocking recall: {found}/{len(true_matches)} = {recall:.4f}")

  save_candidate_pairs(candidates, candidate_pairs_path)
  print(f"\nCandidate pairs saved to {candidate_pairs_path}")


def run_single_llm(max_pairs: int | None = None) -> None:
  """Run the single-LLM entity resolution pipeline on test-split pairs."""
  data = load_abt_buy()

  candidates = prepare_test_candidates(data["test_pairs_df"])
  ground_truth = build_ground_truth_set(data["test_pairs_df"])
  print(f"Test pairs: {len(candidates)}, ground truth matches: {len(ground_truth)}")

  client = create_openai_client()

  predictions, total_tokens, elapsed = run_single_llm_pipeline(
    client=client,
    candidate_pairs=candidates,
    source_df=data["source_df"],
    target_df=data["target_df"],
    max_pairs=max_pairs,
  )

  metrics = compute_er_metrics(predictions, ground_truth, total_tokens, elapsed)
  print_metrics(metrics, pipeline_name="Single LLM (GPT-oss-120b)")


def run_multi_agent(max_pairs: int | None = None, verbose: bool = False) -> None:
  """Run the multi-agent entity resolution pipeline on test-split pairs."""
  data = load_abt_buy()

  candidates = prepare_test_candidates(data["test_pairs_df"])
  ground_truth = build_ground_truth_set(data["test_pairs_df"])
  print(f"Test pairs: {len(candidates)}, ground truth matches: {len(ground_truth)}")

  predictions, total_tokens, elapsed = run_multi_agent_pipeline(
    candidate_pairs=candidates,
    source_df=data["source_df"],
    target_df=data["target_df"],
    max_pairs=max_pairs,
    verbose=verbose,
  )

  metrics = compute_er_metrics(predictions, ground_truth, total_tokens, elapsed)
  print_metrics(metrics, pipeline_name="Multi-Agent LLM (GPT-oss-120b)")


def run_magellan_tune() -> None:
  """Run GridSearchCV to find best RF hyperparameters and save to JSON."""
  data = load_abt_buy()

  result = tune_rf_params(
    source_df=data["source_df"],
    target_df=data["target_df"],
    train_pairs_df=data["train_pairs_df"],
    valid_pairs_df=data["valid_pairs_df"],
  )

  print(f"Best params: {result['best_params']}")
  print(f"Validation F1: {result['valid_f1']}")


def run_evaluation(num_runs: int = 5) -> None:
  """Run full evaluation of all pipelines on test-split pairs."""
  data = load_abt_buy()
  results = run_full_evaluation(data, num_runs=num_runs)
  save_results_json(results)
  generate_results_markdown(results)
  print("\nResults saved to data/evaluation_results.json and preliminary_results.md")


def run_magellan(max_pairs: int | None = None, debug: bool = False) -> None:
  """Run the Magellan (Random Forest) entity resolution pipeline on test-split pairs."""
  data = load_abt_buy()

  candidates = prepare_test_candidates(data["test_pairs_df"])
  ground_truth = build_ground_truth_set(data["test_pairs_df"])
  print(f"Test pairs: {len(candidates)}, ground truth matches: {len(ground_truth)}")

  predictions, total_tokens, elapsed = run_magellan_pipeline(
    candidate_pairs=candidates,
    source_df=data["source_df"],
    target_df=data["target_df"],
    train_pairs_df=data["train_pairs_df"],
    valid_pairs_df=data["valid_pairs_df"],
    max_pairs=max_pairs,
    debug=debug,
  )

  metrics = compute_er_metrics(predictions, ground_truth, total_tokens, elapsed)
  print_metrics(metrics, pipeline_name="Magellan (Random Forest)")


def main() -> None:
  parser = argparse.ArgumentParser(description="Entity Resolution Pipelines")
  parser.add_argument(
    "pipeline",
    choices=["blocking", "single-llm", "multi-agent", "magellan", "magellan-tune", "evaluate-all"],
    help="Pipeline to run",
  )
  parser.add_argument(
    "--max-pairs",
    type=int,
    default=None,
    help="Limit number of candidate pairs to process. Default is all pairs.",
  )
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose logging",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Print debug info (features, sample data, importances) for Magellan pipeline",
  )
  parser.add_argument(
    "--runs",
    type=int,
    default=5,
    choices=range(1, 6),
    metavar="[1-5]",
    help="Number of evaluation runs per pipeline, 1-5 (for evaluate-all)",
  )
  args = parser.parse_args()

  if args.debug:
    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
  elif args.verbose:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  if args.pipeline == "blocking":
    run_blocking()
  elif args.pipeline == "single-llm":
    run_single_llm(max_pairs=args.max_pairs)
  elif args.pipeline == "multi-agent":
    run_multi_agent(max_pairs=args.max_pairs, verbose=args.verbose)
  elif args.pipeline == "magellan-tune":
    run_magellan_tune()
  elif args.pipeline == "magellan":
    run_magellan(max_pairs=args.max_pairs, debug=args.debug)
  elif args.pipeline == "evaluate-all":
    run_evaluation(num_runs=args.runs)


if __name__ == "__main__":
  main()
