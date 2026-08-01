"""Results output for evaluation: JSON and Markdown."""

import json
from pathlib import Path


def save_results_json(results: dict, path: Path = Path("data/evaluation_results.json")) -> None:
  """Save full evaluation results as JSON."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    json.dump(results, f, indent=2)


def _fmt(mean: float, std: float) -> str:
  """Format a metric as 'X.XXXX +/- X.XXXX'."""
  return f"{mean:.4f} +/- {std:.4f}"


def _fmt_runtime(mean: float, std: float) -> str:
  """Format runtime as 'X.XX +/- X.XX'."""
  return f"{mean:.2f} +/- {std:.2f}"


def _fmt_int(mean: float, std: float) -> str:
  """Format integer-valued metric."""
  return f"{int(mean):,} +/- {int(std):,}"


def _pipeline_col(p: dict, key: str, fmt_fn) -> str:
  """Get formatted column value for a pipeline."""
  return fmt_fn(p["mean"][key], p["std"][key])


def generate_results_markdown(results: dict, path: Path = Path("results/preliminary_results.md")) -> None:
  """Generate RESULTS.md with evaluation comparison tables."""
  p = results["pipelines"]
  mag, sllm, ma = p["magellan"], p["single_llm"], p["multi_agent"]

  lines = [
    "# Preliminary Results",
    "",
    "## Dataset Summary",
    "| Metric | Value |",
    "|--------|-------|",
    f"| Test pairs evaluated | {results['num_test_pairs']:,} |",
    "| Data sources | 2 (Abt, Buy) |",
    f"| Ground truth matches | {results['ground_truth_matches']} |",
    f"| Evaluation date | {results['evaluation_date']} |",
    f"| Runs per pipeline | {results['num_runs']} |",
    "",
    "## Pipeline Comparison",
    "| Metric | Magellan (RF) | Single LLM | Multi-Agent LLM |",
    "|--------|--------------|------------|-----------------|",
  ]

  for key, label, fn in [
    ("precision", "Precision", _fmt),
    ("recall", "Recall", _fmt),
    ("f1", "F1", _fmt),
    ("total_runtime", "Runtime (s)", _fmt_runtime),
    ("total_tokens", "Total tokens", _fmt_int),
    ("tokens_per_pair", "Tokens/pair", _fmt_runtime),
    ("parse_errors", "Parse errors", _fmt_int),
  ]:
    row = f"| {label} | {_pipeline_col(mag, key, fn)} | {_pipeline_col(sllm, key, fn)} | {_pipeline_col(ma, key, fn)} |"
    lines.append(row)

  lines += [
    "",
    "## Quality Metrics",
    "| Metric | Magellan (RF) | Single LLM | Multi-Agent LLM |",
    "|--------|--------------|------------|-----------------|",
  ]

  for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
    row = f"| {label} | {mag['mean'][key]:.4f} | {sllm['mean'][key]:.4f} | {ma['mean'][key]:.4f} |"
    lines.append(row)

  lines += [""]

  with open(path, "w") as f:
    f.write("\n".join(lines))
