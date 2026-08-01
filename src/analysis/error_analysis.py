"""Qualitative error analysis for entity resolution pipelines.

Loads cached predictions, compares against ground truth, and generates
detailed error reports showing why each mistake was made.

Usage:
  uv run python -m src.analysis.error_analysis --pipeline multi-agent --run 0
  uv run python -m src.analysis.error_analysis --pipeline single-llm --run 0
  uv run python -m src.analysis.error_analysis --pipeline multi-agent --run 0 --errors-only
"""

import argparse
import json
from pathlib import Path

from src.data.loader import load_abt_buy
from src.evaluation.metrics import build_ground_truth_set
from src.utils.lookup import build_product_lookup


CACHE_BASE = Path("data/cache")
OUTPUT_DIR = Path("resuts")


def load_cached_predictions(pipeline: str, run: int) -> list[dict]:
    """Load all cached prediction files for a pipeline run."""
    cache_dir = CACHE_BASE / pipeline.replace("-", "_") / f"eval_run_{run}"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    predictions = []
    for path in sorted(cache_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        predictions.append(data["prediction"])
    return predictions


def classify_predictions(
    predictions: list[dict],
    ground_truth: set[tuple[str, str]],
) -> dict[str, list[dict]]:
    """Classify predictions into TP, FP, FN, TN."""
    result = {"tp": [], "fp": [], "fn": [], "tn": []}

    predicted_pairs = set()
    for pred in predictions:
        pair = (str(pred["source_id"]), str(pred["target_id"]))
        predicted_pairs.add(pair)
        is_match = isinstance(pred.get("verdict"), str) and pred["verdict"].upper() == "MATCH"
        is_true = pair in ground_truth

        if is_match and is_true:
            result["tp"].append(pred)
        elif is_match and not is_true:
            result["fp"].append(pred)
        elif not is_match and is_true:
            result["fn"].append(pred)
        else:
            result["tn"].append(pred)

    return result


def detect_override(pred: dict) -> str:
    """Detect orchestrator override pattern for multi-agent predictions.

    Returns a label describing the decision pattern.
    """
    syn = pred.get("syntactic", {})
    sem = pred.get("semantic", {})
    if not syn or not sem:
        return "n/a"

    syn_match = isinstance(syn.get("verdict"), str) and syn["verdict"].upper() == "MATCH"
    sem_match = isinstance(sem.get("verdict"), str) and sem["verdict"].upper() == "MATCH"
    final_match = isinstance(pred.get("verdict"), str) and pred["verdict"].upper() == "MATCH"

    if syn_match and sem_match:
        if final_match:
            return "both-match → MATCH (consensus)"
        return "both-match → NO MATCH (override)"
    if not syn_match and not sem_match:
        if not final_match:
            return "both-no-match → NO MATCH (consensus)"
        return "both-no-match → MATCH (override)"
    if syn_match and not sem_match:
        if final_match:
            return "split (syn=match, sem=no) → MATCH (sided with syntactic)"
        return "split (syn=match, sem=no) → NO MATCH (sided with semantic)"
    # sem_match and not syn_match
    if final_match:
        return "split (syn=no, sem=match) → MATCH (sided with semantic)"
    return "split (syn=no, sem=match) → NO MATCH (sided with syntactic)"


def format_product_snippet(product: dict | None, max_len: int = 80) -> str:
    """Format a product as a short readable string."""
    if product is None:
        return "(missing)"
    name = product["name"][:max_len]
    price = product["price"]
    return f"{name} [${price}]"


def print_console_summary(
    pipeline: str,
    run: int,
    classified: dict[str, list[dict]],
) -> None:
    """Print a concise error summary to console."""
    tp, fp, fn, tn = len(classified["tp"]), len(classified["fp"]), len(classified["fn"]), len(classified["tn"])
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  Error Analysis: {pipeline} (run {run})")
    print(f"{'=' * 60}")
    print(f"  Total pairs: {total}  |  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    # Parse error count
    parse_errors = sum(1 for cat in classified.values() for p in cat if p.get("parse_error"))
    if parse_errors:
        print(f"  Parse errors: {parse_errors}")

    # Multi-agent override breakdown
    if pipeline == "multi-agent":
        print(f"\n  --- FP Breakdown ({fp} false positives) ---")
        _print_override_breakdown(classified["fp"])
        print(f"\n  --- FN Breakdown ({fn} false negatives) ---")
        _print_override_breakdown(classified["fn"])

    print(f"{'=' * 60}\n")


def _print_override_breakdown(errors: list[dict]) -> None:
    """Print override pattern counts for a list of errors."""
    patterns: dict[str, int] = {}
    for pred in errors:
        pattern = detect_override(pred)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"    {count:4d}  {pattern}")


def generate_markdown(
    pipeline: str,
    run: int,
    classified: dict[str, list[dict]],
    source_lookup: dict[str, dict],
    target_lookup: dict[str, dict],
) -> str:
    """Generate detailed markdown error report."""
    lines = [
        f"# Error Analysis: {pipeline} (run {run})\n",
    ]

    tp, fp, fn = len(classified["tp"]), len(classified["fp"]), len(classified["fn"])
    tn = len(classified["tn"])
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total pairs | {total} |")
    lines.append(f"| TP | {tp} |")
    lines.append(f"| FP | {fp} |")
    lines.append(f"| FN | {fn} |")
    lines.append(f"| TN | {tn} |")
    lines.append(f"| Precision | {precision:.4f} |")
    lines.append(f"| Recall | {recall:.4f} |")
    lines.append(f"| F1 | {f1:.4f} |")
    lines.append("")

    if pipeline == "multi-agent":
        lines.append("## Override Pattern Breakdown\n")
        lines.append("### False Positives\n")
        _append_override_table(lines, classified["fp"])
        lines.append("\n### False Negatives\n")
        _append_override_table(lines, classified["fn"])
        lines.append("")

    # Detailed errors
    lines.append("## False Positives (predicted MATCH, actual NO MATCH)\n")
    _append_error_details(lines, classified["fp"], source_lookup, target_lookup, pipeline)

    lines.append("## False Negatives (predicted NO MATCH, actual MATCH)\n")
    _append_error_details(lines, classified["fn"], source_lookup, target_lookup, pipeline)

    return "\n".join(lines)


def _append_override_table(lines: list[str], errors: list[dict]) -> None:
    """Append an override pattern frequency table."""
    patterns: dict[str, int] = {}
    for pred in errors:
        pattern = detect_override(pred)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    if not patterns:
        lines.append("None\n")
        return
    lines.append("| Count | Pattern |")
    lines.append("|------:|---------|")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        lines.append(f"| {count} | {pattern} |")
    lines.append("")


def _append_error_details(
    lines: list[str],
    errors: list[dict],
    source_lookup: dict[str, dict],
    target_lookup: dict[str, dict],
    pipeline: str,
) -> None:
    """Append detailed per-error information."""
    if not errors:
        lines.append("None\n")
        return

    for i, pred in enumerate(errors, 1):
        sid, tid = str(pred["source_id"]), str(pred["target_id"])
        src = source_lookup.get(sid)
        tgt = target_lookup.get(tid)

        lines.append(f"### Error {i}: pair ({sid}, {tid})\n")
        lines.append(f"- **Source (Abt):** {format_product_snippet(src, 120)}")
        lines.append(f"- **Target (Buy):** {format_product_snippet(tgt, 120)}")
        lines.append(f"- **Final verdict:** {pred.get('verdict')} (confidence: {pred.get('confidence')})")
        lines.append(f"- **Reasoning:** {pred.get('reasoning')}")

        if pred.get("parse_error"):
            lines.append(f"- **Parse error:** Yes")

        if pipeline == "multi-agent":
            syn = pred.get("syntactic", {})
            sem = pred.get("semantic", {})
            lines.append(f"- **Override pattern:** {detect_override(pred)}")
            lines.append(f"- **Syntactic agent:** {syn.get('verdict')} (conf: {syn.get('confidence')})")
            lines.append(f"  - {syn.get('reasoning')}")
            lines.append(f"- **Semantic agent:** {sem.get('verdict')} (conf: {sem.get('confidence')})")
            lines.append(f"  - {sem.get('reasoning')}")
            if syn.get("parse_error") or sem.get("parse_error"):
                lines.append(f"  - Agent parse errors: syn={syn.get('parse_error')}, sem={sem.get('parse_error')}")

        lines.append("")


def main():
    parser = argparse.ArgumentParser(description="Qualitative error analysis for ER pipelines")
    parser.add_argument("--pipeline", required=True, choices=["multi-agent", "single-llm"])
    parser.add_argument("--run", type=int, default=0, help="Eval run index (default: 0)")
    parser.add_argument("--errors-only", action="store_true", help="Only show FP and FN in markdown")
    args = parser.parse_args()

    print(f"Loading dataset...")
    data = load_abt_buy()
    ground_truth = build_ground_truth_set(data["test_pairs_df"])
    source_lookup = build_product_lookup(data["source_df"])
    target_lookup = build_product_lookup(data["target_df"])

    print(f"Loading cached predictions for {args.pipeline} run {args.run}...")
    predictions = load_cached_predictions(args.pipeline, args.run)
    print(f"  Loaded {len(predictions)} predictions")

    classified = classify_predictions(predictions, ground_truth)
    print_console_summary(args.pipeline, args.run, classified)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"error_analysis_{args.pipeline.replace('-', '_')}_run{args.run}.md"
    md = generate_markdown(args.pipeline, args.run, classified, source_lookup, target_lookup)
    out_path.write_text(md)
    print(f"Detailed report written to: {out_path}")


if __name__ == "__main__":
    main()
