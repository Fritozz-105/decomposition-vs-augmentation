"""
Agent-agreement and orchestrator-arbitration analysis across all five
multi-agent runs, computed from the released prediction caches.

Produces the counts quoted in the Discussion:
  - decisions where the two reviewer agents agree vs disagree
  - orchestrator overrides of an agent consensus (and how many were correct)
  - arbitration accuracy on disagreements vs the fixed policy of always
    following the syntactic agent
  - multi-agent false negatives arising in arbitration cases
  - per-run and aggregate F1 for each reviewer agent alone

Outputs:
  - results/arbitration_analysis.md

Run with: uv run python scripts/analyze_arbitration.py
"""

import json
import statistics
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTI_AGENT_ROOT = PROJECT_ROOT / "results" / "multi_agent"
TEST_CSV = PROJECT_ROOT / "data" / "raw" / "test.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "arbitration_analysis.md"
N_RUNS = 5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict[tuple[str, str], int]:
    df = pd.read_csv(path)
    return {
        (str(row["ltable_id"]), str(row["rtable_id"])): int(row["label"])
        for _, row in df.iterrows()
    }


def load_run(cache_dir: Path) -> dict[tuple[str, str], dict]:
    """Load one multi-agent run: final verdict plus both agent verdicts."""
    preds = {}
    for f in cache_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        p = data["prediction"]
        syn = p.get("syntactic", {}) or {}
        sem = p.get("semantic", {}) or {}
        preds[(str(p["source_id"]), str(p["target_id"]))] = {
            "verdict": p["verdict"],
            "syn_verdict": syn.get("verdict", "NO MATCH"),
            "sem_verdict": sem.get("verdict", "NO MATCH"),
        }
    return preds


def verdict_to_int(v: str) -> int:
    return 1 if str(v).upper() == "MATCH" else 0


def f1_from_preds(
    preds: dict[tuple[str, str], int], gt: dict[tuple[str, str], int]
) -> float:
    tp = sum(1 for k, p in preds.items() if p == 1 and gt[k] == 1)
    fp = sum(1 for k, p in preds.items() if p == 1 and gt[k] == 0)
    fn = sum(1 for k, p in preds.items() if p == 0 and gt[k] == 1)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(gt: dict[tuple[str, str], int], runs: list[dict]) -> dict:
    total = agree = disagree = 0
    overrides = override_correct = 0
    orch_correct = syn_policy_correct = 0
    fn_total = fn_in_disagreement = 0
    syn_f1s: list[float] = []
    sem_f1s: list[float] = []

    for preds in runs:
        syn_preds: dict[tuple[str, str], int] = {}
        sem_preds: dict[tuple[str, str], int] = {}
        for key, rec in preds.items():
            if key not in gt:
                continue
            label = gt[key]
            final = verdict_to_int(rec["verdict"])
            syn = verdict_to_int(rec["syn_verdict"])
            sem = verdict_to_int(rec["sem_verdict"])
            syn_preds[key] = syn
            sem_preds[key] = sem

            total += 1
            if final == 0 and label == 1:
                fn_total += 1
            if syn == sem:
                agree += 1
                if final != syn:
                    overrides += 1
                    if final == label:
                        override_correct += 1
            else:
                disagree += 1
                if final == label:
                    orch_correct += 1
                if syn == label:
                    syn_policy_correct += 1
                if final == 0 and label == 1:
                    fn_in_disagreement += 1

        syn_f1s.append(f1_from_preds(syn_preds, gt))
        sem_f1s.append(f1_from_preds(sem_preds, gt))

    return {
        "total_decisions": total,
        "agents_agree": agree,
        "agents_disagree": disagree,
        "overrides_of_consensus": overrides,
        "overrides_correct": override_correct,
        "orchestrator_correct_on_disagreements": orch_correct,
        "follow_syntactic_correct_on_disagreements": syn_policy_correct,
        "multi_agent_false_negatives": fn_total,
        "false_negatives_in_disagreements": fn_in_disagreement,
        "syntactic_f1_per_run": syn_f1s,
        "semantic_f1_per_run": sem_f1s,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(stats: dict) -> str:
    d = stats["agents_disagree"]
    syn_f1s = stats["syntactic_f1_per_run"]
    sem_f1s = stats["semantic_f1_per_run"]
    lines = [
        "# Agent Agreement and Orchestrator Arbitration (all 5 runs)",
        "",
        "Generated by `scripts/analyze_arbitration.py` from the released",
        "multi-agent caches under `results/multi_agent/eval_run_{0..4}`.",
        "",
        "## Agent agreement",
        "",
        f"- Decisions analyzed: **{stats['total_decisions']}** "
        f"(1,916 pairs x {N_RUNS} runs)",
        f"- Agents agree: **{stats['agents_agree']}**",
        f"- Agents disagree: **{d}**",
        "",
        "## Orchestrator behavior on agent consensus",
        "",
        f"- Overrides of a unanimous agent verdict: "
        f"**{stats['overrides_of_consensus']}** "
        f"({stats['overrides_correct']} correct)",
        "",
        "## Arbitration on disagreements",
        "",
        f"- Orchestrator correct: "
        f"**{stats['orchestrator_correct_on_disagreements']}/{d}** "
        f"({stats['orchestrator_correct_on_disagreements'] / d:.1%})",
        f"- Always-follow-syntactic policy correct: "
        f"**{stats['follow_syntactic_correct_on_disagreements']}/{d}** "
        f"({stats['follow_syntactic_correct_on_disagreements'] / d:.1%})",
        f"- Multi-agent false negatives (all runs): "
        f"**{stats['multi_agent_false_negatives']}**, of which "
        f"**{stats['false_negatives_in_disagreements']}** arise in "
        "arbitration cases",
        "",
        "## Per-agent standalone F1 (agent verdict scored directly)",
        "",
        "| Run | Syntactic | Semantic |",
        "|---|---:|---:|",
    ]
    for i, (s, m) in enumerate(zip(syn_f1s, sem_f1s)):
        lines.append(f"| {i} | {s:.4f} | {m:.4f} |")
    lines += [
        f"| mean +/- sample SD | {statistics.mean(syn_f1s):.4f} +/- "
        f"{statistics.stdev(syn_f1s):.4f} | {statistics.mean(sem_f1s):.4f} "
        f"+/- {statistics.stdev(sem_f1s):.4f} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    gt = load_ground_truth(TEST_CSV)
    runs = [
        load_run(MULTI_AGENT_ROOT / f"eval_run_{i}") for i in range(N_RUNS)
    ]
    stats = analyze(gt, runs)
    report = format_report(stats)
    print(report)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report)
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
