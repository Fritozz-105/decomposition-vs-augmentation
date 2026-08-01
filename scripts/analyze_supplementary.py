"""
Supplementary analyses for the paper, computed from cached pair data.

Produces:
  A. Ensemble analysis (union / intersection / high-confidence agreement)
  B. Precision-Recall curves via confidence threshold sweep
  C. Syntactic-only and semantic-only counterfactual

Outputs:
  - results/supplementary_analysis.md      (numerical tables + narrative)
  - paper/figures/pr_curves.pdf            (P-R curves figure)
  - paper/figures/ensemble_bars.pdf        (ensemble F1 comparison)
  - paper/figures/cost_f1_scatter.pdf      (Pareto cost vs F1)

Run with: uv run python scripts/analyze_supplementary.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Embed TrueType rather than matplotlib's default Type 3 fonts. ACM camera-ready
# submissions reject Type 3, and the default silently produced one in pr_curves.pdf.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SINGLE_LLM_DIR = PROJECT_ROOT / "results" / "single_llm" / "eval_run_0"
MULTI_AGENT_DIR = PROJECT_ROOT / "results" / "multi_agent" / "eval_run_0"
TEST_CSV = PROJECT_ROOT / "data" / "raw" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict[tuple[str, str], int]:
    df = pd.read_csv(path)
    return {
        (str(row["ltable_id"]), str(row["rtable_id"])): int(row["label"])
        for _, row in df.iterrows()
    }


def load_single_llm(cache_dir: Path) -> dict[tuple[str, str], dict]:
    preds = {}
    for f in cache_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        p = data["prediction"]
        preds[(str(p["source_id"]), str(p["target_id"]))] = {
            "verdict": p["verdict"],
            "confidence": float(p.get("confidence", 0.0)),
        }
    return preds


def load_multi_agent(cache_dir: Path) -> dict[tuple[str, str], dict]:
    """Load multi-agent preds including per-agent verdicts + confidences."""
    preds = {}
    for f in cache_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        p = data["prediction"]
        syn = p.get("syntactic", {}) or {}
        sem = p.get("semantic", {}) or {}
        preds[(str(p["source_id"]), str(p["target_id"]))] = {
            "verdict": p["verdict"],
            "confidence": float(p.get("confidence", 0.0)),
            "syn_verdict": syn.get("verdict", "NO MATCH"),
            "syn_confidence": float(syn.get("confidence", 0.0)),
            "sem_verdict": sem.get("verdict", "NO MATCH"),
            "sem_confidence": float(sem.get("confidence", 0.0)),
        }
    return preds


def verdict_to_int(v: str) -> int:
    return 1 if str(v).upper() == "MATCH" else 0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def prf1(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# A. Ensemble analysis
# ---------------------------------------------------------------------------

def ensemble_analysis(pairs, single, multi, gt) -> dict:
    y_true, y_single, y_multi = [], [], []
    y_union, y_intersection = [], []
    y_high_conf_agree = []          # both MATCH with conf >= 0.9

    for key in pairs:
        label = gt[key]
        s = single[key]
        m = multi[key]
        s_v = verdict_to_int(s["verdict"])
        m_v = verdict_to_int(m["verdict"])

        y_true.append(label)
        y_single.append(s_v)
        y_multi.append(m_v)
        y_union.append(1 if (s_v == 1 or m_v == 1) else 0)
        y_intersection.append(1 if (s_v == 1 and m_v == 1) else 0)

        hca = 1 if (
            s_v == 1 and m_v == 1
            and s["confidence"] >= 0.9 and m["confidence"] >= 0.9
        ) else 0
        y_high_conf_agree.append(hca)

    return {
        "single": prf1(y_true, y_single),
        "multi": prf1(y_true, y_multi),
        "union": prf1(y_true, y_union),
        "intersection": prf1(y_true, y_intersection),
        "high_conf_agree": prf1(y_true, y_high_conf_agree),
    }


# ---------------------------------------------------------------------------
# B. Precision-recall curve via confidence threshold sweep
# ---------------------------------------------------------------------------

def pr_curve_from_verdicts(pairs, preds, gt, n_points: int = 101):
    """
    Interpret confidence as a score for whichever verdict the model produced.
    For threshold sweep we convert to a match-probability score:
        if verdict == MATCH: score = confidence
        else:                score = 1 - confidence
    Then sweep threshold tau in [0, 1] and predict MATCH iff score >= tau.
    """
    y_true, scores = [], []
    for key in pairs:
        label = gt[key]
        p = preds[key]
        v = verdict_to_int(p["verdict"])
        conf = p["confidence"]
        score = conf if v == 1 else 1.0 - conf
        y_true.append(label)
        scores.append(score)

    y_true = np.array(y_true)
    scores = np.array(scores)

    thresholds = np.linspace(0.0, 1.0, n_points)
    precisions, recalls, f1s = [], [], []
    for tau in thresholds:
        y_pred = (scores >= tau).astype(int)
        m = prf1(y_true.tolist(), y_pred.tolist())
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    return {
        "thresholds": thresholds,
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "f1": np.array(f1s),
    }


# ---------------------------------------------------------------------------
# C. Per-agent counterfactual
# ---------------------------------------------------------------------------

def per_agent_counterfactual(pairs, multi, gt) -> dict:
    y_true = [gt[k] for k in pairs]
    y_syn = [verdict_to_int(multi[k]["syn_verdict"]) for k in pairs]
    y_sem = [verdict_to_int(multi[k]["sem_verdict"]) for k in pairs]
    y_final = [verdict_to_int(multi[k]["verdict"]) for k in pairs]

    return {
        "syntactic_only": prf1(y_true, y_syn),
        "semantic_only": prf1(y_true, y_sem),
        "orchestrated": prf1(y_true, y_final),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_pr_curves(single_curve, multi_curve, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(single_curve["recall"], single_curve["precision"],
            label="Single-LLM", linewidth=2)
    ax.plot(multi_curve["recall"], multi_curve["precision"],
            label="Multi-Agent", linewidth=2, linestyle="--")

    # Mark operating points (threshold = 0.5, the default)
    idx = np.argmin(np.abs(single_curve["thresholds"] - 0.5))
    ax.scatter([single_curve["recall"][idx]],
               [single_curve["precision"][idx]],
               s=60, zorder=5, marker="o")
    idx = np.argmin(np.abs(multi_curve["thresholds"] - 0.5))
    ax.scatter([multi_curve["recall"][idx]],
               [multi_curve["precision"][idx]],
               s=60, zorder=5, marker="s")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("Precision-Recall: threshold sweep on confidence")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_bars(ens: dict, out_path: Path) -> None:
    names = ["Single-LLM", "Multi-Agent", "Union\n(either MATCH)",
             "Intersection\n(both MATCH)"]
    keys = ["single", "multi", "union", "intersection"]
    precisions = [ens[k]["precision"] for k in keys]
    recalls = [ens[k]["recall"] for k in keys]
    f1s = [ens[k]["f1"] for k in keys]

    x = np.arange(len(names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x,         recalls,    width, label="Recall")
    ax.bar(x + width, f1s,        width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower center", ncol=3, fontsize=8)
    ax.set_title("Ensemble policies on the disagreement set")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cost_f1_scatter(out_path: Path) -> None:
    """Log-scale tokens/pair vs F1 for all three pipelines."""
    pipelines = [
        ("Magellan",     1,    0.6066),      # 0 tokens → use 1 for log plot
        ("Single-LLM",   656,  0.9065),
        ("Multi-Agent",  5720, 0.8973),
    ]
    # Offsets in points (x, y) so labels don't overlap
    offsets = {
        "Magellan":    (12, -4),
        "Single-LLM":  (0, 12),
        "Multi-Agent": (0, -14),
    }
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for name, tokens, f1 in pipelines:
        ax.scatter(tokens, f1, s=50, zorder=5)
        ax.annotate(
            name, (tokens, f1),
            textcoords="offset points",
            xytext=offsets[name],
            fontsize=9,
            ha="center",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Tokens per pair (log scale; Magellan = 0)")
    ax.set_ylabel("F1")
    ax.set_xlim(0.5, 20000)
    ax.set_ylim(0.55, 0.95)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Cost vs accuracy across pipelines")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown reporting
# ---------------------------------------------------------------------------

def format_metrics_row(label: str, m: dict) -> str:
    return (f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")


def write_markdown(
    ens: dict,
    per_agent: dict,
    single_curve: dict,
    multi_curve: dict,
    n_pairs: int,
    n_positives: int,
    out_path: Path,
) -> None:
    lines = []
    lines.append("# Supplementary Analysis (run 0)")
    lines.append("")
    lines.append(f"- Pairs analyzed: **{n_pairs}**")
    lines.append(f"- True positives in set: **{n_positives}**")
    lines.append("")

    # A. Ensemble
    lines.append("## A. Ensemble Analysis")
    lines.append("")
    lines.append("| Policy | TP | FP | FN | TN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(format_metrics_row("Single-LLM", ens["single"]))
    lines.append(format_metrics_row("Multi-Agent", ens["multi"]))
    lines.append(format_metrics_row("Union (either MATCH)", ens["union"]))
    lines.append(format_metrics_row("Intersection (both MATCH)", ens["intersection"]))
    lines.append(format_metrics_row("High-confidence agreement (both MATCH, conf >= 0.9)",
                                    ens["high_conf_agree"]))
    lines.append("")
    lines.append("**Interpretation.** "
                 "The *union* policy upper-bounds recall (at the price of precision) "
                 "and the *intersection* policy upper-bounds precision. "
                 "If the union F1 exceeds both individual pipelines, the two pipelines "
                 "have complementary errors in the strong sense.")
    lines.append("")

    # C. Per-agent
    lines.append("## B. Per-agent Counterfactual")
    lines.append("")
    lines.append("| Configuration | TP | FP | FN | TN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(format_metrics_row("Syntactic agent alone", per_agent["syntactic_only"]))
    lines.append(format_metrics_row("Semantic agent alone",  per_agent["semantic_only"]))
    lines.append(format_metrics_row("Orchestrated (paper)",  per_agent["orchestrated"]))
    lines.append("")
    lines.append("**Interpretation.** "
                 "If the syntactic agent's standalone F1 exceeds the orchestrated F1, "
                 "the orchestration layer is degrading the best single agent by mixing "
                 "in the weaker semantic agent's verdicts.")
    lines.append("")

    # B. PR curve summary
    lines.append("## C. Precision-Recall Curve (confidence threshold sweep)")
    lines.append("")
    lines.append("For each pipeline we reinterpret confidence as a match-probability "
                 "score (score = confidence if verdict = MATCH else 1 - confidence) "
                 "and sweep the decision threshold over [0, 1]. The table below shows "
                 "the maximum F1 each pipeline can reach by tuning its threshold, "
                 "and the threshold at which that F1 occurs.")
    lines.append("")
    lines.append("| Pipeline | Max F1 | Argmax tau | Precision at max F1 | Recall at max F1 |")
    lines.append("|---|---|---|---|---|")
    for name, curve in [("Single-LLM", single_curve), ("Multi-Agent", multi_curve)]:
        idx = int(np.argmax(curve["f1"]))
        lines.append(
            f"| {name} | {curve['f1'][idx]:.4f} | {curve['thresholds'][idx]:.2f} | "
            f"{curve['precision'][idx]:.4f} | {curve['recall'][idx]:.4f} |"
        )
    lines.append("")
    lines.append("See `paper/figures/pr_curves.pdf` for the full curves.")
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(TEST_CSV)
    single = load_single_llm(SINGLE_LLM_DIR)
    multi = load_multi_agent(MULTI_AGENT_DIR)

    # Intersection of keys that appear in gt AND both pipelines
    pairs = sorted(set(gt) & set(single) & set(multi))
    n_positives = sum(1 for k in pairs if gt[k] == 1)
    print(f"Pairs available in all three sources: {len(pairs)}")
    print(f"True positives in this set: {n_positives}")

    # A. Ensemble
    ens = ensemble_analysis(pairs, single, multi, gt)
    print("\n=== Ensemble Analysis ===")
    for name in ["single", "multi", "union", "intersection", "high_conf_agree"]:
        m = ens[name]
        print(f"  {name:20s} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f}  TP={m['tp']} FP={m['fp']} FN={m['fn']}")

    # B. PR curves
    single_curve = pr_curve_from_verdicts(pairs, single, gt)
    multi_curve = pr_curve_from_verdicts(pairs, multi, gt)
    print("\n=== Max F1 via threshold sweep ===")
    for name, c in [("Single-LLM", single_curve), ("Multi-Agent", multi_curve)]:
        idx = int(np.argmax(c["f1"]))
        print(f"  {name:15s} max F1={c['f1'][idx]:.4f} at tau={c['thresholds'][idx]:.2f} "
              f"(P={c['precision'][idx]:.4f}, R={c['recall'][idx]:.4f})")

    # C. Per-agent
    per_agent = per_agent_counterfactual(pairs, multi, gt)
    print("\n=== Per-agent Counterfactual ===")
    for name, m in per_agent.items():
        print(f"  {name:20s} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f}")

    # Figures
    plot_pr_curves(single_curve, multi_curve, FIG_DIR / "pr_curves.pdf")
    plot_ensemble_bars(ens, FIG_DIR / "ensemble_bars.pdf")
    plot_cost_f1_scatter(FIG_DIR / "cost_f1_scatter.pdf")
    print(f"\nFigures written to {FIG_DIR}")

    # Markdown report
    write_markdown(ens, per_agent, single_curve, multi_curve,
                   len(pairs), n_positives,
                   OUTPUT_DIR / "supplementary_analysis.md")
    print(f"Report written to {OUTPUT_DIR / 'supplementary_analysis.md'}")


if __name__ == "__main__":
    main()
