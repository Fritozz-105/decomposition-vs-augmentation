"""
Comprehensive error analysis across all 3 pipelines (single LLM, multi-agent).
Uses eval_run_0 for both pipelines. Outputs results to stdout and
results/final_error_analysis.md.

Run with: uv run python scripts/analyze_final_results.py
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Read the released caches under results/, not data/cache: data/cache is a
# gitignored working area and its eval_run_* copies no longer exist.
SINGLE_LLM_DIR = PROJECT_ROOT / "results" / "single_llm" / "eval_run_0"
MULTI_AGENT_DIR = PROJECT_ROOT / "results" / "multi_agent" / "eval_run_0"
TEST_CSV = PROJECT_ROOT / "data" / "raw" / "test.csv"
TABLE_A = PROJECT_ROOT / "data" / "raw" / "tableA.csv"
TABLE_B = PROJECT_ROOT / "data" / "raw" / "tableB.csv"
OUTPUT_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict[tuple[str, str], int]:
    """Return {(source_id, target_id): label} from test.csv."""
    df = pd.read_csv(path)
    return {
        (str(row["ltable_id"]), str(row["rtable_id"])): int(row["label"])
        for _, row in df.iterrows()
    }


def load_products(path: Path) -> dict[str, dict]:
    """Return {id: {name, description, price}} from a product CSV."""
    df = pd.read_csv(path)
    products = {}
    for _, row in df.iterrows():
        products[str(row["id"])] = {
            "name": str(row["name"]) if pd.notna(row["name"]) else "",
            "description": str(row["description"]) if pd.notna(row["description"]) else "",
            "price": row["price"] if pd.notna(row["price"]) else None,
        }
    return products


def load_predictions(cache_dir: Path) -> dict[tuple[str, str], dict]:
    """Load all JSON prediction files from a cache directory."""
    preds = {}
    for f in cache_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        pred = data["prediction"]
        sid = str(pred["source_id"])
        tid = str(pred["target_id"])
        preds[(sid, tid)] = pred
    return preds


def verdict_to_int(verdict: str) -> int:
    return 1 if verdict.upper() == "MATCH" else 0


# ---------------------------------------------------------------------------
# 1. Cohen's Kappa
# ---------------------------------------------------------------------------

def compute_kappa(
    single_preds: dict, multi_preds: dict, gt: dict
) -> str:
    """Compute Cohen's kappa between single LLM and multi-agent."""
    common_keys = sorted(set(single_preds) & set(multi_preds))
    y_single = [verdict_to_int(single_preds[k]["verdict"]) for k in common_keys]
    y_multi = [verdict_to_int(multi_preds[k]["verdict"]) for k in common_keys]

    kappa = cohen_kappa_score(y_single, y_multi)

    # Agreement matrix
    cm = confusion_matrix(y_single, y_multi, labels=[0, 1])
    total = len(common_keys)
    agree = sum(1 for a, b in zip(y_single, y_multi) if a == b)

    lines = []
    lines.append("## 1. Cohen's Kappa: Single LLM vs Multi-Agent")
    lines.append("")
    lines.append(f"- **Pairs compared:** {total}")
    lines.append(f"- **Agreement count:** {agree} ({agree/total*100:.1f}%)")
    lines.append(f"- **Cohen's Kappa:** {kappa:.4f}")
    lines.append("")
    lines.append("### Agreement Matrix")
    lines.append("")
    lines.append("| | Multi-Agent: NO MATCH | Multi-Agent: MATCH |")
    lines.append("|---|---|---|")
    lines.append(
        f"| **Single LLM: NO MATCH** | {cm[0][0]} | {cm[0][1]} |"
    )
    lines.append(
        f"| **Single LLM: MATCH** | {cm[1][0]} | {cm[1][1]} |"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Per-Pipeline Error Summary
# ---------------------------------------------------------------------------

def classify_errors(
    preds: dict,
    gt: dict,
    products_a: dict,
    products_b: dict,
) -> dict:
    """Classify predictions into TP, FP, FN, TN and collect error details."""
    tp, fp, fn, tn = [], [], [], []
    for key in sorted(preds):
        if key not in gt:
            continue
        pred_label = verdict_to_int(preds[key]["verdict"])
        true_label = gt[key]
        sid, tid = key
        entry = {
            "source_id": sid,
            "target_id": tid,
            "source_name": products_a.get(sid, {}).get("name", "?"),
            "target_name": products_b.get(tid, {}).get("name", "?"),
            "source_price": products_a.get(sid, {}).get("price"),
            "target_price": products_b.get(tid, {}).get("price"),
            "source_desc": products_a.get(sid, {}).get("description", ""),
            "target_desc": products_b.get(tid, {}).get("description", ""),
            "verdict": preds[key]["verdict"],
            "confidence": preds[key].get("confidence", "?"),
            "reasoning": preds[key].get("reasoning", ""),
        }
        if pred_label == 1 and true_label == 1:
            tp.append(entry)
        elif pred_label == 1 and true_label == 0:
            fp.append(entry)
        elif pred_label == 0 and true_label == 1:
            fn.append(entry)
        else:
            tn.append(entry)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def truncate(s: str, n: int = 200) -> str:
    s = str(s).replace("\n", " ").replace("\r", "")
    return s[:n] + "..." if len(s) > n else s


def format_price(p) -> str:
    if p is None:
        return "N/A"
    try:
        return f"${float(p):.2f}"
    except (ValueError, TypeError):
        return str(p)


def error_summary_section(
    pipeline_name: str,
    result: dict,
) -> str:
    lines = []
    tp, fp, fn, tn = result["tp"], result["fp"], result["fn"], result["tn"]
    total = len(tp) + len(fp) + len(fn) + len(tn)

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    lines.append(f"### {pipeline_name}")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| TP | {len(tp)} |")
    lines.append(f"| FP | {len(fp)} |")
    lines.append(f"| FN | {len(fn)} |")
    lines.append(f"| TN | {len(tn)} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append(f"| Precision | {precision:.4f} |")
    lines.append(f"| Recall | {recall:.4f} |")
    lines.append(f"| F1 | {f1:.4f} |")
    lines.append("")

    # False Positives
    lines.append(f"#### False Positives ({len(fp)})")
    lines.append("")
    if fp:
        lines.append(
            "| # | Source ID | Source Name | Target ID | Target Name "
            "| Src Price | Tgt Price | Confidence | Reasoning |"
        )
        lines.append(
            "|---|----------|------------|-----------|------------|"
            "-----------|-----------|------------|-----------|"
        )
        for i, e in enumerate(fp, 1):
            lines.append(
                f"| {i} | {e['source_id']} | {truncate(e['source_name'], 60)} "
                f"| {e['target_id']} | {truncate(e['target_name'], 60)} "
                f"| {format_price(e['source_price'])} "
                f"| {format_price(e['target_price'])} "
                f"| {e['confidence']} "
                f"| {truncate(e['reasoning'])} |"
            )
        lines.append("")
    else:
        lines.append("*No false positives.*\n")

    # False Negatives
    lines.append(f"#### False Negatives ({len(fn)})")
    lines.append("")
    if fn:
        lines.append(
            "| # | Source ID | Source Name | Target ID | Target Name "
            "| Src Price | Tgt Price | Confidence | Reasoning |"
        )
        lines.append(
            "|---|----------|------------|-----------|------------|"
            "-----------|-----------|------------|-----------|"
        )
        for i, e in enumerate(fn, 1):
            lines.append(
                f"| {i} | {e['source_id']} | {truncate(e['source_name'], 60)} "
                f"| {e['target_id']} | {truncate(e['target_name'], 60)} "
                f"| {format_price(e['source_price'])} "
                f"| {format_price(e['target_price'])} "
                f"| {e['confidence']} "
                f"| {truncate(e['reasoning'])} |"
            )
        lines.append("")
    else:
        lines.append("*No false negatives.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Error Category Classification
# ---------------------------------------------------------------------------

COLOR_WORDS = re.compile(
    r"\b(black|white|silver|red|blue|green|gold|pink|gray|grey|purple|"
    r"brown|orange|yellow|beige|ivory|chrome|graphite|titanium|platinum|"
    r"midnight|space gray|rose gold|matte|glossy)\b",
    re.IGNORECASE,
)

MODEL_VARIANT_RE = re.compile(
    r"\b([A-Z]{1,5}[\-]?\d{2,}[A-Z]?)\b", re.IGNORECASE
)


def classify_fp(entry: dict) -> str:
    """Classify a false-positive error."""
    sn = entry["source_name"].lower()
    tn = entry["target_name"].lower()
    reasoning = entry["reasoning"].lower()
    sd = entry["source_desc"].lower()
    td = entry["target_desc"].lower()
    combined = f"{sn} {tn} {reasoning} {sd} {td}"

    # Color variant
    src_colors = set(COLOR_WORDS.findall(sn + " " + sd))
    tgt_colors = set(COLOR_WORDS.findall(tn + " " + td))
    if src_colors and tgt_colors and src_colors != tgt_colors:
        return "color variant"

    # Form factor
    form_keywords = [
        "micro sd", "mini sd", "compact flash", "sdhc", "sdxc",
        "desktop", "laptop", "notebook", "tower", "mini",
        "portable", "internal", "external", "slim", "ultra-slim",
    ]
    src_forms = [kw for kw in form_keywords if kw in sn or kw in sd]
    tgt_forms = [kw for kw in form_keywords if kw in tn or kw in td]
    if src_forms and tgt_forms and set(src_forms) != set(tgt_forms):
        return "form factor"

    # Accessory confusion
    accessory_words = [
        "case", "cover", "cable", "charger", "adapter", "mount",
        "stand", "dock", "battery", "strap", "screen protector",
        "stylus", "holder", "sleeve", "bag", "pouch", "remote",
        "replacement", "refill", "cartridge", "ink", "toner",
    ]
    src_acc = any(w in sn or w in sd for w in accessory_words)
    tgt_acc = any(w in tn or w in td for w in accessory_words)
    if src_acc != tgt_acc:
        return "accessory confusion"

    # Model variant — extract model numbers and compare
    src_models = set(MODEL_VARIANT_RE.findall(sn))
    tgt_models = set(MODEL_VARIANT_RE.findall(tn))
    if src_models and tgt_models and src_models != tgt_models:
        # Check if they share a common prefix (same product line)
        for sm in src_models:
            for tm in tgt_models:
                prefix_len = min(len(sm), len(tm)) // 2
                if prefix_len >= 2 and sm[:prefix_len].lower() == tm[:prefix_len].lower():
                    return "model variant"

    # Generic vs specific
    generic_words = ["generic", "universal", "compatible", "replacement", "oem"]
    src_gen = any(w in sn or w in sd for w in generic_words)
    tgt_gen = any(w in tn or w in td for w in generic_words)
    if src_gen != tgt_gen:
        return "generic vs specific"

    # Check reasoning for clues
    if any(w in reasoning for w in ["color", "colour"]):
        return "color variant"
    if any(w in reasoning for w in ["model number", "model variant", "suffix", "version"]):
        return "model variant"
    if any(w in reasoning for w in ["form factor", "size differ", "micro", "compact"]):
        return "form factor"
    if any(w in reasoning for w in ["accessory", "cable", "case", "adapter"]):
        return "accessory confusion"

    return "other"


def classify_fn(entry: dict) -> str:
    """Classify a false-negative error."""
    sn = entry["source_name"].lower()
    tn = entry["target_name"].lower()
    reasoning = entry["reasoning"].lower()
    sd = entry["source_desc"].lower()
    td = entry["target_desc"].lower()

    sp = entry["source_price"]
    tp = entry["target_price"]

    # Sparse description
    if (not sd or sd == "nan" or len(sd) < 10) or (not td or td == "nan" or len(td) < 10):
        return "sparse description"

    # Price mismatch
    if sp is not None and tp is not None:
        try:
            sp_f, tp_f = float(sp), float(tp)
            if sp_f > 0 and tp_f > 0:
                ratio = max(sp_f, tp_f) / min(sp_f, tp_f)
                if ratio > 1.5:
                    if any(w in reasoning for w in ["price", "cost", "$"]):
                        return "price mismatch"
        except (ValueError, TypeError):
            pass

    # Model number format differences
    src_models = set(MODEL_VARIANT_RE.findall(sn))
    tgt_models = set(MODEL_VARIANT_RE.findall(tn))
    if src_models and tgt_models:
        # Normalize: strip hyphens and lowercase
        src_norm = {m.replace("-", "").lower() for m in src_models}
        tgt_norm = {m.replace("-", "").lower() for m in tgt_models}
        if src_norm & tgt_norm:
            # Same model after normalization but different raw text
            if src_models != tgt_models:
                return "model number format"

    # Name mismatch — very different names
    src_words = set(sn.split())
    tgt_words = set(tn.split())
    if len(src_words) > 0 and len(tgt_words) > 0:
        overlap = len(src_words & tgt_words) / max(len(src_words), len(tgt_words))
        if overlap < 0.3:
            return "name mismatch"

    # Reasoning-based classification
    if any(w in reasoning for w in ["price", "cost", "expensive", "cheap"]):
        return "price mismatch"
    if any(w in reasoning for w in [
        "different name", "naming", "different brand", "brand mismatch"
    ]):
        return "name mismatch"
    if any(w in reasoning for w in [
        "model number", "model format", "model differ", "identifier"
    ]):
        return "model number format"
    if any(w in reasoning for w in [
        "no description", "missing description", "lacking detail",
        "sparse", "insufficient"
    ]):
        return "sparse description"

    return "other"


def error_categories_section(
    single_result: dict,
    multi_result: dict,
) -> str:
    lines = []
    lines.append("## 3. Error Category Classification")
    lines.append("")

    for name, result in [("Single LLM", single_result), ("Multi-Agent", multi_result)]:
        lines.append(f"### {name}")
        lines.append("")

        # FP categories
        fp_cats = Counter(classify_fp(e) for e in result["fp"])
        lines.append(f"#### False Positive Categories ({len(result['fp'])} total)")
        lines.append("")
        if fp_cats:
            lines.append("| Category | Count | % |")
            lines.append("|----------|-------|---|")
            for cat, cnt in fp_cats.most_common():
                lines.append(
                    f"| {cat} | {cnt} | {cnt/len(result['fp'])*100:.1f}% |"
                )
            lines.append("")
        else:
            lines.append("*No false positives to classify.*\n")

        # FN categories
        fn_cats = Counter(classify_fn(e) for e in result["fn"])
        lines.append(f"#### False Negative Categories ({len(result['fn'])} total)")
        lines.append("")
        if fn_cats:
            lines.append("| Category | Count | % |")
            lines.append("|----------|-------|---|")
            for cat, cnt in fn_cats.most_common():
                lines.append(
                    f"| {cat} | {cnt} | {cnt/len(result['fn'])*100:.1f}% |"
                )
            lines.append("")
        else:
            lines.append("*No false negatives to classify.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Multi-Agent Architecture Analysis
# ---------------------------------------------------------------------------

def multi_agent_analysis(
    multi_preds: dict, gt: dict
) -> str:
    lines = []
    lines.append("## 4. Multi-Agent Architecture Analysis")
    lines.append("")

    agree_count = 0
    disagree_count = 0
    syn_correct_on_disagree = 0
    sem_correct_on_disagree = 0
    override_count = 0
    override_correct = 0
    total_with_agents = 0

    # Track per-agent stats
    syn_tp = syn_fp = syn_fn = syn_tn = 0
    sem_tp = sem_fp = sem_fn = sem_tn = 0

    for key, pred in multi_preds.items():
        if key not in gt:
            continue
        true_label = gt[key]
        orch_verdict = verdict_to_int(pred["verdict"])

        syn = pred.get("syntactic")
        sem = pred.get("semantic")
        if not syn or not sem:
            continue
        total_with_agents += 1

        syn_v = verdict_to_int(syn["verdict"])
        sem_v = verdict_to_int(sem["verdict"])

        # Agent-level confusion matrix
        if syn_v == 1 and true_label == 1:
            syn_tp += 1
        elif syn_v == 1 and true_label == 0:
            syn_fp += 1
        elif syn_v == 0 and true_label == 1:
            syn_fn += 1
        else:
            syn_tn += 1

        if sem_v == 1 and true_label == 1:
            sem_tp += 1
        elif sem_v == 1 and true_label == 0:
            sem_fp += 1
        elif sem_v == 0 and true_label == 1:
            sem_fn += 1
        else:
            sem_tn += 1

        if syn_v == sem_v:
            agree_count += 1
            # Did orchestrator override consensus?
            if orch_verdict != syn_v:
                override_count += 1
                if orch_verdict == true_label:
                    override_correct += 1
        else:
            disagree_count += 1
            if syn_v == true_label:
                syn_correct_on_disagree += 1
            if sem_v == true_label:
                sem_correct_on_disagree += 1

    lines.append(f"**Total pairs with agent data:** {total_with_agents}")
    lines.append("")

    lines.append("### Agent Agreement")
    lines.append("")
    lines.append(f"| Metric | Count | % |")
    lines.append(f"|--------|-------|---|")
    pct_agree = agree_count / total_with_agents * 100 if total_with_agents else 0
    pct_disagree = disagree_count / total_with_agents * 100 if total_with_agents else 0
    lines.append(f"| Agents agreed | {agree_count} | {pct_agree:.1f}% |")
    lines.append(f"| Agents disagreed | {disagree_count} | {pct_disagree:.1f}% |")
    lines.append("")

    lines.append("### When Agents Disagreed")
    lines.append("")
    if disagree_count > 0:
        lines.append(f"- **Syntactic agent correct:** {syn_correct_on_disagree} "
                      f"({syn_correct_on_disagree/disagree_count*100:.1f}%)")
        lines.append(f"- **Semantic agent correct:** {sem_correct_on_disagree} "
                      f"({sem_correct_on_disagree/disagree_count*100:.1f}%)")
    else:
        lines.append("*Agents always agreed.*")
    lines.append("")

    lines.append("### Orchestrator Override of Agent Consensus")
    lines.append("")
    lines.append(f"- **Override count:** {override_count}")
    if override_count > 0:
        lines.append(
            f"- **Override correct:** {override_correct} "
            f"({override_correct/override_count*100:.1f}%)"
        )
    lines.append("")

    # Per-agent performance
    lines.append("### Per-Agent Performance")
    lines.append("")
    lines.append("| Metric | Syntactic Agent | Semantic Agent |")
    lines.append("|--------|-----------------|----------------|")
    lines.append(f"| TP | {syn_tp} | {sem_tp} |")
    lines.append(f"| FP | {syn_fp} | {sem_fp} |")
    lines.append(f"| FN | {syn_fn} | {sem_fn} |")
    lines.append(f"| TN | {syn_tn} | {sem_tn} |")
    syn_prec = syn_tp / (syn_tp + syn_fp) if (syn_tp + syn_fp) else 0
    sem_prec = sem_tp / (sem_tp + sem_fp) if (sem_tp + sem_fp) else 0
    syn_rec = syn_tp / (syn_tp + syn_fn) if (syn_tp + syn_fn) else 0
    sem_rec = sem_tp / (sem_tp + sem_fn) if (sem_tp + sem_fn) else 0
    syn_f1 = (2 * syn_prec * syn_rec / (syn_prec + syn_rec)
              if (syn_prec + syn_rec) else 0)
    sem_f1 = (2 * sem_prec * sem_rec / (sem_prec + sem_rec)
              if (sem_prec + sem_rec) else 0)
    lines.append(f"| Precision | {syn_prec:.4f} | {sem_prec:.4f} |")
    lines.append(f"| Recall | {syn_rec:.4f} | {sem_rec:.4f} |")
    lines.append(f"| F1 | {syn_f1:.4f} | {sem_f1:.4f} |")
    lines.append("")

    # Patterns each agent catches uniquely
    lines.append("### Unique Contributions")
    lines.append("")
    syn_unique_tp = 0
    sem_unique_tp = 0
    syn_unique_tn = 0
    sem_unique_tn = 0
    for key, pred in multi_preds.items():
        if key not in gt:
            continue
        true_label = gt[key]
        syn = pred.get("syntactic")
        sem = pred.get("semantic")
        if not syn or not sem:
            continue
        syn_v = verdict_to_int(syn["verdict"])
        sem_v = verdict_to_int(sem["verdict"])
        if syn_v == 1 and sem_v == 0 and true_label == 1:
            syn_unique_tp += 1
        if sem_v == 1 and syn_v == 0 and true_label == 1:
            sem_unique_tp += 1
        if syn_v == 0 and sem_v == 1 and true_label == 0:
            syn_unique_tn += 1
        if sem_v == 0 and syn_v == 1 and true_label == 0:
            sem_unique_tn += 1

    lines.append(
        f"- **Syntactic correctly identified match when semantic missed:** "
        f"{syn_unique_tp}"
    )
    lines.append(
        f"- **Semantic correctly identified match when syntactic missed:** "
        f"{sem_unique_tp}"
    )
    lines.append(
        f"- **Syntactic correctly rejected when semantic false-alarmed:** "
        f"{syn_unique_tn}"
    )
    lines.append(
        f"- **Semantic correctly rejected when syntactic false-alarmed:** "
        f"{sem_unique_tn}"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Cross-Pipeline Disagreement Analysis
# ---------------------------------------------------------------------------

def cross_pipeline_disagreements(
    single_preds: dict,
    multi_preds: dict,
    gt: dict,
    products_a: dict,
    products_b: dict,
) -> str:
    lines = []
    lines.append("## 5. Cross-Pipeline Disagreement Analysis")
    lines.append("")

    common_keys = sorted(set(single_preds) & set(multi_preds))
    disagreements = []
    for key in common_keys:
        s_v = verdict_to_int(single_preds[key]["verdict"])
        m_v = verdict_to_int(multi_preds[key]["verdict"])
        if s_v != m_v and key in gt:
            true_label = gt[key]
            sid, tid = key
            disagreements.append({
                "source_id": sid,
                "target_id": tid,
                "source_name": products_a.get(sid, {}).get("name", "?"),
                "target_name": products_b.get(tid, {}).get("name", "?"),
                "true_label": true_label,
                "single_verdict": single_preds[key]["verdict"],
                "single_confidence": single_preds[key].get("confidence", "?"),
                "single_reasoning": single_preds[key].get("reasoning", ""),
                "multi_verdict": multi_preds[key]["verdict"],
                "multi_confidence": multi_preds[key].get("confidence", "?"),
                "multi_reasoning": multi_preds[key].get("reasoning", ""),
                "single_correct": s_v == true_label,
                "multi_correct": m_v == true_label,
            })

    lines.append(f"**Total disagreements:** {len(disagreements)}")
    lines.append("")

    single_wins = sum(1 for d in disagreements if d["single_correct"])
    multi_wins = sum(1 for d in disagreements if d["multi_correct"])
    lines.append(f"- **Single LLM correct (multi-agent wrong):** {single_wins}")
    lines.append(f"- **Multi-agent correct (single LLM wrong):** {multi_wins}")
    lines.append("")

    if disagreements:
        lines.append("### Detailed Disagreements")
        lines.append("")
        lines.append(
            "| # | Source Name | Target Name | True | Single Verdict "
            "| Multi Verdict | Correct |"
        )
        lines.append(
            "|---|------------|-------------|------|---------------"
            "|---------------|---------|"
        )
        for i, d in enumerate(disagreements, 1):
            true_str = "MATCH" if d["true_label"] == 1 else "NO MATCH"
            correct_who = (
                "Single" if d["single_correct"]
                else "Multi" if d["multi_correct"]
                else "Neither"
            )
            lines.append(
                f"| {i} | {truncate(d['source_name'], 45)} "
                f"| {truncate(d['target_name'], 45)} "
                f"| {true_str} "
                f"| {d['single_verdict']} ({d['single_confidence']}) "
                f"| {d['multi_verdict']} ({d['multi_confidence']}) "
                f"| {correct_who} |"
            )
        lines.append("")

        # Detailed reasoning for each disagreement
        lines.append("### Disagreement Details")
        lines.append("")
        for i, d in enumerate(disagreements, 1):
            true_str = "MATCH" if d["true_label"] == 1 else "NO MATCH"
            correct_who = (
                "Single LLM" if d["single_correct"]
                else "Multi-Agent" if d["multi_correct"]
                else "Neither"
            )
            lines.append(f"**{i}. {d['source_name']} vs {d['target_name']}**")
            lines.append(f"- True label: {true_str} | Correct: **{correct_who}**")
            lines.append(
                f"- Single LLM: {d['single_verdict']} "
                f"(conf {d['single_confidence']}): "
                f"{truncate(d['single_reasoning'])}"
            )
            lines.append(
                f"- Multi-Agent: {d['multi_verdict']} "
                f"(conf {d['multi_confidence']}): "
                f"{truncate(d['multi_reasoning'])}"
            )
            lines.append("")

    # Summarize patterns
    lines.append("### Summary of Strengths")
    lines.append("")

    # Analyze what single LLM does better
    single_better_on_match = sum(
        1 for d in disagreements
        if d["single_correct"] and d["true_label"] == 1
    )
    single_better_on_nomatch = sum(
        1 for d in disagreements
        if d["single_correct"] and d["true_label"] == 0
    )
    multi_better_on_match = sum(
        1 for d in disagreements
        if d["multi_correct"] and d["true_label"] == 1
    )
    multi_better_on_nomatch = sum(
        1 for d in disagreements
        if d["multi_correct"] and d["true_label"] == 0
    )

    lines.append("| Metric | Single LLM better | Multi-Agent better |")
    lines.append("|--------|--------------------|--------------------|")
    lines.append(
        f"| Correctly matching (FN avoidance) | {single_better_on_match} "
        f"| {multi_better_on_match} |"
    )
    lines.append(
        f"| Correctly rejecting (FP avoidance) | {single_better_on_nomatch} "
        f"| {multi_better_on_nomatch} |"
    )
    lines.append(
        f"| **Total wins** | **{single_wins}** | **{multi_wins}** |"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. What Each Architecture Does Well
# ---------------------------------------------------------------------------

def architecture_strengths(
    single_result: dict,
    multi_result: dict,
    single_preds: dict,
    multi_preds: dict,
    gt: dict,
) -> str:
    lines = []
    lines.append("## 6. Architecture Strengths and Weaknesses")
    lines.append("")

    s_tp = len(single_result["tp"])
    s_fp = len(single_result["fp"])
    s_fn = len(single_result["fn"])
    s_tn = len(single_result["tn"])
    m_tp = len(multi_result["tp"])
    m_fp = len(multi_result["fp"])
    m_fn = len(multi_result["fn"])
    m_tn = len(multi_result["tn"])

    s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) else 0
    s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) else 0
    s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec) if (s_prec + s_rec) else 0
    m_prec = m_tp / (m_tp + m_fp) if (m_tp + m_fp) else 0
    m_rec = m_tp / (m_tp + m_fn) if (m_tp + m_fn) else 0
    m_f1 = 2 * m_prec * m_rec / (m_prec + m_rec) if (m_prec + m_rec) else 0

    lines.append("### Head-to-Head Comparison")
    lines.append("")
    lines.append("| Metric | Single LLM | Multi-Agent | Advantage |")
    lines.append("|--------|------------|-------------|-----------|")
    lines.append(
        f"| Precision | {s_prec:.4f} | {m_prec:.4f} "
        f"| {'Single' if s_prec > m_prec else 'Multi' if m_prec > s_prec else 'Tied'} |"
    )
    lines.append(
        f"| Recall | {s_rec:.4f} | {m_rec:.4f} "
        f"| {'Single' if s_rec > m_rec else 'Multi' if m_rec > s_rec else 'Tied'} |"
    )
    lines.append(
        f"| F1 | {s_f1:.4f} | {m_f1:.4f} "
        f"| {'Single' if s_f1 > m_f1 else 'Multi' if m_f1 > s_f1 else 'Tied'} |"
    )
    lines.append(
        f"| FP Count | {s_fp} | {m_fp} "
        f"| {'Single' if s_fp < m_fp else 'Multi' if m_fp < s_fp else 'Tied'} |"
    )
    lines.append(
        f"| FN Count | {s_fn} | {m_fn} "
        f"| {'Single' if s_fn < m_fn else 'Multi' if m_fn < s_fn else 'Tied'} |"
    )
    lines.append("")

    # Confidence analysis
    lines.append("### Confidence Distribution")
    lines.append("")

    for name, preds, result in [
        ("Single LLM", single_preds, single_result),
        ("Multi-Agent", multi_preds, multi_result),
    ]:
        confs = []
        correct_confs = []
        incorrect_confs = []
        for key in preds:
            if key not in gt:
                continue
            pred_label = verdict_to_int(preds[key]["verdict"])
            true_label = gt[key]
            conf = preds[key].get("confidence", 0)
            if conf is None:
                conf = 0
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 0
            confs.append(conf)
            if pred_label == true_label:
                correct_confs.append(conf)
            else:
                incorrect_confs.append(conf)

        avg_conf = sum(confs) / len(confs) if confs else 0
        avg_correct = sum(correct_confs) / len(correct_confs) if correct_confs else 0
        avg_incorrect = (
            sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0
        )
        lines.append(f"**{name}:**")
        lines.append(f"- Average confidence (all): {avg_conf:.3f}")
        lines.append(f"- Average confidence (correct): {avg_correct:.3f}")
        lines.append(f"- Average confidence (incorrect): {avg_incorrect:.3f}")
        lines.append(
            f"- Calibration gap: {abs(avg_correct - avg_incorrect):.3f}"
        )
        lines.append("")

    # Token usage
    lines.append("### Token Usage")
    lines.append("")
    single_tokens = []
    multi_tokens = []
    for key in single_preds:
        t = single_preds[key].get("tokens")
        if t is None:
            # Check parent structure — some formats have tokens at top level
            pass
        else:
            try:
                single_tokens.append(int(t))
            except (ValueError, TypeError):
                pass
    for key in multi_preds:
        t = multi_preds[key].get("tokens")
        if t is None:
            pass
        else:
            try:
                multi_tokens.append(int(t))
            except (ValueError, TypeError):
                pass

    # Reload raw JSON to get top-level tokens
    if not single_tokens:
        for f in SINGLE_LLM_DIR.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            if "tokens" in data:
                single_tokens.append(data["tokens"])
    if not multi_tokens:
        for f in MULTI_AGENT_DIR.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            if "tokens" in data:
                multi_tokens.append(data["tokens"])

    if single_tokens:
        lines.append(
            f"- **Single LLM:** avg {sum(single_tokens)/len(single_tokens):.0f} "
            f"tokens/pair, total {sum(single_tokens):,} tokens"
        )
    if multi_tokens:
        lines.append(
            f"- **Multi-Agent:** avg {sum(multi_tokens)/len(multi_tokens):.0f} "
            f"tokens/pair, total {sum(multi_tokens):,} tokens"
        )
    if single_tokens and multi_tokens:
        ratio = sum(multi_tokens) / sum(single_tokens) if sum(single_tokens) else 0
        lines.append(f"- **Multi-Agent / Single LLM ratio:** {ratio:.1f}x")
    lines.append("")

    # Strengths / weaknesses narrative
    lines.append("### Single LLM Strengths")
    lines.append("")
    lines.append("- Lower token cost per pair")
    if s_f1 > m_f1:
        lines.append(f"- Higher F1 score ({s_f1:.4f} vs {m_f1:.4f})")
    if s_prec > m_prec:
        lines.append(f"- Better precision ({s_prec:.4f} vs {m_prec:.4f})")
    if s_rec > m_rec:
        lines.append(f"- Better recall ({s_rec:.4f} vs {m_rec:.4f})")
    lines.append("- Simpler architecture, less latency per pair")
    lines.append("")

    lines.append("### Single LLM Weaknesses")
    lines.append("")
    if s_prec <= m_prec:
        lines.append(f"- Lower precision ({s_prec:.4f} vs {m_prec:.4f})")
    if s_rec <= m_rec:
        lines.append(f"- Lower recall ({s_rec:.4f} vs {m_rec:.4f})")
    lines.append("- No structured tool use for verification")
    lines.append("- Single point of failure in reasoning")
    lines.append("")

    lines.append("### Multi-Agent Strengths")
    lines.append("")
    if m_prec > s_prec:
        lines.append(f"- Better precision ({m_prec:.4f} vs {s_prec:.4f})")
    if m_rec > s_rec:
        lines.append(f"- Better recall ({m_rec:.4f} vs {s_rec:.4f})")
    lines.append("- Structured tool use provides quantitative evidence")
    lines.append("- Dual-perspective (syntactic + semantic) catches different error types")
    lines.append("- Orchestrator can resolve agent disagreements")
    lines.append("")

    lines.append("### Multi-Agent Weaknesses")
    lines.append("")
    if m_f1 < s_f1:
        lines.append(f"- Lower F1 score ({m_f1:.4f} vs {s_f1:.4f})")
    if m_prec <= s_prec:
        lines.append(f"- Lower precision ({m_prec:.4f} vs {s_prec:.4f})")
    if m_rec <= s_rec:
        lines.append(f"- Lower recall ({m_rec:.4f} vs {s_rec:.4f})")
    lines.append("- Significantly higher token cost per pair")
    lines.append("- More complex architecture increases latency")
    lines.append("- Agent consensus does not always lead to correct answer")
    lines.append("")

    # Why single LLM may edge out
    lines.append("### Why Single LLM May Edge Out Multi-Agent")
    lines.append("")
    lines.append(
        "The single LLM approach benefits from unified reasoning where all "
        "available evidence is weighed holistically in a single pass. The "
        "multi-agent system, while providing structured verification through "
        "dedicated syntactic and semantic tools, introduces potential failure "
        "modes: agent disagreements may confuse the orchestrator, the rigid "
        "tool-based analysis may miss nuanced contextual clues that a single "
        "LLM captures through its broader reasoning, and the orchestrator must "
        "synthesize potentially conflicting agent reports. The overhead of "
        "coordination does not always translate to better decisions, "
        "particularly when the single LLM already has strong general reasoning "
        "capabilities for entity matching."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")

    gt = load_ground_truth(TEST_CSV)
    products_a = load_products(TABLE_A)
    products_b = load_products(TABLE_B)
    single_preds = load_predictions(SINGLE_LLM_DIR)
    multi_preds = load_predictions(MULTI_AGENT_DIR)

    # Also load raw JSON for token data (top-level tokens field)
    single_tokens_raw = {}
    for f in SINGLE_LLM_DIR.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        parts = f.stem.split("_")
        key = (parts[0], parts[1])
        single_tokens_raw[key] = data.get("tokens", 0)

    multi_tokens_raw = {}
    for f in MULTI_AGENT_DIR.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        parts = f.stem.split("_")
        key = (parts[0], parts[1])
        multi_tokens_raw[key] = data.get("tokens", 0)

    print(f"Ground truth pairs: {len(gt)}")
    print(f"Single LLM predictions: {len(single_preds)}")
    print(f"Multi-Agent predictions: {len(multi_preds)}")
    print()

    # Classify errors
    single_result = classify_errors(single_preds, gt, products_a, products_b)
    multi_result = classify_errors(multi_preds, gt, products_a, products_b)

    # Build report
    sections = []
    sections.append("# Final Error Analysis: Single LLM vs Multi-Agent (Run 0)")
    sections.append("")
    sections.append(
        f"**Generated:** 2026-04-07 | "
        f"**Pairs:** {len(gt)} | "
        f"**Single LLM cached:** {len(single_preds)} | "
        f"**Multi-Agent cached:** {len(multi_preds)}"
    )
    sections.append("")

    # Section 1: Cohen's Kappa
    sections.append(compute_kappa(single_preds, multi_preds, gt))

    # Section 2: Per-Pipeline Error Summary
    sections.append("## 2. Per-Pipeline Error Summary")
    sections.append("")
    sections.append(error_summary_section("Single LLM", single_result))
    sections.append(error_summary_section("Multi-Agent", multi_result))

    # Section 3: Error Categories
    sections.append(error_categories_section(single_result, multi_result))

    # Section 4: Multi-Agent Architecture Analysis
    sections.append(multi_agent_analysis(multi_preds, gt))

    # Section 5: Cross-Pipeline Disagreements
    sections.append(
        cross_pipeline_disagreements(
            single_preds, multi_preds, gt, products_a, products_b
        )
    )

    # Section 6: Architecture Strengths
    sections.append(
        architecture_strengths(
            single_result, multi_result, single_preds, multi_preds, gt
        )
    )

    report = "\n".join(sections)

    # Print to stdout
    print(report)

    # Save to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "final_error_analysis.md"
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
