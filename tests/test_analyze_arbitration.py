"""Tests for the agent-agreement and arbitration analysis script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_arbitration import analyze, f1_from_preds, load_run


def _write_pair(
    directory: Path,
    source_id: str,
    target_id: str,
    final: str,
    syn: str,
    sem: str,
) -> None:
    record = {
        "prediction": {
            "source_id": source_id,
            "target_id": target_id,
            "verdict": final,
            "syntactic": {"verdict": syn},
            "semantic": {"verdict": sem},
        }
    }
    (directory / f"{source_id}_{target_id}.json").write_text(json.dumps(record))


def test_analyze_classifies_consensus_override_and_arbitration(tmp_path: Path) -> None:
    """Overrides of an agent consensus and arbitration outcomes are each counted correctly."""
    run_dir = tmp_path / "eval_run_0"
    run_dir.mkdir()
    # Agents agree MATCH, orchestrator follows, label 1: plain agreement, TP.
    _write_pair(run_dir, "1", "10", "MATCH", "MATCH", "MATCH")
    # Agents agree MATCH, orchestrator overrides to NO MATCH, label 1: wrong override, FN.
    _write_pair(run_dir, "2", "20", "NO MATCH", "MATCH", "MATCH")
    # Agents agree NO MATCH, orchestrator overrides to MATCH, label 1: correct override.
    _write_pair(run_dir, "3", "30", "MATCH", "NO MATCH", "NO MATCH")
    # Agents disagree, orchestrator sides with semantic NO MATCH, label 1: FN in arbitration.
    _write_pair(run_dir, "4", "40", "NO MATCH", "MATCH", "NO MATCH")
    gt = {("1", "10"): 1, ("2", "20"): 1, ("3", "30"): 1, ("4", "40"): 1}

    stats = analyze(gt, [load_run(run_dir)])

    assert stats["total_decisions"] == 4
    assert stats["agents_agree"] == 3
    assert stats["agents_disagree"] == 1
    assert stats["overrides_of_consensus"] == 2
    assert stats["overrides_correct"] == 1
    assert stats["orchestrator_correct_on_disagreements"] == 0
    assert stats["follow_syntactic_correct_on_disagreements"] == 1
    assert stats["multi_agent_false_negatives"] == 2
    assert stats["false_negatives_in_disagreements"] == 1


def test_per_agent_f1_scores_agent_verdicts_not_final(tmp_path: Path) -> None:
    """Per-agent F1 is computed from each agent's own verdict, ignoring the orchestrator."""
    run_dir = tmp_path / "eval_run_0"
    run_dir.mkdir()
    # Syntactic is right on both pairs, semantic wrong on both, final verdicts mixed.
    _write_pair(run_dir, "1", "10", "NO MATCH", "MATCH", "NO MATCH")
    _write_pair(run_dir, "2", "20", "MATCH", "NO MATCH", "MATCH")
    gt = {("1", "10"): 1, ("2", "20"): 0}

    stats = analyze(gt, [load_run(run_dir)])

    assert stats["syntactic_f1_per_run"] == [1.0]
    assert stats["semantic_f1_per_run"] == [0.0]


def test_load_run_defaults_missing_agent_block_to_no_match(tmp_path: Path) -> None:
    """A cache record without an agent block loads as NO MATCH instead of crashing."""
    run_dir = tmp_path / "eval_run_0"
    run_dir.mkdir()
    record = {
        "prediction": {
            "source_id": "9",
            "target_id": "90",
            "verdict": "MATCH",
            "syntactic": {"verdict": "MATCH"},
        }
    }
    (run_dir / "9_90.json").write_text(json.dumps(record))

    preds = load_run(run_dir)

    assert preds[("9", "90")]["sem_verdict"] == "NO MATCH"


def test_f1_from_preds_matches_hand_computation() -> None:
    """F1 equals the hand-computed value for a known confusion split."""
    gt = {("a", "1"): 1, ("b", "2"): 1, ("c", "3"): 0, ("d", "4"): 0}
    preds = {("a", "1"): 1, ("b", "2"): 0, ("c", "3"): 1, ("d", "4"): 0}
    # tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5, F1=0.5.
    assert f1_from_preds(preds, gt) == 0.5
