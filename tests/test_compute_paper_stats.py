"""Tests for the canonical paper-statistics pipeline.

Task A1 of the Phase 7 plan: no paper number may be computed from a cache that has
not first been proven to contain exactly the 1,916 labeled test keys, once each.
The previous script silently intersected "keys present in both caches", which hid
missing records instead of reporting them -- these tests exist to keep that
failure mode from coming back.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from scripts.compute_paper_stats import (
    CacheRecord,
    CacheValidationError,
    Confusion,
    PIPELINES,
    RUNS,
    build_report,
    canonical_key_labels,
    cohen_kappa,
    confusion_for,
    load_validated_run,
    mcnemar_cells,
    read_cache_records,
    render_markdown,
    BootstrapInputs,
    RuntimeEvidenceError,
    ZeroF1DenominatorError,
    bootstrap_effect_for_draw,
    exact_permutation_test,
    load_runtime_measurements,
    pair_weights,
    run_cache_dir,
    runtime_artifact_path,
    summarize_runtime,
    run_pigeonhole_bootstrap,
    weighted_f1,
    sample_sd,
    serialize,
    studentized_mean_difference,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _record(source_id: str, target_id: str, verdict: str, tokens: int = 100) -> CacheRecord:
    """A minimal in-memory record for hand-checkable metric tests."""
    return CacheRecord(
        source_id=source_id,
        target_id=target_id,
        verdict=verdict,
        confidence=0.9,
        parse_error=False,
        tokens=tokens,
    )


def _bootstrap_inputs(n_entities: int = 40, mirror_runs: bool = False) -> BootstrapInputs:
    """A crossed bipartite fixture: every source and target appears in exactly two pairs.

    Sized so that zeroing out every positive pair is negligibly unlikely, which keeps
    the determinism tests on the happy path. The deliberately degenerate case lives
    in ``_degenerate_bootstrap_inputs``.
    """
    pair_source_index: list[int] = []
    pair_target_index: list[int] = []
    labels: list[int] = []
    for i in range(n_entities):
        # Each source i pairs with target i (positive) and target (i+1) (negative).
        pair_source_index += [i, i]
        pair_target_index += [i, (i + 1) % n_entities]
        labels += [1, 0]

    n_pairs = len(labels)

    def run(flip_at: int) -> tuple[int, ...]:
        """Predict every positive correctly except one, flipped at ``flip_at``."""
        return tuple(
            0 if index == flip_at else labels[index] for index in range(n_pairs)
        )

    single = (run(-1), run(0), run(2))
    multi = tuple(reversed(single)) if mirror_runs else (run(4), run(6), run(8))

    return BootstrapInputs(
        labels=tuple(labels),
        pair_source_index=tuple(pair_source_index),
        pair_target_index=tuple(pair_target_index),
        n_sources=n_entities,
        n_targets=n_entities,
        predictions={"single_llm": single, "multi_agent": multi},
    )


def _degenerate_bootstrap_inputs() -> BootstrapInputs:
    """A 6-pair, 3-source, 3-target fixture small enough to zero out every positive.

    With only three entities per side, a replicate can easily draw multiplicities that
    give every positive pair zero weight, making weighted F1 undefined. Used to prove
    the bootstrap surfaces that instead of coercing it to zero.
    """
    labels = (1, 0, 1, 0, 1, 0)
    perfect = (1, 0, 1, 0, 1, 0)
    return BootstrapInputs(
        labels=labels,
        pair_source_index=(0, 0, 1, 1, 2, 2),
        pair_target_index=(0, 1, 1, 2, 2, 0),
        n_sources=3,
        n_targets=3,
        predictions={"single_llm": (perfect,), "multi_agent": (perfect,)},
    )


@lru_cache(maxsize=1)
def _production_report() -> dict[str, object]:
    """Build the real report once and share it across assertions (19k file reads)."""
    if not run_cache_dir(PROJECT_ROOT, "single_llm", 0).is_dir():
        pytest.skip("production caches not present")
    return build_report(PROJECT_ROOT)


# --------------------------------------------------------------------------- #
# Synthetic-cache helpers
# --------------------------------------------------------------------------- #


def write_record(
    directory: Path,
    filename: str,
    *,
    source_id: str = "1",
    target_id: str = "2",
    verdict: str = "MATCH",
    confidence: float = 0.9,
    parse_error: bool = False,
    tokens: int | None = 100,
    drop: tuple[str, ...] = (),
) -> Path:
    """Write one cache record, optionally omitting fields to simulate corruption."""
    directory.mkdir(parents=True, exist_ok=True)
    prediction: dict[str, object] = {
        "source_id": source_id,
        "target_id": target_id,
        "verdict": verdict,
        "confidence": confidence,
        "parse_error": parse_error,
    }
    for field in drop:
        prediction.pop(field, None)
    payload: dict[str, object] = {"prediction": prediction}
    if tokens is not None:
        payload["tokens"] = tokens
    path = directory / filename
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def two_pair_labels() -> dict[tuple[str, str], int]:
    """A minimal canonical key-label map: one positive, one negative."""
    return {("1", "2"): 1, ("3", "4"): 0}


# --------------------------------------------------------------------------- #
# read_cache_records: per-record integrity
# --------------------------------------------------------------------------- #


def test_duplicate_key_is_rejected_with_both_filenames(tmp_path: Path) -> None:
    """Two files resolving to the same (source, target) key fail, naming both files."""
    write_record(tmp_path, "1_2.json", source_id="1", target_id="2")
    write_record(tmp_path, "1_2_again.json", source_id="1", target_id="2")

    with pytest.raises(CacheValidationError) as excinfo:
        read_cache_records(tmp_path)

    message = str(excinfo.value)
    assert "duplicate" in message.lower()
    assert "('1', '2')" in message
    assert "1_2.json" in message and "1_2_again.json" in message


def test_malformed_verdict_is_rejected_with_file_and_value(tmp_path: Path) -> None:
    """A verdict outside {MATCH, NO MATCH} fails, naming the file and the bad value."""
    write_record(tmp_path, "1_2.json", verdict="MAYBE")

    with pytest.raises(CacheValidationError) as excinfo:
        read_cache_records(tmp_path)

    message = str(excinfo.value)
    assert "verdict" in message.lower()
    assert "MAYBE" in message
    assert "1_2.json" in message


def test_missing_prediction_field_is_rejected_with_field_name(tmp_path: Path) -> None:
    """A record missing source_id fails, naming the absent field and the file."""
    write_record(tmp_path, "1_2.json", drop=("source_id",))

    with pytest.raises(CacheValidationError) as excinfo:
        read_cache_records(tmp_path)

    message = str(excinfo.value)
    assert "source_id" in message
    assert "1_2.json" in message


def test_unparseable_json_is_rejected_with_file(tmp_path: Path) -> None:
    """A file that is not valid JSON fails with the filename, not a bare JSONDecodeError."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "broken.json").write_text("{not json")

    with pytest.raises(CacheValidationError) as excinfo:
        read_cache_records(tmp_path)

    assert "broken.json" in str(excinfo.value)


def test_missing_tokens_field_is_rejected(tmp_path: Path) -> None:
    """Token counts are reported in the paper, so a record without them fails."""
    write_record(tmp_path, "1_2.json", tokens=None)

    with pytest.raises(CacheValidationError) as excinfo:
        read_cache_records(tmp_path)

    message = str(excinfo.value)
    assert "tokens" in message.lower()
    assert "1_2.json" in message


def test_valid_records_load_with_parsed_fields(tmp_path: Path) -> None:
    """A well-formed cache yields records keyed by (source, target) with fields parsed."""
    write_record(tmp_path, "1_2.json", source_id="1", target_id="2", verdict="MATCH", tokens=590)
    write_record(
        tmp_path, "3_4.json", source_id="3", target_id="4", verdict="NO MATCH", tokens=610
    )

    records = read_cache_records(tmp_path)

    assert set(records) == {("1", "2"), ("3", "4")}
    assert records[("1", "2")].matched is True
    assert records[("3", "4")].matched is False
    assert records[("1", "2")].tokens == 590
    assert records[("3", "4")].parse_error is False


# --------------------------------------------------------------------------- #
# validate_run_cache: key-set completeness against the canonical split
# --------------------------------------------------------------------------- #


def test_missing_key_is_rejected_with_count_and_example(
    tmp_path: Path, two_pair_labels: dict[tuple[str, str], int]
) -> None:
    """A cache short one labeled key fails, reporting how many and which are absent."""
    write_record(tmp_path, "1_2.json", source_id="1", target_id="2")

    with pytest.raises(CacheValidationError) as excinfo:
        load_validated_run(tmp_path, two_pair_labels)

    message = str(excinfo.value)
    assert "missing" in message.lower()
    assert "1" in message
    assert "('3', '4')" in message


def test_extra_key_is_rejected_with_count_and_example(
    tmp_path: Path, two_pair_labels: dict[tuple[str, str], int]
) -> None:
    """A cache holding a key absent from the labeled split fails as an extra key."""
    write_record(tmp_path, "1_2.json", source_id="1", target_id="2")
    write_record(tmp_path, "3_4.json", source_id="3", target_id="4")
    write_record(tmp_path, "9_9.json", source_id="9", target_id="9")

    with pytest.raises(CacheValidationError) as excinfo:
        load_validated_run(tmp_path, two_pair_labels)

    message = str(excinfo.value)
    assert "extra" in message.lower()
    assert "('9', '9')" in message


def test_label_key_mismatch_reports_both_directions(
    tmp_path: Path, two_pair_labels: dict[tuple[str, str], int]
) -> None:
    """A cache that both drops a labeled key and adds an unlabeled one reports both."""
    write_record(tmp_path, "1_2.json", source_id="1", target_id="2")
    write_record(tmp_path, "9_9.json", source_id="9", target_id="9")

    with pytest.raises(CacheValidationError) as excinfo:
        load_validated_run(tmp_path, two_pair_labels)

    message = str(excinfo.value)
    assert "missing" in message.lower()
    assert "extra" in message.lower()


def test_exact_key_set_validates(
    tmp_path: Path, two_pair_labels: dict[tuple[str, str], int]
) -> None:
    """A cache matching the labeled split exactly returns all records."""
    write_record(tmp_path, "1_2.json", source_id="1", target_id="2", verdict="MATCH")
    write_record(tmp_path, "3_4.json", source_id="3", target_id="4", verdict="NO MATCH")

    records = load_validated_run(tmp_path, two_pair_labels)

    assert set(records) == set(two_pair_labels)


def test_swapped_orientation_is_not_silently_accepted(
    tmp_path: Path, two_pair_labels: dict[tuple[str, str], int]
) -> None:
    """(target, source) is a different key than (source, target) and must not validate."""
    write_record(tmp_path, "2_1.json", source_id="2", target_id="1")
    write_record(tmp_path, "3_4.json", source_id="3", target_id="4")

    with pytest.raises(CacheValidationError):
        load_validated_run(tmp_path, two_pair_labels)


# --------------------------------------------------------------------------- #
# Production caches
# --------------------------------------------------------------------------- #


def test_canonical_key_labels_is_the_1916_pair_test_split() -> None:
    """The canonical map is exactly the 1,916 labeled test pairs, 206 of them positive."""
    labels = canonical_key_labels()

    assert len(labels) == 1916
    assert sum(labels.values()) == 206


@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("run", RUNS)
def test_production_cache_has_exactly_the_1916_test_keys(pipeline: str, run: int) -> None:
    """Each of the ten authoritative run caches validates at exactly 1,916 keys."""
    directory = run_cache_dir(PROJECT_ROOT, pipeline, run)
    if not directory.is_dir():
        pytest.skip(f"cache not present: {directory}")

    records = load_validated_run(directory, canonical_key_labels())

    assert len(records) == 1916


# --------------------------------------------------------------------------- #
# Metric correctness
# --------------------------------------------------------------------------- #


def test_confusion_metrics_are_hand_checkable() -> None:
    """Precision, recall, and F1 come straight from the confusion cells."""
    cm = Confusion(tp=3, fp=1, fn=2, tn=4)

    assert cm.n == 10
    assert cm.precision == 3 / 4
    assert cm.recall == 3 / 5
    assert cm.f1 == (2 * 3) / (2 * 3 + 1 + 2)
    assert cm.accuracy == 7 / 10


def test_empty_denominators_do_not_divide_by_zero() -> None:
    """A configuration that predicts nothing reports zero, not NaN."""
    cm = Confusion(tp=0, fp=0, fn=5, tn=5)

    assert cm.precision == 0.0
    assert cm.f1 == 0.0
    assert cm.recall == 0.0


@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("run", RUNS)
def test_f1_agrees_bit_for_bit_with_sklearn(pipeline: str, run: int) -> None:
    """The hand-rolled 2tp/(2tp+fp+fn) form matches sklearn exactly on real data.

    Guards the arithmetic: the plan's appendix used the 2PR/(P+R) form, which
    differs by one unit in the last place on three runs. sklearn is the tie-break.
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    directory = run_cache_dir(PROJECT_ROOT, pipeline, run)
    if not directory.is_dir():
        pytest.skip(f"cache not present: {directory}")

    labels = canonical_key_labels()
    keys = sorted(labels)
    records = load_validated_run(directory, labels)

    cm = confusion_for(records, labels, keys)
    expected = sklearn_metrics.f1_score(
        [labels[k] for k in keys], [1 if records[k].matched else 0 for k in keys]
    )

    assert cm.f1 == expected


def test_sample_sd_uses_ddof_one() -> None:
    """The canonical SD is the sample definition, not the population one."""
    # For [1, 2, 3]: sample SD = 1.0, population SD = sqrt(2/3) ~ 0.816.
    assert sample_sd([1.0, 2.0, 3.0]) == 1.0


def test_sample_sd_rejects_a_single_observation() -> None:
    """One observation has no sample SD; that must raise rather than return 0."""
    with pytest.raises(ValueError):
        sample_sd([1.0])


def test_kappa_is_one_for_identical_and_zero_for_chance() -> None:
    """Cohen's kappa anchors: perfect agreement is 1.0, chance-level is ~0."""
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0


def test_mcnemar_cells_partition_every_pair(two_pair_labels: dict[tuple[str, str], int]) -> None:
    """The four discordance cells sum to the number of pairs examined."""
    labels = {("1", "2"): 1, ("3", "4"): 0, ("5", "6"): 1}
    left = {
        ("1", "2"): _record("1", "2", "MATCH"),
        ("3", "4"): _record("3", "4", "MATCH"),
        ("5", "6"): _record("5", "6", "MATCH"),
    }
    right = {
        ("1", "2"): _record("1", "2", "MATCH"),
        ("3", "4"): _record("3", "4", "NO MATCH"),
        ("5", "6"): _record("5", "6", "NO MATCH"),
    }
    keys = sorted(labels)

    cells = mcnemar_cells(left, right, labels, keys)

    assert cells["both_correct"] == 1  # ("1","2"): both say MATCH, label 1
    assert cells["c_multi_only_correct"] == 1  # ("3","4"): only right is correct
    assert cells["b_single_only_correct"] == 1  # ("5","6"): only left is correct
    assert cells["neither_correct"] == 0
    total = (
        cells["both_correct"]
        + cells["neither_correct"]
        + cells["b_single_only_correct"]
        + cells["c_multi_only_correct"]
    )
    assert total == len(keys)


# --------------------------------------------------------------------------- #
# Canonical output
# --------------------------------------------------------------------------- #


def test_report_excludes_the_invalidated_tests() -> None:
    """The canonical payload must not carry pooled McNemar or a paired t-test."""
    payload = _production_report()

    blob = json.dumps(payload).lower()
    assert "pooled_mcnemar" in payload["excluded_by_design"]
    assert "paired_t_test" in payload["excluded_by_design"]
    # No computed p-value from either removed test may appear anywhere.
    assert "t_statistic" not in blob
    assert "pooled_mcnemar_p" not in blob


def test_report_reproduces_the_locked_f1_vectors() -> None:
    """Per-run F1 vectors match the plan's locked values to floating-point tolerance."""
    payload = _production_report()

    single = payload["aggregates"]["f1"]["single_llm"]["values"]
    multi = payload["aggregates"]["f1"]["multi_agent"]["values"]

    expected_single = [
        0.9061784897025172,
        0.9041095890410958,
        0.9016018306636155,
        0.9103448275862068,
        0.9103448275862068,
    ]
    expected_multi = [
        0.9056603773584906,
        0.8962264150943396,
        0.8962264150943396,
        0.8942307692307693,
        0.8941176470588236,
    ]

    assert single == pytest.approx(expected_single, abs=1e-15)
    assert multi == pytest.approx(expected_multi, abs=1e-15)


def test_report_reproduces_the_locked_effect_size() -> None:
    """delta_f1 = mean(Single-LLM) - mean(Multi-Agent) matches the locked value."""
    payload = _production_report()
    f1 = payload["aggregates"]["f1"]

    delta = f1["single_llm"]["mean"] - f1["multi_agent"]["mean"]

    assert delta == pytest.approx(0.009223588148576045, abs=1e-15)


def test_report_reproduces_the_locked_token_evidence() -> None:
    """Tokens/pair reproduce 656.1 +/- 0.3 and 5720.0 +/- 13.0 at the paper's precision."""
    payload = _production_report()
    tokens = payload["aggregates"]["tokens_per_pair"]

    assert round(tokens["single_llm"]["mean"], 1) == 656.1
    assert round(tokens["single_llm"]["sample_sd"], 1) == 0.3
    assert round(tokens["multi_agent"]["mean"], 1) == 5720.0
    assert round(tokens["multi_agent"]["sample_sd"], 1) == 13.0
    ratio = tokens["multi_agent"]["mean"] / tokens["single_llm"]["mean"]
    assert round(ratio, 2) == 8.72


def test_report_records_the_known_parse_error_counts() -> None:
    """Parse errors are 0 per Single-LLM run and 4,4,4,5,4 for Multi-Agent."""
    payload = _production_report()

    assert [c["parse_errors"] for c in payload["caches"]["single_llm"]] == [0, 0, 0, 0, 0]
    assert [c["parse_errors"] for c in payload["caches"]["multi_agent"]] == [4, 4, 4, 5, 4]


def test_agreement_is_summarized_over_all_25_run_pairings() -> None:
    """Agreement and kappa summarize all 5x5 pairings, not run i against run i.

    Gate-1 finding 3A: pairing single_llm run i with multi_agent run i is the same
    arbitrary alignment that disqualified the paired t-test, so per-index rows were
    properties of one bijection rather than of the configurations.
    """
    payload = _production_report()
    agreement = payload["agreement"]

    invariant = agreement["pairing_invariant"]
    assert invariant["n_pairings"] == 25
    assert len(agreement["all_pairings"]) == 25
    # Every ordered (single_run, multi_run) combination appears exactly once.
    assert {(row["single_run"], row["multi_run"]) for row in agreement["all_pairings"]} == {
        (s, m) for s in RUNS for m in RUNS
    }
    # The old arbitrary-alignment view must be gone, not merely supplemented.
    assert "per_run" not in agreement
    assert "pooled" not in agreement


def test_pairing_invariant_kappa_spread_exceeds_the_single_alignment_spread() -> None:
    """The honest kappa SD is wider than the one the submitted paper reported.

    The manuscript quotes kappa = 0.942 +/- 0.005, which is the spread across five
    arbitrarily aligned pairs. Across all 25 pairings the spread is larger, which is
    the number the camera-ready must use.
    """
    payload = _production_report()
    kappa = payload["agreement"]["pairing_invariant"]["cohen_kappa"]

    assert round(kappa["mean"], 3) == 0.943
    assert kappa["sample_sd"] > 0.005
    assert kappa["min"] < kappa["max"]


def test_micro_average_disclaims_its_denominator() -> None:
    """The pooled rate is labelled descriptive, with its denominator disclaimed."""
    payload = _production_report()
    micro = payload["agreement"]["micro_average"]

    assert round(micro["agreement_rate"] * 100, 1) == 98.8
    assert micro["decisions"] == 25 * 1916
    note = micro["note"].lower()
    assert "not an independent sample size" in note
    assert "no binomial inference" in note


def test_bootstrap_matches_an_independent_scalar_reference() -> None:
    """The vectorized bootstrap agrees with a hand-rolled scalar computation.

    Gate-1 finding 2D: the other bootstrap tests check determinism and provenance
    strings, so a transposition or gather-index error in the matmul path could pass
    them all while moving the interval. This pins the arithmetic against an
    independent implementation using fixed multiplicities and run selections.
    """
    inputs = _bootstrap_inputs(n_entities=6)
    source_multiplicity = [2, 0, 1, 3, 0, 0]
    target_multiplicity = [1, 1, 0, 2, 2, 0]
    selected_runs = {"single_llm": [0, 2], "multi_agent": [1, 1]}

    weights = pair_weights(
        inputs.pair_source_index,
        inputs.pair_target_index,
        source_multiplicity,
        target_multiplicity,
    )
    # Reference: weighted F1 per selected run, averaged within configuration.
    expected = {}
    for pipeline, runs in selected_runs.items():
        scores = [
            weighted_f1(inputs.labels, inputs.predictions[pipeline][r], weights) for r in runs
        ]
        expected[pipeline] = sum(scores) / len(scores)
    expected_effect = expected["single_llm"] - expected["multi_agent"]

    actual_effect = bootstrap_effect_for_draw(
        inputs, source_multiplicity, target_multiplicity, selected_runs
    )

    assert actual_effect == pytest.approx(expected_effect, rel=1e-12)


def test_checked_in_outputs_equal_a_fresh_build() -> None:
    """results/paper_stats.{json,md} on disk match a freshly computed report.

    Gate-1 finding 4C: the determinism test built the report twice in memory and
    never compared against the committed files, so the suite could pass while the
    canonical output was stale.

    The `producer` block is excluded from the JSON comparison: it fingerprints
    the generating checkout (git head, lockfile hash), which legitimately
    differs across clones and commits. Every scientific field is still compared
    exactly.
    """
    payload = _production_report()

    json_path = PROJECT_ROOT / "results" / "paper_stats.json"
    md_path = PROJECT_ROOT / "results" / "paper_stats.md"
    assert json_path.is_file(), "canonical JSON has never been generated"
    assert md_path.is_file(), "canonical Markdown has never been generated"

    checked_in = json.loads(json_path.read_text())
    fresh = json.loads(serialize(payload))
    checked_in.pop("producer", None)
    fresh.pop("producer", None)
    assert checked_in == fresh, (
        "results/paper_stats.json is stale; regenerate with "
        "`uv run python -m scripts.compute_paper_stats`"
    )
    assert md_path.read_text() == render_markdown(payload), (
        "results/paper_stats.md is stale; regenerate with "
        "`uv run python -m scripts.compute_paper_stats`"
    )


def test_camera_ready_gate_requires_the_caches_to_be_present() -> None:
    """The caches are primary evidence: their absence must fail, not skip.

    Gate-1 finding 4C: every other production test skips when the caches are
    missing, so the suite could pass green in an environment holding none of the
    evidence behind the paper. This one test refuses to skip.
    """
    missing = [
        str(run_cache_dir(PROJECT_ROOT, pipeline, run).relative_to(PROJECT_ROOT))
        for pipeline in PIPELINES
        for run in RUNS
        if not run_cache_dir(PROJECT_ROOT, pipeline, run).is_dir()
    ]

    assert not missing, f"primary prediction caches are absent: {missing}"


def test_rerun_is_byte_identical(tmp_path: Path) -> None:
    """Regenerating the canonical JSON and Markdown produces identical bytes."""
    payload = _production_report()

    first_json = serialize(payload)
    first_md = render_markdown(payload)

    second = build_report(PROJECT_ROOT)
    assert serialize(second) == first_json
    assert render_markdown(second) == first_md


# --------------------------------------------------------------------------- #
# Task A2 -- exact execution-level permutation inference
# --------------------------------------------------------------------------- #

LOCKED_SINGLE_F1 = (
    0.9061784897025172,
    0.9041095890410958,
    0.9016018306636155,
    0.9103448275862068,
    0.9103448275862068,
)
LOCKED_MULTI_F1 = (
    0.9056603773584906,
    0.8962264150943396,
    0.8962264150943396,
    0.8942307692307693,
    0.8941176470588236,
)


def test_studentized_difference_is_welch_style() -> None:
    """The statistic divides the mean gap by the unpooled (Welch) standard error."""
    a = [1.0, 2.0, 3.0]
    b = [0.0, 1.0, 2.0]
    # mean gap 1.0; both sample variances are 1.0, so SE = sqrt(1/3 + 1/3).
    expected = 1.0 / ((1.0 / 3 + 1.0 / 3) ** 0.5)

    assert studentized_mean_difference(a, b) == pytest.approx(expected)


def test_studentized_difference_flips_sign_when_groups_swap() -> None:
    """Swapping the two groups negates the statistic but not its magnitude."""
    forward = studentized_mean_difference(LOCKED_SINGLE_F1, LOCKED_MULTI_F1)
    backward = studentized_mean_difference(LOCKED_MULTI_F1, LOCKED_SINGLE_F1)

    assert forward == pytest.approx(-backward)
    assert forward > 0


def test_permutation_test_reproduces_the_locked_values() -> None:
    """The locked F1 arrays give T=3.3555, 6 of 252 extreme allocations, p=6/252."""
    result = exact_permutation_test(LOCKED_SINGLE_F1, LOCKED_MULTI_F1)

    assert result.total_allocations == 252
    assert result.extreme_allocations == 6
    assert result.p_value == 6 / 252
    assert result.statistic == pytest.approx(3.3555, abs=5e-5)
    assert result.delta == pytest.approx(0.009223588148576045, abs=1e-15)


def test_permutation_p_value_is_the_inclusive_proportion() -> None:
    """p is extreme/total with no Monte Carlo correction added to a full enumeration."""
    result = exact_permutation_test(LOCKED_SINGLE_F1, LOCKED_MULTI_F1)

    assert result.p_value == result.extreme_allocations / result.total_allocations


def test_permutation_test_is_invariant_to_run_ordering() -> None:
    """Shuffling run order within each group cannot change an unpaired test."""
    baseline = exact_permutation_test(LOCKED_SINGLE_F1, LOCKED_MULTI_F1)

    shuffled_single = (
        LOCKED_SINGLE_F1[3],
        LOCKED_SINGLE_F1[0],
        LOCKED_SINGLE_F1[4],
        LOCKED_SINGLE_F1[2],
        LOCKED_SINGLE_F1[1],
    )
    shuffled_multi = (
        LOCKED_MULTI_F1[2],
        LOCKED_MULTI_F1[4],
        LOCKED_MULTI_F1[1],
        LOCKED_MULTI_F1[3],
        LOCKED_MULTI_F1[0],
    )
    shuffled = exact_permutation_test(shuffled_single, shuffled_multi)

    assert shuffled.p_value == baseline.p_value
    assert shuffled.extreme_allocations == baseline.extreme_allocations
    assert shuffled.statistic == pytest.approx(baseline.statistic)


def test_permutation_test_is_symmetric_under_group_swap() -> None:
    """Two-sided p is identical when the group labels are exchanged."""
    forward = exact_permutation_test(LOCKED_SINGLE_F1, LOCKED_MULTI_F1)
    backward = exact_permutation_test(LOCKED_MULTI_F1, LOCKED_SINGLE_F1)

    assert backward.p_value == forward.p_value
    assert backward.extreme_allocations == forward.extreme_allocations
    assert backward.delta == pytest.approx(-forward.delta)


@pytest.mark.parametrize(
    "left,right",
    [
        ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0, 9.0)),  # 4 vs 5
        ((1.0, 2.0, 3.0, 4.0, 5.0), (6.0, 7.0, 8.0, 9.0)),  # 5 vs 4
        ((1.0,), (2.0,)),  # too few for a sample SD
    ],
)
def test_permutation_test_rejects_non_five_five_input(
    left: tuple[float, ...], right: tuple[float, ...]
) -> None:
    """This study's test is defined for exactly five runs per configuration."""
    with pytest.raises(ValueError):
        exact_permutation_test(left, right)


def test_permutation_test_reports_a_maximal_p_for_identical_groups() -> None:
    """Identical groups have zero effect, so every allocation is at least as extreme."""
    same = (0.9, 0.91, 0.92, 0.93, 0.94)

    result = exact_permutation_test(same, same)

    assert result.delta == 0.0
    assert result.p_value == 1.0


def test_report_carries_the_permutation_inference_with_its_scope() -> None:
    """The canonical payload records the test, its assumptions, and its claim scope."""
    payload = _production_report()
    inference = payload["inference"]["permutation_test"]

    assert inference["total_allocations"] == 252
    assert inference["extreme_allocations"] == 6
    assert inference["p_value"] == 6 / 252
    assert inference["delta_f1"] == pytest.approx(0.009223588148576045, abs=1e-15)
    assert inference["effect_definition"] == "mean(single_llm) - mean(multi_agent)"
    assert "exchangeable" in inference["assumptions"].lower()
    # The claim must be explicitly bounded to this fixed test set.
    assert "fixed" in inference["claim_scope"].lower()
    assert "generaliz" in inference["claim_scope"].lower()


# --------------------------------------------------------------------------- #
# Task A3 -- crossed entity/run pigeonhole bootstrap
# --------------------------------------------------------------------------- #


def test_pair_weight_is_the_product_of_source_and_target_multiplicity() -> None:
    """A pair's bootstrap weight is source multiplicity times target multiplicity.

    Hand-checkable bipartite case: sources {s1, s2}, targets {t1, t2}, pairs
    (s1,t1), (s1,t2), (s2,t1). If the source draw picks s1 twice and the target
    draw picks t1 three times, then (s1,t1) weighs 2*3=6, (s1,t2) weighs 2*0=0,
    and (s2,t1) weighs 0*3=0.
    """
    pair_source_index = [0, 0, 1]
    pair_target_index = [0, 1, 0]
    source_multiplicity = [2, 0]
    target_multiplicity = [3, 0]

    weights = pair_weights(
        pair_source_index, pair_target_index, source_multiplicity, target_multiplicity
    )

    assert list(weights) == [6, 0, 0]


def test_weighted_f1_reduces_to_plain_f1_at_unit_weights() -> None:
    """With every weight 1, the weighted F1 equals the ordinary F1."""
    labels = [1, 1, 0, 0]
    predictions = [1, 0, 1, 0]
    weights = [1, 1, 1, 1]

    # tp=1, fn=1, fp=1 -> F1 = 2/(2+1+1) = 0.5
    assert weighted_f1(labels, predictions, weights) == pytest.approx(0.5)


def test_weighted_f1_honours_multiplicity() -> None:
    """Duplicating a pair's weight duplicates its contribution to the confusion cells."""
    labels = [1, 0]
    predictions = [1, 1]
    # tp=3, fp=1 -> F1 = 6/(6+1+0)
    assert weighted_f1(labels, predictions, [3, 1]) == pytest.approx(6 / 7)


def test_weighted_f1_surfaces_a_zero_denominator() -> None:
    """A replicate with no positive labels and no positive predictions must raise."""
    with pytest.raises(ZeroF1DenominatorError):
        weighted_f1([0, 0], [0, 0], [5, 5])


def test_bootstrap_surfaces_an_undefined_replicate_instead_of_coercing_it() -> None:
    """On a degenerate fixture the bootstrap raises, naming the replicate index.

    Three entities per side let a draw zero-weight every positive pair. Silently
    scoring that replicate as F1 = 0 would bias the interval downward, so it must
    surface. The production data (737 sources, 700 targets, 206 positives) never
    reaches this branch, which is exactly why it needs a test.
    """
    with pytest.raises(ZeroF1DenominatorError) as excinfo:
        run_pigeonhole_bootstrap(
            _degenerate_bootstrap_inputs(), replicates=500, seed=20260730
        )

    message = str(excinfo.value)
    assert "replicate" in message.lower()
    assert "coerce" in message.lower()


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    """Two runs with the same seed and replicate count give identical output."""
    payload_a = run_pigeonhole_bootstrap(
        _bootstrap_inputs(), replicates=200, seed=20260730
    )
    payload_b = run_pigeonhole_bootstrap(
        _bootstrap_inputs(), replicates=200, seed=20260730
    )

    assert payload_a == payload_b


def test_bootstrap_changes_with_a_different_seed() -> None:
    """A different seed gives a different draw, proving the seed is actually used."""
    a = run_pigeonhole_bootstrap(_bootstrap_inputs(), replicates=200, seed=20260730)
    b = run_pigeonhole_bootstrap(_bootstrap_inputs(), replicates=200, seed=1)

    assert a["mean_bootstrap_effect"] != b["mean_bootstrap_effect"]


def test_bootstrap_is_invariant_to_chunk_size() -> None:
    """Chunking is a memory strategy, not part of the statistic."""
    whole = run_pigeonhole_bootstrap(
        _bootstrap_inputs(), replicates=200, seed=20260730, chunk_size=200
    )
    split = run_pigeonhole_bootstrap(
        _bootstrap_inputs(), replicates=200, seed=20260730, chunk_size=32
    )

    assert whole == split


def test_bootstrap_resamples_runs_independently_per_configuration() -> None:
    """Run indices are drawn separately for each configuration, not shared.

    If the two configurations shared one run-index draw, a dataset whose per-run F1
    vectors are identical but oppositely ordered would show zero effect variance.
    Independent draws produce a spread.
    """
    inputs = _bootstrap_inputs(mirror_runs=True)

    result = run_pigeonhole_bootstrap(inputs, replicates=400, seed=20260730)

    assert result["range_lower"] != result["range_upper"]


def test_bootstrap_records_its_full_provenance() -> None:
    """The output states algorithm, RNG, seed, replicates, and failure counts."""
    result = run_pigeonhole_bootstrap(
        _bootstrap_inputs(), replicates=200, seed=20260730
    )

    assert result["replicates"] == 200
    assert result["seed"] == 20260730
    assert result["rng"] == "numpy.random.Generator(PCG64)"
    assert result["discarded_replicates"] == 0
    assert "pigeonhole" in result["algorithm"].lower()
    assert "percentile" in result["interval_method"]
    assert result["range_lower"] < result["range_upper"]


def test_report_carries_the_bootstrap_as_a_sensitivity_not_the_primary_test() -> None:
    """The payload labels the bootstrap a generalization sensitivity analysis."""
    payload = _production_report()
    bootstrap = payload["inference"]["fixed_graph_sensitivity"]

    assert bootstrap["replicates"] == 50000
    assert bootstrap["seed"] == 20260730
    assert bootstrap["discarded_replicates"] == 0
    assert "sensitivity" in bootstrap["role"].lower()
    assert "not the primary" in bootstrap["role"].lower()
    # Owen's pigeonhole bootstrap must be credited with its DOI.
    assert bootstrap["citation"]["doi"] == "10.1214/07-AOAS122"
    # The nonlinearity caveat is required: F1 is a ratio, Owen's result is for means.
    assert "nonlinear" in bootstrap["caveat"].lower()


def test_report_states_the_crossed_record_structure_that_motivates_the_bootstrap() -> None:
    """The payload records the 737/700 unique-record counts justifying the method."""
    payload = _production_report()
    structure = payload["inference"]["fixed_graph_sensitivity"]["record_structure"]

    assert structure["unique_sources"] == 737
    assert structure["unique_targets"] == 700
    assert structure["max_source_multiplicity"] == 14
    assert structure["max_target_multiplicity"] == 20


# --------------------------------------------------------------------------- #
# Task A4 -- runtime evidence
# --------------------------------------------------------------------------- #

LOCKED_RUNTIME = {
    "magellan_rf": ([9.9, 10.0, 10.0, 10.2, 10.1], 10.04, 0.1140175425),
    "single_llm": ([2401.2, 2672.2, 3066.1, 2199.1, 2356.0], 2538.92, 340.4359220176),
    "multi_agent": (
        [17143.29, 16820.33, 17048.41, 16414.62, 16965.69],
        16878.468,
        285.1112267519,
    ),
}


def _runtime_payload(**overrides: object) -> dict[str, object]:
    """A minimal well-formed runtime artifact, with fields overridable per test."""
    payload: dict[str, object] = {
        "provenance": {
            "source_file": "time.txt",
            "source_sha256": "abc",
            "measurement_scope": "full test split",
            "execution_mode": "sequential",
            "hardware": "unknown",
            "uncached_claim": "NOT ASSERTED",
            "excluded_source": "data/evaluation_results.json is not used",
        },
        "configurations": [
            {"name": "single_llm", "label": "Single-LLM",
             "seconds": [10.0, 12.0, 11.0, 13.0, 10.5]},
            {"name": "multi_agent", "label": "Multi-Agent",
             "seconds": [60.0, 66.0, 62.0, 70.0, 64.0]},
        ],
    }
    payload.update(overrides)
    return payload


def _write_runtime(tmp_path: Path, payload: object) -> tuple[Path, Path]:
    """Write an artifact plus a matching synthetic timing log.

    The loader now cross-checks the artifact against the log (gate-1 finding 4D),
    so a fixture has to supply both. The log is generated from the payload's own
    vectors, then its digest is written back in, so a well-formed fixture agrees
    with its log and each test isolates the single defect it injects.
    """
    log_lines = ["Running..."]
    prefix = {"magellan_rf": "- Magellan run", "single_llm": "- Single LLM run",
              "multi_agent": "- Multi-Agent run"}
    for configuration in payload["configurations"]:
        for index, value in enumerate(configuration["seconds"]):
            shown = 0.0 if value is None else value
            log_lines.append(
                f"{prefix[configuration['name']]} {index + 1}/5 done: F1=0.9 in {shown}s"
            )
    log = tmp_path / "time.txt"
    log.write_text("\n".join(log_lines) + "\n")

    import hashlib
    payload["provenance"]["source_sha256"] = hashlib.sha256(
        log.read_text().encode()
    ).hexdigest()

    path = tmp_path / "runtime_measurements.json"
    path.write_text(json.dumps(payload))
    return path, log


def test_runtime_artifact_preserves_the_exact_vectors() -> None:
    """The production artifact holds the timing vectors verbatim from time.txt."""
    payload = load_runtime_measurements(runtime_artifact_path(PROJECT_ROOT))

    by_name = {c["name"]: c for c in payload["configurations"]}
    for name, (seconds, _, _) in LOCKED_RUNTIME.items():
        assert by_name[name]["seconds"] == seconds


def test_runtime_summary_matches_the_locked_means_and_sample_sds() -> None:
    """Means and sample SDs reproduce the plan's locked full-precision runtime table."""
    summary = summarize_runtime(load_runtime_measurements(runtime_artifact_path(PROJECT_ROOT)))

    for name, (_, expected_mean, expected_sd) in LOCKED_RUNTIME.items():
        block = summary["per_configuration"][name]
        assert block["mean"] == pytest.approx(expected_mean, abs=1e-9)
        assert block["sample_sd"] == pytest.approx(expected_sd, abs=1e-9)
        assert block["sd_definition"] == "sample (ddof=1)"


def test_runtime_ratio_is_multi_agent_over_single_llm() -> None:
    """The 6.6x headline is explicitly Multi-Agent divided by Single-LLM, not the inverse."""
    summary = summarize_runtime(load_runtime_measurements(runtime_artifact_path(PROJECT_ROOT)))
    ratio = summary["ratio_multi_over_single"]

    assert ratio["definition"] == "mean(multi_agent) / mean(single_llm)"
    assert ratio["value"] == pytest.approx(6.6478928048, abs=1e-9)
    assert ratio["rounded"] == "6.6x"
    assert ratio["value"] > 1  # Multi-Agent is the slower configuration


def test_runtime_sample_sds_are_not_the_population_sds() -> None:
    """The submitted paper's 304.5 / 255.0 were population SDs and must not reappear."""
    summary = summarize_runtime(load_runtime_measurements(runtime_artifact_path(PROJECT_ROOT)))

    single_sd = summary["per_configuration"]["single_llm"]["sample_sd"]
    multi_sd = summary["per_configuration"]["multi_agent"]["sample_sd"]

    assert round(single_sd, 1) != 304.5
    assert round(multi_sd, 1) != 255.0
    assert round(single_sd, 1) == 340.4
    assert round(multi_sd, 1) == 285.1


def test_runtime_artifact_discloses_the_cache_and_curation_story() -> None:
    """The artifact must disclose that two Single-LLM durations are author-recorded.

    Single-LLM runs 1 and 2 restarted and hit the response cache, so their
    harness prints were meaningless; the author recorded the actual wall-clock
    times and curated the log. That provenance must stay disclosed, not erased.
    """
    payload = load_runtime_measurements(runtime_artifact_path(PROJECT_ROOT))

    provenance = payload["provenance"]
    assert "cache" in provenance["uncached_claim"].lower()
    assert "author recorded" in provenance["uncached_claim"]
    assert "log_curation" in provenance
    # The unusable cache-hit source must be explicitly ruled out.
    assert "evaluation_results.json" in provenance["excluded_source"]
    # Hardware was never recorded, so it must stay unknown rather than backfilled.
    assert provenance["hardware"] == "unknown"


def test_runtime_artifact_warns_that_its_f1_column_is_stale() -> None:
    """time.txt's Multi-Agent F1 of 0.8665 disagrees with the caches and must be flagged."""
    payload = load_runtime_measurements(runtime_artifact_path(PROJECT_ROOT))

    warning = payload["provenance"]["stale_f1_warning"]
    assert "0.8665" in warning
    assert "stale" in warning.lower()


def test_runtime_loader_rejects_a_nonpositive_sample(tmp_path: Path) -> None:
    """A zero or negative wall-clock reading is not a measurement."""
    payload = _runtime_payload()
    payload["configurations"][0]["seconds"] = [10.0, 0.0, 11.0, 13.0, 10.5]

    with pytest.raises(RuntimeEvidenceError) as excinfo:
        load_runtime_measurements(*_write_runtime(tmp_path, payload))

    assert "positive" in str(excinfo.value).lower()


def test_runtime_loader_rejects_a_missing_sample(tmp_path: Path) -> None:
    """A null in the timing vector fails rather than being dropped."""
    payload = _runtime_payload()
    payload["configurations"][0]["seconds"] = [10.0, None, 11.0, 13.0, 10.5]

    with pytest.raises(RuntimeEvidenceError) as excinfo:
        load_runtime_measurements(*_write_runtime(tmp_path, payload))

    assert "missing" in str(excinfo.value).lower()


def test_runtime_loader_rejects_a_single_sample(tmp_path: Path) -> None:
    """One observation cannot yield a sample SD, so it is rejected up front."""
    payload = _runtime_payload()
    payload["configurations"][0]["seconds"] = [10.0]

    with pytest.raises(RuntimeEvidenceError):
        load_runtime_measurements(*_write_runtime(tmp_path, payload))


def test_runtime_loader_rejects_an_empty_configuration_list(tmp_path: Path) -> None:
    """An artifact with no configurations is not usable evidence."""
    with pytest.raises(RuntimeEvidenceError):
        load_runtime_measurements(*_write_runtime(tmp_path, _runtime_payload(configurations=[])))


def test_runtime_loader_reports_a_missing_artifact(tmp_path: Path) -> None:
    """An absent artifact fails with its path rather than a bare FileNotFoundError."""
    with pytest.raises(RuntimeEvidenceError) as excinfo:
        load_runtime_measurements(tmp_path / "absent.json")

    assert "absent.json" in str(excinfo.value)


def test_runtime_summary_needs_both_llm_configurations(tmp_path: Path) -> None:
    """The ratio is undefined without both configurations present."""
    payload = _runtime_payload()
    payload["configurations"] = [payload["configurations"][0]]

    with pytest.raises(RuntimeEvidenceError):
        summarize_runtime(load_runtime_measurements(*_write_runtime(tmp_path, payload)))


def test_report_carries_the_runtime_evidence() -> None:
    """The canonical payload exposes the runtime table the paper prints."""
    payload = _production_report()
    runtime = payload["runtime"]

    assert runtime["per_configuration"]["multi_agent"]["mean"] == pytest.approx(
        16878.468, abs=1e-9
    )
    assert runtime["ratio_multi_over_single"]["rounded"] == "6.6x"
    assert runtime["provenance"]["source_file"] == "time.txt"
    # The cache-and-curation disclosure must reach the canonical output.
    assert "cache" in runtime["provenance"]["uncached_claim"].lower()
    assert "log_curation" in runtime["provenance"]
    # Gate-1 finding 4E: the stale-F1 warning must reach the canonical output.
    assert "0.8665" in runtime["provenance"]["stale_f1_warning"]
    assert "4F" in runtime["like_for_like_caveat"]


def test_markdown_is_rendered_from_the_json_payload() -> None:
    """The Markdown view quotes values that exist in the payload, not hardcoded text."""
    payload = _production_report()

    markdown = render_markdown(payload)

    assert payload["dataset"]["key_label_sha256"] in markdown
    assert "1916" in markdown
    assert "sample SD" in markdown
