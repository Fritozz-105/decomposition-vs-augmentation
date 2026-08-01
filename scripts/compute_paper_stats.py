"""Canonical paper statistics for the KDD-UC '26 camera-ready.

Reads the ten authoritative prediction caches under
`results/{single_llm,multi_agent}/eval_run_{0..4}` and the labeled Abt-Buy test
split, and emits every number the paper reports to `results/paper_stats.json`.
`results/paper_stats.md` is rendered *from* that JSON, so the human view can never
drift from the machine-readable source.

Two design rules come from the Phase 7 plan and are load-bearing:

1. **Validate before aggregating.** Every cache must be proven to hold exactly the
   1,916 labeled test keys, once each, before any statistic is computed. The
   previous version of this script intersected "keys present in both caches",
   which turned a missing-record bug into a silently smaller denominator.
2. **Sample standard deviation everywhere** (`ddof=1`). The submitted manuscript
   reported population SDs for the same vectors; those numbers are superseded.

No pooled McNemar test and no paired run-level t-test appear here. Both were
invalid for this design -- the discordant pairs are pseudoreplicated across runs,
and pairing arbitrary run indices makes the p-value an artifact of run ordering.
Per-run McNemar cells are retained below as descriptive diagnostics only. The
paper's inferential claim comes from the exact execution-level permutation test.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from src.data.loader import load_abt_buy
from src.utils.paths import find_project_root

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PIPELINES: tuple[str, ...] = ("single_llm", "multi_agent")
RUNS: tuple[int, ...] = (0, 1, 2, 3, 4)

# The cached verdict vocabulary. "NO MATCH" is space-separated, not underscored;
# anything else means the parser or the cache changed and must not be coerced.
VALID_VERDICTS: frozenset[str] = frozenset({"MATCH", "NO MATCH"})

SCHEMA_VERSION = 1

PairKey = tuple[str, str]


class CacheValidationError(RuntimeError):
    """A prediction cache is not the exact, complete artifact the paper claims."""


# --------------------------------------------------------------------------- #
# Cache records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CacheRecord:
    """One cached pair decision."""

    source_id: str
    target_id: str
    verdict: str
    confidence: float | None
    parse_error: bool
    tokens: int

    @property
    def key(self) -> PairKey:
        return (self.source_id, self.target_id)

    @property
    def matched(self) -> bool:
        return self.verdict == "MATCH"


def run_cache_dir(root: Path, pipeline: str, run: int) -> Path:
    """Path of one authoritative run cache."""
    return root / "results" / pipeline / f"eval_run_{run}"


def _parse_record(path: Path) -> CacheRecord:
    """Parse one cache file, raising CacheValidationError with the filename on any defect."""
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CacheValidationError(f"{path.name}: not valid JSON ({exc})") from exc

    if not isinstance(payload, dict) or "prediction" not in payload:
        raise CacheValidationError(f"{path.name}: missing top-level 'prediction' object")

    prediction = payload["prediction"]
    if not isinstance(prediction, dict):
        raise CacheValidationError(f"{path.name}: 'prediction' is not an object")

    for field in ("source_id", "target_id", "verdict"):
        if field not in prediction:
            raise CacheValidationError(f"{path.name}: prediction is missing '{field}'")

    verdict = prediction["verdict"]
    if verdict not in VALID_VERDICTS:
        raise CacheValidationError(
            f"{path.name}: malformed verdict {verdict!r}; "
            f"expected one of {sorted(VALID_VERDICTS)}"
        )

    # Token counts feed the paper's cost comparison, so a record without them is
    # not usable evidence even though the verdict alone would be.
    if "tokens" not in payload:
        raise CacheValidationError(f"{path.name}: missing 'tokens' count")
    tokens = payload["tokens"]
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        raise CacheValidationError(
            f"{path.name}: 'tokens' must be a non-negative integer, got {tokens!r}"
        )

    confidence = prediction.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise CacheValidationError(
            f"{path.name}: 'confidence' must be numeric or absent, got {confidence!r}"
        )

    # The multi-agent cache also carries a numeric `prediction` mirror of the
    # verdict. Where present it must agree; a disagreement means the two fields
    # disagree about what the system actually decided.
    mirror = prediction.get("prediction")
    if mirror is not None:
        expected = 1 if verdict == "MATCH" else 0
        if mirror != expected:
            raise CacheValidationError(
                f"{path.name}: numeric prediction {mirror!r} contradicts verdict {verdict!r}"
            )

    return CacheRecord(
        source_id=str(prediction["source_id"]),
        target_id=str(prediction["target_id"]),
        verdict=verdict,
        confidence=None if confidence is None else float(confidence),
        parse_error=bool(prediction.get("parse_error", False)),
        tokens=tokens,
    )


def read_cache_records(directory: Path) -> dict[PairKey, CacheRecord]:
    """Parse every ``*.json`` in ``directory``, rejecting duplicates and malformed records."""
    if not directory.is_dir():
        raise CacheValidationError(f"cache directory does not exist: {directory}")

    records: dict[PairKey, CacheRecord] = {}
    seen_in: dict[PairKey, str] = {}

    for path in sorted(directory.glob("*.json")):
        record = _parse_record(path)
        if record.key in records:
            raise CacheValidationError(
                f"{directory.name}: duplicate key {record.key} appears in both "
                f"{seen_in[record.key]} and {path.name}"
            )
        records[record.key] = record
        seen_in[record.key] = path.name

    if not records:
        raise CacheValidationError(f"cache directory holds no *.json records: {directory}")

    return records


def _sample(keys: Sequence[PairKey], limit: int = 3) -> str:
    shown = ", ".join(repr(k) for k in keys[:limit])
    return shown if len(keys) <= limit else f"{shown}, ... (+{len(keys) - limit} more)"


def validate_run_cache(
    directory: Path,
    records: Mapping[PairKey, CacheRecord],
    expected: Mapping[PairKey, int],
) -> None:
    """Raise unless ``records`` covers ``expected`` exactly -- no missing, no extra keys."""
    found = set(records)
    wanted = set(expected)

    missing = sorted(wanted - found)
    extra = sorted(found - wanted)

    if not missing and not extra:
        return

    problems: list[str] = []
    if missing:
        problems.append(f"{len(missing)} missing labeled key(s): {_sample(missing)}")
    if extra:
        problems.append(f"{len(extra)} extra unlabeled key(s): {_sample(extra)}")

    raise CacheValidationError(
        f"{directory}: cache does not match the canonical test split "
        f"(expected {len(wanted)} keys, found {len(found)}); " + "; ".join(problems)
    )


def load_validated_run(
    directory: Path, expected: Mapping[PairKey, int]
) -> dict[PairKey, CacheRecord]:
    """Read a run cache and prove it matches the canonical split before returning it."""
    records = read_cache_records(directory)
    validate_run_cache(directory, records, expected)
    return records


# --------------------------------------------------------------------------- #
# Canonical labels
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def canonical_key_labels() -> dict[PairKey, int]:
    """The exact labeled Abt-Buy test split as a (source_id, target_id) -> label map."""
    test_pairs = load_abt_buy()["test_pairs_df"]

    labels: dict[PairKey, int] = {}
    for row in test_pairs.itertuples(index=False):
        key = (str(row.ltable_id), str(row.rtable_id))
        if key in labels:
            raise CacheValidationError(
                f"the labeled test split itself contains duplicate key {key}"
            )
        labels[key] = int(row.label)
    return labels


def key_label_digest(labels: Mapping[PairKey, int]) -> str:
    """Stable digest of the canonical key-label map, for the reproducibility manifest."""
    h = hashlib.sha256()
    for key in sorted(labels):
        h.update(f"{key[0]}\t{key[1]}\t{labels[key]}\n".encode())
    return h.hexdigest()


def records_digest(records: Mapping[PairKey, CacheRecord]) -> str:
    """Stable digest of one run's decisions, independent of file layout."""
    h = hashlib.sha256()
    for key in sorted(records):
        r = records[key]
        h.update(f"{key[0]}\t{key[1]}\t{r.verdict}\t{r.tokens}\t{int(r.parse_error)}\n".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Confusion:
    """Binary confusion counts against the labeled split."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "n": self.n,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


def confusion_for(
    records: Mapping[PairKey, CacheRecord],
    labels: Mapping[PairKey, int],
    keys: Sequence[PairKey],
) -> Confusion:
    """Confusion counts over ``keys`` in the given order."""
    tp = fp = fn = tn = 0
    for key in keys:
        predicted = records[key].matched
        actual = labels[key] == 1
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence")
    return sum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1), the paper's canonical definition."""
    if len(values) < 2:
        raise ValueError("sample standard deviation needs at least two observations")
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def summarize(values: Sequence[float]) -> dict[str, object]:
    """Mean, sample SD, and the underlying vector at full precision."""
    return {
        "values": list(values),
        "mean": mean(values),
        "sample_sd": sample_sd(values),
        "n": len(values),
        "sd_definition": "sample (ddof=1)",
    }


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa between two binary decision vectors."""
    if len(a) != len(b):
        raise ValueError("kappa needs equal-length vectors")
    n = len(a)
    if n == 0:
        raise ValueError("kappa of empty vectors")

    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def mcnemar_cells(
    left: Mapping[PairKey, CacheRecord],
    right: Mapping[PairKey, CacheRecord],
    labels: Mapping[PairKey, int],
    keys: Sequence[PairKey],
) -> dict[str, int]:
    """Discordance cells: b = only-left-correct, c = only-right-correct.

    Descriptive diagnostic only. These are deliberately NOT pooled across runs and
    NOT converted into a p-value here: the same 1,916 pairs recur in all five runs,
    so pooling the discordances pseudoreplicates them.
    """
    b = c = both = neither = 0
    for key in keys:
        actual = labels[key] == 1
        left_ok = left[key].matched == actual
        right_ok = right[key].matched == actual
        if left_ok and not right_ok:
            b += 1
        elif right_ok and not left_ok:
            c += 1
        elif left_ok and right_ok:
            both += 1
        else:
            neither += 1
    return {
        "b_single_only_correct": b,
        "c_multi_only_correct": c,
        "both_correct": both,
        "neither_correct": neither,
        "discordant": b + c,
    }


# --------------------------------------------------------------------------- #
# Exact execution-level inference
# --------------------------------------------------------------------------- #

# This study ran each configuration five times. The permutation test below is
# defined for exactly that design; a different shape means the caller is holding
# something other than the paper's run-level F1 vectors.
RUNS_PER_CONFIGURATION = 5


def studentized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float:
    """Welch-style studentized mean difference: gap over the unpooled standard error.

    Gate-1 finding 1A: studentizing does NOT buy finite-sample exactness. Exactness
    comes from exchangeability under the null. What studentizing adds is asymptotic
    robustness for the weaker null of equal means when the two configurations have
    different run-to-run spreads (Chung & Romano, arXiv:1304.5939), which is why it
    is preferred here -- not because an unstudentized statistic would be invalid.
    """
    if len(left) < 2 or len(right) < 2:
        raise ValueError("each group needs at least two observations")

    gap = mean(left) - mean(right)
    standard_error = math.sqrt(
        sample_sd(left) ** 2 / len(left) + sample_sd(right) ** 2 / len(right)
    )
    if standard_error == 0.0:
        # Degenerate: no within-group variation at all. A zero gap is "not extreme";
        # a nonzero gap under zero variance is maximally extreme.
        return 0.0 if gap == 0.0 else math.inf * (1 if gap > 0 else -1)
    return gap / standard_error


@dataclass(frozen=True)
class PermutationResult:
    """Outcome of the exact two-sided unpaired permutation test."""

    statistic: float
    delta: float
    total_allocations: int
    extreme_allocations: int
    p_value: float

    def as_dict(self) -> dict[str, object]:
        return {
            "test": "exact two-sided unpaired studentized permutation test",
            "unit_of_inference": (
                "one stochastic execution (run). Executions are ASSUMED independent "
                "and identically distributed; no seeds, invocation ids, timestamps, or "
                "provider builds were captured for these runs, so independence is an "
                "assumption rather than an established property."
            ),
            "statistic": self.statistic,
            "statistic_definition": (
                "(mean(single_llm) - mean(multi_agent)) / "
                "sqrt(s_single^2/5 + s_multi^2/5), sample variances (ddof=1)"
            ),
            "delta_f1": self.delta,
            "effect_definition": "mean(single_llm) - mean(multi_agent)",
            "effect_direction": (
                "single_llm higher" if self.delta > 0 else
                "multi_agent higher" if self.delta < 0 else "no difference"
            ),
            "total_allocations": self.total_allocations,
            "extreme_allocations": self.extreme_allocations,
            "p_value": self.p_value,
            "p_value_definition": (
                "inclusive proportion of all allocations whose |statistic| is at least "
                "the observed |statistic|; complete enumeration, so no Monte Carlo "
                "correction is applied"
            ),
            "assumptions": (
                "Conditions on the fixed 1,916-pair Abt-Buy test set. The p-value is "
                "exact under the null that the ten executions are EXCHANGEABLE, i.e. "
                "identically distributed across configurations -- a stronger null than "
                "equality of means. Studentizing gives asymptotic robustness for the "
                "weaker equal-means null but does not make the test finite-sample exact "
                "for it. Does not assume normality and does not pair run indices."
            ),
            "claim_scope": (
                "Under the exchangeability null above, supports only: across repeated "
                "executions on this fixed Abt-Buy test "
                "set, the Single-LLM configuration had higher mean F1 than the "
                "Multi-Agent configuration. It does not establish cross-catalog, "
                "cross-dataset, or cross-model generalization, and it compares bundled "
                "configurations rather than isolating agent decomposition."
            ),
        }


def exact_permutation_test(
    left: Sequence[float], right: Sequence[float]
) -> PermutationResult:
    """Enumerate every five/five allocation of the ten observed run-level values.

    With five runs per configuration there are C(10,5) = 252 allocations, so the
    p-value is exact by complete enumeration rather than sampled.
    """
    if len(left) != RUNS_PER_CONFIGURATION or len(right) != RUNS_PER_CONFIGURATION:
        raise ValueError(
            f"expected {RUNS_PER_CONFIGURATION} observations per configuration, "
            f"got {len(left)} and {len(right)}"
        )

    observed = studentized_mean_difference(left, right)
    pooled = list(left) + list(right)
    indices = range(len(pooled))

    total = 0
    extreme = 0
    for group in itertools.combinations(indices, RUNS_PER_CONFIGURATION):
        chosen = set(group)
        a = [pooled[i] for i in indices if i in chosen]
        b = [pooled[i] for i in indices if i not in chosen]
        total += 1
        if abs(studentized_mean_difference(a, b)) >= abs(observed):
            extreme += 1

    return PermutationResult(
        statistic=observed,
        delta=mean(left) - mean(right),
        total_allocations=total,
        extreme_allocations=extreme,
        p_value=extreme / total,
    )


# --------------------------------------------------------------------------- #
# Crossed entity/run pigeonhole bootstrap (generalization sensitivity)
# --------------------------------------------------------------------------- #

# Why this exists rather than a pair-level bootstrap: the 1,916 test pairs are
# built from only 737 unique source records and 700 unique target records, with
# 85.4% of pairs sharing a source and 86.6% sharing a target. Resampling pairs
# independently would treat heavily overlapping observations as independent.

BOOTSTRAP_SEED = 20260730
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_CHUNK_SIZE = 2_500

OWEN_CITATION = {
    "author": "Art B. Owen",
    "title": "The Pigeonhole Bootstrap",
    "journal": "Annals of Applied Statistics",
    "volume": "1",
    "number": "2",
    "pages": "386--411",
    "year": "2007",
    "doi": "10.1214/07-AOAS122",
}


class ZeroF1DenominatorError(RuntimeError):
    """A bootstrap replicate produced an undefined F1 rather than a value to coerce."""


@dataclass(frozen=True)
class BootstrapInputs:
    """Everything the crossed bootstrap needs, in one fixed pair ordering."""

    labels: tuple[int, ...]
    pair_source_index: tuple[int, ...]
    pair_target_index: tuple[int, ...]
    n_sources: int
    n_targets: int
    # pipeline -> per-run prediction vectors, each aligned with ``labels``
    predictions: Mapping[str, tuple[tuple[int, ...], ...]]


def pair_weights(
    pair_source_index: Sequence[int],
    pair_target_index: Sequence[int],
    source_multiplicity: Sequence[int],
    target_multiplicity: Sequence[int],
) -> list[int]:
    """Weight each pair by source multiplicity times target multiplicity."""
    return [
        source_multiplicity[s] * target_multiplicity[t]
        for s, t in zip(pair_source_index, pair_target_index)
    ]


def weighted_f1(
    labels: Sequence[int], predictions: Sequence[int], weights: Sequence[float]
) -> float:
    """F1 over weighted confusion cells; raises rather than coercing an undefined value."""
    tp = fp = fn = 0.0
    for label, prediction, weight in zip(labels, predictions, weights):
        if prediction and label:
            tp += weight
        elif prediction and not label:
            fp += weight
        elif not prediction and label:
            fn += weight

    denominator = 2 * tp + fp + fn
    if denominator == 0:
        raise ZeroF1DenominatorError(
            "weighted F1 is undefined: this replicate selected no positive labels and "
            "no positive predictions (2*tp + fp + fn == 0)"
        )
    return (2 * tp) / denominator


def _prediction_masks(
    labels: "np.ndarray", runs: tuple[tuple[int, ...], ...]
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Per-run boolean confusion masks as float matrices of shape (n_runs, n_pairs)."""
    import numpy as np

    predicted = np.asarray(runs, dtype=bool)
    actual = labels.astype(bool)
    return (
        (predicted & actual).astype(np.float64),
        (predicted & ~actual).astype(np.float64),
        (~predicted & actual).astype(np.float64),
    )


def _weighted_f1_matrix(
    weights: "np.ndarray",
    masks: tuple["np.ndarray", "np.ndarray", "np.ndarray"],
) -> tuple["np.ndarray", "np.ndarray"]:
    """Weighted F1 for every (replicate, run) pair, plus its denominator.

    The single place the vectorized arithmetic lives, so the production loop and the
    scalar-oracle test in the suite exercise identical code (gate-1 finding 2D).
    """
    tp_mask, fp_mask, fn_mask = masks
    tp = weights @ tp_mask.T
    fp = weights @ fp_mask.T
    fn = weights @ fn_mask.T
    denominator = 2 * tp + fp + fn
    return 2 * tp, denominator


def bootstrap_effect_for_draw(
    inputs: BootstrapInputs,
    source_multiplicity: Sequence[int],
    target_multiplicity: Sequence[int],
    selected_runs: Mapping[str, Sequence[int]],
) -> float:
    """One replicate's effect from explicit draws, via the vectorized code path.

    Exists so a test can pin the matmul/gather arithmetic against an independent
    scalar computation with predetermined multiplicities and run selections.
    """
    import numpy as np

    labels = np.asarray(inputs.labels, dtype=np.int64)
    weights = np.asarray(
        [
            pair_weights(
                inputs.pair_source_index,
                inputs.pair_target_index,
                source_multiplicity,
                target_multiplicity,
            )
        ],
        dtype=np.float64,
    )

    means: list[float] = []
    for pipeline in ("single_llm", "multi_agent"):
        numerator, denominator = _weighted_f1_matrix(
            weights, _prediction_masks(labels, inputs.predictions[pipeline])
        )
        chosen = list(selected_runs[pipeline])
        if not np.all(denominator[0, chosen] > 0):
            raise ZeroF1DenominatorError("undefined weighted F1 for the supplied draw")
        means.append(float((numerator[0, chosen] / denominator[0, chosen]).mean()))
    return means[0] - means[1]


def run_pigeonhole_bootstrap(
    inputs: BootstrapInputs,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
) -> dict[str, object]:
    """Two-way entity-cluster bootstrap of the F1 difference between configurations.

    Per replicate: resample the unique source records and the unique target records
    independently with replacement, weight each observed pair by the product of its
    two multiplicities, then independently resample five run indices with
    replacement *within each configuration* and average the weighted F1 across the
    selected runs. The stored statistic is
    ``mean_f1(single_llm) - mean_f1(multi_agent)``.

    Processed in bounded chunks so peak memory stays independent of ``replicates``.
    """
    import numpy as np

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    labels = np.asarray(inputs.labels, dtype=np.int64)
    source_index = np.asarray(inputs.pair_source_index, dtype=np.int64)
    target_index = np.asarray(inputs.pair_target_index, dtype=np.int64)

    pipelines = tuple(inputs.predictions)
    masks = {p: _prediction_masks(labels, inputs.predictions[p]) for p in pipelines}
    n_runs = {p: len(inputs.predictions[p]) for p in pipelines}

    # Observed effect at unit weights, for comparison against the bootstrap mean.
    observed = {
        p: mean(
            [
                weighted_f1(inputs.labels, run, [1] * len(inputs.labels))
                for run in inputs.predictions[p]
            ]
        )
        for p in pipelines
    }
    point_estimate = observed[pipelines[0]] - observed[pipelines[1]]

    # One generator per kind of draw, each consumed strictly in replicate order.
    # A single shared generator would interleave the four draw kinds per chunk, so
    # chunk_size would silently change the resulting interval -- the chunking is a
    # memory strategy and must not be part of the statistic.
    streams = np.random.SeedSequence(seed).spawn(2 + len(pipelines))
    source_rng = np.random.default_rng(streams[0])
    target_rng = np.random.default_rng(streams[1])
    run_rngs = {
        pipeline: np.random.default_rng(streams[2 + i])
        for i, pipeline in enumerate(pipelines)
    }

    effects = np.empty(replicates, dtype=np.float64)
    uniform_sources = np.full(inputs.n_sources, 1.0 / inputs.n_sources)
    uniform_targets = np.full(inputs.n_targets, 1.0 / inputs.n_targets)

    filled = 0
    while filled < replicates:
        size = min(chunk_size, replicates - filled)

        # Drawing n ids with replacement from n is exactly a multinomial count vector.
        source_counts = source_rng.multinomial(inputs.n_sources, uniform_sources, size=size)
        target_counts = target_rng.multinomial(inputs.n_targets, uniform_targets, size=size)
        weights = (
            source_counts[:, source_index] * target_counts[:, target_index]
        ).astype(np.float64)

        chunk_means = []
        for pipeline in pipelines:
            numerator, denominator = _weighted_f1_matrix(weights, masks[pipeline])

            selected = run_rngs[pipeline].integers(
                0, n_runs[pipeline], size=(size, n_runs[pipeline])
            )
            rows = np.arange(size)[:, None]
            chosen_den = denominator[rows, selected]
            if not np.all(chosen_den > 0):
                bad = int(np.argmax(~(chosen_den > 0)) // n_runs[pipeline]) + filled
                raise ZeroF1DenominatorError(
                    f"replicate {bad} produced an undefined weighted F1 for "
                    f"{pipeline!r} (2*tp + fp + fn == 0); refusing to coerce it to zero"
                )
            chosen_f1 = numerator[rows, selected] / chosen_den
            chunk_means.append(chosen_f1.mean(axis=1))

        effects[filled : filled + size] = chunk_means[0] - chunk_means[1]
        filled += size

    lower, upper = np.percentile(effects, [2.5, 97.5])
    return {
        "algorithm": (
            "two-way entity-cluster (pigeonhole-style) bootstrap over unique source and "
            "target records, crossed with independent within-configuration resampling of "
            "run indices"
        ),
        "rng": "numpy.random.Generator(PCG64)",
        "seed": seed,
        "replicates": replicates,
        "effect_definition": f"mean_f1({pipelines[0]}) - mean_f1({pipelines[1]})",
        "point_estimate_source": point_estimate,
        "mean_bootstrap_effect": float(effects.mean()),
        "range_lower": float(lower),
        "range_upper": float(upper),
        "interval_method": (
            "percentile (2.5%, 97.5%) -- a sensitivity range, not a calibrated "
            "confidence interval"
        ),
        "discarded_replicates": 0,
        "range_includes_zero": bool(lower <= 0.0 <= upper),
    }


def bootstrap_inputs_from_caches(
    labels: Mapping[PairKey, int],
    keys: Sequence[PairKey],
    loaded: Mapping[str, Mapping[int, Mapping[PairKey, CacheRecord]]],
) -> BootstrapInputs:
    """Assemble bootstrap inputs in the canonical sorted-key ordering."""
    source_ids = sorted({k[0] for k in keys})
    target_ids = sorted({k[1] for k in keys})
    source_slot = {s: i for i, s in enumerate(source_ids)}
    target_slot = {t: i for i, t in enumerate(target_ids)}

    return BootstrapInputs(
        labels=tuple(labels[k] for k in keys),
        pair_source_index=tuple(source_slot[k[0]] for k in keys),
        pair_target_index=tuple(target_slot[k[1]] for k in keys),
        n_sources=len(source_ids),
        n_targets=len(target_ids),
        predictions={
            pipeline: tuple(
                tuple(1 if loaded[pipeline][run][k].matched else 0 for k in keys)
                for run in RUNS
            )
            for pipeline in PIPELINES
        },
    )


# --------------------------------------------------------------------------- #
# Runtime evidence
# --------------------------------------------------------------------------- #


class RuntimeEvidenceError(RuntimeError):
    """The runtime artifact is missing, malformed, or holds unusable samples."""


def _file_digest(path: Path) -> str:
    """SHA-256 of a file, or "absent" so a missing optional input is visible."""
    if not path.is_file():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def producer_provenance(root: Path) -> dict[str, object]:
    """Identify the code and environment that produced this output.

    Gate-1 finding 4B: content hashes of the inputs are not enough. Without the
    producer's own hash and the dependency lock, a reader cannot tell which version
    of the analysis generated a number.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        head, dirty = "unknown", True

    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "absent"

    return {
        "producer_script": "scripts/compute_paper_stats.py",
        "producer_script_sha256": _file_digest(Path(__file__).resolve()),
        "lockfile_sha256": _file_digest(root / "uv.lock"),
        "python_version": sys.version.split()[0],
        "numpy_version": numpy_version,
        "git_head": head,
        "git_dirty": dirty,
        "git_note": (
            "git_head and lockfile_sha256 describe the checkout that generated "
            "this file and are expected to differ across clones and commits. "
            "The producer_script_sha256 above is the authoritative identifier "
            "for this analysis code."
        ),
    }


def runtime_artifact_path(root: Path) -> Path:
    return root / "results" / "runtime_measurements.json"


# Terminal duration on a timing log line. The optional "Xs -> " prefix
# tolerates an older annotation format in which a cache-affected harness print
# preceded the author-recorded actual duration; anchoring at the end of the
# line takes the actual one in either format.
_TIMING_LINE = re.compile(r"in\s+(?:[0-9.]+s\s*->\s*)?([0-9.]+)s\s*$")


def parse_timing_log(text: str) -> dict[str, list[float]]:
    """Extract per-configuration terminal durations from the raw `time.txt` log.

    Exists so the transcribed artifact can be checked against primary evidence
    instead of against a second hardcoded copy of the same vector.
    """
    prefixes = {
        "- Magellan run": "magellan_rf",
        "- Single LLM run": "single_llm",
        "- Multi-Agent run": "multi_agent",
    }
    parsed: dict[str, list[float]] = {name: [] for name in prefixes.values()}
    for line in text.splitlines():
        stripped = line.strip()
        for prefix, name in prefixes.items():
            if stripped.startswith(prefix):
                match = _TIMING_LINE.search(stripped)
                if match is None:
                    raise RuntimeEvidenceError(
                        f"timing log line has no parseable duration: {stripped!r}"
                    )
                parsed[name].append(float(match.group(1)))
                break
    return parsed


def load_runtime_measurements(
    path: Path, timing_log: Path | None = None
) -> dict[str, object]:
    """Load the transcribed wall-clock observations and verify them against `time.txt`.

    ``timing_log`` defaults to the ``provenance.source_file`` recorded in the
    artifact, resolved next to the repository root. Pass it explicitly in tests.
    """
    if not path.is_file():
        raise RuntimeEvidenceError(f"runtime artifact not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeEvidenceError(f"{path.name}: not valid JSON ({exc})") from exc

    configurations = payload.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise RuntimeEvidenceError(f"{path.name}: 'configurations' must be a non-empty list")

    seen: set[str] = set()
    for configuration in configurations:
        name = configuration.get("name", "<unnamed>")
        if name in seen:
            raise RuntimeEvidenceError(f"duplicate configuration name: {name!r}")
        seen.add(name)

        seconds = configuration.get("seconds")
        if not isinstance(seconds, list) or not seconds:
            raise RuntimeEvidenceError(f"{name}: 'seconds' must be a non-empty list")
        if len(seconds) != len(RUNS):
            raise RuntimeEvidenceError(
                f"{name}: expected {len(RUNS)} timing samples, got {len(seconds)}"
            )
        for index, value in enumerate(seconds):
            if value is None:
                raise RuntimeEvidenceError(f"{name}: timing sample {index} is missing")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RuntimeEvidenceError(
                    f"{name}: timing sample {index} is not numeric ({value!r})"
                )
            if value <= 0:
                raise RuntimeEvidenceError(
                    f"{name}: timing sample {index} is not positive ({value!r}); "
                    "a wall-clock measurement cannot be zero or negative"
                )

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeEvidenceError(f"{path.name}: 'provenance' block is required")

    log_path = timing_log if timing_log is not None else path.parent.parent / provenance["source_file"]
    if not log_path.is_file():
        raise RuntimeEvidenceError(f"timing log not found: {log_path}")

    log_text = log_path.read_text()
    actual_digest = hashlib.sha256(log_text.encode()).hexdigest()
    recorded_digest = provenance.get("source_sha256")
    if actual_digest != recorded_digest:
        raise RuntimeEvidenceError(
            f"{log_path.name} has changed since transcription: recorded "
            f"{recorded_digest}, actual {actual_digest}. The artifact is no longer "
            "a faithful transcription and must be re-derived."
        )

    parsed = parse_timing_log(log_text)
    for configuration in configurations:
        name = configuration["name"]
        if parsed.get(name) != list(configuration["seconds"]):
            raise RuntimeEvidenceError(
                f"{name}: transcribed vector {configuration['seconds']} does not match "
                f"the durations parsed from {log_path.name}: {parsed.get(name)}"
            )

    return payload


def summarize_runtime(payload: Mapping[str, object]) -> dict[str, object]:
    """Full-precision runtime summary plus the Multi-Agent / Single-LLM ratio."""
    configurations = payload["configurations"]
    summaries: dict[str, object] = {}
    for configuration in configurations:
        seconds = configuration["seconds"]
        summaries[configuration["name"]] = {
            "label": configuration["label"],
            **summarize(seconds),
        }

    if "multi_agent" not in summaries or "single_llm" not in summaries:
        raise RuntimeEvidenceError(
            "the runtime ratio needs both 'multi_agent' and 'single_llm' configurations"
        )

    multi_mean = summaries["multi_agent"]["mean"]
    single_mean = summaries["single_llm"]["mean"]
    ratio = multi_mean / single_mean

    return {
        "per_configuration": summaries,
        "ratio_multi_over_single": {
            "value": ratio,
            "definition": "mean(multi_agent) / mean(single_llm)",
            "rounded": f"{ratio:.1f}x",
        },
        # Gate-1 finding 4E: the whole provenance block propagates. Copying a
        # curated subset silently dropped stale_f1_warning -- the single most
        # important caveat on this evidence -- from the canonical output.
        "provenance": dict(payload["provenance"]),
        "sd_definition": "sample (ddof=1)",
        "like_for_like_caveat": (
            "Gate-1 finding 4F, unresolved: nothing in the repository links these "
            "timing executions to the prediction caches behind the accuracy numbers. "
            "The co-located F1 column in time.txt reads 0.8665 for all five "
            "Multi-Agent lines and does not match those caches. Retaining the "
            "durations while discarding that F1 is selective use of one source. The "
            "vectors are kept by explicit user decision (2026-07-30) with provenance "
            "disclosed; they are NOT established as a like-for-like comparison of the "
            "same three systems."
        ),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def build_report(root: Path) -> dict[str, object]:
    """Validate all ten caches and compute every canonical paper statistic."""
    labels = canonical_key_labels()
    keys = sorted(labels)  # fixed order: makes every downstream vector reproducible

    loaded: dict[str, dict[int, dict[PairKey, CacheRecord]]] = {}
    cache_meta: dict[str, list[dict[str, object]]] = {}

    for pipeline in PIPELINES:
        loaded[pipeline] = {}
        cache_meta[pipeline] = []
        for run in RUNS:
            directory = run_cache_dir(root, pipeline, run)
            records = load_validated_run(directory, labels)
            loaded[pipeline][run] = records

            tokens = [records[k].tokens for k in keys]
            cache_meta[pipeline].append(
                {
                    "run": run,
                    "directory": str(directory.relative_to(root)),
                    "n_records": len(records),
                    "records_sha256": records_digest(records),
                    "parse_errors": sum(1 for k in keys if records[k].parse_error),
                    "tokens_total": sum(tokens),
                    "tokens_per_pair": sum(tokens) / len(keys),
                }
            )

    per_run: dict[str, list[dict[str, object]]] = {}
    f1_by_pipeline: dict[str, list[float]] = {}
    for pipeline in PIPELINES:
        per_run[pipeline] = []
        f1_by_pipeline[pipeline] = []
        for run in RUNS:
            cm = confusion_for(loaded[pipeline][run], labels, keys)
            per_run[pipeline].append({"run": run, **cm.as_dict()})
            f1_by_pipeline[pipeline].append(cm.f1)

    # Cross-configuration agreement over ALL 25 run pairings, not run i vs run i.
    #
    # Gate-1 finding 3A: pairing single_llm run i with multi_agent run i is exactly
    # the arbitrary alignment that disqualified the paired t-test. The runs are
    # independent executions; run 0 of one configuration has no correspondence to
    # run 0 of the other. Per-index kappa and McNemar cells were therefore
    # properties of one arbitrary bijection, and reordering either configuration's
    # runs would change them. Summarising over the full 5x5 cross removes the
    # dependence on that choice.
    cross_pairings: list[dict[str, object]] = []
    for single_run in RUNS:
        for multi_run in RUNS:
            single = loaded["single_llm"][single_run]
            multi = loaded["multi_agent"][multi_run]
            s_vec = [single[k].matched for k in keys]
            m_vec = [multi[k].matched for k in keys]
            agree = sum(1 for x, y in zip(s_vec, m_vec) if x == y)
            cross_pairings.append(
                {
                    "single_run": single_run,
                    "multi_run": multi_run,
                    "agreements": agree,
                    "agreement_rate": agree / len(keys),
                    "cohen_kappa": cohen_kappa(s_vec, m_vec),
                    "mcnemar_cells": mcnemar_cells(single, multi, labels, keys),
                }
            )

    agreement_rates = [row["agreement_rate"] for row in cross_pairings]
    kappas = [row["cohen_kappa"] for row in cross_pairings]
    pooled_agree = sum(row["agreements"] for row in cross_pairings)
    pooled_total = len(cross_pairings) * len(keys)

    permutation = exact_permutation_test(
        f1_by_pipeline["single_llm"], f1_by_pipeline["multi_agent"]
    )

    runtime = summarize_runtime(load_runtime_measurements(runtime_artifact_path(root)))

    bootstrap_inputs = bootstrap_inputs_from_caches(labels, keys, loaded)
    bootstrap = run_pigeonhole_bootstrap(bootstrap_inputs)
    source_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for key in keys:
        source_counts[key[0]] = source_counts.get(key[0], 0) + 1
        target_counts[key[1]] = target_counts.get(key[1], 0) + 1

    positives = sum(labels.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/compute_paper_stats.py",
        "source_manifest": {
            "note": (
                "Gate-1 finding 4A: the previous blanket claim that every value derives "
                "from the caches was false. Values have four distinct provenances, "
                "separated here."
            ),
            "cache_derived": [
                "per_run", "aggregates.precision", "aggregates.recall", "aggregates.f1",
                "aggregates.tokens_per_pair", "agreement", "caches",
                "inference.permutation_test", "inference.fixed_graph_sensitivity "
                "(resampled from cache predictions)",
            ],
            "dataset_derived": ["dataset"],
            "hardcoded_analysis_choices": [
                "inference.fixed_graph_sensitivity.seed",
                "inference.fixed_graph_sensitivity.replicates",
                "inference.fixed_graph_sensitivity.citation",
                "schema_version", "formulas",
            ],
            "rng_output": [
                "inference.fixed_graph_sensitivity.range_lower",
                "inference.fixed_graph_sensitivity.range_upper",
                "inference.fixed_graph_sensitivity.mean_bootstrap_effect",
            ],
            "manually_transcribed": [
                "runtime (transcribed from time.txt; the loader now verifies the "
                "transcription against that file's hash and parsed durations)",
            ],
        },
        "producer": producer_provenance(root),
        "dataset": {
            "name": "Abt-Buy",
            "source": "matchbench/Abt-Buy (HuggingFace)",
            "split": "test",
            "n_pairs": len(labels),
            "n_positive": positives,
            "n_negative": len(labels) - positives,
            "key_label_sha256": key_label_digest(labels),
        },
        "caches": cache_meta,
        "per_run": per_run,
        "agreement": {
            "pairing_invariant": {
                "n_pairings": len(cross_pairings),
                "pairing_scheme": (
                    "all 5 single_llm runs x all 5 multi_agent runs; no run-index "
                    "alignment is assumed between configurations"
                ),
                "agreement_rate": {
                    "mean": mean(agreement_rates),
                    "sample_sd": sample_sd(agreement_rates),
                    "min": min(agreement_rates),
                    "max": max(agreement_rates),
                },
                "cohen_kappa": {
                    "mean": mean(kappas),
                    "sample_sd": sample_sd(kappas),
                    "min": min(kappas),
                    "max": max(kappas),
                },
                "note": (
                    "Replaces the previous per-run-index rows. Pairing run i with run i "
                    "was the same arbitrary alignment that disqualified the paired "
                    "t-test; these runs are independent executions with no "
                    "correspondence between their indices."
                ),
            },
            "micro_average": {
                "decisions": pooled_total,
                "agreements": pooled_agree,
                "agreement_rate": pooled_agree / pooled_total,
                "note": (
                    "Descriptive micro-average over all 25 run pairings. The "
                    f"denominator ({pooled_total}) is NOT an independent sample size "
                    "and supports no binomial inference: the same 1,916 pairs recur in "
                    "every pairing, and those pairs share only 737 unique source and "
                    "700 unique target records. Quote it as a descriptive rate or not "
                    "at all."
                ),
            },
            "all_pairings": cross_pairings,
        },
        "aggregates": {
            "f1": {p: summarize(f1_by_pipeline[p]) for p in PIPELINES},
            "precision": {
                p: summarize([row["precision"] for row in per_run[p]]) for p in PIPELINES
            },
            "recall": {p: summarize([row["recall"] for row in per_run[p]]) for p in PIPELINES},
            "tokens_per_pair": {
                p: summarize([row["tokens_per_pair"] for row in cache_meta[p]])
                for p in PIPELINES
            },
        },
        "inference": {
            "permutation_test": permutation.as_dict(),
            "fixed_graph_sensitivity": {
                **bootstrap,
                "role": (
                    "Fixed-test-graph entity-reweighting and execution sensitivity "
                    "analysis, NOT the primary inference and NOT a cross-catalog result. "
                    "Gate-1 finding 2A: product weighting only reweights the 1,916 "
                    "OBSERVED edges. It cannot create new source-target intersections, "
                    "new label prevalence, or any prediction on an unseen record, so it "
                    "licenses no claim about another catalog. Gate-1 finding 2C: it "
                    "combines record-reweighting and execution variability into one "
                    "joint range, so zero-inclusion cannot be attributed to either "
                    "source alone."
                ),
                "caveat": (
                    "Described as pigeonhole-STYLE: F1 is a nonlinear ratio, whereas "
                    "Owen's consistency result is stated for means under crossed random "
                    "effects. Gate-1 finding 2B: Owen's pigeonhole variance is itself "
                    "approximate and generally conservative even for means, so this is "
                    "reported as a 95% bootstrap percentile SENSITIVITY RANGE, not a "
                    "calibrated 95% confidence interval. F1 is smooth while its "
                    "denominator stays positive, so nonlinearity alone does not "
                    "invalidate the percentile method, and no BC/BCa correction was "
                    "added because that would not supply the missing multiway "
                    "justification."
                ),
                "citation": OWEN_CITATION,
                "record_structure": {
                    "n_pairs": len(keys),
                    "unique_sources": bootstrap_inputs.n_sources,
                    "unique_targets": bootstrap_inputs.n_targets,
                    "max_source_multiplicity": max(source_counts.values()),
                    "max_target_multiplicity": max(target_counts.values()),
                    "share_pairs_sharing_a_source": sum(
                        1 for k in keys if source_counts[k[0]] > 1
                    ) / len(keys),
                    "share_pairs_sharing_a_target": sum(
                        1 for k in keys if target_counts[k[1]] > 1
                    ) / len(keys),
                    "note": (
                        "This overlap is why pair-level iid resampling is invalid for this "
                        "benchmark."
                    ),
                },
            },
        },
        "runtime": runtime,
        "formulas": {
            "precision": "tp / (tp + fp)",
            "recall": "tp / (tp + fn)",
            "f1": "2*tp / (2*tp + fp + fn)",
            "sample_sd": "sqrt(sum((x - mean)^2) / (n - 1))",
            "cohen_kappa": "(p_observed - p_expected) / (1 - p_expected)",
        },
        "excluded_by_design": {
            "pooled_mcnemar": (
                "Removed: the same 1,916 pairs recur across all five runs, so pooling "
                "discordant pairs pseudoreplicates them."
            ),
            "paired_t_test": (
                "Removed: pairs arbitrary run indices across two independent "
                "configurations; the p-value moves with run ordering."
            ),
            "pair_level_iid_bootstrap": (
                "Removed: the 1,916 pairs are built from only 737 unique source and 700 "
                "unique target records, so pair-level iid resampling is invalid."
            ),
        },
    }


def render_markdown(payload: Mapping[str, object]) -> str:
    """Render the human-readable view strictly from the canonical JSON payload."""
    dataset = payload["dataset"]
    caches = payload["caches"]
    per_run = payload["per_run"]
    aggregates = payload["aggregates"]
    agreement = payload["agreement"]

    lines: list[str] = [
        "# Canonical paper statistics",
        "",
        "Generated by `scripts/compute_paper_stats.py` from `results/paper_stats.json`.",
        "Do not edit by hand; regenerate instead.",
        "",
        "## Dataset",
        "",
        f"- Split: **{dataset['split']}** of {dataset['name']} ({dataset['source']})",
        f"- Pairs: **{dataset['n_pairs']}** "
        f"({dataset['n_positive']} positive, {dataset['n_negative']} negative)",
        f"- Key-label digest: `{dataset['key_label_sha256']}`",
        "",
        "## Cache validation",
        "",
        "Every run below was proven to contain exactly the labeled test keys, once each,",
        "before any statistic was computed.",
        "",
        "| Pipeline | Run | Records | Parse errors | Tokens/pair | Decisions digest |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for pipeline in PIPELINES:
        for row in caches[pipeline]:
            lines.append(
                f"| {pipeline} | {row['run']} | {row['n_records']} | {row['parse_errors']} "
                f"| {row['tokens_per_pair']:.1f} | `{row['records_sha256'][:12]}` |"
            )

    lines += ["", "## Per-run metrics", "", "| Pipeline | Run | TP | FP | FN | TN | P | R | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pipeline in PIPELINES:
        for row in per_run[pipeline]:
            lines.append(
                f"| {pipeline} | {row['run']} | {row['tp']} | {row['fp']} | {row['fn']} "
                f"| {row['tn']} | {row['precision']:.4f} | {row['recall']:.4f} "
                f"| {row['f1']:.4f} |"
            )

    lines += ["", "## Aggregates across 5 runs (sample SD, ddof=1)", "", "| Metric | Pipeline | Mean | Sample SD |", "|---|---|---:|---:|"]
    for metric in ("precision", "recall", "f1", "tokens_per_pair"):
        for pipeline in PIPELINES:
            block = aggregates[metric][pipeline]
            fmt = "{:.1f}" if metric == "tokens_per_pair" else "{:.4f}"
            lines.append(
                f"| {metric} | {pipeline} | {fmt.format(block['mean'])} "
                f"| {fmt.format(block['sample_sd'])} |"
            )

    invariant = agreement["pairing_invariant"]
    micro = agreement["micro_average"]
    rate = invariant["agreement_rate"]
    kappa = invariant["cohen_kappa"]
    lines += [
        "",
        "## Configuration agreement",
        "",
        f"Summarised over all **{invariant['n_pairings']}** Single-LLM x Multi-Agent run",
        "pairings. No run-index alignment is assumed between the two configurations:",
        "pairing run *i* with run *i* was the same arbitrary choice that disqualified the",
        "paired t-test.",
        "",
        "| Quantity | Mean | Sample SD | Min | Max |",
        "|---|---:|---:|---:|---:|",
        f"| Agreement rate | {rate['mean'] * 100:.2f}% | {rate['sample_sd'] * 100:.2f}% "
        f"| {rate['min'] * 100:.2f}% | {rate['max'] * 100:.2f}% |",
        f"| Cohen's kappa | {kappa['mean']:.4f} | {kappa['sample_sd']:.4f} "
        f"| {kappa['min']:.4f} | {kappa['max']:.4f} |",
        "",
        f"Descriptive micro-average: **{micro['agreements']}/{micro['decisions']}** "
        f"= {micro['agreement_rate'] * 100:.1f}%.",
        "",
        f"That denominator ({micro['decisions']}) is **not** an independent sample size and",
        "supports no binomial inference: the same 1,916 pairs recur in every pairing, and",
        "those pairs are built from only 737 unique source and 700 unique target records.",
        "",
        "Per-pairing McNemar cells are in the JSON as descriptive counts only. They carry",
        "no p-value: pooling discordances across pairings would pseudoreplicate the same",
        "1,916 pairs.",
        "",
        "## Deliberately excluded",
        "",
    ]
    for name, reason in payload["excluded_by_design"].items():
        lines.append(f"- **{name}** — {reason}")

    return "\n".join(lines) + "\n"


def serialize(payload: Mapping[str, object]) -> str:
    """Canonical JSON form: sorted keys and a trailing newline, so reruns are byte-identical."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for paper_stats.{json,md} (default: <repo>/results)",
    )
    args = parser.parse_args(argv)

    root = find_project_root()
    out_dir = args.out_dir or (root / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_report(root)

    (out_dir / "paper_stats.json").write_text(serialize(payload))
    (out_dir / "paper_stats.md").write_text(render_markdown(payload))

    f1 = payload["aggregates"]["f1"]
    print(f"validated {len(PIPELINES) * len(RUNS)} caches at {payload['dataset']['n_pairs']} keys each")
    for pipeline in PIPELINES:
        block = f1[pipeline]
        print(f"  {pipeline:<12} F1 {block['mean']:.4f} +/- {block['sample_sd']:.4f}")
    print(f"wrote {out_dir / 'paper_stats.json'}")
    print(f"wrote {out_dir / 'paper_stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
