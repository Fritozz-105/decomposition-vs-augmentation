"""Hyperparameter tuning script for the Magellan Random Forest classifier. Runs GridSearchCV using the train split for fitting and the validation split for scoring."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from src.pipelines.magellan.features import compute_features_for_pairs
from src.utils import build_product_lookup, build_label_series

logger = logging.getLogger(__name__)

DEFAULT_PARAMS_PATH = Path("data/magellan_best_params.json")

RF_PARAM_GRID = {
  "n_estimators": [10, 50, 100, 150, 200],
  "max_depth": [None, 10, 20, 30],
  "min_samples_split": [2, 5, 10],
  "max_features": ["sqrt", "log2", None],
  "class_weight": ["balanced", "balanced_subsample", None],
}


def find_best_threshold(
  y_true: np.ndarray,
  y_proba: np.ndarray,
  thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
  """Sweep thresholds on predicted probabilities to find the one maximizing F1.

  Returns:
    (best_threshold, best_f1)
  """
  if thresholds is None:
    thresholds = np.arange(0.05, 0.96, 0.05)

  best_f1, best_thresh = 0.0, 0.5
  for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    if f1 > best_f1:
      best_f1, best_thresh = f1, float(thresh)

  return best_thresh, best_f1


def tune_rf_params(
  source_df: pd.DataFrame,
  target_df: pd.DataFrame,
  train_pairs_df: pd.DataFrame,
  valid_pairs_df: pd.DataFrame,
  param_grid: dict | None = None,
  save_path: Path | None = DEFAULT_PARAMS_PATH,
) -> dict:
  """
  Run GridSearchCV to find best RF hyperparameters. Uses train split for fitting, validation split for scoring (F1).

  Returns:
    Dict with best params and validation F1 score.
  """
  if param_grid is None:
    param_grid = RF_PARAM_GRID

  src_lookup = build_product_lookup(source_df)
  tgt_lookup = build_product_lookup(target_df)

  train_renamed = train_pairs_df.rename(
    columns={"ltable_id": "source_id", "rtable_id": "target_id"}
  )
  valid_renamed = valid_pairs_df.rename(
    columns={"ltable_id": "source_id", "rtable_id": "target_id"}
  )

  logger.info("Computing training features for %d pairs...", len(train_renamed))
  X_train, train_pair_ids = compute_features_for_pairs(train_renamed, src_lookup, tgt_lookup)

  logger.info("Computing validation features for %d pairs...", len(valid_renamed))
  X_val, val_pair_ids = compute_features_for_pairs(valid_renamed, src_lookup, tgt_lookup)

  if len(X_train) == 0:
    raise RuntimeError("No training pairs could have features extracted.")
  if len(X_val) == 0:
    raise RuntimeError("No validation pairs could have features extracted.")

  train_labels = build_label_series(train_pairs_df)
  valid_labels = build_label_series(valid_pairs_df)

  y_train = np.array([train_labels.get(pid, 0) for pid in train_pair_ids])
  y_val = np.array([valid_labels.get(pid, 0) for pid in val_pair_ids])

  logger.info(
    "Train: %d pairs (%d pos) | Valid: %d pairs (%d pos)",
    len(y_train), int(y_train.sum()),
    len(y_val), int(y_val.sum()),
  )

  # PredefinedSplit: -1 = always in training, 0 = validation fold
  X_combined = np.vstack([X_train, X_val])
  y_combined = np.concatenate([y_train, y_val])
  test_fold = np.concatenate([
    np.full(len(X_train), -1, dtype=int),
    np.full(len(X_val), 0, dtype=int),
  ])

  n_combos = 1
  for vals in param_grid.values():
    n_combos *= len(vals)
  logger.info("Running GridSearchCV over %d param combinations...", n_combos)

  grid = GridSearchCV(
    RandomForestClassifier(random_state=77, n_jobs=-1),
    param_grid,
    cv=PredefinedSplit(test_fold),
    scoring="f1",
    n_jobs=-1,
    refit=False,
  )
  grid.fit(X_combined, y_combined)

  # Find best threshold: train model with best params on train only, predict on valid
  best_rf = RandomForestClassifier(random_state=77, n_jobs=-1, **grid.best_params_)
  best_rf.fit(X_train, y_train)
  if 1 in best_rf.classes_:
    val_proba = best_rf.predict_proba(X_val)[:, list(best_rf.classes_).index(1)]
  else:
    val_proba = np.zeros(len(X_val))
  best_thresh, thresh_f1 = find_best_threshold(y_val, val_proba)
  logger.info("Best threshold: %.2f (valid F1=%.4f)", best_thresh, thresh_f1)

  result = {
    "best_params": grid.best_params_,
    "valid_f1": round(grid.best_score_, 4),
    "best_threshold": round(best_thresh, 4),
    "threshold_f1": round(thresh_f1, 4),
  }

  logger.info("Best params (valid F1=%.4f): %s", result["valid_f1"], result["best_params"])

  if save_path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Saved best params to %s", save_path)

  return result


def load_best_params(path: Path = DEFAULT_PARAMS_PATH) -> tuple[dict | None, float]:
  """Load saved best params and threshold from JSON. Returns (None, 0.5) if file doesn't exist."""
  if not path.exists():
    return None, 0.5
  data = json.loads(path.read_text())
  return data.get("best_params"), data.get("best_threshold", 0.5)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  from src.data import load_abt_buy

  print("Loading Abt-Buy dataset...")
  data = load_abt_buy()

  result = tune_rf_params(
    source_df=data["source_df"],
    target_df=data["target_df"],
    train_pairs_df=data["train_pairs_df"],
    valid_pairs_df=data["valid_pairs_df"],
  )

  print(f"\nBest params: {result['best_params']}")
  print(f"Validation F1: {result['valid_f1']}")
  print(f"Saved to: {DEFAULT_PARAMS_PATH}")
