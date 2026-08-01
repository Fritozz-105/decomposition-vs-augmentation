"""Magellan-style entity resolution pipeline (Random Forest with manual features)."""

from src.pipelines.magellan.pipeline import run_magellan_pipeline
from src.pipelines.magellan.tuning import tune_rf_params

__all__ = ["run_magellan_pipeline", "tune_rf_params"]
