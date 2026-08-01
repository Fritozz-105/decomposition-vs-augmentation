"""Single-LLM entity resolution pipeline with disk caching."""

import json
import logging
import time
from pathlib import Path
import pandas as pd
from openai import OpenAI
from src.pipelines.single_llm.prompts import (
  SYSTEM_PROMPT,
  format_user_prompt,
  parse_llm_response,
)
from src.utils import build_product_lookup
from src.utils.llm_client import get_model_name

MODEL_NAME = get_model_name()
CACHE_DIR = Path("data/cache/single_llm")

logger = logging.getLogger(__name__)


def run_single_llm_pipeline(
  client: OpenAI,
  candidate_pairs: pd.DataFrame,
  source_df: pd.DataFrame,
  target_df: pd.DataFrame,
  max_pairs: int | None = None,
  cache_dir: Path = CACHE_DIR,
) -> tuple[list[dict], int, float]:
  """
  Run single-LLM entity resolution on candidate pairs.

  Args:
    client: Configured OpenAI client.
    candidate_pairs: DataFrame with source_id, target_id columns.
    source_df: Source product table.
    target_df: Target product table.
    max_pairs: Limit number of pairs to process (None = all).
    cache_dir: Directory for caching LLM responses.

  Returns:
    Tuple of (predictions list, total tokens, elapsed seconds).
  """
  cache_dir.mkdir(parents=True, exist_ok=True)

  source_lookup = build_product_lookup(source_df)
  target_lookup = build_product_lookup(target_df)

  pairs = candidate_pairs.head(max_pairs) if max_pairs else candidate_pairs
  total = len(pairs)

  predictions: list[dict] = []
  total_tokens = 0
  start_time = time.time()

  # Iterate through all pairs in blocks and ask LLM if they match
  for i, (_, row) in enumerate(pairs.iterrows()):
    src_id = str(row["source_id"])
    tgt_id = str(row["target_id"])

    cached = _load_cache(cache_dir, src_id, tgt_id)
    if cached is not None:
      predictions.append(cached["prediction"])
      total_tokens += cached.get("tokens", 0)
      _log_progress(i + 1, total, cached=True)
      continue

    src = source_lookup.get(src_id)
    tgt = target_lookup.get(tgt_id)

    if src is None or tgt is None:
      logger.warning("Missing product data: source=%s target=%s", src_id, tgt_id)
      continue

    # Call LLM
    user_prompt = format_user_prompt(
      src["name"], src["description"], src["price"],
      tgt["name"], tgt["description"], tgt["price"],
    )

    response_text, tokens = _call_llm(client, user_prompt)
    total_tokens += tokens

    parsed = parse_llm_response(response_text)
    prediction = {
      "source_id": src_id,
      "target_id": tgt_id,
      **parsed,
    }
    predictions.append(prediction)

    _save_cache(cache_dir, src_id, tgt_id, prediction, tokens)
    _log_progress(i + 1, total, cached=False)

  elapsed = time.time() - start_time

  return predictions, total_tokens, elapsed


def _call_llm(client: OpenAI, user_prompt: str) -> tuple[str, int]:
  """Call the LLM and return (response text, tokens used)."""
  response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": user_prompt},
    ],
    temperature=0,
    response_format={"type": "json_object"},
  )
  text = response.choices[0].message.content or ""
  tokens = response.usage.total_tokens if response.usage else 0

  return text, tokens


def _cache_path(cache_dir: Path, source_id: str, target_id: str) -> Path:
  """Return the cache file path for a given pair."""
  return cache_dir / f"{source_id}_{target_id}.json"


def _load_cache(cache_dir: Path, source_id: str, target_id: str) -> dict | None:
  """Load a cached prediction, or return None if not cached."""
  path = _cache_path(cache_dir, source_id, target_id)
  if not path.exists():
    return None
  with open(path) as f:
    return json.load(f)


def _save_cache(
  cache_dir: Path,
  source_id: str,
  target_id: str,
  prediction: dict,
  tokens: int,
) -> None:
  """Save a prediction to the disk cache."""
  path = _cache_path(cache_dir, source_id, target_id)
  with open(path, "w") as f:
    json.dump({"prediction": prediction, "tokens": tokens}, f)


def _log_progress(completed: int, total: int, cached: bool = False) -> None:
  """Log progress every 100 pairs."""
  if completed % 100 == 0 or completed == total:
    tag = " (cached)" if cached else ""
    logger.info("Progress: %d/%d pairs%s", completed, total, tag)
