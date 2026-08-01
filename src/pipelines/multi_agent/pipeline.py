"""Multi-agent entity resolution pipeline with disk caching."""

import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from src.pipelines.multi_agent.mcp_tools import SYNTACTIC_TOOLS, SEMANTIC_TOOLS
from src.pipelines.multi_agent.agents import (
  run_agent_loop,
  format_pair_prompt,
  SYNTACTIC_SYSTEM_PROMPT,
  SEMANTIC_SYSTEM_PROMPT,
)
from src.pipelines.multi_agent.consensus import apply_consensus
from src.utils import create_openai_client, build_product_lookup

CACHE_DIR = Path("data/cache/multi_agent")

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, src_id: str, tgt_id: str) -> Path:
  """Return the cache file path for a given pair."""
  return cache_dir / f"{src_id}_{tgt_id}.json"


def _load_cache(cache_dir: Path, src_id: str, tgt_id: str) -> dict | None:
  """Load a cached prediction, or return None if not cached."""
  path = _cache_path(cache_dir, src_id, tgt_id)
  if not path.exists():
    return None
  with open(path) as f:
    return json.load(f)


def _save_cache(cache_dir: Path, src_id: str, tgt_id: str, data: dict) -> None:
  """Save a prediction to the disk cache."""
  path = _cache_path(cache_dir, src_id, tgt_id)
  with open(path, "w") as f:
    json.dump(data, f)


def _agent_consensus_fallback(syntactic: dict, semantic: dict) -> dict:
  """Fall back to agent consensus when orchestrator response fails to parse."""
  syn_verdict = syntactic.get("verdict", "NO MATCH")
  sem_verdict = semantic.get("verdict", "NO MATCH")
  syn_conf = syntactic.get("confidence", 0.0)
  sem_conf = semantic.get("confidence", 0.0)

  if syn_verdict == sem_verdict:
    return {
      "verdict": syn_verdict,
      "confidence": (syn_conf + sem_conf) / 2,
      "reasoning": f"orchestrator fallback: both agents agreed on {syn_verdict}",
      "parse_error": False,
    }

  # Agents split: use the higher-confidence agent's verdict.
  if syn_conf >= sem_conf:
    winner, loser = "syntactic", "semantic"
    verdict, confidence = syn_verdict, syn_conf
  else:
    winner, loser = "semantic", "syntactic"
    verdict, confidence = sem_verdict, sem_conf

  return {
    "verdict": verdict,
    "confidence": confidence,
    "reasoning": f"orchestrator fallback: agents split, sided with {winner} (conf={confidence:.2f})",
    "parse_error": False,
  }


async def _process_pair(
  row,
  src_lookup: dict,
  tgt_lookup: dict,
  mcp_client,
  openai_client,
) -> tuple[dict, int]:
  """Process a single candidate pair through both agents + consensus."""
  src_id = str(row["source_id"])
  tgt_id = str(row["target_id"])

  src = src_lookup.get(src_id)
  tgt = tgt_lookup.get(tgt_id)

  if src is None or tgt is None:
    # Parse error when at least one product is missing. Can't compare.
    logger.warning("Missing product data: source=%s target=%s", src_id, tgt_id)
    return {
      "source_id": src_id,
      "target_id": tgt_id,
      "prediction": 0,
      "verdict": "NO MATCH",
      "confidence": 0.0,
      "reasoning": "missing product data",
      "parse_error": True,
      "tokens": 0,
    }, 0

  # Create structured prompt for the pair of products.
  user_prompt = format_pair_prompt(
    src["name"], src["description"], src["price"],
    tgt["name"], tgt["description"], tgt["price"],
  )

  syntactic_result, syn_tokens = await run_agent_loop(
    SYNTACTIC_SYSTEM_PROMPT, user_prompt, openai_client, mcp_client, SYNTACTIC_TOOLS,
  )
  semantic_result, sem_tokens = await run_agent_loop(
    SEMANTIC_SYSTEM_PROMPT, user_prompt, openai_client, mcp_client, SEMANTIC_TOOLS,
  )

  # Orchestrator LLM reviews agent verdicts and may override.
  consensus, consensus_tokens = await apply_consensus(
    syntactic_result, semantic_result, user_prompt, openai_client,
  )
  total_tokens = syn_tokens + sem_tokens + consensus_tokens

  # Fallback: if orchestrator response failed to parse, use agent consensus. Issue here with parse failures causing a lot of False Negatives, so this is a safety net to improve robustness.ß
  if consensus.get("parse_error", False):
    consensus = _agent_consensus_fallback(syntactic_result, semantic_result)
    logger.info("Orchestrator parse failure for %s_%s, falling back to agent consensus: %s",
                src_id, tgt_id, consensus["verdict"])

  parse_error = syntactic_result.get("parse_error", False) or semantic_result.get("parse_error", False)

  return {
    "source_id": src_id,
    "target_id": tgt_id,
    "prediction": 1 if consensus["verdict"] == "MATCH" else 0,
    "verdict": consensus["verdict"],
    "confidence": consensus["confidence"],
    "reasoning": consensus["reasoning"],
    "syntactic": syntactic_result,
    "semantic": semantic_result,
    "tokens": total_tokens,
    "parse_error": parse_error,
  }, total_tokens


async def _run_pipeline_async(
  candidates: pd.DataFrame,
  src_lookup: dict,
  tgt_lookup: dict,
  max_pairs: int | None,
  verbose: bool,
  cache_dir: Path,
) -> tuple[list[dict], int]:
  """Run the multi-agent pipeline asynchronously."""
  cache_dir.mkdir(parents=True, exist_ok=True)

  pairs = candidates.head(max_pairs) if max_pairs else candidates
  total = len(pairs)

  openai_client = create_openai_client()

  predictions: list[dict] = []
  total_tokens = 0

  transport = StdioTransport(command="uv", args=["run", "python", "-m", "src.pipelines.multi_agent.mcp_tools"])
  try:
    client_ctx = Client(transport)
    mcp_client = await client_ctx.__aenter__()
  except (OSError, FileNotFoundError) as e:
    raise RuntimeError(
      f"Failed to start MCP tool server. Ensure 'uv' is on PATH and src.pipelines.multi_agent.mcp_tools is importable: {e}"
    ) from e
  try:
    # Iterate through pairs sequentially (agent loops are async but pairs are processed one at a time for caching).
    for i, (_, row) in enumerate(pairs.iterrows()):
      src_id = str(row["source_id"])
      tgt_id = str(row["target_id"])

      cached = _load_cache(cache_dir, src_id, tgt_id)
      if cached is not None:
        predictions.append(cached["prediction"])
        total_tokens += cached.get("tokens", 0)
        if verbose and (i + 1) % 10 == 0:
          print(f"  [{i + 1}/{total}] (cached)")
        continue

      result, tokens = await _process_pair(row, src_lookup, tgt_lookup, mcp_client, openai_client)
      predictions.append(result)
      total_tokens += tokens

      _save_cache(cache_dir, src_id, tgt_id, {"prediction": result, "tokens": tokens})

      if verbose and (i + 1) % 10 == 0:
        print(f"  [{i + 1}/{total}] verdict={result['verdict']} conf={result['confidence']:.2f}")
  finally:
    await client_ctx.__aexit__(None, None, None)

  return predictions, total_tokens


def run_multi_agent_pipeline(
  candidate_pairs: pd.DataFrame,
  source_df: pd.DataFrame,
  target_df: pd.DataFrame,
  max_pairs: int | None = None,
  verbose: bool = False,
  cache_dir: Path = CACHE_DIR,
) -> tuple[list[dict], int, float]:
  """
  Run multi-agent entity resolution on candidate pairs.

  Returns:
    Tuple of (predictions list, total tokens, elapsed seconds).
  """
  src_lookup = build_product_lookup(source_df)
  tgt_lookup = build_product_lookup(target_df)

  start_time = time.time()
  predictions, total_tokens = asyncio.run(
    _run_pipeline_async(candidate_pairs, src_lookup, tgt_lookup, max_pairs, verbose, cache_dir)
  )
  elapsed = time.time() - start_time

  return predictions, total_tokens, elapsed
