"""Shared LLM response parsing for entity resolution pipelines."""

import json
import math
import re


def parse_verdict_response(raw_response: str | None) -> dict:
  """
  Extracts verdict, confidence, and reasoning from a JSON object in the response text. Unknown verdicts (anything other than "MATCH" or "NO MATCH") are treated as parse errors.

  Args:
    raw_response: The raw text response from the LLM.

  Returns:
    dict with keys: verdict (str), confidence (float), reasoning (str), parse_error (bool)
  """
  parse_error = {"verdict": "NO MATCH", "confidence": 0.0, "reasoning": "parse error", "parse_error": True}

  if not raw_response or not raw_response.strip():
    return parse_error

  text = raw_response.strip()

  # Strip markdown code fences
  fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
  if fence_match:
    text = fence_match.group(1).strip()

  try:
    data = json.loads(text)
  except json.JSONDecodeError:
    # Try to find JSON object in the text. Strip extra.
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
      try:
        data = json.loads(json_match.group())
      except json.JSONDecodeError:
        return parse_error
    else:
      return parse_error

  # If no verdict field, treat as parse error.
  verdict_raw = str(data.get("verdict", "NO MATCH")).upper().strip()
  if verdict_raw not in ("MATCH", "NO MATCH"):
    return parse_error

  confidence_raw = data.get("confidence", 0.0)
  try:
    confidence = float(confidence_raw)
    if math.isnan(confidence):
      confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
  except (ValueError, TypeError):
    confidence = 0.0

  reasoning = str(data.get("reasoning", ""))

  return {"verdict": verdict_raw, "confidence": confidence, "reasoning": reasoning, "parse_error": False}
