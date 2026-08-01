"""Tests for src/pipelines/single_llm/prompts.py."""

from src.pipelines.single_llm.prompts import (
  SYSTEM_PROMPT,
  format_user_prompt,
  parse_llm_response,
)


def test_system_prompt_mentions_json():
  """Prompt must request JSON output so responses are machine-parseable."""
  assert "JSON" in SYSTEM_PROMPT


def test_system_prompt_mentions_verdict():
  """Prompt must define the verdict field so the LLM knows the expected format."""
  assert "verdict" in SYSTEM_PROMPT


def test_format_includes_all_fields():
  """All product fields (name, description, price) appear in the formatted prompt."""
  result = format_user_prompt(
    "Sony DVD", "A dvd player", "29.99",
    "Sony Player", "DVD player by Sony", "31.00",
  )
  assert "Sony DVD" in result
  assert "A dvd player" in result
  assert "29.99" in result
  assert "Sony Player" in result
  assert "DVD player by Sony" in result
  assert "31.00" in result


def test_format_nan_price_becomes_na():
  """NaN prices are displayed as 'N/A' to avoid confusing the LLM with raw nan strings."""
  result = format_user_prompt("A", "desc", "nan", "B", "desc", "nan")
  assert "N/A" in result
  assert "nan" not in result.split("Price: ")[1].split("\n")[0]


def test_format_contains_product_labels():
  """Products are labeled A and B so the LLM can reference them unambiguously."""
  result = format_user_prompt("A", "", "10", "B", "", "20")
  assert "Product A:" in result
  assert "Product B:" in result


def test_parse_valid_match():
  """Correctly parses a well-formed MATCH response into verdict=True with confidence and reasoning."""
  raw = '{"verdict": "MATCH", "confidence": 0.95, "reasoning": "same product"}'
  result = parse_llm_response(raw)
  assert result["verdict"] == "MATCH"
  assert result["confidence"] == 0.95
  assert result["reasoning"] == "same product"
  assert result["parse_error"] is False


def test_parse_valid_no_match():
  """Correctly parses NO MATCH verdict into verdict='NO MATCH'."""
  raw = '{"verdict": "NO MATCH", "confidence": 0.2, "reasoning": "different"}'
  result = parse_llm_response(raw)
  assert result["verdict"] == "NO MATCH"


def test_parse_markdown_fences():
  """Handles LLM responses wrapped in markdown code fences, which GPT models often produce."""
  raw = '```json\n{"verdict": "MATCH", "confidence": 0.8, "reasoning": "ok"}\n```'
  result = parse_llm_response(raw)
  assert result["verdict"] == "MATCH"
  assert result["parse_error"] is False


def test_parse_malformed_json():
  """Returns safe defaults with parse_error=True when the LLM returns non-JSON text."""
  raw = "This is not valid JSON at all"
  result = parse_llm_response(raw)
  assert result["verdict"] == "NO MATCH"
  assert result["parse_error"] is True


def test_parse_confidence_clamped_above_1():
  """Clamps confidence to 1.0 if the LLM returns a value above the valid range."""
  raw = '{"verdict": "MATCH", "confidence": 1.5, "reasoning": "test"}'
  result = parse_llm_response(raw)
  assert result["confidence"] == 1.0


def test_parse_confidence_clamped_below_0():
  """Clamps confidence to 0.0 if the LLM returns a negative value."""
  raw = '{"verdict": "MATCH", "confidence": -0.5, "reasoning": "test"}'
  result = parse_llm_response(raw)
  assert result["confidence"] == 0.0


def test_parse_empty_response():
  """Returns safe defaults with parse_error=True for empty/blank LLM responses."""
  result = parse_llm_response("")
  assert result["verdict"] == "NO MATCH"
  assert result["parse_error"] is True


def test_parse_match_verdict_is_string():
  """parse_llm_response returns verdict as string 'MATCH', not boolean True."""
  result = parse_llm_response('{"verdict": "MATCH", "confidence": 0.9, "reasoning": "same"}')
  assert result["verdict"] == "MATCH"
  assert isinstance(result["verdict"], str)


def test_parse_no_match_verdict_is_string():
  """parse_llm_response returns verdict as string 'NO MATCH', not boolean False."""
  result = parse_llm_response('{"verdict": "NO MATCH", "confidence": 0.2, "reasoning": "diff"}')
  assert result["verdict"] == "NO MATCH"
  assert isinstance(result["verdict"], str)
