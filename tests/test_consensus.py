"""Tests for LLM-based orchestrator consensus."""

from unittest.mock import MagicMock
import asyncio

from src.pipelines.multi_agent.consensus import (
  apply_consensus,
  format_orchestrator_prompt,
  ORCHESTRATOR_SYSTEM_PROMPT,
)


def _make_mock_client(response_text: str, tokens: int = 100):
  """Create a mock OpenAI client that returns the given response text."""
  mock_message = MagicMock()
  mock_message.content = response_text

  mock_usage = MagicMock()
  mock_usage.total_tokens = tokens

  mock_choice = MagicMock()
  mock_choice.message = mock_message

  mock_response = MagicMock()
  mock_response.choices = [mock_choice]
  mock_response.usage = mock_usage

  mock_client = MagicMock()
  mock_client.chat.completions.create.return_value = mock_response
  return mock_client


class TestFormatOrchestratorPrompt:
  """Prompt formatting must include product pair and both agents' results."""

  def test_includes_product_pair(self):
    """The original product pair context should appear in the prompt."""
    result = format_orchestrator_prompt(
      "Product A: iPhone\nProduct B: Galaxy",
      {"verdict": "MATCH", "confidence": 0.9, "reasoning": "similar"},
      {"verdict": "MATCH", "confidence": 0.8, "reasoning": "prices close"},
    )
    assert "iPhone" in result
    assert "Galaxy" in result

  def test_includes_syntactic_results(self):
    """Syntactic agent's verdict and reasoning should appear in the prompt."""
    result = format_orchestrator_prompt(
      "pair",
      {"verdict": "MATCH", "confidence": 0.9, "reasoning": "names overlap"},
      {"verdict": "NO MATCH", "confidence": 0.6, "reasoning": "prices differ"},
    )
    assert "Syntactic Agent" in result
    assert "names overlap" in result
    assert "0.9" in result

  def test_includes_semantic_results(self):
    """Semantic agent's verdict and reasoning should appear in the prompt."""
    result = format_orchestrator_prompt(
      "pair",
      {"verdict": "MATCH", "confidence": 0.9, "reasoning": "a"},
      {"verdict": "NO MATCH", "confidence": 0.6, "reasoning": "embeddings low"},
    )
    assert "Semantic Agent" in result
    assert "embeddings low" in result
    assert "0.6" in result


class TestOrchestratorSystemPrompt:
  """System prompt must instruct the orchestrator to review and potentially override."""

  def test_not_empty(self):
    """Orchestrator system prompt must be non-empty."""
    assert len(ORCHESTRATOR_SYSTEM_PROMPT) > 0

  def test_mentions_override(self):
    """Prompt should explicitly allow the orchestrator to override agents."""
    assert "override" in ORCHESTRATOR_SYSTEM_PROMPT

  def test_mentions_json_format(self):
    """Prompt should specify the expected JSON response format."""
    assert "verdict" in ORCHESTRATOR_SYSTEM_PROMPT
    assert "confidence" in ORCHESTRATOR_SYSTEM_PROMPT


class TestApplyConsensus:
  """LLM-based consensus must call the orchestrator and parse its response."""

  def test_orchestrator_returns_match(self):
    """Orchestrator returning MATCH should be parsed into a MATCH result."""
    client = _make_mock_client(
      '{"verdict": "MATCH", "confidence": 0.92, "reasoning": "agents agree and evidence is strong"}',
      tokens=150,
    )
    result, tokens = asyncio.run(apply_consensus(
      {"verdict": "MATCH", "confidence": 0.9, "reasoning": "names match"},
      {"verdict": "MATCH", "confidence": 0.8, "reasoning": "prices match"},
      "Product A: X\nProduct B: Y",
      client,
    ))
    assert result["verdict"] == "MATCH"
    assert result["confidence"] == 0.92
    assert result["parse_error"] is False
    assert tokens == 150

  def test_orchestrator_returns_no_match(self):
    """Orchestrator returning NO MATCH should be parsed correctly."""
    client = _make_mock_client(
      '{"verdict": "NO MATCH", "confidence": 0.85, "reasoning": "names differ significantly"}',
    )
    result, tokens = asyncio.run(apply_consensus(
      {"verdict": "MATCH", "confidence": 0.5, "reasoning": "weak"},
      {"verdict": "NO MATCH", "confidence": 0.7, "reasoning": "different"},
      "pair",
      client,
    ))
    assert result["verdict"] == "NO MATCH"
    assert result["parse_error"] is False

  def test_orchestrator_overrides_agents(self):
    """Orchestrator can disagree with both agents if reasoning is flawed."""
    client = _make_mock_client(
      '{"verdict": "NO MATCH", "confidence": 0.8, "reasoning": "agents agree but names are clearly different products"}',
    )
    result, _ = asyncio.run(apply_consensus(
      {"verdict": "MATCH", "confidence": 0.6, "reasoning": "some overlap"},
      {"verdict": "MATCH", "confidence": 0.5, "reasoning": "prices close"},
      "pair",
      client,
    ))
    assert result["verdict"] == "NO MATCH"

  def test_orchestrator_parse_error(self):
    """Garbage LLM output should produce a parse error with safe NO MATCH default."""
    client = _make_mock_client("I'm not sure what to say here")
    result, _ = asyncio.run(apply_consensus(
      {"verdict": "MATCH", "confidence": 0.9, "reasoning": "a"},
      {"verdict": "MATCH", "confidence": 0.8, "reasoning": "b"},
      "pair",
      client,
    ))
    assert result["parse_error"] is True
    assert result["verdict"] == "NO MATCH"

  def test_tokens_returned(self):
    """Token count from the orchestrator LLM call should be returned."""
    client = _make_mock_client(
      '{"verdict": "MATCH", "confidence": 0.9, "reasoning": "ok"}',
      tokens=200,
    )
    _, tokens = asyncio.run(apply_consensus(
      {"verdict": "MATCH", "confidence": 0.9, "reasoning": "a"},
      {"verdict": "MATCH", "confidence": 0.8, "reasoning": "b"},
      "pair",
      client,
    ))
    assert tokens == 200

  def test_llm_called_with_correct_model(self):
    """The orchestrator should call the LLM with the correct model name."""
    client = _make_mock_client(
      '{"verdict": "MATCH", "confidence": 0.9, "reasoning": "ok"}',
    )
    asyncio.run(apply_consensus(
      {"verdict": "MATCH", "confidence": 0.9, "reasoning": "a"},
      {"verdict": "MATCH", "confidence": 0.8, "reasoning": "b"},
      "pair",
      client,
    ))
    call_args = client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-oss-120b"
