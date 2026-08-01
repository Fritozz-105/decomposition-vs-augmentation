"""Tests for agent prompts, tool conversion, and response parsing."""

from src.pipelines.multi_agent.agents import (
  mcp_to_openai_tools,
  parse_agent_response,
  format_pair_prompt,
)


class TestMcpToOpenaiTools:
  """MCP-to-OpenAI schema conversion for function calling."""

  def test_basic_conversion(self):
    """A well-formed MCP tool should map to a valid OpenAI function schema."""
    class MockTool:
      name = "test_tool"
      description = "A test tool"
      inputSchema = {"type": "object", "properties": {"x": {"type": "string"}}}

    result = mcp_to_openai_tools([MockTool()])
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "test_tool"
    assert result[0]["function"]["description"] == "A test tool"
    assert result[0]["function"]["parameters"]["type"] == "object"

  def test_empty_list(self):
    """An empty tool list should produce an empty schema list."""
    assert mcp_to_openai_tools([]) == []

  def test_no_description(self):
    """A tool with no description should default to an empty string."""
    class MockTool:
      name = "tool"
      description = None
      inputSchema = {}

    result = mcp_to_openai_tools([MockTool()])
    assert result[0]["function"]["description"] == ""


class TestParseAgentResponse:
  """Response parser must handle valid JSON, markdown fences, and garbage."""

  def test_valid_match(self):
    """Clean JSON with MATCH verdict should parse without error."""
    r = parse_agent_response('{"verdict": "MATCH", "confidence": 0.95, "reasoning": "same product"}')
    assert r["verdict"] == "MATCH"
    assert r["confidence"] == 0.95
    assert r["reasoning"] == "same product"
    assert r["parse_error"] is False

  def test_valid_no_match(self):
    """Clean JSON with NO MATCH verdict should parse without error."""
    r = parse_agent_response('{"verdict": "NO MATCH", "confidence": 0.8, "reasoning": "different"}')
    assert r["verdict"] == "NO MATCH"
    assert r["parse_error"] is False

  def test_markdown_fence(self):
    """JSON wrapped in markdown code fences should still be parsed."""
    r = parse_agent_response('```json\n{"verdict": "MATCH", "confidence": 0.9, "reasoning": "ok"}\n```')
    assert r["verdict"] == "MATCH"
    assert r["parse_error"] is False

  def test_garbage_input(self):
    """Non-JSON text should produce a parse error with safe defaults."""
    r = parse_agent_response("some chars lol")
    assert r["parse_error"] is True
    assert r["verdict"] == "NO MATCH"
    assert r["confidence"] == 0.0

  def test_empty_input(self):
    """Empty string should produce a parse error."""
    r = parse_agent_response("")
    assert r["parse_error"] is True

  def test_none_input(self):
    """None input should produce a parse error."""
    r = parse_agent_response(None)
    assert r["parse_error"] is True

  def test_invalid_verdict(self):
    """A verdict not in {MATCH, NO MATCH} should produce a parse error."""
    r = parse_agent_response('{"verdict": "MAYBE", "confidence": 0.5, "reasoning": "unsure"}')
    assert r["parse_error"] is True

  def test_confidence_clamping(self):
    """Confidence above 1.0 should be clamped to 1.0."""
    r = parse_agent_response('{"verdict": "MATCH", "confidence": 1.5, "reasoning": "very sure"}')
    assert r["confidence"] == 1.0

  def test_json_in_text(self):
    """JSON embedded in surrounding text should be extracted and parsed."""
    r = parse_agent_response('Here is my answer: {"verdict": "MATCH", "confidence": 0.7, "reasoning": "similar"}')
    assert r["verdict"] == "MATCH"
    assert r["parse_error"] is False


class TestFormatPairPrompt:
  """Pair prompt must include all product fields for agent analysis."""

  def test_contains_product_names(self):
    """Both product names should appear in the formatted prompt."""
    result = format_pair_prompt("iPhone 12", "phone", "$999", "Galaxy S21", "phone", "$899")
    assert "iPhone 12" in result
    assert "Galaxy S21" in result

  def test_contains_prices(self):
    """Both prices should appear in the formatted prompt."""
    result = format_pair_prompt("A", "desc", "$50", "B", "desc", "$60")
    assert "$50" in result
    assert "$60" in result

  def test_returns_string(self):
    """Prompt must be a string for direct use in LLM messages."""
    result = format_pair_prompt("A", "d1", "p1", "B", "d2", "p2")
    assert isinstance(result, str)
