"""Agent prompts, agentic loop, and response parsing for multi-agent ER."""

import json
import logging

from src.utils.llm_client import get_model_name

logger = logging.getLogger(__name__)

MODEL = get_model_name()

SYNTACTIC_SYSTEM_PROMPT = """\
You are a syntactic similarity analyst for product entity resolution.
Determine whether two product listings refer to the same physical product.

You have tools to compute edit distance, Jaccard similarity, and BM25
relevance score. Call ALL three tools on the product names, then decide.

Respond ONLY with a JSON object:
{"verdict": "MATCH" or "NO MATCH", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""

SEMANTIC_SYSTEM_PROMPT = """\
You are a semantic similarity analyst for product entity resolution.
Determine whether two product listings refer to the same physical product.

You have tools to compare prices and compute embedding-based cosine similarity.
Call BOTH tools, then decide.

Respond ONLY with a JSON object:
{"verdict": "MATCH" or "NO MATCH", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""


def format_pair_prompt(
  name_a: str, desc_a: str, price_a: str,
  name_b: str, desc_b: str, price_b: str,
) -> str:
  """Format a candidate pair into a user prompt for the agent."""
  return (
    f"Product A:\n"
    f"  Name: {name_a}\n"
    f"  Description: {desc_a}\n"
    f"  Price: {price_a}\n\n"
    f"Product B:\n"
    f"  Name: {name_b}\n"
    f"  Description: {desc_b}\n"
    f"  Price: {price_b}\n\n"
    f"Are Product A and Product B the same product?"
  )


def mcp_to_openai_tools(tools: list) -> list:
  """Convert FastMCP Tool objects to OpenAI function-calling tool schema."""
  return [
    {
      "type": "function",
      "function": {
        "name": t.name,
        "description": t.description or "",
        "parameters": t.inputSchema,
      },
    }
    for t in tools
  ]


def parse_agent_response(content: str | None) -> dict:
  """
  Parse an agent's final text response into a structured result. Delegates to the shared parse_verdict_response function.

  Returns dict with: verdict (str), confidence (float), reasoning (str), parse_error (bool).
  """
  from src.utils.parsing import parse_verdict_response
  return parse_verdict_response(content)


async def run_agent_loop(
  system_prompt: str,
  user_prompt: str,
  openai_client,
  mcp_client,
  tool_names: set,
  max_iterations: int = 10,
) -> tuple[dict, int]:
  """
  Run an agentic tool-use loop for one agent on one candidate pair.

  Returns (result_dict, total_tokens).
  """
  # Filter MCP tools to only those this agent should use
  all_tools = await mcp_client.list_tools()
  agent_tools = [t for t in all_tools if t.name in tool_names]
  openai_tools = mcp_to_openai_tools(agent_tools)

  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ]

  total_tokens = 0

  for _ in range(max_iterations):
    response = openai_client.chat.completions.create(
      model=MODEL,
      messages=messages,
      tools=openai_tools,
      tool_choice="auto",
      temperature=0,
      response_format={"type": "json_object"},
    )

    msg = response.choices[0].message
    total_tokens += response.usage.total_tokens if response.usage else 0

    if not msg.tool_calls:
      return parse_agent_response(msg.content), total_tokens

    # Append assistant message with tool calls
    messages.append({
      "role": "assistant",
      "content": msg.content,
      "tool_calls": [
        {
          "id": tc.id,
          "type": "function",
          "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in msg.tool_calls
      ],
    })

    # Execute each tool call via MCP client
    tool_schemas = {t.name: t.inputSchema for t in agent_tools}
    for tc in msg.tool_calls:
      try:
        args = json.loads(tc.function.arguments)
      except json.JSONDecodeError:
        messages.append({
          "role": "tool",
          "tool_call_id": tc.id,
          "content": "error: malformed arguments",
        })
        continue
      # Strip unexpected args the LLM may hallucinate
      schema = tool_schemas.get(tc.function.name, {})
      valid_keys = set(schema.get("properties", {}).keys())
      required_keys = set(schema.get("required", []))
      args = {k: v for k, v in args.items() if k in valid_keys}

      # Check for missing required arguments
      missing = required_keys - set(args.keys())
      if missing:
        messages.append({
          "role": "tool",
          "tool_call_id": tc.id,
          "content": f"error: missing required arguments: {', '.join(sorted(missing))}",
        })
        continue

      logger.debug("Tool call: %s(%s)", tc.function.name, json.dumps(args, default=str))
      try:
        result = await mcp_client.call_tool(tc.function.name, args)
        result_text = result.content[0].text if result.content else ""
      except Exception as e:
        result_text = f"error: tool call failed: {e}"
      logger.debug("Tool result: %s → %s", tc.function.name, result_text)
      messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result_text,
      })

  # Max iterations exceeded, create a Parse Error.
  return (
    {"verdict": "NO MATCH", "confidence": 0.0, "reasoning": "max iterations", "parse_error": True},
    total_tokens,
  )
