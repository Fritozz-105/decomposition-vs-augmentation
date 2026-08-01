"""LLM-based orchestrator for multi-agent entity resolution consensus."""

from src.pipelines.multi_agent.agents import parse_agent_response, MODEL

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator for a multi-agent product entity resolution system.
Two specialized agents have independently analyzed a product pair to determine
if they refer to the same real-world product.

Review both agents' verdicts, confidence scores, and reasoning. You have the
final say. You may agree with the agents or override their decision if you
believe their reasoning is flawed, inconsistent with the evidence, or if the
confidence scores do not justify the verdict.

Consider:
- Do the agents' reasoning chains make sense given the product data?
- Are confidence scores justified by the evidence cited?
- If agents disagree, which reasoning is more convincing?
- If agents agree but with weak reasoning, is the verdict still correct?

Respond ONLY with a JSON object:
{"verdict": "MATCH" or "NO MATCH", "confidence": 0.0-1.0, "reasoning": "brief explanation of your decision"}"""


def format_orchestrator_prompt(
  user_prompt: str,
  syntactic: dict,
  semantic: dict,
) -> str:
  """Format the orchestrator prompt with the product pair and agent results."""
  return (
    f"{user_prompt}\n\n"
    f"--- Agent Results ---\n\n"
    f"Syntactic Agent (text similarity analysis):\n"
    f"  Verdict: {syntactic['verdict']}\n"
    f"  Confidence: {syntactic['confidence']}\n"
    f"  Reasoning: {syntactic['reasoning']}\n\n"
    f"Semantic Agent (meaning and price analysis):\n"
    f"  Verdict: {semantic['verdict']}\n"
    f"  Confidence: {semantic['confidence']}\n"
    f"  Reasoning: {semantic['reasoning']}\n\n"
    f"Based on the product data and both agents' analysis, provide your final verdict."
  )


async def apply_consensus(
  syntactic: dict,
  semantic: dict,
  user_prompt: str,
  openai_client,
) -> tuple[dict, int]:
  """
  LLM-based consensus: orchestrator reviews agent verdicts and may override.

  Returns (result_dict, tokens_used).
  """
  prompt = format_orchestrator_prompt(user_prompt, syntactic, semantic)

  response = openai_client.chat.completions.create(
    model=MODEL,
    messages=[
      {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
      {"role": "user", "content": prompt},
    ],
    temperature=0,
    response_format={"type": "json_object"},
  )

  text = response.choices[0].message.content or ""
  tokens = response.usage.total_tokens if response.usage else 0

  result = parse_agent_response(text)
  return result, tokens
