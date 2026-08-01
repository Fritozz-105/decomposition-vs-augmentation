"""Prompt templates and response parsing for single-LLM entity resolution."""

from src.utils.parsing import parse_verdict_response


SYSTEM_PROMPT = """You are an expert at entity resolution. Your task is to determine whether two product listings refer to the same real-world product.

Analyze the product names, descriptions, and prices carefully. Consider:
- Similar product names with minor variations (typos, abbreviations, word order)
- Matching key specifications (model numbers, sizes, capacities)
- Compatible price ranges (exact match not required, but large differences suggest different products)
- Similar product descriptions with overlapping features or specifications

Respond with ONLY a JSON object in this exact format:
{"verdict": "MATCH" or "NO MATCH", "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}

Example response:
{"verdict": "MATCH", "confidence": 0.9, "reasoning": "Names are very similar and descriptions mention the same model number."}
{"verdict": "NO MATCH", "confidence": 0.8, "reasoning": "Names are different and prices differ by more than 50%."}

Do NOT include any text outside the JSON object. Do NOT use markdown formatting. Only use numbers for confidence, do not put percents or letters."""


def _clean_price(price: str | float) -> str:
  """Convert price to display string, handling NaN and missing values."""
  if price is None:
    return "N/A"
  price_str = str(price)
  if price_str.lower() in ("nan", "none", ""):
    return "N/A"
  return price_str


def format_user_prompt(
  source_name: str,
  source_description: str,
  source_price: str | float,
  target_name: str,
  target_description: str,
  target_price: str | float,
) -> str:
  """Format a pair of products into a user prompt for the LLM."""
  return f"""Product A:
  Name: {source_name}
  Description: {source_description}
  Price: {_clean_price(source_price)}

Product B:
  Name: {target_name}
  Description: {target_description}
  Price: {_clean_price(target_price)}

Are Product A and Product B the same product?"""


def parse_llm_response(raw_response: str) -> dict:
  """
  Parse an LLM response into a structured prediction dict.

  Delegates to the shared parse_verdict_response function.

  Args:
    raw_response: The raw text response from the LLM.

  Returns:
    dict with keys: verdict (str), confidence (float), reasoning (str), parse_error (bool)
  """
  return parse_verdict_response(raw_response)
