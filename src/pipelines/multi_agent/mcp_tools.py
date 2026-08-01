"""MCP tool server for multi-agent entity resolution."""

import re
import difflib

from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi
from fastmcp import FastMCP

from src.utils.paths import find_project_root

load_dotenv(find_project_root() / ".env")

mcp = FastMCP("er-tools")

_embedding_model = None


def _get_embedding_model():
  """Lazy-load the sentence-transformers model (singleton)."""
  global _embedding_model
  if _embedding_model is None:
    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
  return _embedding_model


# Syntactic Agent Tools

@mcp.tool()
def compute_edit_distance(name1: str, name2: str) -> float:
  """Compute normalized edit distance between two product names. Returns 0.0 for identical strings, ~1.0 for completely different strings."""
  return 1.0 - difflib.SequenceMatcher(None, name1, name2).ratio()


@mcp.tool()
def compute_jaccard_similarity(name1: str, name2: str) -> float:
  """Compute Jaccard similarity between two product names based on word tokens. Returns 1.0 for identical, 0.0 for disjoint."""
  tokens1 = set(name1.lower().split())
  tokens2 = set(name2.lower().split())
  if not tokens1 and not tokens2:
    return 0.0
  intersection = tokens1 & tokens2
  union = tokens1 | tokens2
  return len(intersection) / len(union)


@mcp.tool()
def compute_bm25_score(name1: str, name2: str) -> float:
  """Compute BM25 relevance score between two product names. Returns a normalized score in [0, 1] where higher means more similar."""
  if not name1 or not name1.strip() or not name2 or not name2.strip():
    return 0.0
  tokens1 = name1.lower().split()
  tokens2 = name2.lower().split()
  # Background empty docs give IDF enough variance to produce positive scores
  corpus = [tokens1, tokens2] + [[] for _ in range(3)]
  bm25 = BM25Okapi(corpus)
  # Symmetric cross-score: text2 as query against doc0 (text1) and vice versa
  score_1 = bm25.get_scores(tokens2)[0]
  score_2 = bm25.get_scores(tokens1)[1]
  avg_score = (score_1 + score_2) / 2
  # Normalize to [0, 1] via score / (score + 1)
  return round(avg_score / (avg_score + 1), 4) if avg_score > 0 else 0.0


# Semantic Agent Tools

@mcp.tool()
def compare_prices(price1: str, price2: str) -> dict:
  """Compare two price strings. Returns ratio, absolute difference, and similarity flag (ratio >= 0.8)."""
  def parse_price(p: str) -> float | None:
    match = re.search(r"[\d]+\.?\d*", p)
    return float(match.group()) if match else None

  a = parse_price(price1)
  b = parse_price(price2)

  if a is not None and b is not None:
    max_val = max(a, b)
    ratio = min(a, b) / max_val if max_val > 0 else 1.0
    return {
      "price_a": a,
      "price_b": b,
      "ratio": round(ratio, 4),
      "absolute_diff": round(abs(a - b), 2),
      "similar": ratio >= 0.8,
    }

  return {
    "price_a": None,
    "price_b": None,
    "ratio": None,
    "absolute_diff": None,
    "similar": None,
    "note": "unparseable",
  }


@mcp.tool()
def compute_embedding_cosine(text1: str, text2: str) -> float:
  """Compute cosine similarity between two product names using sentence embeddings (all-MiniLM-L6-v2). Pass the full product names."""
  if not text1 or not text2:
    return 0.0
  model = _get_embedding_model()
  embs = model.encode([text1, text2], normalize_embeddings=True)
  return float(np.dot(embs[0], embs[1]))


SYNTACTIC_TOOLS = {"compute_edit_distance", "compute_jaccard_similarity", "compute_bm25_score"}
SEMANTIC_TOOLS = {"compare_prices", "compute_embedding_cosine"}

if __name__ == "__main__":
  mcp.run()
