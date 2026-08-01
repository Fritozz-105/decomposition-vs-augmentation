"""Magellan-faithful feature engineering for entity resolution.
Replicates the exact feature set that py_entitymatching's: name(STR_BT_1W_5W): 8 features, description (STR_GT_10W): 2 features, price (NUM): 4 features
"""

import math
from collections import Counter

import jellyfish
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

FEATURE_NAMES = [
  # name: STR_BT_1W_5W (8)
  "name_jaccard_qgm3",
  "name_cosine_tokens",
  "name_jaccard_tokens",
  "name_monge_elkan",
  "name_lev_dist",
  "name_lev_sim",
  "name_needleman_wunsch",
  "name_smith_waterman",
  # description: STR_GT_10W (2)
  "desc_jaccard_qgm3",
  "desc_cosine_tfidf",
  # price: NUM (4)
  "price_exact_match",
  "price_abs_norm",
  "price_lev_dist",
  "price_lev_sim",
]


# ngram Tokenizer functions
def qgram_tokenize(s: str, q: int = 3) -> list[str]:
  """Return q-character grams from a lowercased string."""
  s = s.lower()
  if len(s) < q:
    return [s] if s else []
  return [s[i:i + q] for i in range(len(s) - q + 1)]


# String similarity functions for features
def jaccard_tokens(s1: str, s2: str) -> float:
  """Jaccard similarity on whitespace-tokenized lowercase words."""
  tokens1 = set(s1.lower().split())
  tokens2 = set(s2.lower().split())
  if not tokens1 and not tokens2:
    return 1.0
  if not tokens1 or not tokens2:
    return 0.0
  return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def jaccard_qgrams(s1: str, s2: str, q: int = 3) -> float:
  """Jaccard similarity on character q-grams."""
  t1 = set(qgram_tokenize(s1, q))
  t2 = set(qgram_tokenize(s2, q))
  if not t1 and not t2:
    return 1.0
  if not t1 or not t2:
    return 0.0
  return len(t1 & t2) / len(t1 | t2)


def cosine_tokens(s1: str, s2: str) -> float:
  """Cosine similarity on word-token term-frequency vectors."""
  c1 = Counter(s1.lower().split())
  c2 = Counter(s2.lower().split())
  if not c1 or not c2:
    return 0.0
  dot = sum(c1[t] * c2[t] for t in c1 if t in c2)
  norm1 = math.sqrt(sum(v * v for v in c1.values()))
  norm2 = math.sqrt(sum(v * v for v in c2.values()))
  if norm1 == 0 or norm2 == 0:
    return 0.0
  return dot / (norm1 * norm2)


def cosine_tfidf(s1: str, s2: str) -> float:
  """
  TF-IDF weighted cosine similarity between two texts. Matches py_entitymatching's approach
  for STR_GT_10W attributes.
  """
  if not s1.strip() or not s2.strip():
    return 0.0
  try:
    vec = TfidfVectorizer().fit_transform([s1.lower(), s2.lower()])
  except ValueError:
    return 0.0
  return float(sklearn_cosine(vec[0:1], vec[1:2])[0, 0])


def lev_dist(s1: str, s2: str) -> int:
  """Raw Levenshtein distance (integer >= 0)."""
  return jellyfish.levenshtein_distance(s1.lower(), s2.lower())


def lev_sim(s1: str, s2: str) -> float:
  """Normalized Levenshtein similarity: 1 - dist / max(len)."""
  s1l, s2l = s1.lower(), s2.lower()
  d = jellyfish.levenshtein_distance(s1l, s2l)
  max_len = max(len(s1l), len(s2l))
  if max_len == 0:
    return 1.0
  return 1.0 - d / max_len


def jaro_winkler_sim(s1: str, s2: str) -> float:
  """Jaro-Winkler similarity."""
  return jellyfish.jaro_winkler_similarity(s1.lower(), s2.lower())


def monge_elkan(s1: str, s2: str) -> float:
  """Monge-Elkan with Jaro-Winkler as the secondary similarity."""
  tokens1 = s1.lower().split()
  tokens2 = s2.lower().split()
  if not tokens1 and not tokens2:
    return 1.0
  if not tokens1 or not tokens2:
    return 0.0
  total = 0.0
  for t1 in tokens1:
    best = max(jellyfish.jaro_winkler_similarity(t1, t2) for t2 in tokens2)
    total += best
  return total / len(tokens1)


def needleman_wunsch(s1: str, s2: str, gap_cost: float = 1.0) -> float:
  """Needleman-Wunsch global alignment score (gap=1, match=1, mismatch=0).

  Matches py_stringmatching defaults: gap_cost=1, sim_func=sim_ident.
  Default of Magellan according to anhaidgroup website.
  """
  s1, s2 = s1.lower(), s2.lower()
  n, m = len(s1), len(s2)
  if n == 0 and m == 0:
    return 0.0
  dp = [[0.0] * (m + 1) for _ in range(n + 1)]
  for i in range(1, n + 1):
    dp[i][0] = -i * gap_cost
  for j in range(1, m + 1):
    dp[0][j] = -j * gap_cost
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      match = dp[i - 1][j - 1] + (1.0 if s1[i - 1] == s2[j - 1] else 0.0)
      delete = dp[i - 1][j] - gap_cost
      insert = dp[i][j - 1] - gap_cost
      dp[i][j] = max(match, delete, insert)
  return dp[n][m]


def smith_waterman(s1: str, s2: str, gap_cost: float = 1.0) -> float:
  """Smith-Waterman local alignment score (gap=1, match=1, mismatch=0).

  Matches py_stringmatching defaults: gap_cost=1, sim_func=sim_ident.
  Default of Magellan according to anhaidgroup website.
  """
  s1, s2 = s1.lower(), s2.lower()
  n, m = len(s1), len(s2)
  if n == 0 or m == 0:
    return 0.0
  best = 0.0
  dp = [[0.0] * (m + 1) for _ in range(n + 1)]
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      match = dp[i - 1][j - 1] + (1.0 if s1[i - 1] == s2[j - 1] else 0.0)
      delete = dp[i - 1][j] - gap_cost
      insert = dp[i][j - 1] - gap_cost
      dp[i][j] = max(0.0, match, delete, insert)
      if dp[i][j] > best:
        best = dp[i][j]
  return best


# Numeric features for price
def parse_price(price_str: str) -> float | None:
  """Parse a price string to float, stripping currency symbols."""
  try:
    cleaned = str(price_str).replace("$", "").replace(",", "").strip()
    val = float(cleaned)
    if math.isnan(val) or math.isinf(val):
      return None
    return val
  except (ValueError, AttributeError):
    return None


def price_exact_match(p1: str, p2: str) -> float:
  """1.0 if parsed prices are equal, else 0.0."""
  v1 = parse_price(p1)
  v2 = parse_price(p2)
  if v1 is None or v2 is None:
    return 0.0
  return 1.0 if v1 == v2 else 0.0


def abs_norm(p1: str, p2: str) -> float:
  """Absolute-difference normalized: 1 - |p1-p2| / max(|p1|, |p2|)."""
  v1 = parse_price(p1)
  v2 = parse_price(p2)
  if v1 is None or v2 is None:
    return 0.5
  denom = max(abs(v1), abs(v2))
  if denom == 0:
    return 1.0
  return 1.0 - abs(v1 - v2) / denom


def price_lev_dist(p1: str, p2: str) -> int:
  """Levenshtein distance on raw price strings."""
  return lev_dist(str(p1), str(p2))


def price_lev_sim(p1: str, p2: str) -> float:
  """Levenshtein similarity on raw price strings."""
  return lev_sim(str(p1), str(p2))


# Feature vector assembly
def compute_pair_features(src: dict, tgt: dict) -> list[float]:
  """Compute the 14-feature vector for a single (src, tgt) product pair."""
  return [
    # name: STR_BT_1W_5W (8)
    jaccard_qgrams(src["name"], tgt["name"]),
    cosine_tokens(src["name"], tgt["name"]),
    jaccard_tokens(src["name"], tgt["name"]),
    monge_elkan(src["name"], tgt["name"]),
    lev_dist(src["name"], tgt["name"]),
    lev_sim(src["name"], tgt["name"]),
    needleman_wunsch(src["name"], tgt["name"]),
    smith_waterman(src["name"], tgt["name"]),
    # description: STR_GT_10W (2)
    jaccard_qgrams(src["description"], tgt["description"]),
    cosine_tfidf(src["description"], tgt["description"]),
    # price: NUM (4)
    price_exact_match(src["price"], tgt["price"]),
    abs_norm(src["price"], tgt["price"]),
    price_lev_dist(src["price"], tgt["price"]),
    price_lev_sim(src["price"], tgt["price"]),
  ]


def compute_features_for_pairs(
  pairs_df: pd.DataFrame,
  source_lookup: dict[str, dict],
  target_lookup: dict[str, dict],
  src_col: str = "source_id",
  tgt_col: str = "target_id",
) -> tuple[np.ndarray, list[tuple[str, str]]]:
  """
  Compute 14-feature matrix for all (src, tgt) pairs.

  Returns:
    (X, pair_ids) where X is shape (n, 14) float array and
    pair_ids is list of (source_id, target_id) tuples corresponding to rows.
    Pairs missing from either lookup are silently skipped.
  """
  rows: list[list[float]] = []
  pair_ids: list[tuple[str, str]] = []

  for _, row in pairs_df.iterrows():
    src_id = str(row[src_col])
    tgt_id = str(row[tgt_col])
    src = source_lookup.get(src_id)
    tgt = target_lookup.get(tgt_id)
    if src is None or tgt is None:
      continue
    rows.append(compute_pair_features(src, tgt))
    pair_ids.append((src_id, tgt_id))

  if not rows:
    return np.empty((0, len(FEATURE_NAMES)), dtype=float), []

  return np.array(rows, dtype=float), pair_ids
