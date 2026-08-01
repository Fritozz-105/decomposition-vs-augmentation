"""Tests for MCP tools in multi-agent pipeline."""

from src.pipelines.multi_agent.mcp_tools import (
  compute_edit_distance,
  compute_jaccard_similarity,
  compute_bm25_score,
  compare_prices,
  compute_embedding_cosine,
)


class TestComputeEditDistance:
  """Edit distance measures character-level divergence between product names."""

  def test_identical_strings(self):
    """Identical strings should have zero edit distance."""
    assert compute_edit_distance("apple", "apple") == 0.0

  def test_completely_different(self):
    """Non-overlapping strings should produce a high distance score."""
    result = compute_edit_distance("abc", "xyz")
    assert result > 0.5

  def test_similar_strings(self):
    """Strings differing by one character should have a small distance."""
    result = compute_edit_distance("apple", "appl")
    assert 0.0 < result < 0.5

  def test_empty_strings(self):
    """Two empty strings are identical, so distance should be zero."""
    assert compute_edit_distance("", "") == 0.0


class TestComputeJaccardSimilarity:
  """Jaccard similarity measures word-level token overlap."""

  def test_identical(self):
    """Same token sets should yield perfect similarity."""
    assert compute_jaccard_similarity("hello world", "hello world") == 1.0

  def test_disjoint(self):
    """No shared tokens should yield zero similarity."""
    assert compute_jaccard_similarity("hello", "world") == 0.0

  def test_partial_overlap(self):
    """Partially overlapping token sets should yield a score between 0 and 1."""
    result = compute_jaccard_similarity("hello world", "hello earth")
    assert 0.0 < result < 1.0

  def test_both_empty(self):
    """Two empty strings have no tokens; similarity defaults to zero."""
    assert compute_jaccard_similarity("", "") == 0.0


class TestComputeBm25Score:
  """BM25 score measures term-frequency relevance, normalized to [0, 1]."""

  def test_identical(self):
    """Identical multi-word texts should produce a positive BM25 score."""
    result = compute_bm25_score("apple iphone 12 pro", "apple iphone 12 pro")
    assert result > 0.0

  def test_similar_higher_than_different(self):
    """Similar texts should rank higher than dissimilar texts."""
    similar = compute_bm25_score("apple iphone 12", "apple iphone 12 pro")
    different = compute_bm25_score("apple iphone", "samsung galaxy")
    assert similar > different

  def test_empty_string(self):
    """An empty input should return zero since no terms can match."""
    assert compute_bm25_score("", "hello") == 0.0

  def test_whitespace_only(self):
    """Whitespace-only input should be treated as empty and return zero."""
    assert compute_bm25_score("   ", "hello") == 0.0

  def test_different_texts(self):
    """Unrelated texts should produce a low but valid score in [0, 1]."""
    result = compute_bm25_score("apple iphone", "samsung galaxy")
    assert 0.0 <= result < 1.0

  def test_normalized_range(self):
    """BM25 output must always be within [0, 1] after normalization."""
    result = compute_bm25_score("the quick brown fox", "quick fox jumps")
    assert 0.0 <= result <= 1.0


class TestComparePrices:
  """Price comparison extracts numeric values and checks similarity."""

  def test_parseable_similar(self):
    """Close prices should be flagged as similar (ratio >= 0.8)."""
    result = compare_prices("$99.99", "$89.99")
    assert result["price_a"] == 99.99
    assert result["price_b"] == 89.99
    assert result["ratio"] is not None
    assert result["absolute_diff"] == 10.0
    assert result["similar"] is True

  def test_parseable_different(self):
    """Prices with a 10x difference should not be flagged as similar."""
    result = compare_prices("$10.00", "$100.00")
    assert result["similar"] is False

  def test_unparseable(self):
    """Non-numeric price strings should return None fields with a note."""
    result = compare_prices("N/A", "$50.00")
    assert result["price_a"] is None
    assert result["note"] == "unparseable"

  def test_both_unparseable(self):
    """Two non-numeric prices yield all None comparison fields."""
    result = compare_prices("free", "unknown")
    assert result["ratio"] is None

  def test_equal_prices(self):
    """Identical prices should have ratio 1.0 and be similar."""
    result = compare_prices("$50.00", "$50.00")
    assert result["ratio"] == 1.0
    assert result["similar"] is True


class TestComputeEmbeddingCosine:
  """Embedding cosine uses sentence-transformers for semantic similarity."""

  def test_identical_text(self):
    """Identical texts should produce near-perfect cosine similarity."""
    result = compute_embedding_cosine("apple iphone 12", "apple iphone 12")
    assert result > 0.99

  def test_different_text(self):
    """Semantically unrelated texts should have low similarity."""
    result = compute_embedding_cosine("apple iphone", "wooden table")
    assert result < 0.8

  def test_empty_text(self):
    """An empty input text should return zero similarity."""
    assert compute_embedding_cosine("", "hello") == 0.0

  def test_both_empty(self):
    """Two empty texts should return zero similarity."""
    assert compute_embedding_cosine("", "") == 0.0
