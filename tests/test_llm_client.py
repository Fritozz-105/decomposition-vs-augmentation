"""Tests for src/utils/llm_client.py."""

from unittest.mock import MagicMock, patch
import pytest
from src.utils.llm_client import create_openai_client


def test_missing_api_key_raises():
  """Raises ValueError when OPENAI_API_KEY is not set, preventing silent misconfiguration."""
  with patch.dict("os.environ", {}, clear=True):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
      create_openai_client(env_path="/nonexistent/.env")


@patch("src.utils.llm_client.OpenAI")
def test_client_created_with_key(mock_openai):
  """Creates an OpenAI client using the API key from environment variables."""
  mock_openai.return_value = MagicMock()
  env = {"OPENAI_API_KEY": "test-key-123"}
  with patch.dict("os.environ", env, clear=True):
    client = create_openai_client(env_path="/nonexistent/.env")
    mock_openai.assert_called_once_with(api_key="test-key-123", base_url=None)


@patch("src.utils.llm_client.OpenAI")
def test_base_url_passed_when_set(mock_openai):
  """Passes NAVIGATOR_API_BASE as base_url so the client hits the UF API endpoint."""
  mock_openai.return_value = MagicMock()
  env = {"OPENAI_API_KEY": "test-key", "NAVIGATOR_API_BASE": "https://custom.api/v1"}
  with patch.dict("os.environ", env, clear=True):
    client = create_openai_client(env_path="/nonexistent/.env")
    mock_openai.assert_called_once_with(api_key="test-key", base_url="https://custom.api/v1")
