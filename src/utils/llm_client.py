"""OpenAI client factory using environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.paths import find_project_root

DEFAULT_MODEL = "gpt-oss-120b"

_env_loaded = False


def _ensure_env(env_path: str | Path | None = None) -> None:
  global _env_loaded
  if not _env_loaded:
    if env_path is None:
      env_path = find_project_root() / ".env"
    load_dotenv(env_path)
    _env_loaded = True


def get_model_name() -> str:
  """Return the model name from MODEL_NAME env var, falling back to the default."""
  _ensure_env()
  return os.environ.get("MODEL_NAME", DEFAULT_MODEL)


def create_openai_client(env_path: str | Path | None = None) -> OpenAI:
  """
  Create an OpenAI client using credentials from .env file.

  Args:
    env_path: Path to .env file. Defaults to project root .env.

  Returns:
    Configured OpenAI client.
  """
  _ensure_env(env_path)

  api_key = os.environ.get("OPENAI_API_KEY")
  if not api_key:
    raise ValueError(
      "OPENAI_API_KEY not found. Set it in .env or as an environment variable."
    )

  base_url = os.environ.get("NAVIGATOR_API_BASE")

  return OpenAI(api_key=api_key, base_url=base_url)
