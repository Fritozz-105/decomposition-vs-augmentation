"""Project root discovery utility."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
  """Walk up from start (default: this file's dir) looking for pyproject.toml."""
  current = (start or Path(__file__)).resolve().parent
  while current != current.parent:
    if (current / "pyproject.toml").exists():
      return current
    current = current.parent
  raise FileNotFoundError("Could not find project root (no pyproject.toml found)")
