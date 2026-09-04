from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by finding pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Installed-package fallback: src/substation_detection/paths.py -> repo root.
    return Path(__file__).resolve().parents[2]


ROOT = find_repo_root(Path(__file__))
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"
