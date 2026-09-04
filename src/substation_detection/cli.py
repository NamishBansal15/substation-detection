from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .paths import GRAPHS_DIR, ROOT
from .validation import validate_repository


def _run_script(path: Path, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(path), *args],
        cwd=ROOT,
        check=True,
    )


def cmd_check() -> int:
    summary = validate_repository()
    print("Repository validation passed.")
    print(f"  images:               {summary.images:,}")
    print(f"  prediction rows:      {summary.prediction_rows:,}")
    print(f"  models:               {summary.models:,}")
    print(f"  states/districts:     {summary.states_or_districts:,}")
    print(f"  FERC regions:         {summary.ferc_regions:,}")
    return 0


def cmd_figures() -> int:
    validate_repository()

    scripts = [
        ("model_performance_graph.py", ()),
        ("substation_inference_bar_scatter_graph.py", ()),
        ("substation_inference_choropleth_graphs.py", ()),
        ("substation_inference_choropleth_graphs.py", ("--vertical",)),
    ]

    for name, args in scripts:
        label = f"{name} {' '.join(args)}".strip()
        print(f"Running {label} ...", flush=True)
        _run_script(GRAPHS_DIR / name, *args)

    print("Figure reproduction complete.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce and validate the substation-detection analysis.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("check", "figures", "all"),
        default="check",
        help="check data integrity, regenerate figures, or do both (default: check)",
    )
    args = parser.parse_args(argv)
    if args.command == "check":
        return cmd_check()
    if args.command == "figures":
        return cmd_figures()
    cmd_check()
    return cmd_figures()


if __name__ == "__main__":
    raise SystemExit(main())
