from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import DATA_DIR, RESULTS_DIR

RETAINED_COMPONENTS = ("Transformer", "Reactor", "Circuit Breaker", "Alt Energy")
EXCLUDED_COMPONENTS = ("Control", "Power Lines")


@dataclass(frozen=True)
class ValidationSummary:
    images: int
    prediction_rows: int
    models: int
    states_or_districts: int
    ferc_regions: int


def validate_repository(data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> ValidationSummary:
    """Validate the processed inputs/results required for local reproduction."""
    required = [
        data_dir / "component_predictions.csv",
        data_dir / "image_metadata.csv",
        data_dir / "nerc_gdf.geojson",
        data_dir / "substation_coordinates.parquet",
        data_dir / "us_state_populations.csv",
        results_dir / "model_performance.csv",
        results_dir / "model_bootstrapping_CIs.csv",
        results_dir / "state_component_counts.csv",
        results_dir / "ferc_region_component_counts.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required repository files:\n- " + "\n- ".join(missing))

    predictions = pd.read_csv(data_dir / "component_predictions.csv")
    metadata = pd.read_csv(data_dir / "image_metadata.csv")
    performance = pd.read_csv(results_dir / "model_performance.csv")
    states = pd.read_csv(results_dir / "state_component_counts.csv")
    ferc = pd.read_csv(results_dir / "ferc_region_component_counts.csv")

    for column in ("id", *RETAINED_COMPONENTS, *EXCLUDED_COMPONENTS):
        if column not in predictions.columns:
            raise ValueError(f"component_predictions.csv is missing required column: {column}")
    for column in ("id", "latitude", "longitude"):
        if column not in metadata.columns:
            raise ValueError(f"image_metadata.csv is missing required column: {column}")
    if predictions["id"].duplicated().any():
        raise ValueError("component_predictions.csv contains duplicate ids")
    if metadata["id"].duplicated().any():
        raise ValueError("image_metadata.csv contains duplicate ids")
    if set(predictions["id"]) != set(metadata["id"]):
        raise ValueError("Prediction and metadata ids do not match")

    expected_total = ferc[list(RETAINED_COMPONENTS)].sum(axis=1)
    if not expected_total.equals(ferc["TOTAL"]):
        raise ValueError("FERC TOTAL does not equal the sum of retained component classes")

    return ValidationSummary(
        images=len(metadata),
        prediction_rows=len(predictions),
        models=len(performance),
        states_or_districts=len(states),
        ferc_regions=len(ferc),
    )
