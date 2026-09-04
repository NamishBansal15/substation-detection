import pandas as pd

from substation_detection.paths import DATA_DIR, RESULTS_DIR
from substation_detection.validation import RETAINED_COMPONENTS, validate_repository


def test_repository_validation():
    summary = validate_repository()
    assert summary.images > 0
    assert summary.prediction_rows == summary.images
    assert summary.models == 16


def test_ferc_totals_are_component_sums():
    df = pd.read_csv(RESULTS_DIR / "ferc_region_component_counts.csv")
    pd.testing.assert_series_equal(
        df[list(RETAINED_COMPONENTS)].sum(axis=1),
        df["TOTAL"],
        check_names=False,
    )


def test_prediction_ids_match_metadata_ids():
    pred = pd.read_csv(DATA_DIR / "component_predictions.csv")
    meta = pd.read_csv(DATA_DIR / "image_metadata.csv")
    assert set(pred["id"]) == set(meta["id"])
