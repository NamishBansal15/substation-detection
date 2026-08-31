from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_METADATA_PATH = DATA_DIR / "image_metadata.csv"
PREDICTIONS_PATH = DATA_DIR / "component_predictions.csv"
NERC_SHAPE = DATA_DIR / "nerc_gdf.geojson"

COMPONENT_COLS = [
    "Transformer",
    "Reactor",
    "Circuit Breaker",
    "Alt Energy",
]


def main():
    # Load files
    meta = pd.read_csv(IMAGE_METADATA_PATH)
    preds = pd.read_csv(PREDICTIONS_PATH)
    nerc = gpd.read_file(NERC_SHAPE)

    # Standardize region column name
    if "REGIONS" in nerc.columns and "NERC" not in nerc.columns:
        nerc = nerc.rename(columns={"REGIONS": "NERC"})

    # Merge coordinates with component predictions
    df = meta.merge(preds, on="id", how="left")

    # Make sure component columns are numeric
    for col in COMPONENT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Convert each record to a geographic point
    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )

    # Match each point to a FERC/NERC planning region
    nerc = nerc.to_crs("EPSG:4326")

    joined = gpd.sjoin(
        points,
        nerc[["NERC", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Sum the four retained component classes by region
    ferc_counts = (
        joined.groupby("NERC")[COMPONENT_COLS]
        .sum()
        .reset_index()
    )

    # Add total count
    ferc_counts["TOTAL"] = ferc_counts[COMPONENT_COLS].sum(axis=1)

    # Sort largest regions first
    ferc_counts = ferc_counts.sort_values(
        "TOTAL",
        ascending=False
    ).reset_index(drop=True)

    # Rename column for cleaner public-facing CSV
    ferc_counts = ferc_counts.rename(
        columns={"NERC": "FERC_REGION"}
    )

    # Save
    output_path = RESULTS_DIR / "ferc_region_component_counts.csv"
    ferc_counts.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print()
    print(ferc_counts)


if __name__ == "__main__":
    main()