"""
Question 3 figures — consolidated VS Code version.

Expected project layout:
project/
├── question3_graphs_vscode.py
├── data/
│   ├── image_metadata.csv
│   ├── component_predictions.csv
│   ├── us_state_populations.csv
│   ├── nerc_gdf.geojson
│   ├── cb_2018_us_state_500k.shp
│   ├── cb_2018_us_state_500k.dbf
│   ├── cb_2018_us_state_500k.shx
│   └── cb_2018_us_state_500k.prj
└── results/

Install once from the VS Code zsh terminal:
python3 -m pip install pandas numpy matplotlib geopandas contextily shapely adjustText

This script:
1. Loads/prepares all data once.
2. Creates bar_and_scatter_2x2.png.
3. Creates choropleth_maps_2x2.png.
4. Uses Times New Roman (with Times/DejaVu Serif fallback).
5. Keeps the original bar colors and uses Viridis for continuous data-driven color.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import contextily as ctx
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_METADATA_PATH = DATA_DIR / "image_metadata.csv"
PREDICTIONS_PATH = DATA_DIR / "component_predictions.csv"
POPULATION_PATH = DATA_DIR / "us_state_populations.csv"
NERC_SHAPE = DATA_DIR / "nerc_gdf.geojson"
STATE_SHAPE = DATA_DIR / "cb_2018_us_state_500k.shp"


# ============================================================
# GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})

VIRIDIS = plt.get_cmap("viridis")

COMPONENT_COLS = [
    "Transformer",
    "Reactor",
    "Circuit Breaker",
    "Alt Energy",
]

# Original categorical bar palette from the notebook.
# These are intentionally NOT replaced with Viridis because component type
# is categorical rather than a low-to-high numeric scale.
COMPONENT_COLORS = {
    "Transformer": "#4e79a7",
    "Reactor": "#f28e2b",
    "Circuit Breaker": "#e15759",
    "Alt Energy": "#76b7b2",
}


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare_data():
    required = [
        IMAGE_METADATA_PATH,
        PREDICTIONS_PATH,
        POPULATION_PATH,
        NERC_SHAPE,
        STATE_SHAPE,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_text = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "The following required files are missing:\n"
            f"{missing_text}\n\n"
            "Place them in the data/ folder beside this script."
        )

    meta = pd.read_csv(IMAGE_METADATA_PATH)
    preds = pd.read_csv(PREDICTIONS_PATH)
    pop_df = pd.read_csv(POPULATION_PATH)

    df = meta.merge(preds, on="id", how="left")

    for col in COMPONENT_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )

    states = gpd.read_file(STATE_SHAPE)[["STUSPS", "geometry", "ALAND"]]
    states = states.rename(columns={"STUSPS": "STATE"}).to_crs("EPSG:4326")
    states = states.merge(pop_df, on="STATE", how="left")

    states["POPULATION"] = states["POPULATION"].fillna(0)
    states["AREA_SQM"] = states["ALAND"]
    states["AREA_SQKM"] = states["ALAND"] / 1_000_000
    states["AREA_SQMI"] = states["ALAND"] / 2_589_988.11
    states["POP_DENSITY"] = (
        states["POPULATION"] / states["AREA_SQKM"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    nerc_shapes = gpd.read_file(NERC_SHAPE)
    if "REGIONS" in nerc_shapes.columns and "NERC" not in nerc_shapes.columns:
        nerc_shapes = nerc_shapes.rename(columns={"REGIONS": "NERC"})
    nerc_shapes = nerc_shapes.to_crs("EPSG:4326")

    # Spatial assignments.
    gdf_points = gdf_points.reset_index(drop=True)
    gdf_points = gpd.sjoin(
        gdf_points,
        states[
            [
                "STATE",
                "POPULATION",
                "AREA_SQKM",
                "AREA_SQMI",
                "AREA_SQM",
                "POP_DENSITY",
                "geometry",
            ]
        ],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    gdf_points = gpd.sjoin(
        gdf_points,
        nerc_shapes[["NERC", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    exclude = ["AK", "HI", "PR", "GU", "VI", "MP", "AS"]
    states = states[~states["STATE"].isin(exclude)].copy()
    gdf_points = gdf_points[~gdf_points["STATE"].isin(exclude)].copy()
    gdf_points = gdf_points.dropna(subset=["STATE", "NERC"]).reset_index(drop=True)

    # Project mapping layers to Web Mercator for contextily.
    states_3857 = states.to_crs(epsg=3857)
    nerc_3857 = nerc_shapes.to_crs(epsg=3857)
    points_3857 = gdf_points.to_crs(epsg=3857)

    return meta, gdf_points, points_3857, states, states_3857, nerc_shapes, nerc_3857


def aggregate_data(gdf_points, states, nerc_shapes):
    state_comp = gdf_points.groupby("STATE")[COMPONENT_COLS].sum()
    state_comp["TOTAL"] = state_comp[COMPONENT_COLS].sum(axis=1)

    state_info = states[
        ["STATE", "POPULATION", "AREA_SQKM", "AREA_SQMI", "AREA_SQM", "POP_DENSITY"]
    ].drop_duplicates("STATE").set_index("STATE")

    state_comp = state_comp.merge(
        state_info, left_index=True, right_index=True, how="left"
    )

    state_comp["COMP_DENSITY_PER_CAPITA"] = (
        state_comp["TOTAL"] / state_comp["POPULATION"] * 1_000_000
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    state_comp["COMP_DENSITY_PER_AREA_KM"] = (
        state_comp["TOTAL"] / state_comp["AREA_SQKM"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # NERC totals.
    nerc_comp = gdf_points.groupby("NERC")[COMPONENT_COLS].sum().reset_index()
    nerc_comp["TOTAL"] = nerc_comp[COMPONENT_COLS].sum(axis=1)

    # Dissolve first so region area is not double-counted across fragments.
    nerc_dissolved = nerc_shapes[["NERC", "geometry"]].dissolve(by="NERC").reset_index()
    nerc_equal_area = nerc_dissolved.to_crs(epsg=5070)
    nerc_equal_area["AREA_SQKM"] = nerc_equal_area.geometry.area / 1_000_000

    # Approximate NERC population from assigned records while retaining original
    # notebook behavior.
    nerc_population = (
        gdf_points.groupby("NERC")["POPULATION"]
        .sum()
        .reset_index(name="NERC_POPULATION")
    )

    nerc_comp = nerc_comp.merge(
        nerc_equal_area[["NERC", "AREA_SQKM"]], on="NERC", how="left"
    )
    nerc_comp = nerc_comp.merge(nerc_population, on="NERC", how="left")

    nerc_comp["COMP_DENSITY_PER_AREA_KM"] = (
        nerc_comp["TOTAL"] / nerc_comp["AREA_SQKM"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    nerc_comp["NERC_POP_DENSITY"] = (
        nerc_comp["NERC_POPULATION"] / nerc_comp["AREA_SQKM"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return state_comp, nerc_comp, nerc_dissolved


# ============================================================
# FIGURE 1 — BARS + SCATTERS
# ============================================================

def make_bar_scatter_figure(state_comp, nerc_comp):
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 21,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 15,
    })

    # Explicit spacing is more predictable than constrained_layout here,
    # especially because panels (c) and (d) have appended colorbars.
    fig = plt.figure(figsize=(24, 18.5), facecolor="white")
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.0, 1.08],
        hspace=0.24,
        wspace=0.20,
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, top=0.975, bottom=0.075,
        hspace=0.24, wspace=0.20
    )
    formatter = mticker.StrMethodFormatter("{x:,.0f}")

    # ---------- (a) Top states ----------
    ax1 = fig.add_subplot(gs[0, 0])
    top_states = (
        state_comp.sort_values("TOTAL", ascending=False)
        .head(10)[COMPONENT_COLS]
    )

    top_states.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        color=[COMPONENT_COLORS[c] for c in COMPONENT_COLS],
        edgecolor="black",
        linewidth=0.6,
    )

    totals = top_states.sum(axis=1)
    offset = totals.max() * 0.01
    for i, total in enumerate(totals):
        ax1.text(
            i, total + offset, f"{int(total):,}",
            ha="center", va="bottom", fontsize=16, fontweight="bold"
        )

    ax1.set_title("(a) Top 10 States by Component Count", weight="bold")
    ax1.set_ylabel("Total Component Count")
    ax1.set_xlabel("State")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.legend(title="Component", frameon=False, loc="upper right")

    # ---------- (b) NERC/FERC regions ----------
    ax2 = fig.add_subplot(gs[0, 1])

    region_map = {
        "NorthernGridConnected": "NGC",
        "WestConnectNonEnrolled": "WCNE",
        "NorthernGridUnconnected": "NGU",
    }

    nerc_sorted = (
        nerc_comp.sort_values("TOTAL", ascending=False)
        .set_index("NERC")[COMPONENT_COLS]
    )
    nerc_sorted.index = nerc_sorted.index.to_series().replace(region_map)

    nerc_sorted.plot(
        kind="bar",
        stacked=True,
        ax=ax2,
        color=[COMPONENT_COLORS[c] for c in COMPONENT_COLS],
        edgecolor="black",
        linewidth=0.6,
    )

    totals = nerc_sorted.sum(axis=1)
    offset = totals.max() * 0.01
    for i, total in enumerate(totals):
        ax2.text(
            i, total + offset, f"{int(total):,}",
            ha="center", va="bottom", fontsize=16, fontweight="bold"
        )

    ax2.set_title("(b) FERC Regions by Component Count", weight="bold")
    ax2.set_ylabel("Total Component Count")
    ax2.set_xlabel("Region")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.legend(title="Component", frameon=False, loc="upper right")

    # ---------- (c) State scatter ----------
    ax3 = fig.add_subplot(gs[1, 0])

    state_regions = {
        "ME":"Northeast","VT":"Northeast","NH":"Northeast","MA":"Northeast",
        "RI":"Northeast","CT":"Northeast","NY":"Northeast","PA":"Northeast",
        "NJ":"Northeast","DE":"Northeast","MD":"Northeast",
        "FL":"South","GA":"South","SC":"South","NC":"South","VA":"South",
        "WV":"South","KY":"South","TN":"South","MS":"South","AL":"South",
        "AR":"South","LA":"South","TX":"South","OK":"South",
        "OH":"Midwest","IN":"Midwest","IL":"Midwest","MI":"Midwest",
        "WI":"Midwest","MN":"Midwest","IA":"Midwest","MO":"Midwest",
        "ND":"Midwest","SD":"Midwest","NE":"Midwest","KS":"Midwest",
        "MT":"West","WY":"West","CO":"West","NM":"West","AZ":"West",
        "UT":"West","NV":"West","ID":"West","WA":"West","OR":"West","CA":"West",
    }
    region_markers = {
        "Northeast": "o",
        "South": "s",
        "Midwest": "^",
        "West": "D",
        "Other": "X",
    }

    state_plot = state_comp.copy()
    state_plot["REGION_GROUP"] = (
        state_plot.index.to_series().map(state_regions).fillna("Other")
    )

    max_total = max(state_plot["TOTAL"].max(), 1)
    sizes = state_plot["TOTAL"] / max_total * 1250 + 35
    norm_c = Normalize(vmin=0, vmax=max(state_plot["COMP_DENSITY_PER_CAPITA"].max(), 1e-9))

    scatter_for_cbar = None
    handles = []

    for region, marker in region_markers.items():
        region_data = state_plot[state_plot["REGION_GROUP"] == region]
        if region_data.empty:
            continue

        scatter_for_cbar = ax3.scatter(
            region_data["POP_DENSITY"],
            region_data["TOTAL"],
            s=sizes.loc[region_data.index],
            c=region_data["COMP_DENSITY_PER_CAPITA"],
            cmap="viridis",
            norm=norm_c,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
            marker=marker,
        )
        handles.append(
            plt.Line2D(
                [0], [0], marker=marker, linestyle="",
                markerfacecolor="gray", markeredgecolor="black",
                markersize=11, label=region
            )
        )

    ax3.set_title("(c) State Components vs. Population Density", weight="bold")
    ax3.set_xlabel("Population Density (per Sq Km)")
    ax3.set_ylabel("Total Components")
    ax3.grid(linestyle=":", alpha=0.35)
    ax3.legend(
        handles=handles, title="Region", frameon=False, loc="upper left",
        fontsize=13, title_fontsize=14, labelspacing=0.35, handletextpad=0.45,
        borderaxespad=0.35
    )
    ax3.margins(x=0.04)
    ax3.set_ylim(0, state_plot["TOTAL"].max() * 1.14)

    if scatter_for_cbar is not None:
        cbar = fig.colorbar(
            scatter_for_cbar, ax=ax3, orientation="horizontal",
            pad=0.12, fraction=0.045, aspect=42
        )
        cbar.set_label("Components per 1M People", labelpad=6)
        cbar.ax.tick_params(labelsize=13, pad=2)

    # ---------- (d) NERC scatter ----------
    ax4 = fig.add_subplot(gs[1, 1])

    max_total = max(nerc_comp["TOTAL"].max(), 1)
    n_size = nerc_comp["TOTAL"] / max_total * 1250 + 35
    norm_d = Normalize(
        vmin=0,
        vmax=max(nerc_comp["COMP_DENSITY_PER_AREA_KM"].max(), 1e-9)
    )

    scatter_d = ax4.scatter(
        nerc_comp["NERC_POP_DENSITY"],
        nerc_comp["TOTAL"],
        s=n_size,
        c=nerc_comp["COMP_DENSITY_PER_AREA_KM"],
        cmap="viridis",
        norm=norm_d,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
    )

    for _, row in nerc_comp.iterrows():
        display_name = region_map.get(row["NERC"], row["NERC"])
        ax4.annotate(
            display_name,
            (row["NERC_POP_DENSITY"], row["TOTAL"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12.5,
            fontweight="bold",
        )

    ax4.set_title("(d) FERC Components vs. Population Density", weight="bold")
    ax4.set_xlabel("FERC Population Density (per Sq Km)")
    ax4.set_ylabel("Total Components")
    ax4.grid(linestyle=":", alpha=0.35)

    ax4.margins(x=0.05)
    ax4.set_ylim(0, nerc_comp["TOTAL"].max() * 1.14)

    cbar = fig.colorbar(
        scatter_d, ax=ax4, orientation="horizontal",
        pad=0.12, fraction=0.045, aspect=42
    )
    cbar.set_label("Components per Sq Km (FERC)", labelpad=6)
    cbar.ax.tick_params(labelsize=13, pad=2)

    output = OUTPUT_DIR / "bar_and_scatter_2x2.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


# ============================================================
# FIGURE 2 — CHOROPLETH MAPS
# ============================================================

SMALL_STATES = ["VT", "NH", "MA", "RI", "CT", "NJ", "DE", "MD", "DC"]
LABEL_BBOX = dict(
    boxstyle="round,pad=0.15",
    fc="white",
    ec="none",
    alpha=0.78,
)


def add_basemap_safely(ax):
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            alpha=0.55,
            attribution=False,
        )
    except Exception as exc:
        print(f"Basemap unavailable; continuing without it ({exc}).")


def one_label_per_group(gdf, group_col):
    # Use projected area for choosing the largest fragment.
    temp = gdf.to_crs(epsg=5070).copy()
    temp["_area"] = temp.geometry.area
    keep_idx = (
        temp.sort_values("_area", ascending=False)
        .drop_duplicates(group_col)
        .index
    )
    return gdf.loc[keep_idx].copy()


def draw_small_state_legend_on_axis(
    legend_ax, df, value_col, states, cmap, norm,
    value_fmt="{:,.0f}", fontsize=8, state_col="STATE"
):
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    present = [
        s for s in states
        if not df.loc[df[state_col] == s].empty
        and pd.notna(df.loc[df[state_col] == s, value_col].iloc[0])
    ]

    if not present:
        return

    row_h = 1.0 / len(present)
    box_size = row_h * 0.52

    for i, state in enumerate(present):
        val = df.loc[df[state_col] == state, value_col].iloc[0]
        y = 1 - (i + 0.5) * row_h

        rect = patches.Rectangle(
            (0.04, y - box_size / 2),
            box_size,
            box_size,
            facecolor=cmap(norm(val)),
            edgecolor="black",
            linewidth=0.5,
            transform=legend_ax.transAxes,
        )
        legend_ax.add_patch(rect)

        legend_ax.text(
            0.04 + box_size + 0.08,
            y,
            f"{state}\n{value_fmt.format(val)}",
            fontsize=fontsize,
            va="center",
            ha="left",
            transform=legend_ax.transAxes,
        )


def make_choropleth_figure(gdf_points, states_3857, nerc_3857):
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # Point data already assigned to state/NERC; aggregate components.
    point_3857 = gdf_points.to_crs(epsg=3857)

    state_values = (
        gdf_points.groupby("STATE")[COMPONENT_COLS]
        .sum()
        .sum(axis=1)
        .rename("TOTAL")
        .reset_index()
    )
    state_map = states_3857.merge(state_values, on="STATE", how="left")
    state_map["TOTAL"] = state_map["TOTAL"].fillna(0)

    state_equal = state_map.to_crs(epsg=5070)
    state_map["AREA_SQKM"] = state_equal.geometry.area.to_numpy() / 1_000_000
    state_map["COMP_DENSITY_PER_AREA_KM"] = (
        state_map["TOTAL"] / state_map["AREA_SQKM"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    nerc_values = (
        gdf_points.groupby("NERC")[COMPONENT_COLS]
        .sum()
        .sum(axis=1)
        .rename("TOTAL")
        .reset_index()
    )
    nerc_map = nerc_3857.merge(nerc_values, on="NERC", how="left")
    nerc_map["TOTAL"] = nerc_map["TOTAL"].fillna(0)

    nerc_dissolved = nerc_3857[["NERC", "geometry"]].dissolve(by="NERC").reset_index()
    nerc_area = nerc_dissolved.to_crs(epsg=5070)
    area_lookup = pd.Series(
        nerc_area.geometry.area.to_numpy() / 1_000_000,
        index=nerc_area["NERC"],
    )
    nerc_map["AREA_SQKM"] = nerc_map["NERC"].map(area_lookup)
    nerc_map["COMP_DENSITY_PER_AREA_KM"] = (
        nerc_map["TOTAL"] / nerc_map["AREA_SQKM"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1, 1, 0.22],
        height_ratios=[1, 1],
        wspace=0.04,
        hspace=0.22,
    )

    cmap = plt.get_cmap("viridis")

    # ---------- (a) FERC totals ----------
    ax = fig.add_subplot(gs[0, 0])
    norm = Normalize(vmin=0, vmax=max(nerc_map["TOTAL"].max(), 1))
    nerc_map.plot(
        column="TOTAL", cmap=cmap, norm=norm,
        linewidth=0.7, edgecolor="black", ax=ax
    )
    add_basemap_safely(ax)
    ax.set_title("(a) Total Component Counts per FERC Region", weight="bold")
    ax.axis("off")

    labels = one_label_per_group(nerc_map, "NERC")
    for _, row in labels.iterrows():
        if row["TOTAL"] <= 0:
            continue
        p = row.geometry.representative_point()
        ax.text(
            p.x, p.y, f"{row['NERC']}\n{int(row['TOTAL']):,}",
            ha="center", va="center", fontsize=7.5, bbox=LABEL_BBOX
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.025)
    cbar.set_label("Total Component Count")

    # ---------- (b) State totals ----------
    ax = fig.add_subplot(gs[0, 1])
    legend_ax = fig.add_subplot(gs[0, 2])
    norm = Normalize(vmin=0, vmax=max(state_map["TOTAL"].max(), 1))

    state_map.plot(
        column="TOTAL", cmap=cmap, norm=norm,
        linewidth=0.7, edgecolor="black", ax=ax
    )
    add_basemap_safely(ax)
    ax.set_title("(b) Total Component Counts per State", weight="bold")
    ax.axis("off")

    for _, row in state_map.iterrows():
        if row["STATE"] in SMALL_STATES or row["TOTAL"] <= 0:
            continue
        p = row.geometry.representative_point()
        ax.text(
            p.x, p.y, f"{int(row['TOTAL']):,}",
            ha="center", va="center", fontsize=8, bbox=LABEL_BBOX
        )

    draw_small_state_legend_on_axis(
        legend_ax, state_map, "TOTAL", SMALL_STATES,
        cmap, norm, value_fmt="{:,.0f}"
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.04)
    cbar.set_label("Total Component Count")

    # ---------- (c) FERC density ----------
    ax = fig.add_subplot(gs[1, 0])
    norm = Normalize(
        vmin=0,
        vmax=max(nerc_map["COMP_DENSITY_PER_AREA_KM"].max(), 1e-9)
    )

    nerc_map.plot(
        column="COMP_DENSITY_PER_AREA_KM", cmap=cmap, norm=norm,
        linewidth=0.7, edgecolor="black", ax=ax
    )
    add_basemap_safely(ax)
    ax.set_title("(c) FERC Component Density (per Sq Km)", weight="bold")
    ax.axis("off")

    labels = one_label_per_group(nerc_map, "NERC")
    for _, row in labels.iterrows():
        val = row["COMP_DENSITY_PER_AREA_KM"]
        if val <= 0:
            continue
        p = row.geometry.representative_point()
        ax.text(
            p.x, p.y, f"{row['NERC']}\n{val:.3f}",
            ha="center", va="center", fontsize=7.5, bbox=LABEL_BBOX
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.025)
    cbar.set_label("Components per Sq Km")

    # ---------- (d) State density ----------
    ax = fig.add_subplot(gs[1, 1])
    legend_ax = fig.add_subplot(gs[1, 2])

    density_max = max(state_map["COMP_DENSITY_PER_AREA_KM"].max(), 1e-9)
    norm = Normalize(vmin=0, vmax=density_max)

    state_map.plot(
        column="COMP_DENSITY_PER_AREA_KM", cmap=cmap, norm=norm,
        linewidth=0.7, edgecolor="black", ax=ax
    )
    add_basemap_safely(ax)
    ax.set_title("(d) State Component Density (per Sq Km)", weight="bold")
    ax.axis("off")

    for _, row in state_map.iterrows():
        val = row["COMP_DENSITY_PER_AREA_KM"]
        if row["STATE"] in SMALL_STATES or val <= 0:
            continue
        p = row.geometry.representative_point()
        ax.text(
            p.x, p.y, f"{val:.3f}",
            ha="center", va="center", fontsize=8, bbox=LABEL_BBOX
        )

    draw_small_state_legend_on_axis(
        legend_ax, state_map, "COMP_DENSITY_PER_AREA_KM", SMALL_STATES,
        cmap, norm, value_fmt="{:.3f}"
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.04)
    cbar.set_label("Components per Sq Km")

    output = OUTPUT_DIR / "choropleth_maps_2x2.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


# ============================================================
# MAIN
# ============================================================

def main():
    (
        meta,
        gdf_points,
        points_3857,
        states,
        states_3857,
        nerc_shapes,
        nerc_3857,
    ) = load_and_prepare_data()

    state_comp, nerc_comp, _ = aggregate_data(gdf_points, states, nerc_shapes)

    make_bar_scatter_figure(state_comp, nerc_comp)
    make_choropleth_figure(gdf_points, states_3857, nerc_3857)

    initial_total = len(meta)
    assigned_total = len(gdf_points)
    not_assigned = initial_total - assigned_total

    print("\n" + "=" * 56)
    print("REPORT")
    print("=" * 56)
    print(f"Initial substation records: {initial_total:,}")
    print(f"Assigned to contiguous-US state + FERC region: {assigned_total:,}")
    print(f"Not assigned: {not_assigned:,}")

    if initial_total:
        print(f"Unassigned share: {not_assigned / initial_total:.2%}")

    print(f"\nFigures written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
