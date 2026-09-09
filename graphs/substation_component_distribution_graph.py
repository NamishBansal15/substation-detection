"""Substation-level component distributions overall and by planning region.

The figure uses model-detected counts for the four retained downstream classes:
Transformer, Reactor, Circuit Breaker, and Alt Energy. ``Overall`` is the
row-wise sum of those four classes. Control and Power Lines are intentionally
excluded to match the manuscript's downstream geographic analysis.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

COMPONENTS = ["Transformer", "Reactor", "Circuit Breaker", "Alt Energy"]
SERIES = ["Overall", *COMPONENTS]
COLORS = {
    "Overall": "#6b6b6b",
    "Transformer": "#4e79a7",
    "Reactor": "#f28e2b",
    "Circuit Breaker": "#e15759",
    "Alt Energy": "#76b7b2",
}
REGION_LABELS = {
    "NorthernGridConnected": "NorthernGrid Connected",
    "NorthernGridUnconnected": "NorthernGrid Unconnected",
    "WestConnectNonEnrolled": "WestConnect Non-Enrolled",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_data() -> tuple[pd.DataFrame, gpd.GeoDataFrame, list[str]]:
    pred = pd.read_csv(DATA_DIR / "component_predictions.csv")
    meta = pd.read_csv(DATA_DIR / "image_metadata.csv")
    regions = gpd.read_file(DATA_DIR / "nerc_gdf.geojson")

    for col in COMPONENTS:
        pred[col] = pd.to_numeric(pred[col], errors="coerce").fillna(0)
    pred["Overall"] = pred[COMPONENTS].sum(axis=1)

    points = gpd.GeoDataFrame(
        meta[["id", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(meta.longitude, meta.latitude),
        crs="EPSG:4326",
    )
    regions = regions.to_crs("EPSG:4326") if regions.crs else regions.set_crs("EPSG:4326")
    joined = gpd.sjoin(points, regions[["REGIONS", "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore").merge(
        pred[["id", *SERIES]], on="id", how="left"
    )

    available = set(joined["REGIONS"].dropna().astype(str))
    region_order: list[str] = []
    for region in regions["REGIONS"].dropna().astype(str):
        if region in available and region not in region_order:
            region_order.append(region)
    return pred, joined, region_order


def _breakmarks(left, right) -> None:
    d = .006
    kw = dict(color="#9a9a9a", clip_on=False, linewidth=.65, solid_capstyle="round")
    left.plot((1-d, 1+d), (-d, +d), transform=left.transAxes, **kw)
    right.plot((-d, +d), (-d, +d), transform=right.transAxes, **kw)


def _style_broken_axis(left, right) -> None:
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", left=False, labelleft=False)
    for ax in (left, right):
        ax.grid(axis="x", linestyle="--", linewidth=.65, alpha=.16)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_linewidth(.8)
    left.spines["left"].set_color("#b0b0b0")
    left.spines["left"].set_linewidth(.7)
    _breakmarks(left, right)


def _recolor(bp, names: list[str], alpha: float = .52) -> None:
    for i, name in enumerate(names):
        color = COLORS[name]
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(alpha)
        bp["boxes"][i].set_edgecolor(color)
        bp["medians"][i].set_color(color)
        for whisker in bp["whiskers"][2*i:2*i+2]:
            whisker.set_color(color)
        for cap in bp["caps"][2*i:2*i+2]:
            cap.set_color(color)


def write_summary(pred: pd.DataFrame, joined: gpd.GeoDataFrame, region_order: list[str]) -> None:
    rows = []
    for name in SERIES:
        s = pred[name]
        rows.append({
            "scope": "All images", "series": name, "n": len(s),
            "mean": s.mean(), "median": s.median(),
            "q1": s.quantile(.25), "q3": s.quantile(.75), "max": s.max(),
            "zero_fraction": (s == 0).mean(),
        })
    for region in region_order:
        s = joined.loc[joined["REGIONS"].astype(str) == region, "Overall"].dropna()
        rows.append({
            "scope": region, "series": "Overall", "n": len(s),
            "mean": s.mean(), "median": s.median(),
            "q1": s.quantile(.25), "q3": s.quantile(.75), "max": s.max(),
            "zero_fraction": (s == 0).mean(),
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "substation_component_distribution_summary.csv", index=False)


def main() -> None:
    pred, joined, region_order = load_data()
    series_means = {c: pred[c].mean() for c in SERIES}
    regional_means = {
        r: joined.loc[joined["REGIONS"].astype(str) == r, "Overall"].mean()
        for r in region_order
    }
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(13.6, 14.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.8], hspace=.30)

    # (a) All-image distributions.
    gsa = outer[0].subgridspec(1, 2, width_ratios=[6, 1], wspace=.025)
    a1 = fig.add_subplot(gsa[0, 0])
    a2 = fig.add_subplot(gsa[0, 1], sharey=a1)
    positions = np.arange(len(SERIES), 0, -1)
    values = [pred[c].to_numpy() for c in SERIES]

    for ax in (a1, a2):
        bp = ax.boxplot(
            values, positions=positions, vert=False, widths=.42, patch_artist=True,
            showfliers=False, whis=1.5, capwidths=.25,
            medianprops={"linewidth": 2.5}, boxprops={"linewidth": 1.9},
            whiskerprops={"linewidth": 1.7}, capprops={"linewidth": 1.7},
        )
        _recolor(bp, SERIES)
        for y, name in zip(positions, SERIES):
            arr = pred[name].to_numpy()
            ax.scatter(arr, y + rng.uniform(-.12, .12, len(arr)), s=4.7, alpha=.14,
                       color=COLORS[name], edgecolors="none", rasterized=True, zorder=1)

    for y, name in zip(positions, SERIES):
        mean = series_means[name]
        target = a1 if mean <= 10 else a2
        target.scatter([mean], [y], marker="D", s=44, facecolor="white",
                       edgecolor=COLORS[name], linewidth=1.5, zorder=8)

    a1.set_xlim(-.35, 10); a2.set_xlim(10, 30)
    a1.set_xticks([0, 2, 4, 6, 8, 10]); a2.set_xticks([15, 20, 25, 30])
    a1.set_yticks(positions)
    a1.set_yticklabels([f"{c}\nmean = {series_means[c]:.2f}" for c in SERIES], linespacing=1.05)
    a1.get_yticklabels()[0].set_fontweight("bold")
    a1.set_title("(a) Overall substation-level distributions", loc="left", fontweight="bold")
    a1.set_xlabel("Detected components per substation image")
    _style_broken_axis(a1, a2)

    handles = [Patch(facecolor=COLORS[c], edgecolor=COLORS[c], alpha=.55, label=c) for c in SERIES]
    a1.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.12), ncol=5,
              frameon=False, columnspacing=1.45, handlelength=1.3, borderaxespad=0)

    # (b) Planning-region distributions.
    gsb = outer[1].subgridspec(1, 2, width_ratios=[6, 1], wspace=.025)
    b1 = fig.add_subplot(gsb[0, 0])
    b2 = fig.add_subplot(gsb[0, 1], sharey=b1)
    base = np.arange(len(region_order), 0, -1) * 1.38
    offsets = {"Overall": .46, "Transformer": .23, "Reactor": 0.0,
               "Circuit Breaker": -.23, "Alt Energy": -.46}

    for ax in (b1, b2):
        for name in SERIES:
            arrays = [
                joined.loc[joined["REGIONS"].astype(str) == r, name].dropna().to_numpy()
                for r in region_order
            ]
            pp = base + offsets[name]
            bp = ax.boxplot(
                arrays, positions=pp, vert=False, widths=.155, patch_artist=True,
                showfliers=False, whis=1.5, capwidths=.10, manage_ticks=False,
                medianprops={"linewidth": 1.9}, boxprops={"linewidth": 1.45},
                whiskerprops={"linewidth": 1.3}, capprops={"linewidth": 1.3},
            )
            _recolor(bp, [name] * len(arrays), alpha=.50)
            for y, arr in zip(pp, arrays):
                if len(arr):
                    ax.scatter(arr, y + rng.uniform(-.052, .052, len(arr)), s=3.3, alpha=.105,
                               color=COLORS[name], edgecolors="none", rasterized=True, zorder=1)

    for region, y in zip(region_order, base + offsets["Overall"]):
        mean = regional_means[region]
        target = b1 if mean <= 10 else b2
        target.scatter([mean], [y], marker="D", s=28, facecolor="white",
                       edgecolor=COLORS["Overall"], linewidth=1.25, zorder=8)

    b1.set_xlim(-.35, 10); b2.set_xlim(10, 30)
    b1.set_xticks([0, 2, 4, 6, 8, 10]); b2.set_xticks([15, 20, 25, 30])
    b1.set_yticks(base)
    b1.set_yticklabels([
        f"{REGION_LABELS.get(r, r)}\noverall mean = {regional_means[r]:.2f}" for r in region_order
    ], linespacing=1.05)
    b1.set_ylim(base[-1] - .85, base[0] + .85); b2.set_ylim(b1.get_ylim())
    b1.set_xlabel("Detected components per substation image")
    b1.set_title("(b) Distributions by FERC planning region", loc="left", fontweight="bold")
    _style_broken_axis(b1, b2)

    fig.subplots_adjust(left=.275, right=.985, top=.94, bottom=.055)
    output = RESULTS_DIR / "substation_component_distributions.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_summary(pred, joined, region_order)
    print(f"Wrote {output}")
    print(f"Wrote {RESULTS_DIR / 'substation_component_distribution_summary.csv'}")


if __name__ == "__main__":
    main()
