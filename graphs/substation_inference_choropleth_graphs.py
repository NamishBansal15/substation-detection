"""Create a clean four-panel component-count choropleth from the supplied data."""

from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace

_CACHE = Path(__file__).resolve().parent / "map_cache"
_CACHE.mkdir(exist_ok=True)
os.environ["TEMP"] = str(_CACHE)
os.environ["TMP"] = str(_CACHE)
# contextily 1.x creates a short-lived TemporaryDirectory during import; on
# Windows that directory can be cleaned up before joblib opens it. Keep a
# project-local cache alive for the duration of this script instead.
tempfile.TemporaryDirectory = lambda *args, **kwargs: SimpleNamespace(name=str(_CACHE))
tempfile.mkdtemp = lambda *args, **kwargs: str(_CACHE)

import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.sampledata.us_states import data as state_data
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import MultiPolygon, Point, Polygon

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

ROOT = Path(__file__).resolve().parent
COMPONENTS = ["Transformer", "Reactor", "Circuit Breaker", "Alt Energy", "Control", "Power Lines"]
SMALL = ["VT", "NH", "MA", "RI", "CT", "NJ", "DE", "MD", "DC"]


def polygons_from_bokeh(item):
    """Turn NaN-separated Bokeh state coordinates into a valid (Multi)Polygon."""
    lon, lat = np.asarray(item["lons"]), np.asarray(item["lats"])
    breaks = np.flatnonzero(np.isnan(lon) | np.isnan(lat))
    parts, start = [], 0
    for stop in np.r_[breaks, len(lon)]:
        xy = np.column_stack((lon[start:stop], lat[start:stop]))
        if len(xy) >= 3:
            poly = Polygon(xy).buffer(0)
            if not poly.is_empty:
                parts.extend(list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly])
        start = stop + 1
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def load_states():
    rows = [{"STATE": code, "name": item["name"], "geometry": polygons_from_bokeh(item)}
            for code, item in state_data.items() if code not in {"AK", "HI"}]
    states = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    totals = pd.read_csv(ROOT / "results" / "state_totals.csv")
    totals["TOTAL"] = totals[COMPONENTS].sum(axis=1)
    states = states.merge(totals[["STATE", "TOTAL"]], on="STATE", how="left").fillna({"TOTAL": 0})
    area = states.to_crs("EPSG:5070").area / 1e6
    states["DENSITY"] = (states["TOTAL"] / area) * 100
    return states


def load_regions():
    regions = gpd.read_file(ROOT / "data" / "nerc_gdf.geojson")
    meta = pd.read_csv(ROOT / "data" / "image_metadata.csv")
    pred = pd.read_csv(ROOT / "data" / "component_predictions.csv")
    pred["TOTAL"] = pred[COMPONENTS].fillna(0).sum(axis=1)
    points = meta.merge(pred[["id", "TOTAL"]], on="id", how="inner")
    points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.longitude, points.latitude), crs=4326)
    joined = gpd.sjoin(points, regions[["REGIONS", "geometry"]], predicate="within", how="inner")
    values = joined.groupby("REGIONS", as_index=False)["TOTAL"].sum()
    regions = regions.merge(values, on="REGIONS", how="left").fillna({"TOTAL": 0})
    # A planning region can consist of several polygon rows.  Use its dissolved
    # land area so every fragment receives one consistent region-level density.
    dissolved = regions[["REGIONS", "geometry"]].dissolve(by="REGIONS").to_crs("EPSG:5070")
    area_km2 = (dissolved.area / 1e6).to_dict()
    regions["DENSITY"] = regions.apply(lambda r: (r.TOTAL / area_km2[r.REGIONS]) * 100, axis=1)
    return regions

COUNT_CMAP = plt.colormaps["viridis_r"]
DENSITY_CMAP = plt.colormaps["viridis_r"]


def value_text(value, density):
    return f"{value:.3f}" if density else f"{value:,.0f}"


def draw_states(ax, states, column, title, cmap, density=False, vertical_legend=False):
    vmax = 1.0 if density else states[column].max()
    norm = Normalize(0, vmax)
    prepare_basemap(ax, xmax=-61.5)
    states.plot(ax=ax, column=column, cmap=cmap, norm=norm, edgecolor="white", linewidth=.85,
                alpha=.88, zorder=2)
    states.boundary.plot(ax=ax, color="#45515a", linewidth=.35)
    ax.set_xlim(-125.5, -61.5); ax.set_ylim(24, 50.3)
    for _, row in states.iterrows():
        if row.STATE in SMALL or row.TOTAL <= 0:
            continue
        p = row.geometry.representative_point()
        label = f"{row.STATE}\n{value_text(row[column], density)}"
        ax.text(p.x, p.y, label, ha="center", va="center", fontsize=10.5, color="#17232b",
                linespacing=.9, path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white", alpha=.9)])
    # Fixed, evenly spaced callout rail inspired by the supplied example.
    y_positions = np.linspace(46.5, 31.7, len(SMALL))
    lookup = states.set_index("STATE")
    for code, y in zip(SMALL, y_positions):
        if code not in lookup.index:
            continue
        row = lookup.loc[code]; p = row.geometry.representative_point()
        ax.annotate(f"{code}  {value_text(row[column], density)}", xy=(p.x, p.y), xytext=(-68.1, y),
                    ha="left", va="center", fontsize=10.5, color="#17232b",
                    arrowprops=dict(arrowstyle="-", color="#52606a", lw=.65,
                                    connectionstyle="arc3,rad=0"))
    add_colorbar(ax, cmap, norm, "Components per 100 km²" if density else "Total components", density,
                 vertical_legend)
    style_axis(ax, title)


REGION_OFFSETS = {
    "NorthernGridConnected": (-2.0, 1.0), "NorthernGridUnconnected": (-1.5, 1.3),
    "WestConnectNonEnrolled": (1.0, 1.4), "WestConnect": (-1.0, -1.2),
    "MISO": (1.0, .8), "PJM": (1.3, .3), "SERTP": (.7, -.7), "SCRTP": (1.0, -.7),
}


def draw_regions(ax, regions, column, title, cmap, density=False, vertical_legend=False):
    vmax = regions[column].quantile(.98) if density else regions[column].max()
    norm = Normalize(0, vmax)
    prepare_basemap(ax, xmax=-61.5)
    regions.plot(ax=ax, column=column, cmap=cmap, norm=norm, edgecolor="#3d4b53", linewidth=.8,
                 alpha=.88, zorder=2)
    groups = []
    for name, group in regions.groupby("REGIONS"):
        geom = group.geometry.union_all(); p = geom.representative_point()
        dx, dy = REGION_OFFSETS.get(name, (0, 0)); value = group[column].iloc[0]
        groups.append((name, p.x, p.y, p.x + dx, p.y + dy, value))
    positions = dodge_region_labels(groups)
    for name, x, y, _, _, value in groups:
        tx, ty = positions[name]
        ax.annotate(f"{name}\n{value_text(value, density)}", xy=(x, y), xytext=(tx, ty),
                    ha="center", va="center", fontsize=10, color="#17232b",
                    arrowprops=dict(arrowstyle="-", color="#52606a", lw=.65), zorder=4,
                    path_effects=[patheffects.withStroke(linewidth=2.8, foreground="white", alpha=.9)])
    ax.set_xlim(-125.5, -61.5); ax.set_ylim(24, 50.3)
    add_colorbar(ax, cmap, norm, "Components per 100 km²" if density else "Total components", density,
                 vertical_legend)
    style_axis(ax, title)


def add_colorbar(ax, cmap, norm, label, density, vertical=False):
    # Keep the legend inside the panel's quiet southern margin. This avoids
    # Matplotlib reserving a separate vertical band between subplot rows.
    if vertical:
        cax = ax.inset_axes([1.045, .17, .022, .66], zorder=6)
        orientation = "vertical"
    else:
        cax = ax.inset_axes([.17, .075, .66, .035], zorder=6)
        orientation = "horizontal"
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation=orientation)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=13, length=4, colors="#33434d")
    cb.set_label(label, fontsize=16, color="#24343e", labelpad=6)
    if density and not vertical:
        cb.ax.xaxis.set_major_formatter(lambda x, _: f"{x:.3f}")
    elif density:
        cb.ax.yaxis.set_major_formatter(lambda x, _: f"{x:.3f}")


def style_axis(ax, title):
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color="#17232b", pad=10)
    ax.set_axis_off(); ax.set_facecolor("#f7f8f6")


def prepare_basemap(ax, xmax=-65):
    """Add quiet CARTO context while preserving lon/lat plotting coordinates."""
    ax.set_xlim(-125.5, xmax); ax.set_ylim(24, 50.3)
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldGrayCanvas,
                        zoom=4, attribution=False, alpha=.72, zorder=0)
    except Exception as exc:
        # The figure remains reproducible offline; the soft geographic canvas is
        # the fallback when a tile server is unavailable.
        ax.set_facecolor("#dfe8eb")
        print(f"Basemap unavailable ({exc}); using offline canvas")


def dodge_region_labels(groups):
    """Repel planning-region labels in normalized map space, retaining leaders."""
    xmin, xmax, ymin, ymax = -125.5, -65, 24, 50.3
    pos = {name: np.array([(tx-xmin)/(xmax-xmin), (ty-ymin)/(ymax-ymin)], float)
           for name, _, _, tx, ty, _ in groups}
    sizes = {name: np.array([max(.095, .0105 * len(name)), .082]) for name, *_ in groups}
    names = list(pos)
    for _ in range(350):
        moved = False
        for i, a in enumerate(names):
            for b in names[i+1:]:
                delta = pos[b] - pos[a]
                overlap = (sizes[a] + sizes[b]) / 2 - np.abs(delta)
                if overlap[0] > 0 and overlap[1] > 0:
                    # Prefer vertical separation, with a small horizontal nudge.
                    sy = 1 if delta[1] >= 0 else -1
                    sx = 1 if delta[0] >= 0 else -1
                    push = overlap[1] / 2 + .002
                    pos[a][1] -= sy * push; pos[b][1] += sy * push
                    pos[a][0] -= sx * .0015; pos[b][0] += sx * .0015
                    moved = True
        for name in names:
            half = sizes[name] / 2
            pos[name] = np.clip(pos[name], half + [.005, .015], 1 - half - [.005, .015])
        if not moved:
            break
    return {name: (p[0]*(xmax-xmin)+xmin, p[1]*(ymax-ymin)+ymin) for name, p in pos.items()}


def main():
    states, regions = load_states(), load_regions()
    fig, axes = plt.subplots(2, 2, figsize=(20.5, 10.2), facecolor="#f4f3ef",
                             gridspec_kw={"width_ratios": [1, 1.13]})
    draw_regions(axes[0, 0], regions, "TOTAL", "A  ·  Total components by planning region", COUNT_CMAP)
    draw_states(axes[0, 1], states, "TOTAL", "B  ·  Total components by state", COUNT_CMAP)
    draw_regions(axes[1, 0], regions, "DENSITY", "C  ·  Components per 100 km² by planning region", DENSITY_CMAP, True)
    draw_states(axes[1, 1], states, "DENSITY", "D  ·  Components per 100 km² by state", DENSITY_CMAP, True)
    fig.suptitle("Grid component distribution across the contiguous United States", x=.015, ha="left",
                 fontsize=19, fontweight="bold", color="#14242e", y=.995)
    fig.text(.995, .004, "Basemap © Esri, DeLorme, NAVTEQ", ha="right", fontsize=5.5, color="#7b858a")
    # Compact outer margins while preserving just enough row space for colorbars.
    fig.subplots_adjust(left=.012, right=.995, top=.915, bottom=.018, wspace=.018, hspace=.075)
    out = ROOT / "current_image.png"
    fig.savefig(out, dpi=240, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def vertical_main():
    """Render the same four maps as a single column with four rows."""
    states, regions = load_states(), load_regions()
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 19.5), facecolor="#f4f3ef")
    draw_regions(axes[0], regions, "TOTAL", "A  ·  Total components by planning region", COUNT_CMAP,
                 vertical_legend=True)
    draw_states(axes[1], states, "TOTAL", "B  ·  Total components by state", COUNT_CMAP,
                vertical_legend=True)
    draw_regions(axes[2], regions, "DENSITY", "C  ·  Components per 100 km² by planning region", DENSITY_CMAP,
                 True, vertical_legend=True)
    draw_states(axes[3], states, "DENSITY", "D  ·  Components per 100 km² by state", DENSITY_CMAP,
                True, vertical_legend=True)
    fig.suptitle("Grid component distribution across the contiguous United States", x=.02, ha="left",
                 fontsize=23, fontweight="bold", color="#14242e", y=.997)
    fig.text(.995, .003, "Basemap © Esri, DeLorme, NAVTEQ", ha="right", fontsize=6.5, color="#7b858a")
    # Reserve a clean right-hand rail exclusively for the vertical legends.
    fig.subplots_adjust(left=.012, right=.90, top=.955, bottom=.012, hspace=.085)
    out = ROOT / "current_image_vertical.png"
    fig.savefig(out, dpi=240, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    vertical_main() if "--vertical" in sys.argv else main()
