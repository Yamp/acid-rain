#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.10,<3.11", "numpy>=2,<3"]
# ///
"""Plot saved shallow-water diagnostics without building or running the solver."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS = ROOT / "benchmarks/rapids-after/diagnostics"


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Empty diagnostic capture: {path}")
    nx = max(int(row["x"]) for row in rows) + 1
    ny = max(int(row["y"]) for row in rows) + 1
    if len(rows) != nx * ny:
        raise ValueError("Capture must contain every grid cell exactly once")
    fields = {key: np.full((ny, nx), np.nan) for key in ("elevation", "depth", "foam", "u", "v")}
    seen = set()
    for row in rows:
        cell = (int(row["x"]), int(row["y"]))
        if cell in seen or min(cell) < 0:
            raise ValueError(f"Invalid or duplicate cell: {cell}")
        seen.add(cell)
        for key, field in fields.items():
            field[cell[1], cell[0]] = float(row[key])
    if not all(np.isfinite(field).all() for field in fields.values()):
        raise ValueError("Capture contains a missing or nonfinite value")
    if (fields["depth"] <= 0).any() or ((fields["foam"] < 0) | (fields["foam"] > 1)).any():
        raise ValueError("Depth must be positive and foam coverage must lie in [0, 1]")
    return np.linspace(0, 2, nx + 1), np.linspace(0, 2, ny + 1), fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DIAGNOSTICS / "rapids-state-30.csv")
    parser.add_argument("--output", type=Path, default=DIAGNOSTICS / "flow.png")
    parser.add_argument("--seconds", type=float, default=30, help="Physical time represented by the input capture")
    args = parser.parse_args()
    x_edges, y_edges, fields = load_grid(args.input)
    x, y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)
    reverse = fields["u"] < 0
    reverse_fraction = reverse.mean()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14, "figure.facecolor": "#f7f8fa", "axes.facecolor": "#f7f8fa"})
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    figure.subplots_adjust(left=0.07, right=0.90, top=0.87, bottom=0.13, wspace=0.26, hspace=0.37)
    figure.suptitle(f"River rapids · {args.seconds:g} s of physical time", x=0.07, y=0.965, ha="left", fontsize=21, fontweight="bold")
    figure.text(0.07, 0.927, f"2 m × 2 m shallow-water domain · {fields['depth'].shape[1]} × {fields['depth'].shape[0]} cell-centre samples", color="#485363", fontsize=12)
    surface_palette = LinearSegmentedColormap.from_list("water_surface", ["#33475c", "#6f83ad", "#bdcfd7", "#f1dfc9"])
    panels = [
        (fields["elevation"] * 100, "Surface elevation", "cm relative to the 0.10 m datum", surface_palette, None, None),
        (fields["depth"] * 100, "Water depth above the bed", "cm", "cividis", None, None),
        (fields["foam"], "Advected foam coverage", "dimensionless coverage", "Purples", 0, 1),
        (reverse.astype(float), f"Surface return flow\n{reverse_fraction:.2%} of sampled area", "surface streamwise velocity", ListedColormap(["#e2e8ec", "#c884aa"]), 0, 1),
    ]
    for ax, (data, title, label, palette, lower, upper) in zip(axes.flat, panels):
        plot = ax.pcolormesh(x_edges, y_edges, data, cmap=palette, vmin=lower, vmax=upper, shading="flat", rasterized=True)
        ax.set(title=title, xlabel="Downstream x [m]", ylabel="Cross-channel y [m]", aspect="equal", xlim=(0, 2), ylim=(0, 2))
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.tick_params(length=3, colors="#394655")
        for spine in ax.spines.values():
            spine.set_color("#b1bbc5")
        colorbar = figure.colorbar(plot, ax=ax, fraction=0.042, pad=0.035)
        colorbar.set_label(label)
        if ax is axes[1, 1]:
            colorbar.set_ticks([0.25, 0.75], labels=["u ≥ 0", "u < 0"])
    step = (slice(2, None, 5), slice(2, None, 5))
    arrows = axes[1, 1].quiver(x[step], y[step], fields["u"][step], fields["v"][step], color="#263545", angles="xy", scale_units="xy", scale=3.5, width=0.0045, headwidth=3.5)
    axes[1, 1].quiverkey(arrows, 0.77, 0.928, 0.5, "0.5 m/s", coordinates="figure", labelpos="E")
    rows, columns = np.indices(reverse.shape)
    returning = reverse & ((rows + columns) % 3 == 0)
    axes[1, 1].quiver(x[returning], y[returning], fields["u"][returning], fields["v"][returning], color="#6b214d", angles="xy", scale_units="xy", scale=3.5, width=0.0045, headwidth=3.5)
    figure.text(0.07, 0.060, "Arrows and return-flow mask: surface tracer velocity, including the empirical hydraulic-roller return closure.", fontsize=10.5, color="#394655")
    figure.text(0.07, 0.035, "Surface/depth: conservative depth-averaged solver. Foam and surface roller return are empirical; 3-D overturning is not resolved.", fontsize=10.5, color="#394655")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved {args.output}")
    print(f"Depth: {fields['depth'].min():.5f}–{fields['depth'].max():.5f} m; reverse surface area: {reverse_fraction:.4%}; mean foam: {fields['foam'].mean():.4f}")


if __name__ == "__main__":
    main()
