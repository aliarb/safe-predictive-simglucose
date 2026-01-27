#!/usr/bin/env python3
"""
Create a "Fig.3-like" results figure comparing Safe-NMPC vs Event-Triggered Safe-NMPC.

Outputs an EPS (and PDF) that is suitable to drop into the paper figs/ folder.

Defaults are set for a ~30-hour window (0h to 30h), but you can also use it for a
6-hour dinner window (e.g. 16h to 22h).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_trace(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time")
    else:
        # fallback for index-as-time CSVs
        df.index = pd.to_datetime(df.index)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", default="results/paper_comparison/Safe_NMPC_detailed.csv", help="Safe-NMPC detailed CSV")
    ap.add_argument(
        "--evt",
        default="results/paper_comparison/Safe_NMPC_Event_Triggered_detailed.csv",
        help="Event-triggered detailed CSV",
    )
    ap.add_argument(
        "--out",
        default="Paper/648fc7ad4f4fdf8e359ed7e2/figs/fig_30hours_event_triggered.eps",
        help="Output EPS path (PDF will also be written next to it).",
    )
    ap.add_argument("--t0", type=float, default=0.0, help="Start hour (relative to trace start).")
    ap.add_argument("--t1", type=float, default=30.0, help="End hour (relative to trace start).")
    ap.add_argument(
        "--meal-hours",
        type=str,
        default="7,12,18,31",
        help="Comma-separated meal marker hours (relative to trace start), e.g. '7,12,18,31'.",
    )
    args = ap.parse_args()

    safe_path = Path(args.safe)
    evt_path = Path(args.evt)
    out_eps = Path(args.out)
    out_pdf = out_eps.with_suffix(".pdf")
    out_eps.parent.mkdir(parents=True, exist_ok=True)

    safe = _load_trace(safe_path)
    evt = _load_trace(evt_path)

    t0 = min(safe.index.min(), evt.index.min())
    safe_h = (safe.index - t0).total_seconds() / 3600.0
    evt_h = (evt.index - t0).total_seconds() / 3600.0

    # Window mask
    safe_m = (safe_h >= args.t0) & (safe_h <= args.t1)
    evt_m = (evt_h >= args.t0) & (evt_h <= args.t1)

    # Style (paper-friendly)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    # Colors consistent with compare_nmpc_for_paper
    c_safe = "#57068C"
    c_evt = "#FF7F0E"

    fig, (ax_bg, ax_u) = plt.subplots(2, 1, figsize=(6.8, 3.0), sharex=True, gridspec_kw={"hspace": 0.15})

    # --- BG subplot ---
    ax_bg.plot(safe_h[safe_m], safe.loc[safe_m, "BG"], color=c_safe, label="Safe-NMPC")
    ax_bg.plot(evt_h[evt_m], evt.loc[evt_m, "BG"], color=c_evt, label="Safe-NMPC (Event-Triggered)")

    # BG zones (paper-style background shading)
    # Keep shading but avoid legend clutter.
    ax_bg.axhspan(0, 70, alpha=0.10, color="#d62728", label="_nolegend_")     # hypo risk
    ax_bg.axhspan(70, 180, alpha=0.08, color="#2ca02c", label="_nolegend_")  # target range
    ax_bg.axhspan(180, 1000, alpha=0.10, color="#d62728", label="_nolegend_")  # hyper risk

    # Safety bounds and target
    ax_bg.axhline(70, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_bg.axhline(180, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_bg.axhline(140, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_bg.set_ylabel("BG (mg/dL)")
    ax_bg.set_ylim(50, 220)
    ax_bg.grid(True, alpha=0.25, linestyle="--")

    # Meal markers
    meal_hours = []
    try:
        meal_hours = [float(x.strip()) for x in str(args.meal_hours).split(",") if x.strip()]
    except Exception:
        meal_hours = []
    for mh in meal_hours:
        if mh < float(args.t0) or mh > float(args.t1):
            continue
        ax_bg.axvline(mh, color="red", linestyle="--", linewidth=1.0, alpha=0.6)
        ax_bg.text(mh, 215, "Meal", ha="center", va="top", fontsize=8, color="red")

    ax_bg.legend(loc="upper left", frameon=True)

    # --- Insulin subplot ---
    if "insulin" in safe.columns:
        ax_u.plot(safe_h[safe_m], safe.loc[safe_m, "insulin"], color=c_safe, label="Insulin (Safe-NMPC)")
    if "insulin" in evt.columns:
        ax_u.plot(evt_h[evt_m], evt.loc[evt_m, "insulin"], color=c_evt, label="Insulin (Event-Triggered)")
    # Optional: overlay event pulse signal if present
    if "pulse_u_per_min" in evt.columns:
        pulse = evt.loc[evt_m, "pulse_u_per_min"].fillna(0.0).to_numpy(dtype=float)
        if np.any(pulse > 0):
            ax_u.plot(evt_h[evt_m], pulse, color="black", linestyle=":", linewidth=1.3, label="Event pulse (U/min)")

    ax_u.set_ylabel("Insulin (U/min)")
    ax_u.set_xlabel("Time (hours)")
    ax_u.grid(True, alpha=0.25, linestyle="--")
    ax_u.legend(loc="upper left", frameon=True)

    ax_u.set_xlim(float(args.t0), float(args.t1))

    fig.tight_layout()
    fig.savefig(out_eps, format="eps")
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_eps}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

