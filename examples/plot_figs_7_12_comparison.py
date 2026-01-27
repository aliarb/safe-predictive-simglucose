#!/usr/bin/env python3
"""
Paper figure replacement for figs/Figs_7_12.pdf:
  (1) BG level comparison (top)
  (2) Cumulative insulin delivered (middle)
  (3) Insulin injection rate (bottom)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Time" not in df.columns:
        raise ValueError(f"Missing 'Time' column in {csv_path}")
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def _hours_since_start(df: pd.DataFrame) -> np.ndarray:
    t0 = df["Time"].iloc[0]
    return (df["Time"] - t0).dt.total_seconds().to_numpy(dtype=float) / 3600.0


def _cumulative_insulin_u(df: pd.DataFrame) -> np.ndarray:
    """
    Integrate insulin *rate* (U/min) over time to get cumulative insulin delivered (U).
    """
    if "insulin" not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    rate = pd.to_numeric(df["insulin"], errors="coerce").to_numpy(dtype=float)
    t = df["Time"]
    dt_min = t.diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float) / 60.0
    dt_min = np.clip(dt_min, 0.0, 60.0)
    return np.nancumsum(np.nan_to_num(rate, nan=0.0) * dt_min)

def _clean_rate_u_per_min(df: pd.DataFrame) -> np.ndarray:
    """Match compare_nmpc_for_paper.py behavior: numeric + NaNs -> 0."""
    if "insulin" not in df.columns:
        return np.full(len(df), 0.0, dtype=float)
    rate = pd.to_numeric(df["insulin"], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)

def _meal_starts_and_amounts(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect meal start indices using CHO > 0 rising edge.
    Also estimate meal grams from the first CHO sample times the median sample-time minutes.
    (In these logs CHO is 'grams per sample', so for 3-min sampling: 15 -> 45g.)
    """
    if "CHO" not in df.columns:
        return np.array([], dtype=int), np.array([], dtype=float)
    cho = pd.to_numeric(df["CHO"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    is_meal = cho > 0.0
    starts = np.flatnonzero(is_meal & ~np.r_[False, is_meal[:-1]]).astype(int)
    if starts.size == 0:
        return starts, np.array([], dtype=float)
    dt_min = df["Time"].diff().dt.total_seconds().median()
    if not np.isfinite(dt_min) or dt_min <= 0:
        dt_min = 180.0
    dt_min = float(dt_min) / 60.0
    grams = cho[starts] * dt_min
    return starts, grams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bb",
        default="results/paper_comparison/Basal_Bolus_detailed.csv",
        help="Basal-Bolus (patient) detailed CSV.",
    )
    ap.add_argument("--pid", default="results/paper_comparison/PID_detailed.csv", help="PID detailed CSV.")
    ap.add_argument("--nmpc", default="results/paper_comparison/Safe_NMPC_detailed.csv", help="Safe-NMPC detailed CSV.")
    ap.add_argument(
        "--evt",
        default="results/paper_comparison/Safe_NMPC_Event_Triggered_detailed.csv",
        help="Event-triggered Safe-NMPC detailed CSV.",
    )
    ap.add_argument(
        "--out",
        default="Paper/648fc7ad4f4fdf8e359ed7e2/figs/Figs_7_12.pdf",
        help="Output PDF path (EPS will also be written next to it).",
    )
    ap.add_argument("--t0", type=float, default=0.0, help="Start hour (relative).")
    ap.add_argument("--t1", type=float, default=32.0, help="End hour (relative).")
    args = ap.parse_args()

    series = [
        ("Patient (Basal-Bolus)", _load(Path(args.bb)), "#2E86AB"),
        ("PID", _load(Path(args.pid)), "#A23B72"),
        ("Safe-NMPC", _load(Path(args.nmpc)), "#57068C"),
        ("Safe-NMPC (Event-Triggered)", _load(Path(args.evt)), "#FF7F0E"),
    ]

    # Reference time axis from first series
    href = _hours_since_start(series[0][1])
    mask = (href >= float(args.t0)) & (href <= float(args.t1))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.6,
        }
    )

    fig, (ax_bg, ax_cum, ax_rate) = plt.subplots(
        3,
        1,
        figsize=(7.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4, 1.4], "hspace": 0.12},
    )

    # BG zones (paper-style): red unsafe, green desired, orange warning
    for ax in (ax_bg,):
        ax.axhspan(0, 70, alpha=0.10, color="#d62728", label="_nolegend_")
        ax.axhspan(70, 140, alpha=0.08, color="#2ca02c", label="_nolegend_")
        ax.axhspan(140, 180, alpha=0.10, color="#ff7f0e", label="_nolegend_")
        ax.axhspan(180, 1000, alpha=0.10, color="#d62728", label="_nolegend_")
        ax.axhline(70, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(180, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(140, color="black", linestyle=":", linewidth=1.0, alpha=0.7)

    # Meal markers from CHO in the reference series (any CHO>0 marks meal start)
    ref_df = series[0][1]
    starts, grams = _meal_starts_and_amounts(ref_df)
    starts = starts[mask[starts]] if starts.size > 0 else starts
    grams = grams[: starts.size] if grams.size >= starts.size else grams
    if starts.size > 0:
        meal_times = href[starts]
        # Match compare_nmpc_for_paper style: dashed lines + small labels
        for ax in (ax_bg, ax_cum, ax_rate):
            for mt in meal_times:
                ax.axvline(mt, color="#8c564b", linestyle="--", linewidth=1.8, alpha=0.65, zorder=0)
        for mt, g in zip(meal_times, grams):
            ax_bg.annotate(
                f"Meal\n{g:.0f}g",
                xy=(mt, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -2),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#8c564b", alpha=0.18, edgecolor="#8c564b"),
                color="black",
                clip_on=False,
            )

    # Plot each controller
    for label, df, color in series:
        h = _hours_since_start(df)
        m = (h >= float(args.t0)) & (h <= float(args.t1))
        bg = pd.to_numeric(df.get("BG"), errors="coerce").to_numpy(dtype=float)
        rate = _clean_rate_u_per_min(df)
        cum = _cumulative_insulin_u(df)

        ax_bg.plot(h[m], bg[m], color=color, label=label)
        ax_cum.plot(h[m], cum[m], color=color, label="_nolegend_")
        ax_rate.plot(h[m], rate[m], color=color, alpha=0.85, label="_nolegend_")

    ax_bg.set_ylabel("BG Level (mg/dL)")
    ax_bg.set_ylim(50, 260)
    ax_bg.grid(True, alpha=0.25, linestyle="--")
    ax_bg.legend(loc="upper right", frameon=True, ncol=2)

    ax_cum.set_ylabel("Insulin\nDelivered (U)")
    ax_cum.grid(True, alpha=0.25, linestyle="--")

    ax_rate.set_ylabel("Insulin\nRate (U/min)")
    ax_rate.set_xlabel("Time (hours)")
    ax_rate.grid(True, alpha=0.25, linestyle="--")

    ax_rate.set_xlim(float(args.t0), float(args.t1))

    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_eps = out_pdf.with_suffix(".eps")
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_eps, format="eps")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_eps}")


if __name__ == "__main__":
    main()

