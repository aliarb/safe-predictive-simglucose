#!/usr/bin/env python3
"""
Small paper-ready figure: GP uncertainty bounds over time.

This plots the event-triggered controller's BG trace along with the GP-derived
uncertainty envelope (min_lower / max_upper) that is logged per control step.

Outputs EPS + PDF for easy inclusion in IEEE-style papers.
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
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time")
    else:
        df.index = pd.to_datetime(df.index)
    return df


def _pick_trigger_index(df_masked: pd.DataFrame, *, contains: str) -> int | None:
    if "event" not in df_masked.columns:
        return None
    s = df_masked["event"].astype(str)
    hits = s.str.contains(contains, case=False, na=False)
    if not bool(hits.any()):
        return None
    # Prefer the first trigger in the shown window (more representative than max/min extremes).
    return int(np.flatnonzero(hits.to_numpy())[0])

def _pick_trigger_index_near(
    df_masked: pd.DataFrame, *, contains: str, hours: np.ndarray, target_hour: float, window_hours: float
) -> int | None:
    """Pick a trigger index near a target hour (within window)."""
    if "event" not in df_masked.columns:
        return None
    s = df_masked["event"].astype(str)
    hits = s.str.contains(contains, case=False, na=False).to_numpy()
    if not bool(np.any(hits)):
        return None
    w = np.abs(hours - float(target_hour)) <= float(window_hours)
    cand = np.flatnonzero(hits & w)
    if cand.size == 0:
        return None
    # choose closest to target_hour
    j = cand[int(np.argmin(np.abs(hours[cand] - float(target_hour))))]
    return int(j)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--evt",
        default="results/paper_comparison/Safe_NMPC_Event_Triggered_detailed.csv",
        help="Event-triggered detailed CSV (must include min_lower/max_upper columns).",
    )
    ap.add_argument(
        "--out",
        default="Paper/648fc7ad4f4fdf8e359ed7e2/figs/gp_uncertainty_small.eps",
        help="Output EPS path (PDF will also be written next to it).",
    )
    ap.add_argument("--t0", type=float, default=0.0, help="Start hour (relative to trace start).")
    ap.add_argument("--t1", type=float, default=30.0, help="End hour (relative to trace start).")
    ap.add_argument("--target", type=float, default=140.0, help="Target BG (mg/dL).")
    ap.add_argument(
        "--ubwc-hour",
        type=float,
        default=8.0,
        help="Hour (relative) near which to place UBWC callout (defaults to 8h).",
    )
    ap.add_argument(
        "--ubwc-window-hours",
        type=float,
        default=1.0,
        help="Search window (+/- hours) around --ubwc-hour for the BG peak to anchor UBWC.",
    )
    ap.add_argument(
        "--hyper-callout-hour",
        type=float,
        default=18.5,
        help="Hour (relative) near which to point the 'Hyper trigger' callout (defaults to 18–19h).",
    )
    ap.add_argument(
        "--hyper-callout-window-hours",
        type=float,
        default=1.0,
        help="Search window (+/- hours) around --hyper-callout-hour for a hyper trigger line.",
    )
    ap.add_argument(
        "--no-annotations",
        action="store_true",
        help="Disable explanation callouts (arrows) on the plot.",
    )
    args = ap.parse_args()

    evt = _load(Path(args.evt))
    if "BG" not in evt.columns or "min_lower" not in evt.columns or "max_upper" not in evt.columns:
        raise SystemExit("CSV must contain BG, min_lower, max_upper columns.")

    t_ref = evt.index.min()
    hours = (evt.index - t_ref).total_seconds() / 3600.0
    mask = (hours >= float(args.t0)) & (hours <= float(args.t1))

    x = hours[mask]
    bg = evt.loc[mask, "BG"].to_numpy(dtype=float)
    lo = evt.loc[mask, "min_lower"].to_numpy(dtype=float)
    hi = evt.loc[mask, "max_upper"].to_numpy(dtype=float)
    evt_masked = evt.loc[mask].copy()

    # Paper-friendly style (small figure like fig_GPRM.eps)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    # Slightly taller for clarity in print
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 1.9))

    # BG zones (paper-style)
    # - red: unsafe (hypo / hyper)
    # - green: desired (70-140)
    # - orange: warning (140-180) – allowed mainly post-meal
    ax.axhspan(0, 70, alpha=0.10, color="#d62728", label="_nolegend_")         # hypo risk
    ax.axhspan(70, 140, alpha=0.08, color="#2ca02c", label="_nolegend_")       # desired
    ax.axhspan(140, 180, alpha=0.10, color="#ff7f0e", label="_nolegend_")      # warning
    ax.axhspan(180, 1000, alpha=0.10, color="#d62728", label="_nolegend_")     # hyper risk

    # Uncertainty envelope (EPS backend renders transparency as opaque → use a very light fill + outline)
    ax.fill_between(
        x,
        lo,
        hi,
        facecolor="#dbe9f6",   # light blue with good contrast on green/orange/red zones
        edgecolor="#1f77b4",   # subtle outline for visibility
        linewidth=0.9,
        alpha=0.45,
        label="GP envelope",
    )

    # Highlight when GP upper bound exceeds the safety upper limit (180 mg/dL)
    ub_exceeds = hi > 180.0
    if np.any(ub_exceeds):
        ax.fill_between(
            x,
            180.0,
            np.minimum(hi, 220.0),
            where=ub_exceeds,
            facecolor="#d62728",
            edgecolor="none",
            alpha=0.18,
            label="_nolegend_",
        )
    ax.plot(x, bg, color="#57068C", label="BG Level")

    # Mark trigger events with vertical lines
    if "event" in evt_masked.columns:
        ev = evt_masked["event"].astype(str)
        hypo = ev.str.contains("trigger_hypo", case=False, na=False).to_numpy()
        hyper = ev.str.contains("trigger_hyper", case=False, na=False).to_numpy()
        # Use subtle but readable colors; keep legend clean.
        if np.any(hyper):
            ax.vlines(x[hyper], ymin=50, ymax=220, colors="#2E86AB", linewidth=1.0, alpha=0.65, label="_nolegend_")
        if np.any(hypo):
            ax.vlines(x[hypo], ymin=50, ymax=220, colors="#A23B72", linewidth=1.0, alpha=0.65, label="Hypo trigger")

    # Mark eating events (meal starts) with vertical lines (different color)
    meal_col = None
    for cand in ("meal_rate", "meal_rate_g_per_min", "meal"):
        if cand in evt_masked.columns:
            meal_col = cand
            break
    if meal_col is not None:
        mr = pd.to_numeric(evt_masked[meal_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        is_meal = mr > 0.0
        # Rising edges => meal start markers (avoids thick blocks if meal lasts multiple minutes)
        starts = np.flatnonzero(is_meal & ~np.r_[False, is_meal[:-1]])
        if starts.size > 0:
            ax.vlines(
                x[starts],
                ymin=50,
                ymax=220,
                colors="#8c564b",  # brown (distinct from trigger colors + BG zones)
                linewidth=1.2,
                alpha=0.7,
                label="_nolegend_",
            )

    ax.axhline(70, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axhline(180, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axhline(float(args.target), color="black", linestyle=":", linewidth=0.9, alpha=0.7)

    ax.set_xlim(float(args.t0), float(args.t1))
    ax.set_ylim(50, 220)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("BG Level (mg/dL)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=True)

    # Explanation callouts pointing to horizon extremes (paper-friendly)
    if not args.no_annotations and len(x) >= 3:
        # Prefer pointing to *actual* trigger activations in this window.
        # - LBWC -> hypo trigger moment
        # - UBWC -> place around ~8h at the BG peak (paper narrative), otherwise fall back
        i_lo = _pick_trigger_index(evt_masked, contains="trigger_hypo")

        # UBWC: prefer around the requested hour at the BG peak (within a window).
        i_hi = None
        ubwc_hour = float(args.ubwc_hour)
        ubwc_win = float(args.ubwc_window_hours)
        if np.isfinite(ubwc_hour) and np.isfinite(ubwc_win) and ubwc_win > 0:
            w = np.abs(x - ubwc_hour) <= ubwc_win
            if bool(np.any(w)):
                # pick the BG peak in that window
                idxs = np.flatnonzero(w)
                j = int(idxs[int(np.nanargmax(bg[w]))])
                i_hi = j

        # Fallbacks if no triggers exist in the shown interval.
        if i_lo is None:
            i_lo = int(np.nanargmin(lo))
        if i_hi is None:
            # secondary preference: first hyper trigger, then global max upper
            i_hi = _pick_trigger_index(evt_masked, contains="trigger_hyper")
        if i_hi is None:
            i_hi = int(np.nanargmax(hi))

        ax.annotate(
            "LBWC",
            xy=(float(x[i_lo]), float(lo[i_lo])),
            xytext=(float(x[i_lo]) + 1.2, float(lo[i_lo]) - 25.0),
            textcoords="data",
            fontsize=8.5,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="0.25", linewidth=1.0),
        )

        ax.annotate(
            "UBWC",
            xy=(float(x[i_hi]), float(hi[i_hi])),
            xytext=(float(x[i_hi]) - 4.2, float(hi[i_hi]) + 18.0),
            textcoords="data",
            fontsize=8.5,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="0.25", linewidth=1.0),
        )

        # Hyper-trigger callout (arrow to a trigger line; keep it out of legend).
        if "event" in evt_masked.columns:
            j_hyper = _pick_trigger_index_near(
                evt_masked,
                contains="trigger_hyper",
                hours=x,
                target_hour=float(args.hyper_callout_hour),
                window_hours=float(args.hyper_callout_window_hours),
            )
            if j_hyper is None:
                j_hyper = _pick_trigger_index(evt_masked, contains="trigger_hyper")
            if j_hyper is not None:
                ax.annotate(
                    "Hyper trigger",
                    # Arrow tip sits *on* the blue vertical trigger line.
                    xy=(float(x[j_hyper]), 205.0),
                    xytext=(float(x[j_hyper]) + 1.0, 214.0),
                    textcoords="data",
                    fontsize=8.5,
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color="#2E86AB", linewidth=1.2, shrinkA=0, shrinkB=0),
                )

    out_eps = Path(args.out)
    out_eps.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_eps.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(out_eps, format="eps")
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_eps}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

