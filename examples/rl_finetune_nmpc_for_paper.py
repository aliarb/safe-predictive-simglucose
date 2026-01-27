#!/usr/bin/env python3
"""
RL-style fine-tuning (evolutionary search) for NMPC parameters, starting from
the current fine-tuned baseline.

Outputs (for IEEE paper):
- results/rl_finetuning/best_params.json
- results/rl_finetuning/training_history.csv
- results/rl_finetuning/training_history.png

Then you can run:
  python examples/compare_nmpc_for_paper.py
which will automatically include the RL-tuned Safe-NMPC if best_params.json exists.
"""

from __future__ import annotations

import json
import os
import argparse
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.nmpc_ctrller import NMPCController


def _configure_stdout_for_live_logs() -> None:
    """
    Make progress prints show up promptly when piping output (e.g. through `tee`).
    Best-effort; safe to ignore if unsupported.
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def run_sim_with_heartbeat(sim_obj: SimObj, label: str, *, heartbeat_seconds: float = 60.0):
    """
    Run sim(sim_obj) while printing periodic "still running" lines so users know
    the script hasn't stalled (useful when NMPC solves are slow).
    """
    start_wall = time.time()
    stop_event = threading.Event()

    def _heartbeat():
        while not stop_event.wait(float(heartbeat_seconds)):
            elapsed_min = (time.time() - start_wall) / 60.0
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] still running: {label} (elapsed {elapsed_min:.1f} min)", flush=True)

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    try:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] starting: {label}", flush=True)
        results = sim(sim_obj)
        return results
    finally:
        stop_event.set()
        t.join(timeout=1.0)
        elapsed_min = (time.time() - start_wall) / 60.0
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] finished: {label} (elapsed {elapsed_min:.1f} min)", flush=True)


# Current fine-tuned baseline (keep these as the center of search)
BASELINE_PARAMS: Dict[str, float] = {
    "q_weight": 2.0,
    "q_terminal_weight": 3.0,
    "r_delta_weight": 0.3,
    "hypo_penalty_weight": 100.0,
    "hyper_penalty_weight": 15.0,
    "barrier_weight": 10.0,
    "zone_transition_smoothness": 5.0,
    "insulin_rate_penalty_weight": 100.0,
    "delta_u_asymmetry": 2.0,
    # PID gains (used in supervisor mode; also used as warm start in optimization mode)
    "pid_P": 0.001,
    "pid_I": 0.00001,
    "pid_D": 0.001,
    # Optional gain scheduling (disabled by default)
    "pid_schedule": 0.0,
    "pid_low_bg": 90.0,
    "pid_high_bg": 180.0,
    "pid_P_low": 0.001,
    "pid_I_low": 0.00001,
    "pid_D_low": 0.001,
    "pid_P_mid": 0.001,
    "pid_I_mid": 0.00001,
    "pid_D_mid": 0.001,
    "pid_P_high": 0.001,
    "pid_I_high": 0.00001,
    "pid_D_high": 0.001,
}


# Multiplicative search bounds around baseline (to prevent wild params)
BOUNDS_MULTIPLIER: Dict[str, tuple[float, float]] = {
    "q_weight": (0.5, 2.0),
    "q_terminal_weight": (0.5, 2.0),
    "r_delta_weight": (0.5, 2.0),
    "hypo_penalty_weight": (0.7, 2.0),   # keep safety strong
    "hyper_penalty_weight": (0.5, 2.0),
    "barrier_weight": (0.5, 3.0),
    "zone_transition_smoothness": (0.5, 2.0),
    "insulin_rate_penalty_weight": (0.5, 3.0),
    "delta_u_asymmetry": (0.5, 2.0),
    # PID gains: tune in log-space around baseline (supervisor performance)
    "pid_P": (0.1, 10.0),
    "pid_I": (0.1, 10.0),
    "pid_D": (0.1, 10.0),
    # Schedule gains: same multipliers
    "pid_P_low": (0.1, 10.0),
    "pid_I_low": (0.1, 10.0),
    "pid_D_low": (0.1, 10.0),
    "pid_P_mid": (0.1, 10.0),
    "pid_I_mid": (0.1, 10.0),
    "pid_D_mid": (0.1, 10.0),
    "pid_P_high": (0.1, 10.0),
    "pid_I_high": (0.1, 10.0),
    "pid_D_high": (0.1, 10.0),
}


@dataclass
class Metrics:
    tir: float
    hypo_pct: float
    hyper_pct: float
    violation_pct: float
    severe_hypo_pct: float
    std_bg: float
    mean_bg: float
    bg_diff_std: float
    mage: float
    insulin_std: float
    above_140_pct: float
    above_140_outside_postmeal_pct: float
    rmse_to_140: float


def _compute_metrics(results_df: pd.DataFrame) -> Metrics:
    # Keep timestamps for meal-window logic
    if isinstance(results_df.index, pd.DatetimeIndex):
        df = results_df.copy()
        df = df.reset_index().rename(columns={"index": "Time"})
    else:
        df = results_df.copy()
        if "Time" not in df.columns:
            df = df.reset_index().rename(columns={"index": "Time"})

    # Robust numeric conversion (protect scoring against NaNs/strings)
    bg = pd.to_numeric(df["BG"], errors="coerce").to_numpy(dtype=float)
    bg = np.nan_to_num(bg, nan=140.0, posinf=350.0, neginf=0.0)

    if "insulin" in df.columns:
        insulin = pd.to_numeric(df["insulin"], errors="coerce").to_numpy(dtype=float)
        insulin = np.nan_to_num(insulin, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        insulin = np.zeros_like(bg)
    tir = float(np.mean((bg >= 70) & (bg <= 180)) * 100.0)
    hypo_pct = float(np.mean(bg < 70) * 100.0)
    severe_hypo_pct = float(np.mean(bg < 54) * 100.0)
    hyper_pct = float(np.mean(bg > 180) * 100.0)
    violation_pct = float(np.mean((bg < 70) | (bg > 180)) * 100.0)
    std_bg = float(np.std(bg))
    mean_bg = float(np.mean(bg))
    rmse_to_140 = float(np.sqrt(np.mean((bg - 140.0) ** 2)))
    bg_diff = np.diff(bg)
    bg_diff_std = float(np.std(bg_diff)) if bg_diff.size else 0.0
    mage = float(np.mean(np.abs(bg_diff))) if bg_diff.size else 0.0
    insulin_std = float(np.std(insulin)) if insulin.size else 0.0

    # Meal-aware “orange zone” accounting:
    # - 140–180 is allowed primarily post-meal; penalize time >140 outside a post-meal window.
    # Scenario meals are at 7h, 12h, 18h after start_time. Here we infer elapsed hours from the Time column.
    # If Time parsing fails, fall back to a conservative "no post-meal window" assumption.
    above_140_mask = bg > 140.0
    above_140_pct = float(np.mean(above_140_mask) * 100.0)

    outside_postmeal_mask = np.ones_like(bg, dtype=bool)
    try:
        t = pd.to_datetime(df["Time"])
        t0 = t.iloc[0]
        hours = (t - t0).dt.total_seconds().to_numpy() / 3600.0
        postmeal_hours = 2.5  # configurable via CLI in main()
        meal_times_h = np.array([7.0, 12.0, 18.0], dtype=float)
        in_postmeal = np.zeros_like(hours, dtype=bool)
        for mt in meal_times_h:
            in_postmeal |= (hours >= mt) & (hours <= (mt + postmeal_hours))
        outside_postmeal_mask = ~in_postmeal
    except Exception:
        outside_postmeal_mask = np.ones_like(bg, dtype=bool)

    above_140_outside_postmeal_pct = float(np.mean(above_140_mask & outside_postmeal_mask) * 100.0)
    return Metrics(
        tir=tir,
        hypo_pct=hypo_pct,
        hyper_pct=hyper_pct,
        violation_pct=violation_pct,
        severe_hypo_pct=severe_hypo_pct,
        std_bg=std_bg,
        mean_bg=mean_bg,
        bg_diff_std=bg_diff_std,
        mage=mage,
        insulin_std=insulin_std,
        above_140_pct=above_140_pct,
        above_140_outside_postmeal_pct=above_140_outside_postmeal_pct,
        rmse_to_140=rmse_to_140,
    )


def _score(m: Metrics) -> float:
    # Safety-first + tracking objective:
    # - Severe hypoglycemia (<54) is catastrophic
    # - Any time <70 is catastrophic
    # - Otherwise: minimize tracking error to 140 mg/dL (primary), then smoothness/insulin variability.
    if m.severe_hypo_pct > 0.0:
        return -5e6 - 2e5 * m.severe_hypo_pct - 1e3 * m.rmse_to_140
    if m.hypo_pct > 0.0:
        return -2e6 - 5e4 * m.hypo_pct - 1e3 * m.rmse_to_140

    # If something went numerically wrong, treat it as a terrible candidate (never crash the run).
    if not all(
        np.isfinite(
            [
                m.tir,
                m.hypo_pct,
                m.severe_hypo_pct,
                m.hyper_pct,
                m.violation_pct,
                m.std_bg,
                m.mean_bg,
                m.bg_diff_std,
                m.mage,
                m.insulin_std,
                m.above_140_pct,
                m.above_140_outside_postmeal_pct,
                m.rmse_to_140,
            ]
        )
    ):
        return -np.inf

    # Tracking-first among safe runs.
    # Note: rmse_to_140 already penalizes both hyper and oscillation; we add light smoothness/insulin terms.
    return (
        -100.0 * m.rmse_to_140
        - 1.0 * m.bg_diff_std
        - 0.2 * m.insulin_std
        - 2.0 * m.violation_pct
        - 0.5 * m.hyper_pct
    )


def _sample_params(rng: np.random.RandomState, sigma: float) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for k, base in BASELINE_PARAMS.items():
        # Some keys (e.g. scheduling toggles / thresholds) are not meant to be sampled
        # multiplicatively. If bounds are missing, keep baseline value.
        if k not in BOUNDS_MULTIPLIER:
            params[k] = float(base) if isinstance(base, (int, float, np.integer, np.floating)) else base
            continue

        lo_m, hi_m = BOUNDS_MULTIPLIER[k]
        # Log-normal multiplier around 1.0
        mult = float(np.exp(rng.normal(loc=0.0, scale=sigma)))
        mult = float(np.clip(mult, lo_m, hi_m))
        params[k] = float(base * mult)

    # Keep within absolute sane ranges (avoid pathological values)
    params["r_delta_weight"] = float(np.clip(params["r_delta_weight"], 0.01, 2.0))
    params["zone_transition_smoothness"] = float(np.clip(params["zone_transition_smoothness"], 1.0, 20.0))
    params["delta_u_asymmetry"] = float(np.clip(params["delta_u_asymmetry"], 1.0, 5.0))
    # PID gain absolute sane ranges
    params["pid_P"] = float(np.clip(params["pid_P"], 1e-5, 0.1))
    params["pid_I"] = float(np.clip(params["pid_I"], 0.0, 1e-3))
    params["pid_D"] = float(np.clip(params["pid_D"], 0.0, 0.1))
    # Schedule gain absolute sane ranges
    for k in ("pid_P_low", "pid_P_mid", "pid_P_high"):
        params[k] = float(np.clip(params[k], 1e-5, 0.1))
    for k in ("pid_I_low", "pid_I_mid", "pid_I_high"):
        params[k] = float(np.clip(params[k], 0.0, 1e-3))
    for k in ("pid_D_low", "pid_D_mid", "pid_D_high"):
        params[k] = float(np.clip(params[k], 0.0, 0.1))

    # Thresholds: keep them in a sensible order/range if user chooses to tune them later
    params["pid_low_bg"] = float(np.clip(params.get("pid_low_bg", 90.0), 70.0, 120.0))
    params["pid_high_bg"] = float(np.clip(params.get("pid_high_bg", 180.0), 140.0, 250.0))
    if params["pid_high_bg"] <= params["pid_low_bg"] + 5.0:
        params["pid_high_bg"] = params["pid_low_bg"] + 10.0
    return params


def _make_controller(params: Dict[str, float], *, use_optimization: bool) -> NMPCController:
    c = NMPCController(
        target_bg=140.0,
        prediction_horizon=60,
        control_horizon=30,
        sample_time=5.0,
        bg_min=70.0,
        bg_max=180.0,
        use_optimization=bool(use_optimization),
        pid_P=float(params.get("pid_P", BASELINE_PARAMS["pid_P"])),
        pid_I=float(params.get("pid_I", BASELINE_PARAMS["pid_I"])),
        pid_D=float(params.get("pid_D", BASELINE_PARAMS["pid_D"])),
        pid_schedule=bool(float(params.get("pid_schedule", 0.0)) > 0.5),
        pid_low_bg=float(params.get("pid_low_bg", BASELINE_PARAMS["pid_low_bg"])),
        pid_high_bg=float(params.get("pid_high_bg", BASELINE_PARAMS["pid_high_bg"])),
        pid_P_low=float(params.get("pid_P_low", BASELINE_PARAMS["pid_P_low"])),
        pid_I_low=float(params.get("pid_I_low", BASELINE_PARAMS["pid_I_low"])),
        pid_D_low=float(params.get("pid_D_low", BASELINE_PARAMS["pid_D_low"])),
        pid_P_mid=float(params.get("pid_P_mid", BASELINE_PARAMS["pid_P_mid"])),
        pid_I_mid=float(params.get("pid_I_mid", BASELINE_PARAMS["pid_I_mid"])),
        pid_D_mid=float(params.get("pid_D_mid", BASELINE_PARAMS["pid_D_mid"])),
        pid_P_high=float(params.get("pid_P_high", BASELINE_PARAMS["pid_P_high"])),
        pid_I_high=float(params.get("pid_I_high", BASELINE_PARAMS["pid_I_high"])),
        pid_D_high=float(params.get("pid_D_high", BASELINE_PARAMS["pid_D_high"])),
        q_weight=params["q_weight"],
        q_terminal_weight=params["q_terminal_weight"],
        r_delta_weight=params["r_delta_weight"],
        hypo_penalty_weight=params["hypo_penalty_weight"],
        hyper_penalty_weight=params["hyper_penalty_weight"],
        barrier_weight=params["barrier_weight"],
        zone_transition_smoothness=params["zone_transition_smoothness"],
        insulin_rate_penalty_weight=params["insulin_rate_penalty_weight"],
        delta_u_asymmetry=params["delta_u_asymmetry"],
        verbose=False,  # silence safety spam during RL search
    )
    return c


def _apply_solver_caps(controller: NMPCController, max_iterations: int, max_optimization_time: float) -> None:
    """
    Hard cap NMPC solver effort so finetuning finishes in predictable time.
    These caps apply per control step.
    """
    # Controller stores these under both MATLAB-ish and alias names in different places.
    controller.max_iterations = int(max_iterations)
    controller.Nopt = int(max_iterations)
    controller.max_optimization_time = float(max_optimization_time)
    controller.max_time = float(max_optimization_time)


def _evaluate(
    params: Dict[str, float],
    duration: timedelta,
    comparison_path: str,
    *,
    max_iterations: int,
    max_optimization_time: float,
    start_hour: float,
    use_optimization: bool,
    heartbeat_seconds: float,
    label: str,
) -> tuple[Metrics, pd.DataFrame]:
    # Deterministic setup (same as paper comparison)
    # NOTE: CustomScenario interprets float times as hours after start_time.
    # So if we want the first meal to occur soon during short tuning runs,
    # we should shift start_time forward (e.g., start at 06:00 so breakfast at 7h
    # happens 1 hour into the sim).
    start_hour_i = int(start_hour)
    start_min_i = int(round((float(start_hour) - start_hour_i) * 60.0))
    start_time = datetime(2025, 1, 1, start_hour_i, start_min_i, 0)
    scenario = CustomScenario(start_time=start_time, scenario=[(7, 45), (12, 70), (18, 80)])

    patient = T1DPatient.withName("adolescent#001")
    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")

    env = T1DSimEnv(patient, sensor, pump, scenario)
    controller = _make_controller(params, use_optimization=use_optimization)
    _apply_solver_caps(controller, max_iterations=max_iterations, max_optimization_time=max_optimization_time)
    sim_obj = SimObj(env, controller, duration, animate=False, path=comparison_path)
    results = run_sim_with_heartbeat(sim_obj, label, heartbeat_seconds=heartbeat_seconds)
    m = _compute_metrics(results)
    return m, results


def main():
    parser = argparse.ArgumentParser(description="RL-style finetuning for Safe-NMPC (IEEE-ready outputs).")
    parser.add_argument("--candidates", type=int, default=25, help="Number of candidate parameter sets to evaluate.")
    parser.add_argument("--tune_hours", type=float, default=24.0, help="Duration (hours) for each candidate eval.")
    parser.add_argument("--final_hours", type=float, default=32.0, help="Duration (hours) for final validation.")
    parser.add_argument(
        "--tune-mode",
        type=str,
        choices=["full", "reduced4"],
        default="reduced4",
        help=(
            "Which parameters to tune. "
            "'reduced4' tunes only pid_P, pid_D, r_delta_weight, hypo_penalty_weight (pid_I fixed). "
            "'full' tunes the broader set (and PID gains if enabled)."
        ),
    )
    parser.add_argument(
        "--start_hour",
        type=float,
        default=6.0,
        help="Simulation start hour (0-23). Use 6.0 so the 7am meal occurs early during short tuning runs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--sigma", type=float, default=0.25, help="Log-normal sampling sigma (exploration).")
    parser.add_argument("--max_iterations", type=int, default=15, help="Hard cap on NMPC optimizer iterations per step.")
    parser.add_argument("--max_time", type=float, default=0.10, help="Hard cap (seconds) per NMPC solve per step.")
    parser.add_argument("--final_max_iterations", type=int, default=20, help="Cap for final validation run.")
    parser.add_argument("--final_max_time", type=float, default=0.15, help="Time cap (seconds) for final validation.")
    parser.add_argument(
        "--use-optimization",
        action="store_true",
        help="Tune the optimization-based NMPC (use_optimization=True). If omitted, tuning may have little effect.",
    )
    parser.add_argument(
        "--tune-pid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune the internal PID gains (pid_P/pid_I/pid_D). Default: enabled.",
    )
    parser.add_argument(
        "--tune-pid-schedule",
        action="store_true",
        help="Tune a 3-zone PID gain schedule (low/normal/high BG) to reduce oscillations while staying safe.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/rl_finetuning",
        help="Output directory for best params + training history.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=60.0,
        help="How often to print a 'still running' heartbeat during each candidate simulation.",
    )
    args = parser.parse_args()

    _configure_stdout_for_live_logs()

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    history_rows: List[Dict[str, Any]] = []

    rng = np.random.RandomState(args.seed)
    sigma = float(args.sigma)

    # Fast tuning sims, then validate best on longer horizon
    tune_duration = timedelta(hours=float(args.tune_hours))
    final_duration = timedelta(hours=float(args.final_hours))

    n_candidates = int(args.candidates)  # keep runtime reasonable
    best_score = -np.inf
    best_params = dict(BASELINE_PARAMS)
    best_metrics = None

    print("=" * 80)
    print("RL-STYLE FINETUNING (EVOLUTIONARY SEARCH) FOR SAFE-NMPC")
    print("=" * 80)
    print(f"Candidates: {n_candidates}, tune_duration: {tune_duration}, final_duration: {final_duration}", flush=True)
    print(f"Solver caps (tune): max_iterations={args.max_iterations}, max_time={args.max_time:.3f}s/step", flush=True)
    print(f"Solver caps (final): max_iterations={args.final_max_iterations}, max_time={args.final_max_time:.3f}s/step", flush=True)
    print(f"Controller mode: {'OPTIMIZE' if args.use_optimization else 'SUPERVISOR'}", flush=True)
    print(f"Tune mode: {args.tune_mode}", flush=True)
    if args.tune_mode == "reduced4":
        print(
            "Reduced search params: pid_P, pid_D, r_delta_weight, hypo_penalty_weight (pid_I fixed)",
            flush=True,
        )
    else:
        print(f"Tuning PID gains: {bool(args.tune_pid)} (pid_P/pid_I/pid_D)", flush=True)
        print(f"Tuning PID gain schedule: {bool(args.tune_pid_schedule)}", flush=True)

    # Safety/quality warning: make sure tuning duration actually includes at least one meal.
    # Meals happen at 7h, 12h, 18h *after* start_time (CustomScenario float semantics).
    first_meal_in_hours = 7.0
    if float(args.tune_hours) <= first_meal_in_hours:
        print(
            f"WARNING: tune_hours={args.tune_hours}h does not reach the first meal (7h after start_time). "
            f"Your tuning will be 'night-only' and can produce misleading/degenerate params. "
            f"Consider --start_hour 6.0 or increasing --tune_hours."
        )

    # Always evaluate baseline first for reference
    base_m, _ = _evaluate(
        dict(BASELINE_PARAMS),
        tune_duration,
        out_dir,
        max_iterations=args.max_iterations,
        max_optimization_time=args.max_time,
        start_hour=float(args.start_hour),
        use_optimization=bool(args.use_optimization),
        heartbeat_seconds=float(args.heartbeat_seconds),
        label="baseline",
    )
    base_s = _score(base_m)
    # Initialize "best" from baseline if it is finite (prevents best_metrics=None when scores are NaN)
    if np.isfinite(base_s):
        best_score = base_s
        best_params = dict(BASELINE_PARAMS)
        best_metrics = base_m
    history_rows.append(
        {
            "idx": -1,
            "score": base_s,
            "tir": base_m.tir,
            "hypo_pct": base_m.hypo_pct,
            "severe_hypo_pct": base_m.severe_hypo_pct,
            "hyper_pct": base_m.hyper_pct,
            "violation_pct": base_m.violation_pct,
            "std_bg": base_m.std_bg,
            "mean_bg": base_m.mean_bg,
            "above_140_pct": base_m.above_140_pct,
            "above_140_outside_postmeal_pct": base_m.above_140_outside_postmeal_pct,
            "rmse_to_140": base_m.rmse_to_140,
            "is_baseline": True,
            "use_optimization": bool(args.use_optimization),
            **BASELINE_PARAMS,
        }
    )
    print(
        f"Baseline (tune window): score={base_s:.2f}, TIR={base_m.tir:.1f}%, "
        f"hypo={base_m.hypo_pct:.2f}%, severe_hypo={base_m.severe_hypo_pct:.2f}%, "
        f"hyper={base_m.hyper_pct:.2f}%, violations={base_m.violation_pct:.2f}%, std={base_m.std_bg:.2f}",
        flush=True,
    )

    # SAFETY GATE: Do not run optimization-mode tuning until the optimization baseline is safe.
    # Otherwise the search will select among already-dangerous behaviors and produce unsafe params.
    if args.use_optimization and (base_m.hypo_pct > 0.0 or base_m.severe_hypo_pct > 0.0 or base_m.violation_pct > 0.0):
        print(
            "ERROR: Optimization-mode baseline violates safety (hypo/violations > 0). "
            "Refusing to tune optimization-mode NMPC because it would optimize inside an unsafe regime.\n"
            "Fix optimization-mode safety first (hard insulin bounds/startup guard/safety check), then re-run tuning.",
            flush=True,
        )
        raise SystemExit(2)

    for i in range(n_candidates):
        params = _sample_params(rng, sigma=sigma)
        if args.tune_mode == "reduced4":
            # User-requested simplified search:
            # Tune 4 params total: pid_P, pid_D, r_delta_weight, hypo_penalty_weight.
            # Fix everything else (including pid_I) to baseline.
            tuned_keys = {"pid_P", "pid_D", "r_delta_weight", "hypo_penalty_weight"}
            for k, v in BASELINE_PARAMS.items():
                if k not in tuned_keys:
                    params[k] = v
            # Ensure scheduling stays off unless user explicitly enables it.
            if not args.tune_pid_schedule:
                params["pid_schedule"] = 0.0
        else:
            # Full search space; PID gains are tuned by default (supervisor mode is sensitive to these).
            # You can disable with --no-tune-pid if you want NMPC-only tuning.
            if not bool(args.tune_pid):
                params["pid_P"] = BASELINE_PARAMS["pid_P"]
                params["pid_I"] = BASELINE_PARAMS["pid_I"]
                params["pid_D"] = BASELINE_PARAMS["pid_D"]
        if not args.tune_pid_schedule:
            params["pid_schedule"] = 0.0
            params["pid_low_bg"] = BASELINE_PARAMS["pid_low_bg"]
            params["pid_high_bg"] = BASELINE_PARAMS["pid_high_bg"]
            params["pid_P_low"] = BASELINE_PARAMS["pid_P_low"]
            params["pid_I_low"] = BASELINE_PARAMS["pid_I_low"]
            params["pid_D_low"] = BASELINE_PARAMS["pid_D_low"]
            params["pid_P_mid"] = BASELINE_PARAMS["pid_P_mid"]
            params["pid_I_mid"] = BASELINE_PARAMS["pid_I_mid"]
            params["pid_D_mid"] = BASELINE_PARAMS["pid_D_mid"]
            params["pid_P_high"] = BASELINE_PARAMS["pid_P_high"]
            params["pid_I_high"] = BASELINE_PARAMS["pid_I_high"]
            params["pid_D_high"] = BASELINE_PARAMS["pid_D_high"]
        else:
            params["pid_schedule"] = 1.0
        m, _ = _evaluate(
            params,
            tune_duration,
            out_dir,
            max_iterations=args.max_iterations,
            max_optimization_time=args.max_time,
            start_hour=float(args.start_hour),
            use_optimization=bool(args.use_optimization),
            heartbeat_seconds=float(args.heartbeat_seconds),
            label=f"candidate {i+1}/{n_candidates}",
        )
        s = _score(m)

        history_rows.append(
            {
                "idx": i,
                "score": s,
                "tir": m.tir,
                "hypo_pct": m.hypo_pct,
                "severe_hypo_pct": m.severe_hypo_pct,
                "hyper_pct": m.hyper_pct,
                "violation_pct": m.violation_pct,
                "std_bg": m.std_bg,
                "mean_bg": m.mean_bg,
                "above_140_pct": m.above_140_pct,
                "above_140_outside_postmeal_pct": m.above_140_outside_postmeal_pct,
                "rmse_to_140": m.rmse_to_140,
                "is_baseline": False,
                "use_optimization": bool(args.use_optimization),
                **params,
            }
        )

        if np.isfinite(s) and (best_metrics is None or s > best_score):
            best_score = s
            best_params = params
            best_metrics = m

        if (i + 1) % 5 == 0 or i == 0:
            if best_metrics is None:
                print(f"[{i+1:02d}/{n_candidates}] best_score=NA (no finite scores yet)", flush=True)
                continue
            print(
                f"[{i+1:02d}/{n_candidates}] best_score={best_score:.2f}, "
                f"best(TIR={best_metrics.tir:.1f}%, hypo={best_metrics.hypo_pct:.2f}%, "
                f"severe_hypo={best_metrics.severe_hypo_pct:.2f}%, hyper={best_metrics.hyper_pct:.2f}%, "
                f"violations={best_metrics.violation_pct:.2f}%, std={best_metrics.std_bg:.2f})",
                flush=True,
            )

    # Validate best on final horizon
    assert best_metrics is not None
    final_m, final_results = _evaluate(
        best_params,
        final_duration,
        out_dir,
        max_iterations=args.final_max_iterations,
        max_optimization_time=args.final_max_time,
        start_hour=float(args.start_hour),
        use_optimization=bool(args.use_optimization),
        heartbeat_seconds=float(args.heartbeat_seconds),
        label="final validation",
    )
    final_s = _score(final_m)
    print("-" * 80)
    print(
        f"BEST ({args.final_hours:.0f}h validation): score={final_s:.2f}, "
        f"RMSE140={final_m.rmse_to_140:.2f}, TIR={final_m.tir:.1f}%, "
        f"hypo={final_m.hypo_pct:.2f}%, hyper={final_m.hyper_pct:.2f}%, std={final_m.std_bg:.2f}, mean={final_m.mean_bg:.1f}",
        flush=True,
    )

    # Save best params
    best_path = os.path.join(out_dir, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"✓ Saved best params: {best_path}")

    # Save best detailed results (final horizon)
    best_csv = os.path.join(out_dir, "best_controller_detailed.csv")
    final_results.to_csv(best_csv)
    print(f"✓ Saved best {args.final_hours:.0f}h trace: {best_csv}")

    # Save history CSV
    hist_df = pd.DataFrame(history_rows)
    hist_csv = os.path.join(out_dir, "training_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    print(f"✓ Saved training history: {hist_csv}")

    # Plot training history (IEEE-friendly)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    non_base = hist_df[~hist_df["is_baseline"]].copy()
    ax.plot(non_base["idx"], non_base["score"], color="#57068C", linewidth=2, label="Candidate score")
    ax.axhline(base_s, color="gray", linestyle="--", linewidth=1.5, label=f"Baseline score ({args.tune_hours:.0f}h)")
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("Score")
    ax.set_title(f"RL-style Finetuning Progress ({args.tune_hours:.0f}h evaluations)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "training_history.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved training plot: {plot_path}")

    print("=" * 80)
    print("DONE. Now run: python examples/compare_nmpc_for_paper.py")
    print("=" * 80)


if __name__ == "__main__":
    main()


