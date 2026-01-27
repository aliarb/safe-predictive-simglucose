#!/usr/bin/env python3
"""
Compare NMPC with constraints against other controllers for paper publication.

This script:
1. Runs simulations with multiple controllers (NMPC with constraints, PID, Basal-Bolus)
2. Calculates comprehensive performance metrics
3. Saves results in multiple formats suitable for papers:
   - CSV tables (for Excel/analysis)
   - LaTeX tables (for paper inclusion)
   - Summary statistics JSON
   - Publication-quality figures (PNG and PDF)
   - Detailed per-controller CSV files
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Ensure the repository root is importable when running this file directly.
# (When executed as `python3 examples/compare_nmpc_for_paper.py`, sys.path[0] is `examples/`.)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.event_triggered_nmpc import EventTriggeredNMPCController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim

def _configure_stdout_for_live_logs() -> None:
    """
    Make progress prints show up promptly when piping output (e.g. through `tee`).
    This is best-effort and safe to ignore if unsupported.
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        # Fallback: rely on print(..., flush=True)
        pass


def run_sim_with_heartbeat(sim_obj: SimObj, label: str, *, heartbeat_seconds: float = 15.0):
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


_configure_stdout_for_live_logs()

parser = argparse.ArgumentParser(
    description="Compare NMPC (with constraints) against other controllers and export paper-ready tables/figures."
)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Use speed-oriented NMPC settings (shorter horizons, coarser ODE integration, capped optimization).",
)
parser.add_argument(
    "--skip-rl-tuned",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Skip the RL-tuned Safe-NMPC run (default: enabled). Use --no-skip-rl-tuned to include it.",
)
parser.add_argument(
    "--include-rl-tuned",
    action="store_true",
    help="Explicitly include the RL-tuned Safe-NMPC run (overrides the default skip behavior).",
)
parser.add_argument(
    "--include-event-triggered",
    action="store_true",
    help="Include Safe-NMPC wrapped with GP-based event-triggered pulse/suspension layer.",
)
parser.add_argument(
    "--event-pulse-max",
    type=float,
    default=2.0,
    help="Max event-trigger bolus-rate pulse (U/min).",
)
parser.add_argument(
    "--event-pulse-minutes",
    type=float,
    default=5.0,
    help="Duration of each event-trigger pulse (minutes).",
)
parser.add_argument(
    "--event-cooldown-minutes",
    type=float,
    default=30.0,
    help="Cooldown after each event-trigger (minutes).",
)
parser.add_argument(
    "--event-suspend-minutes",
    type=float,
    default=15.0,
    help="Duration to suspend insulin after predicted hypoglycemia (minutes).",
)
parser.add_argument(
    "--nmpc-use-optimization",
    action="store_true",
    help="Run NMPC controllers in optimization mode (use_optimization=True). Default is PID-supervisor mode.",
)
parser.add_argument(
    "--best-params-path",
    type=str,
    default="./results/rl_finetuning/best_params.json",
    help="Path to RL-tuned NMPC params JSON (used for Safe-NMPC (RL-Tuned) if included).",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default=None,
    help="Output directory for comparison artifacts (default: ./results/paper_comparison).",
)
parser.add_argument("--prediction-horizon", type=int, default=None, help="NMPC prediction horizon (minutes).")
parser.add_argument("--control-horizon", type=int, default=None, help="NMPC control horizon (minutes).")
parser.add_argument(
    "--ode-time-step",
    type=float,
    default=None,
    help="ODE integration step (minutes). Larger = faster but less accurate.",
)
parser.add_argument(
    "--max-iterations",
    type=int,
    default=None,
    help="Cap NMPC optimization iterations per control step (e.g., 10 or 20).",
)
parser.add_argument(
    "--max-opt-time",
    type=float,
    default=None,
    help="Cap NMPC optimization wall time per control step (seconds).",
)
parser.add_argument(
    "--heartbeat-seconds",
    type=float,
    default=15.0,
    help="How often to print 'still running' heartbeat lines during each simulation.",
)
parser.add_argument(
    "--duration-hours",
    type=float,
    default=32.0,
    help="Simulation duration in hours (default: 32). Use smaller values for quick checks.",
)

args = parser.parse_args()

# Choose effective NMPC settings (fast preset unless user explicitly overrides)
if args.fast:
    _default_pred_h = 30
    _default_ctrl_h = 15
    _default_ode_dt = 5.0
    _default_max_iter = 20
    _default_max_time = 1.0
else:
    _default_pred_h = 60
    _default_ctrl_h = 30
    _default_ode_dt = 1.0
    _default_max_iter = None
    _default_max_time = None

PRED_H = int(args.prediction_horizon if args.prediction_horizon is not None else _default_pred_h)
CTRL_H = int(args.control_horizon if args.control_horizon is not None else _default_ctrl_h)
ODE_DT = float(args.ode_time_step if args.ode_time_step is not None else _default_ode_dt)
MAX_ITER = args.max_iterations if args.max_iterations is not None else _default_max_iter
MAX_TIME = args.max_opt_time if args.max_opt_time is not None else _default_max_time
SIM_DURATION = timedelta(hours=float(args.duration_hours))
SIM_DURATION_HOURS = float(args.duration_hours)

def _apply_nmpc_caps(ctrl: NMPCController) -> None:
    """Apply optional runtime caps to NMPCController (keeps behavior but limits compute)."""
    if MAX_ITER is not None:
        ctrl.max_iterations = int(MAX_ITER)
        ctrl.Nopt = int(MAX_ITER)
    if MAX_TIME is not None:
        ctrl.max_optimization_time = float(MAX_TIME)
        ctrl.max_time = float(MAX_TIME)

print("=" * 80)
print("NMPC WITH CONSTRAINTS vs OTHER CONTROLLERS - PAPER COMPARISON")
print("=" * 80)

# Setup simulation parameters
start_time = datetime(2025, 1, 1, 0, 0, 0)
base_path = './results'
comparison_path = args.out_dir if args.out_dir is not None else './results/paper_comparison'
os.makedirs(comparison_path, exist_ok=True)

# Create custom meal scenario
BASE_MEALS = [(7.0, 45), (12.0, 70), (18.0, 80)]  # (hour after start, grams)


def _build_meal_scenario(duration_hours: float):
    """
    Repeat the 7/12/18 meals for each simulated day up to duration_hours.
    Example for 32h: 7, 12, 18, 31 (next day's 7am) will be included.
    """
    duration_hours = float(duration_hours)
    days = int(np.floor((duration_hours - 1e-9) / 24.0)) + 1
    scenario_list = []
    for d in range(days):
        for t_h, grams in BASE_MEALS:
            t = float(t_h + 24.0 * d)
            if t <= duration_hours:
                scenario_list.append((t, int(grams)))
    return scenario_list


meal_scenario = _build_meal_scenario(SIM_DURATION_HOURS)
scenario = CustomScenario(start_time=start_time, scenario=meal_scenario)

# Create patient, sensor, and pump (same for all controllers)
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')

print(f"\nSimulation Setup:")
print(f"  Patient: {patient.name}")
print(f"  Duration: {args.duration_hours:.1f} hours")
meal_desc = ", ".join([f"{t:.0f}h ({g}g)" for t, g in meal_scenario])
print(f"  Meals: {meal_desc}")
print(f"  Results will be saved to: {comparison_path}")
print("\nNMPC Runtime Settings:")
print(f"  Mode: {'FAST' if args.fast else 'DEFAULT'}")
print(f"  prediction_horizon={PRED_H} min, control_horizon={CTRL_H} min, ode_time_step={ODE_DT} min")
print(f"  max_iterations={MAX_ITER}, max_optimization_time={MAX_TIME} sec")
print("\n" + "=" * 80)

# Dictionary to store all results
all_results = {}
all_stats = {}

# Decide which runs to include (for consistent numbering + tables)
best_params_path = args.best_params_path
include_rl = (bool(args.include_rl_tuned) or (not bool(args.skip_rl_tuned))) and os.path.exists(best_params_path)
include_evt = bool(args.include_event_triggered)
TOTAL_RUNS = 3 + (1 if include_rl else 0) + (1 if include_evt else 0)
run_idx = 0

BB_LABEL = "Patient (Basal-Bolus)"

# ========== 1. BASAL-BOLUS CONTROLLER ==========
run_idx += 1
print(f"\n[{run_idx}/{TOTAL_RUNS}] Running Basal-Bolus Controller...")
env1 = T1DSimEnv(patient, sensor, pump, scenario)
controller1 = BBController(target=140)
sim_obj1 = SimObj(env1, controller1, SIM_DURATION, animate=False, path=comparison_path)
results1 = run_sim_with_heartbeat(sim_obj1, f"Basal-Bolus ({args.duration_hours:.1f}h)", heartbeat_seconds=args.heartbeat_seconds)
all_results[BB_LABEL] = results1
print("   ✓ Complete")

# ========== 2. PID CONTROLLER ==========
run_idx += 1
print(f"\n[{run_idx}/{TOTAL_RUNS}] Running PID Controller...")
env2 = T1DSimEnv(patient, sensor, pump, scenario)
controller2 = PIDController(P=0.001, I=0.00001, D=0.001, target=140)
sim_obj2 = SimObj(env2, controller2, SIM_DURATION, animate=False, path=comparison_path)
results2 = run_sim_with_heartbeat(sim_obj2, f"PID ({args.duration_hours:.1f}h)", heartbeat_seconds=args.heartbeat_seconds)
all_results['PID'] = results2
print("   ✓ Complete")

# ========== 3. NMPC CONTROLLER (WITH CONSTRAINTS) ==========
run_idx += 1
print(f"\n[{run_idx}/{TOTAL_RUNS}] Running NMPC Controller with Constraints...")
env3 = T1DSimEnv(patient, sensor, pump, scenario)
controller3 = NMPCController(
    target_bg=140.0,
    prediction_horizon=PRED_H,
    control_horizon=CTRL_H,
    sample_time=5.0,
    ode_time_step=ODE_DT,
    use_optimization=bool(args.nmpc_use_optimization),
    # Fine-tuned baseline params (center for RL finetuning)
    q_weight=2.0,
    r_weight=0.1,
    bg_min=70.0,
    bg_max=180.0,
    barrier_weight=10.0,  # Constraint enforcement weight
    q_terminal_weight=3.0,
    r_delta_weight=0.3,
    hypo_penalty_weight=100.0,
    hyper_penalty_weight=15.0,
    zone_transition_smoothness=5.0,
    insulin_rate_penalty_weight=100.0,
    delta_u_asymmetry=2.0,
    verbose=False
)
_apply_nmpc_caps(controller3)
sim_obj3 = SimObj(env3, controller3, SIM_DURATION, animate=False, path=comparison_path)
results3 = run_sim_with_heartbeat(sim_obj3, f"Safe-NMPC (constraints) ({args.duration_hours:.1f}h)", heartbeat_seconds=args.heartbeat_seconds)
all_results['Safe-NMPC'] = results3
print("   ✓ Complete")
print(f"      Constraint bounds: [{controller3.bg_min}, {controller3.bg_max}] mg/dL")
print(f"      Barrier weight: {controller3.barrier_weight}")

# ========== 4. SAFE-NMPC (RL-TUNED) ==========
if include_rl:
    run_idx += 1
    print(f"\n[{run_idx}/{TOTAL_RUNS}] Running Safe-NMPC (RL-Tuned)...")
    try:
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        env4 = T1DSimEnv(patient, sensor, pump, scenario)
        controller4 = NMPCController(
            target_bg=140.0,
            prediction_horizon=PRED_H,
            control_horizon=CTRL_H,
            sample_time=5.0,
            ode_time_step=ODE_DT,
            bg_min=70.0,
            bg_max=180.0,
            r_weight=0.1,
            verbose=False,
            use_optimization=bool(args.nmpc_use_optimization),
            **best_params
        )
        _apply_nmpc_caps(controller4)
        sim_obj4 = SimObj(env4, controller4, SIM_DURATION, animate=False, path=comparison_path)
        results4 = run_sim_with_heartbeat(sim_obj4, f"Safe-NMPC (RL-Tuned) ({args.duration_hours:.1f}h)", heartbeat_seconds=args.heartbeat_seconds)
        all_results['Safe-NMPC (RL-Tuned)'] = results4
        print("   ✓ Complete")
        print(f"      Loaded: {best_params_path}")
    except Exception as e:
        print(f"   ⚠  Failed to run RL-tuned NMPC (reason: {e})")
elif os.path.exists(best_params_path) and bool(args.skip_rl_tuned) and (not bool(args.include_rl_tuned)):
    # Keep output clean by default; but mention why RL-tuned isn't in the table.
    print("\nSkipping Safe-NMPC (RL-Tuned) (default). Use --no-skip-rl-tuned or --include-rl-tuned to include it.")

# ========== 5. SAFE-NMPC (EVENT-TRIGGERED) ==========
if args.include_event_triggered:
    run_idx += 1
    print(f"\n[{run_idx}/{TOTAL_RUNS}] Running Safe-NMPC (Event-Triggered, GP)...")
    # Wrap the baseline Safe-NMPC (constraints) controller with an event-trigger pulse/suspension layer.
    # This keeps NMPC/PID maintenance behavior but adds fast, time-limited reactions.
    env5 = T1DSimEnv(patient, sensor, pump, scenario)
    base5 = NMPCController(
        target_bg=140.0,
        prediction_horizon=PRED_H,
        control_horizon=CTRL_H,
        sample_time=5.0,
        ode_time_step=ODE_DT,
        use_optimization=bool(args.nmpc_use_optimization),
        q_weight=2.0,
        r_weight=0.1,
        bg_min=70.0,
        bg_max=180.0,
        barrier_weight=10.0,
        q_terminal_weight=3.0,
        r_delta_weight=0.3,
        hypo_penalty_weight=100.0,
        hyper_penalty_weight=15.0,
        zone_transition_smoothness=5.0,
        insulin_rate_penalty_weight=100.0,
        delta_u_asymmetry=2.0,
        # Use the same supervisor PID gains/schedule as the baseline Safe-NMPC for fair comparison
        pid_P=float(getattr(controller3, "pid_P", 0.001)),
        pid_I=float(getattr(controller3, "pid_I", 0.00001)),
        pid_D=float(getattr(controller3, "pid_D", 0.001)),
        pid_schedule=bool(getattr(controller3, "pid_schedule", False)),
        pid_low_bg=float(getattr(controller3, "pid_low_bg", 90.0)),
        pid_high_bg=float(getattr(controller3, "pid_high_bg", 180.0)),
        pid_P_low=float(getattr(controller3, "pid_P_low", getattr(controller3, "pid_P", 0.001))),
        pid_I_low=float(getattr(controller3, "pid_I_low", getattr(controller3, "pid_I", 0.00001))),
        pid_D_low=float(getattr(controller3, "pid_D_low", getattr(controller3, "pid_D", 0.001))),
        pid_P_mid=float(getattr(controller3, "pid_P_mid", getattr(controller3, "pid_P", 0.001))),
        pid_I_mid=float(getattr(controller3, "pid_I_mid", getattr(controller3, "pid_I", 0.00001))),
        pid_D_mid=float(getattr(controller3, "pid_D_mid", getattr(controller3, "pid_D", 0.001))),
        pid_P_high=float(getattr(controller3, "pid_P_high", getattr(controller3, "pid_P", 0.001))),
        pid_I_high=float(getattr(controller3, "pid_I_high", getattr(controller3, "pid_I", 0.00001))),
        pid_D_high=float(getattr(controller3, "pid_D_high", getattr(controller3, "pid_D", 0.001))),
        verbose=False,
    )
    _apply_nmpc_caps(base5)
    controller5 = EventTriggeredNMPCController(
        base5,
        target_bg=140.0,
        hypo_threshold=70.0,
        hyper_threshold=180.0,
        prediction_horizon_minutes=30.0,
        pulse_max_u_per_min=float(args.event_pulse_max),
        pulse_minutes=float(args.event_pulse_minutes),
        cooldown_minutes=float(args.event_cooldown_minutes),
        suspend_minutes=float(args.event_suspend_minutes),
        uncertainty_k=1.0,
        verbose=False,  # keep compare output clean; debug columns are still saved in CSV
    )
    sim_obj5 = SimObj(env5, controller5, SIM_DURATION, animate=False, path=comparison_path)
    results5 = run_sim_with_heartbeat(sim_obj5, f"Safe-NMPC (Event-Triggered) ({args.duration_hours:.1f}h)", heartbeat_seconds=args.heartbeat_seconds)
    all_results["Safe-NMPC (Event-Triggered)"] = results5
    print("   ✓ Complete")

# Note: Removed duplicate NMPC controller since both produce identical results
# The PID-first architecture means barrier_weight has minimal effect

# ========== CALCULATE COMPREHENSIVE STATISTICS ==========
print("\n" + "=" * 80)
print("CALCULATING PERFORMANCE METRICS")
print("=" * 80)

constraint_bounds = (70.0, 180.0)
bg_min, bg_max = constraint_bounds

# Sample time (assuming 5 minutes based on typical CGM sampling)
sample_time_minutes = 5.0

for name, results in all_results.items():
    # Reset index if Time is the index
    if results.index.name == 'Time' or (isinstance(results.index, pd.DatetimeIndex)):
        results_reset = results.reset_index()
    else:
        results_reset = results
    
    # Get data columns
    bg_data = results_reset['BG'].values if 'BG' in results_reset.columns else results_reset.iloc[:, 1].values
    cgm_data = results_reset['CGM'].values if 'CGM' in results_reset.columns else results_reset.iloc[:, 2].values
    
    # Try different ways to get insulin data
    if 'insulin' in results_reset.columns:
        insulin_data = results_reset['insulin'].values
    elif 'Insulin' in results_reset.columns:
        insulin_data = results_reset['Insulin'].values
    else:
        # Try to find by position or name matching
        col_names = [str(col).lower() for col in results_reset.columns]
        if 'insulin' in col_names:
            insulin_idx = col_names.index('insulin')
            insulin_data = results_reset.iloc[:, insulin_idx].values
        elif len(results_reset.columns) >= 5:
            insulin_data = results_reset.iloc[:, 4].values  # insulin is typically 5th column
        else:
            insulin_data = np.zeros_like(bg_data)
            print(f"⚠️  Warning: Could not find insulin column for {name}, using zeros")
    
    # Convert to numeric and handle any non-numeric values
    insulin_data = pd.to_numeric(insulin_data, errors='coerce')
    insulin_data = np.nan_to_num(insulin_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Basic statistics
    mean_bg = np.mean(bg_data)
    std_bg = np.std(bg_data)
    min_bg = np.min(bg_data)
    max_bg = np.max(bg_data)
    cv_bg = (std_bg / mean_bg) * 100 if mean_bg > 0 else 0  # Coefficient of variation

    # Deviation from desired BG (better than mean BG for oscillation/tracking assessment)
    target_bg = 140.0
    bg_err = bg_data - target_bg
    mae_bg = float(np.mean(np.abs(bg_err)))
    rmse_bg = float(np.sqrt(np.mean(bg_err ** 2)))
    
    # Time in range metrics
    time_in_range = np.sum((bg_data >= bg_min) & (bg_data <= bg_max)) / len(bg_data) * 100
    time_below_70 = np.sum(bg_data < 70) / len(bg_data) * 100
    time_above_180 = np.sum(bg_data > 180) / len(bg_data) * 100
    time_below_54 = np.sum(bg_data < 54) / len(bg_data) * 100  # Severe hypoglycemia
    time_above_250 = np.sum(bg_data > 250) / len(bg_data) * 100  # Severe hyperglycemia
    
    # Constraint violations
    violations_below = np.sum(bg_data < bg_min)
    violations_above = np.sum(bg_data > bg_max)
    total_violations = violations_below + violations_above
    violation_percentage = (total_violations / len(bg_data)) * 100
    
    # Risk metrics
    mean_risk = np.mean(results['Risk'].values) if 'Risk' in results.columns else 0.0
    mean_lbgi = np.mean(results['LBGI'].values) if 'LBGI' in results.columns else 0.0
    mean_hbgi = np.mean(results['HBGI'].values) if 'HBGI' in results.columns else 0.0
    
    # Insulin metrics
    # Convert insulin rate (U/min) to total insulin (U) over simulation
    total_insulin = np.sum(insulin_data) * (sample_time_minutes / 60.0)  # Total U over simulation
    mean_insulin = np.mean(insulin_data)
    max_insulin = np.max(insulin_data)
    min_insulin = np.min(insulin_data)
    std_insulin = np.std(insulin_data)
    # Calculate insulin delivery per day (extrapolate from simulation)
    simulation_duration_hours = len(insulin_data) * sample_time_minutes / 60.0
    insulin_per_day = (total_insulin / simulation_duration_hours * 24) if simulation_duration_hours > 0 else 0
    
    # Glucose variability
    mage = np.mean(np.abs(np.diff(bg_data)))  # Mean absolute glucose excursion
    
    # Target achievement
    time_in_tight_range = np.sum((bg_data >= 70) & (bg_data <= 140)) / len(bg_data) * 100
    
    all_stats[name] = {
        'Controller': name,
        'Mean BG (mg/dL)': mean_bg,
        'MAE to 140 (mg/dL)': mae_bg,
        'RMSE to 140 (mg/dL)': rmse_bg,
        'Std BG (mg/dL)': std_bg,
        'CV BG (%)': cv_bg,
        'Min BG (mg/dL)': min_bg,
        'Max BG (mg/dL)': max_bg,
        'Time in Range 70-180 (%)': time_in_range,
        'Time Below 70 (%)': time_below_70,
        'Time Above 180 (%)': time_above_180,
        'Time Below 54 (%)': time_below_54,
        'Time Above 250 (%)': time_above_250,
        'Time in Tight Range 70-140 (%)': time_in_tight_range,
        'Constraint Violations Below': violations_below,
        'Constraint Violations Above': violations_above,
        'Total Constraint Violations': total_violations,
        'Constraint Violation Rate (%)': violation_percentage,
        'Mean Risk Index': mean_risk,
        'Mean LBGI': mean_lbgi,
        'Mean HBGI': mean_hbgi,
        'Total Insulin (U)': total_insulin,
        'Mean Insulin (U/min)': mean_insulin,
        'Std Insulin (U/min)': std_insulin,
        'Min Insulin (U/min)': min_insulin,
        'Max Insulin (U/min)': max_insulin,
        'Insulin per Day (U/day)': insulin_per_day,
        'MAGE (mg/dL)': mage,
        'bg_data': bg_data,
        'cgm_data': cgm_data,
        'insulin_data': insulin_data,
        'time': results.index
    }

# ========== CREATE RESULTS DATAFRAME ==========
stats_df = pd.DataFrame([{k: v for k, v in stats.items() if k not in ['bg_data', 'cgm_data', 'insulin_data', 'time']} 
                        for stats in all_stats.values()])
stats_df = stats_df.set_index('Controller')

# ========== HELPER FUNCTION FOR LATEX TABLE ==========
def create_formatted_latex_table(df):
    """Create a nicely formatted LaTeX table for papers."""
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance comparison of controllers}\n"
    latex += "\\label{tab:controller_comparison}\n"
    latex += "\\begin{tabular}{l" + "c" * (len(df.columns)) + "}\n"
    latex += "\\toprule\n"
    
    # Header
    latex += "Controller"
    for col in df.columns:
        col_clean = col.replace(' (%)', '').replace(' (mg/dL)', '').replace(' (U)', '').replace(' (U/min)', '')
        latex += f" & {col_clean}"
    latex += " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for idx, row in df.iterrows():
        latex += idx.replace('_', ' ')
        for col in df.columns:
            val = row[col]
            if '%' in col or 'Rate' in col:
                latex += f" & {val:.2f}"
            elif 'Mean' in col or 'Std' in col or 'CV' in col:
                latex += f" & {val:.2f}"
            elif isinstance(val, (int, np.integer)):
                latex += f" & {val}"
            else:
                latex += f" & {val:.2f}"
        latex += " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    return latex

# ========== SAVE RESULTS IN MULTIPLE FORMATS ==========
print("\n" + "=" * 80)
print("SAVING RESULTS FOR PAPER")
print("=" * 80)

# 1. Save comprehensive CSV table
csv_file = os.path.join(comparison_path, 'controller_comparison_table.csv')
stats_df.to_csv(csv_file)
print(f"✓ Saved CSV table: {csv_file}")

# 2. Save LaTeX table (using custom function to avoid jinja2 dependency)
latex_file = os.path.join(comparison_path, 'controller_comparison_table.tex')
try:
    latex_table = stats_df.to_latex(float_format="%.2f", 
                                     caption="Performance comparison of controllers",
                                     label="tab:controller_comparison",
                                     index=True)
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved LaTeX table: {latex_file}")
except (ImportError, Exception) as e:
    # Fallback: use custom function if jinja2 not available
    print(f"⚠  Using custom LaTeX generator (reason: {type(e).__name__})")
    latex_table = create_formatted_latex_table(stats_df)
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved LaTeX table: {latex_file}")

# 3. Save formatted LaTeX table (more readable)
formatted_latex_file = os.path.join(comparison_path, 'controller_comparison_table_formatted.tex')
formatted_latex = create_formatted_latex_table(stats_df)
with open(formatted_latex_file, 'w') as f:
    f.write(formatted_latex)
print(f"✓ Saved formatted LaTeX table: {formatted_latex_file}")

# 4. Save JSON summary
json_file = os.path.join(comparison_path, 'controller_comparison_summary.json')
json_summary = {name: {k: float(v) if isinstance(v, (np.integer, np.floating)) else str(v) 
                       for k, v in stats.items() 
                       if k not in ['bg_data', 'cgm_data', 'insulin_data', 'time']}
                for name, stats in all_stats.items()}
with open(json_file, 'w') as f:
    json.dump(json_summary, f, indent=2)
print(f"✓ Saved JSON summary: {json_file}")

# 5. Save individual controller CSV files
for name, results in all_results.items():
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    controller_csv = os.path.join(comparison_path, f'{safe_name}_detailed.csv')
    results.to_csv(controller_csv)
print(f"✓ Saved individual controller CSV files")

# ========== PRINT SUMMARY TABLE ==========
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON SUMMARY")
print("=" * 80)
print("\nKey Metrics:")
print(stats_df[['RMSE to 140 (mg/dL)', 'Time in Range 70-180 (%)', 
                'Time Below 70 (%)', 'Time Above 180 (%)', 
                'Constraint Violation Rate (%)', 'Mean Risk Index',
                'Mean Insulin (U/min)', 'Insulin per Day (U/day)']].to_string())

# ========== CREATE PUBLICATION-QUALITY FIGURES ==========
print("\n" + "=" * 80)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("=" * 80)

# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3
})

# Color scheme
colors = {
    BB_LABEL: '#2E86AB',
    'PID': '#A23B72',
    'Safe-NMPC': '#57068C',  # NYU/CCNY Purple
    'Safe-NMPC (RL-Tuned)': '#2CA02C'  # distinct green (keep dashed linestyle too)
}
if 'Safe-NMPC (Event-Triggered)' in all_stats:
    colors['Safe-NMPC (Event-Triggered)'] = '#FF7F0E'  # orange

# Create comprehensive figure.
# Use a 5x2 grid to avoid overcrowding/overlapping text.
# Row 0: glucose trace (spans both columns)
# Rows 1-3: bar charts
# Row 4: insulin trace (spans both columns)
fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(
    5, 2,
    hspace=0.55, wspace=0.35,
    left=0.07, right=0.98, top=0.92, bottom=0.07
)

# Subplot 1: Glucose Trajectories
ax1 = fig.add_subplot(gs[0, :])
for name, stats in all_stats.items():
    time_hours = [(t - stats['time'][0]).total_seconds() / 3600 for t in stats['time']]
    linestyle = '--' if name == 'Safe-NMPC (RL-Tuned)' else '-'
    ax1.plot(time_hours, stats['bg_data'], color=colors[name], label=name, linewidth=2, alpha=0.85, linestyle=linestyle)

# Axes limits first (so annotations don't end up in weird places)
ax1.set_xlim(0, SIM_DURATION_HOURS)
ax1.set_ylim(50, 200)

# Add meal markers (text in axes-coordinates so it never overlaps the trace).
# If duration > 24h, repeat daily markers for display (simulation scenario may or may not include meals after day 1).
base_meal_times = [7, 12, 18]  # Hours: 7am, 12pm, 6pm
meal_times = []
days = int(np.floor(SIM_DURATION_HOURS / 24.0)) + 1
for d in range(days):
    for mt in base_meal_times:
        t = mt + 24.0 * d
        if t <= SIM_DURATION_HOURS:
            meal_times.append(t)
meal_amounts = [45, 70, 80]  # Grams of CHO
meal_colors = ['orange', 'red', 'darkred']

for i, meal_time in enumerate(meal_times):
    meal_amount = meal_amounts[i % len(meal_amounts)]
    meal_color = meal_colors[i % len(meal_colors)]
    ax1.axvline(meal_time, color=meal_color, linestyle='--', linewidth=2, alpha=0.7, zorder=0)
    ax1.annotate(
        f"Meal\n{meal_amount}g",
        xy=(meal_time, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -2),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=meal_color, alpha=0.25, edgecolor=meal_color),
        color="black",
        clip_on=False,
    )

# Zone shading (kept for visual clarity) but hidden from the legend to simplify plots.
ax1.axhspan(0, 70, alpha=0.10, color='red', label='_nolegend_')
ax1.axhspan(70, 140, alpha=0.12, color='green', label='_nolegend_')
ax1.axhspan(140, 180, alpha=0.12, color='orange', label='_nolegend_')
ax1.axhspan(180, 1000, alpha=0.10, color='red', label='_nolegend_')
ax1.axhline(140, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Target (140 mg/dL)')
ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel('Blood Glucose (mg/dL)', fontweight='bold')
ax1.set_title(f'(a) Blood Glucose Trajectories Over {SIM_DURATION_HOURS:.0f} Hours', fontweight='bold', pad=10)
ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, ncol=2)
ax1.grid(True, alpha=0.3, linestyle='--')

# Subplot 2: Time in Range
ax2 = fig.add_subplot(gs[1, 0])
controllers = list(all_stats.keys())
tir_values = [all_stats[c]['Time in Range 70-180 (%)'] for c in controllers]
colors_list = [colors[c] for c in controllers]
bars = ax2.bar(controllers, tir_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax2.axhline(70, color='green', linestyle='--', linewidth=2, label='Target (70%)')
ax2.set_ylabel('Time in Range (%)', fontweight='bold')
ax2.set_title('(b) Time in Range (70-180 mg/dL)', fontweight='bold', pad=10)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.legend(loc='upper right')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha='right')
for bar, val in zip(bars, tir_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9, clip_on=True)

# Subplot 3: Constraint Violations
ax3 = fig.add_subplot(gs[1, 1])
violation_values = [all_stats[c]['Constraint Violation Rate (%)'] for c in controllers]
bars = ax3.bar(controllers, violation_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax3.axhline(5, color='red', linestyle='--', linewidth=2, label='Target (<5%)')
ax3.set_ylabel('Constraint Violation Rate (%)', fontweight='bold')
ax3.set_title('(c) Constraint Violation Rate', fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
ax3.legend(loc='upper right')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=25, ha='right')
for bar, val in zip(bars, violation_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9, clip_on=True)

# Subplot 4: Risk Index
ax4 = fig.add_subplot(gs[2, 0])
risk_values = [all_stats[c]['Mean Risk Index'] for c in controllers]
bars = ax4.bar(controllers, risk_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax4.set_ylabel('Mean Risk Index', fontweight='bold')
ax4.set_title('(d) Mean Risk Index (Lower is Better)', fontweight='bold', pad=10)
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=25, ha='right')
for bar, val in zip(bars, risk_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9, clip_on=True)

# Subplot 5: Mean Insulin Rate
ax5 = fig.add_subplot(gs[2, 1])
mean_insulin_values = [all_stats[c]['Mean Insulin (U/min)'] for c in controllers]
bars = ax5.bar(controllers, mean_insulin_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax5.set_ylabel('Mean Insulin Rate (U/min)', fontweight='bold')
ax5.set_title('(e) Mean Insulin Injection Rate', fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=25, ha='right')
for bar, val in zip(bars, mean_insulin_values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9, clip_on=True)

# Subplot 6: Insulin per Day
ax6 = fig.add_subplot(gs[3, 0])
insulin_per_day_values = [all_stats[c]['Insulin per Day (U/day)'] for c in controllers]
bars = ax6.bar(controllers, insulin_per_day_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax6.set_ylabel('Insulin per Day (U/day)', fontweight='bold')
ax6.set_title('(f) Total Daily Insulin', fontweight='bold', pad=10)
ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=25, ha='right')
for bar, val in zip(bars, insulin_per_day_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9, clip_on=True)

# Subplot 7: Deviation from Target (RMSE)
ax7 = fig.add_subplot(gs[3, 1])
rmse_values = [all_stats[c]['RMSE to 140 (mg/dL)'] for c in controllers]
bars = ax7.bar(controllers, rmse_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax7.set_ylabel('RMSE to Target (mg/dL)', fontweight='bold')
ax7.set_title('(g) Tracking Error (RMSE to 140 mg/dL)', fontweight='bold', pad=10)
ax7.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=25, ha='right')
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9, clip_on=True)

# Subplot 8: Insulin Trajectories
ax8 = fig.add_subplot(gs[4, :])
for name, stats in all_stats.items():
    time_hours = [(t - stats['time'][0]).total_seconds() / 3600 for t in stats['time']]
    linestyle = '--' if name == 'Safe-NMPC (RL-Tuned)' else '-'
    ax8.plot(time_hours, stats['insulin_data'], color=colors[name], label=name, linewidth=2, alpha=0.85, linestyle=linestyle)

# Add meal markers to insulin plot as well
for meal_time, meal_amount, meal_color in zip(meal_times, meal_amounts, meal_colors):
    ax8.axvline(meal_time, color=meal_color, linestyle='--', linewidth=2, alpha=0.7, zorder=0)
    ax8.annotate(
        f"Meal\n{meal_amount}g",
        xy=(meal_time, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -2),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=meal_color, alpha=0.25, edgecolor=meal_color),
        color="black",
        clip_on=False,
    )

ax8.set_xlabel('Time (hours)', fontweight='bold')
ax8.set_ylabel('Insulin Rate (U/min)', fontweight='bold')
ax8.set_title('(h) Insulin Injection Rate Over Time', fontweight='bold', pad=10)
ax8.legend(loc='best', frameon=True, fancybox=True, shadow=True)
ax8.grid(True, alpha=0.3, linestyle='--')
ax8.set_xlim(0, SIM_DURATION_HOURS)

# Overall title
fig.suptitle('Controller Performance Comparison: Safe-NMPC vs Baseline Methods',
            fontsize=16, fontweight='bold', y=0.98)

# Final layout pass to reduce label overlap
fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

# Save figures
png_file = os.path.join(comparison_path, 'controller_comparison.png')
plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved PNG figure: {png_file}")

pdf_file = os.path.join(comparison_path, 'controller_comparison.pdf')
plt.savefig(pdf_file, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved PDF figure: {pdf_file}")

plt.close()

# ========== ADDITIONAL PAPER_COMPARISON EXPORTS (BG+INSULIN, RMSE+RISK) ==========
# Requested: save BG level + insulin level in one file, and RMSE + Risk Index in a separate file.

# 1) BG + Insulin traces in a single figure
# IMPORTANT: Use the same plotting logic as in controller_comparison.png
fig_traces, (ax_bg, ax_u) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# (a) Blood glucose trajectories (match controller_comparison: ax1)
for name, stats in all_stats.items():
    time_hours = [(t - stats['time'][0]).total_seconds() / 3600 for t in stats['time']]
    linestyle = '--' if name == 'Safe-NMPC (RL-Tuned)' else '-'
    ax_bg.plot(
        time_hours,
        stats['bg_data'],
        color=colors[name],
        label=name,
        linewidth=2,
        alpha=0.85,
        linestyle=linestyle,
    )

ax_bg.set_xlim(0, SIM_DURATION_HOURS)
ax_bg.set_ylim(50, 200)

# Meal markers + annotations (same logic as controller_comparison: ax1)
for i, meal_time in enumerate(meal_times):
    meal_amount = meal_amounts[i % len(meal_amounts)]
    meal_color = meal_colors[i % len(meal_colors)]
    ax_bg.axvline(meal_time, color=meal_color, linestyle='--', linewidth=2, alpha=0.7, zorder=0)
    ax_bg.annotate(
        f"Meal\n{meal_amount}g",
        xy=(meal_time, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -2),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=meal_color, alpha=0.25, edgecolor=meal_color),
        color="black",
        clip_on=False,
    )

# Zone shading + target line (same as controller_comparison: ax1)
ax_bg.axhspan(0, 70, alpha=0.10, color='red', label='_nolegend_')
ax_bg.axhspan(70, 140, alpha=0.12, color='green', label='_nolegend_')
ax_bg.axhspan(140, 180, alpha=0.12, color='orange', label='_nolegend_')
ax_bg.axhspan(180, 1000, alpha=0.10, color='red', label='_nolegend_')
ax_bg.axhline(140, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Target (140 mg/dL)')
ax_bg.set_ylabel('Blood Glucose (mg/dL)', fontweight='bold')
ax_bg.set_title(f'(a) Blood Glucose Trajectories Over {SIM_DURATION_HOURS:.0f} Hours', fontweight='bold', pad=10)
ax_bg.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, ncol=2)
ax_bg.grid(True, alpha=0.3, linestyle='--')

# (h) Insulin trajectories (match controller_comparison: ax8)
for name, stats in all_stats.items():
    time_hours = [(t - stats['time'][0]).total_seconds() / 3600 for t in stats['time']]
    linestyle = '--' if name == 'Safe-NMPC (RL-Tuned)' else '-'
    ax_u.plot(
        time_hours,
        stats['insulin_data'],
        color=colors[name],
        label=name,
        linewidth=2,
        alpha=0.85,
        linestyle=linestyle,
    )

# Meal markers + annotations (same as controller_comparison: ax8)
for meal_time, meal_amount, meal_color in zip(meal_times, meal_amounts, meal_colors):
    ax_u.axvline(meal_time, color=meal_color, linestyle='--', linewidth=2, alpha=0.7, zorder=0)
    ax_u.annotate(
        f"Meal\n{meal_amount}g",
        xy=(meal_time, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -2),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=meal_color, alpha=0.25, edgecolor=meal_color),
        color="black",
        clip_on=False,
    )

ax_u.set_xlabel('Time (hours)', fontweight='bold')
ax_u.set_ylabel('Insulin Rate (U/min)', fontweight='bold')
ax_u.set_title('(h) Insulin Injection Rate Over Time', fontweight='bold', pad=10)
ax_u.legend(loc='best', frameon=True, fancybox=True, shadow=True)
ax_u.grid(True, alpha=0.3, linestyle='--')
ax_u.set_xlim(0, SIM_DURATION_HOURS)

fig_traces.tight_layout()
traces_png = os.path.join(comparison_path, 'nmpc_bg_insulin.png')
fig_traces.savefig(traces_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved BG+Insulin figure: {traces_png}")
traces_pdf = os.path.join(comparison_path, 'nmpc_bg_insulin.pdf')
fig_traces.savefig(traces_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved BG+Insulin figure (PDF): {traces_pdf}")
plt.close(fig_traces)

# 2) RMSE + Risk Index in a separate file (CSV + small figure)
controllers = list(all_stats.keys())
rmse_values = [all_stats[c]['RMSE to 140 (mg/dL)'] for c in controllers]
risk_values = [all_stats[c]['Mean Risk Index'] for c in controllers]

metrics_df = pd.DataFrame(
    {
        'Controller': controllers,
        'RMSE to 140 (mg/dL)': rmse_values,
        'Mean Risk Index': risk_values,
    }
).set_index('Controller')

metrics_csv = os.path.join(comparison_path, 'nmpc_rmse_risk.csv')
metrics_df.to_csv(metrics_csv)
print(f"✓ Saved RMSE+Risk CSV: {metrics_csv}")

fig_metrics, (ax_rmse, ax_risk) = plt.subplots(1, 2, figsize=(14, 5))

colors_list = [colors.get(c, '#4C4C4C') for c in controllers]
ax_rmse.bar(controllers, rmse_values, color=colors_list, alpha=0.75, edgecolor='black', linewidth=1.0)
ax_rmse.set_title('Tracking Error (RMSE to 140 mg/dL)', fontweight='bold')
ax_rmse.set_ylabel('RMSE (mg/dL)', fontweight='bold')
ax_rmse.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax_rmse.xaxis.get_majorticklabels(), rotation=25, ha='right')

ax_risk.bar(controllers, risk_values, color=colors_list, alpha=0.75, edgecolor='black', linewidth=1.0)
ax_risk.set_title('Mean Risk Index (Lower is Better)', fontweight='bold')
ax_risk.set_ylabel('Mean Risk Index', fontweight='bold')
ax_risk.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax_risk.xaxis.get_majorticklabels(), rotation=25, ha='right')

fig_metrics.tight_layout()
metrics_png = os.path.join(comparison_path, 'nmpc_rmse_risk.png')
fig_metrics.savefig(metrics_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved RMSE+Risk figure: {metrics_png}")
metrics_pdf = os.path.join(comparison_path, 'nmpc_rmse_risk.pdf')
fig_metrics.savefig(metrics_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved RMSE+Risk figure (PDF): {metrics_pdf}")
plt.close(fig_metrics)

# 3) Multi-day 6-panel figure (meals, BG, insulin, internal states, cumulative insulin)
def _run_with_state_history(sim_env: T1DSimEnv, controller, sim_duration: timedelta):
    """
    Run a simulation like SimObj.simulate, but also collect patient internal states
    (from info['patient_state']) at each environment step.
    Returns:
      - df_hist: env.show_history() dataframe (Time-indexed)
      - state_df: dataframe with columns x0..x12 indexed by Time
    """
    controller.reset()
    obs, reward, done, info = sim_env.reset()

    # Collect initial state
    state_hist = [np.array(info["patient_state"], dtype=float)]
    time_hist = [info["time"]]

    while sim_env.time < sim_env.scenario.start_time + sim_duration:
        action = controller.policy(obs, reward, done, **info)
        obs, reward, done, info = sim_env.step(action)
        # state at the end of this step
        state_hist.append(np.array(info["patient_state"], dtype=float))
        time_hist.append(info["time"])

    df_hist = sim_env.show_history()
    state_arr = np.vstack(state_hist)  # (N+1, 13)
    state_df = pd.DataFrame(
        state_arr,
        index=pd.to_datetime(pd.Series(time_hist)),
        columns=[f"x{i}" for i in range(state_arr.shape[1])],
    )
    # Align indices with history (same Time grid)
    try:
        state_df = state_df.reindex(df_hist.index)
    except Exception:
        pass
    return df_hist, state_df


try:
    # Generate the 6-panel multi-day figure using the same comparison style as controller_comparison.png,
    # overlaying all included methods and stacking subplots in one column.
    if "Safe-NMPC" in all_results:
        print("\nGenerating 6-panel multi-day comparison figure (stacked, all methods)...")

        # Decide which controller variants are included
        _included = [BB_LABEL, "PID", "Safe-NMPC"]
        if include_rl and "Safe-NMPC (RL-Tuned)" in all_results:
            _included.append("Safe-NMPC (RL-Tuned)")
        if include_evt and "Safe-NMPC (Event-Triggered)" in all_results:
            _included.append("Safe-NMPC (Event-Triggered)")

        # Build fresh env/controller for each method to capture internal patient states.
        def _make_controller(name: str):
            if name == BB_LABEL:
                return BBController(target=140)
            if name == "PID":
                return PIDController(P=0.001, I=0.00001, D=0.001, target=140)
            if name == "Safe-NMPC":
                c = NMPCController(
                    target_bg=140.0,
                    prediction_horizon=PRED_H,
                    control_horizon=CTRL_H,
                    sample_time=5.0,
                    ode_time_step=ODE_DT,
                    use_optimization=bool(args.nmpc_use_optimization),
                    q_weight=2.0,
                    r_weight=0.1,
                    bg_min=70.0,
                    bg_max=180.0,
                    barrier_weight=10.0,
                    q_terminal_weight=3.0,
                    r_delta_weight=0.3,
                    hypo_penalty_weight=100.0,
                    hyper_penalty_weight=15.0,
                    zone_transition_smoothness=5.0,
                    insulin_rate_penalty_weight=100.0,
                    delta_u_asymmetry=2.0,
                    verbose=False,
                )
                _apply_nmpc_caps(c)
                return c
            if name == "Safe-NMPC (RL-Tuned)":
                with open(best_params_path, "r") as f:
                    best_params = json.load(f)
                c = NMPCController(
                    target_bg=140.0,
                    prediction_horizon=PRED_H,
                    control_horizon=CTRL_H,
                    sample_time=5.0,
                    ode_time_step=ODE_DT,
                    bg_min=70.0,
                    bg_max=180.0,
                    r_weight=0.1,
                    verbose=False,
                    use_optimization=bool(args.nmpc_use_optimization),
                    **best_params,
                )
                _apply_nmpc_caps(c)
                return c
            if name == "Safe-NMPC (Event-Triggered)":
                # Recreate the same baseline + wrapper used earlier in this script.
                base = NMPCController(
                    target_bg=140.0,
                    prediction_horizon=PRED_H,
                    control_horizon=CTRL_H,
                    sample_time=5.0,
                    ode_time_step=ODE_DT,
                    use_optimization=bool(args.nmpc_use_optimization),
                    q_weight=2.0,
                    r_weight=0.1,
                    bg_min=70.0,
                    bg_max=180.0,
                    barrier_weight=10.0,
                    q_terminal_weight=3.0,
                    r_delta_weight=0.3,
                    hypo_penalty_weight=100.0,
                    hyper_penalty_weight=15.0,
                    zone_transition_smoothness=5.0,
                    insulin_rate_penalty_weight=100.0,
                    delta_u_asymmetry=2.0,
                    pid_P=float(getattr(controller3, "pid_P", 0.001)),
                    pid_I=float(getattr(controller3, "pid_I", 0.00001)),
                    pid_D=float(getattr(controller3, "pid_D", 0.001)),
                    pid_schedule=bool(getattr(controller3, "pid_schedule", False)),
                    pid_low_bg=float(getattr(controller3, "pid_low_bg", 90.0)),
                    pid_high_bg=float(getattr(controller3, "pid_high_bg", 180.0)),
                    pid_P_low=float(getattr(controller3, "pid_P_low", getattr(controller3, "pid_P", 0.001))),
                    pid_I_low=float(getattr(controller3, "pid_I_low", getattr(controller3, "pid_I", 0.00001))),
                    pid_D_low=float(getattr(controller3, "pid_D_low", getattr(controller3, "pid_D", 0.001))),
                    pid_P_mid=float(getattr(controller3, "pid_P_mid", getattr(controller3, "pid_P", 0.001))),
                    pid_I_mid=float(getattr(controller3, "pid_I_mid", getattr(controller3, "pid_I", 0.00001))),
                    pid_D_mid=float(getattr(controller3, "pid_D_mid", getattr(controller3, "pid_D", 0.001))),
                    pid_P_high=float(getattr(controller3, "pid_P_high", getattr(controller3, "pid_P", 0.001))),
                    pid_I_high=float(getattr(controller3, "pid_I_high", getattr(controller3, "pid_I", 0.00001))),
                    pid_D_high=float(getattr(controller3, "pid_D_high", getattr(controller3, "pid_D", 0.001))),
                    verbose=False,
                )
                _apply_nmpc_caps(base)
                return EventTriggeredNMPCController(
                    base,
                    target_bg=140.0,
                    hypo_threshold=70.0,
                    hyper_threshold=180.0,
                    prediction_horizon_minutes=30.0,
                    pulse_max_u_per_min=float(args.event_pulse_max),
                    pulse_minutes=float(args.event_pulse_minutes),
                    cooldown_minutes=float(args.event_cooldown_minutes),
                    suspend_minutes=float(args.event_suspend_minutes),
                    uncertainty_k=1.0,
                    verbose=False,
                )
            raise ValueError(f"Unknown controller name: {name}")

        per_method = {}
        for name in _included:
            _p_int = T1DPatient.withName(patient.name)
            _s_int = CGMSensor.withName("Dexcom", seed=1)
            _pump_int = InsulinPump.withName("Insulet")
            _env_int = T1DSimEnv(_p_int, _s_int, _pump_int, scenario)
            _ctrl = _make_controller(name)
            df_hist, state_df = _run_with_state_history(_env_int, _ctrl, SIM_DURATION)
            per_method[name] = (df_hist, state_df)

        # Use a common time axis (hours) from the first method
        df0 = next(iter(per_method.values()))[0]
        t0 = df0.index[0]
        time_hours = (df0.index - t0).total_seconds() / 3600.0

        # Infer step size in minutes for cumulative insulin
        if len(df0.index) >= 2:
            dt_min = float(np.median(np.diff(df0.index.values).astype("timedelta64[s]").astype(float)) / 60.0)
        else:
            dt_min = 5.0

        # Meal profile from the scenario list (hours, grams)
        meal_t = [float(t_h) for t_h, _g in meal_scenario]
        meal_g = [float(_g) for _t, _g in meal_scenario]

        # Panel height tuning for one-column figure:
        # - (a) CHO: half height
        # - (b) BG: 1.25x height
        # - (c)-(f): 0.7x height
        fig6 = plt.figure(figsize=(14, 18))
        gs6 = fig6.add_gridspec(6, 1, hspace=0.45, height_ratios=[0.5, 1.25, 0.7, 0.7, 0.7, 0.7])

        axa = fig6.add_subplot(gs6[0, 0])
        axb = fig6.add_subplot(gs6[1, 0], sharex=axa)
        axc = fig6.add_subplot(gs6[2, 0], sharex=axa)
        axd = fig6.add_subplot(gs6[3, 0], sharex=axa)
        axe = fig6.add_subplot(gs6[4, 0], sharex=axa)
        axf = fig6.add_subplot(gs6[5, 0], sharex=axa)

        # (a) Meal/disturbance profiles
        axa.stem(meal_t, meal_g, basefmt=" ", linefmt="k--", markerfmt="ko")
        axa.set_title("(a) Meal / Disturbance Profile", fontweight="bold")
        axa.set_ylabel("CHO (g)", fontweight="bold")
        axa.set_xlim(0, SIM_DURATION_HOURS)
        axa.grid(True, alpha=0.3, linestyle="--")

        # Display names for paper
        _display_name = {
            BB_LABEL: "Ideal patient",
            "Safe-NMPC": "NMPC",
            "Safe-NMPC (Event-Triggered)": "PEPC",
        }

        # (b) BG regulation under disturbances (same style as controller_comparison ax1)
        for name in _included:
            df_hist, _state_df = per_method[name]
            th = (df_hist.index - t0).total_seconds() / 3600.0
            linestyle = "--" if name == "Safe-NMPC (RL-Tuned)" else "-"
            axb.plot(
                th,
                df_hist["BG"].to_numpy(),
                color=colors[name],
                linewidth=2,
                alpha=0.85,
                linestyle=linestyle,
                label=_display_name.get(name, name),
            )

        axb.axhspan(0, 70, alpha=0.10, color="red", label="_nolegend_")
        axb.axhspan(70, 140, alpha=0.12, color="green", label="_nolegend_")
        axb.axhspan(140, 180, alpha=0.12, color="orange", label="_nolegend_")
        axb.axhspan(180, 1000, alpha=0.10, color="red", label="_nolegend_")
        axb.axhline(140, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Target (140 mg/dL)")
        for i, mt in enumerate(meal_times):
            axb.axvline(mt, color=meal_colors[i % len(meal_colors)], linestyle="--", linewidth=2, alpha=0.7, zorder=0)
        axb.set_title("(b) BG Regulation Under Disturbances", fontweight="bold")
        axb.set_ylabel("BG (mg/dL)", fontweight="bold")
        axb.set_ylim(50, 250)
        axb.grid(True, alpha=0.3, linestyle="--")
        # Use one shared legend only (placed above panel a).

        # (c) Insulin delivery profile (rate) (same style as controller_comparison ax8)
        for name in _included:
            df_hist, _state_df = per_method[name]
            th = (df_hist.index - t0).total_seconds() / 3600.0
            insulin_rate = pd.to_numeric(df_hist["insulin"], errors="coerce").fillna(0.0).to_numpy()
            linestyle = "--" if name == "Safe-NMPC (RL-Tuned)" else "-"
            axc.plot(th, insulin_rate, color=colors[name], linewidth=2, alpha=0.85, linestyle=linestyle, label=_display_name.get(name, name))
        for meal_time, meal_amount, meal_color in zip(meal_times, meal_amounts, meal_colors):
            axc.axvline(meal_time, color=meal_color, linestyle="--", linewidth=2, alpha=0.7, zorder=0)
            axc.annotate(
                f"Meal\n{meal_amount}g",
                xy=(meal_time, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -2),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=meal_color, alpha=0.25, edgecolor=meal_color),
                color="black",
                clip_on=False,
            )
        axc.set_title("(c) Insulin Delivery Profile", fontweight="bold")
        axc.set_ylabel("Insulin (U/min)", fontweight="bold")
        axc.grid(True, alpha=0.3, linestyle="--")
        # Use one shared legend only (placed above panel a).

        # (d) BG deviation from target (140 mg/dL)
        _target_bg = 140.0
        for name in _included:
            df_hist, _state_df = per_method[name]
            th = (df_hist.index - t0).total_seconds() / 3600.0
            linestyle = "--" if name == "Safe-NMPC (RL-Tuned)" else "-"
            bg_dev = df_hist["BG"].to_numpy(dtype=float) - _target_bg
            axd.plot(
                th,
                bg_dev,
                color=colors[name],
                linewidth=2,
                alpha=0.85,
                linestyle=linestyle,
                label=_display_name.get(name, name),
            )
        axd.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Target deviation (0)")
        axd.set_title("(d) BG Deviation from Target", fontweight="bold")
        axd.set_ylabel("BG − 140 (mg/dL)", fontweight="bold")
        axd.grid(True, alpha=0.3, linestyle="--")

        # (e) Internal physiological state: insulin action X (x6) overlaid
        for name in _included:
            df_hist, _state_df = per_method[name]
            th = (df_hist.index - t0).total_seconds() / 3600.0
            linestyle = "--" if name == "Safe-NMPC (RL-Tuned)" else "-"
            axe.plot(th, _state_df["x6"].to_numpy(), color=colors[name], linewidth=2, alpha=0.85, linestyle=linestyle, label=_display_name.get(name, name))
        axe.set_title("(e) Internal State: Insulin Action $X$", fontweight="bold")
        axe.set_ylabel("$X$ (a.u.)", fontweight="bold")
        axe.grid(True, alpha=0.3, linestyle="--")

        # (f) Injected insulin (cumulative) overlaid
        for name in _included:
            df_hist, _state_df = per_method[name]
            th = (df_hist.index - t0).total_seconds() / 3600.0
            insulin_rate = pd.to_numeric(df_hist["insulin"], errors="coerce").fillna(0.0).to_numpy()
            cum_insulin = np.cumsum(insulin_rate * dt_min)
            linestyle = "--" if name == "Safe-NMPC (RL-Tuned)" else "-"
            axf.plot(th, cum_insulin, color=colors[name], linewidth=2, alpha=0.85, linestyle=linestyle, label=_display_name.get(name, name))
        axf.set_title("(f) Injected Insulin (Cumulative)", fontweight="bold")
        axf.set_ylabel("Cumulative insulin (U)", fontweight="bold")
        axf.set_xlabel("Time (hours)", fontweight="bold")
        axf.grid(True, alpha=0.3, linestyle="--")
        # Use one shared legend only (placed above panel a).

        # Single-line legend attached to the top of panel (a)
        handles, labels = axb.get_legend_handles_labels()
        if handles:
            axa.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.22),
                ncol=len(labels),
                frameon=True,
                fancybox=True,
                shadow=False,
                borderaxespad=0.0,
                fontsize=11,
            )

        # Remove global title and reduce top whitespace.
        # Keep room only for the legend above (a).
        fig6.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

        fig6_png = os.path.join(comparison_path, "nmpc_multiday_6panel.png")
        fig6.savefig(fig6_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"✓ Saved 6-panel figure: {fig6_png}")
        fig6_pdf = os.path.join(comparison_path, "nmpc_multiday_6panel.pdf")
        fig6.savefig(fig6_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"✓ Saved 6-panel figure (PDF): {fig6_pdf}")
        plt.close(fig6)
except Exception as _e:
    print(f"⚠  Could not generate 6-panel multi-day figure (reason: {_e})")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {comparison_path}")
print("\nFiles generated:")
print("  - controller_comparison_table.csv (Excel-compatible)")
print("  - controller_comparison_table.tex (LaTeX table)")
print("  - controller_comparison_table_formatted.tex (Formatted LaTeX)")
print("  - controller_comparison_summary.json (JSON summary)")
print("  - controller_comparison.png (High-resolution figure)")
print("  - controller_comparison.pdf (Vector figure)")
print("  - [Controller]_detailed.csv (Individual controller data)")
print("\n" + "=" * 80)

