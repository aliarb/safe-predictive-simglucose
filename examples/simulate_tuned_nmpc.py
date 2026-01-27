#!/usr/bin/env python3
"""
Simulate glucose control using RL-tuned NMPC parameters.

This script loads the tuned parameters from RL training and runs
a full simulation to demonstrate glucose control performance.
"""
import json
import numpy as np
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from datetime import timedelta, datetime
import os

print("=" * 70)
print("GLUCOSE CONTROL SIMULATION WITH RL-TUNED NMPC PARAMETERS")
print("=" * 70)

# Load tuned parameters
params_file = 'results/rl_finetuning/best_params.json'
if not os.path.exists(params_file):
    print(f"\nError: {params_file} not found!")
    print("Please run RL-style finetuning first using:")
    print("  python examples/rl_finetune_nmpc_for_paper.py")
    exit(1)

print(f"\nLoading tuned parameters from {params_file}...")
with open(params_file, 'r') as f:
    tuned_params = json.load(f)

print("\nTuned Parameters:")
for key, value in tuned_params.items():
    print(f"  {key}: {value}")

# Setup simulation parameters
start_time = datetime(2025, 1, 1, 0, 0, 0)
path = './results'

# Create custom meal scenario: meals at 7am, 12pm, 6pm
scenario = CustomScenario(
    start_time=start_time,
    scenario=[(7, 45), (12, 70), (18, 80)]  # (hour, carbs_in_grams)
)

# Create patient, sensor, and pump
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')

print(f"\nSimulation Setup:")
print(f"  Patient: {patient.name}")
print(f"  Simulation duration: 1 day")
print(f"  Meals: 7am (45g), 12pm (70g), 6pm (80g)")
print("\n" + "-" * 70)

# ========== NMPC CONTROLLER WITH TUNED PARAMETERS ==========
print("\nNMPC CONTROLLER (RL-Tuned Parameters)")
print("-" * 70)

# Initialize NMPC controller with tuned parameters (paper configuration)
nmpc_controller = NMPCController(
    target_bg=140.0,
    prediction_horizon=60,
    control_horizon=30,
    sample_time=5.0,
    bg_min=70.0,
    bg_max=180.0,
    r_weight=0.1,
    verbose=False,
    **tuned_params
)

print(f"\nController Configuration:")
print(f"  Target BG: 140.0 mg/dL")
print(f"  Prediction Horizon: 60 minutes")
print(f"  Control Horizon: 30 minutes")
if "q_weight" in tuned_params:
    print(f"  q_weight: {tuned_params['q_weight']:.3f}")
if "q_terminal_weight" in tuned_params:
    print(f"  q_terminal_weight: {tuned_params['q_terminal_weight']:.3f}")
if "r_delta_weight" in tuned_params:
    print(f"  r_delta_weight: {tuned_params['r_delta_weight']:.3f}")
if "hypo_penalty_weight" in tuned_params:
    print(f"  hypo_penalty_weight: {tuned_params['hypo_penalty_weight']:.3f}")
if "hyper_penalty_weight" in tuned_params:
    print(f"  hyper_penalty_weight: {tuned_params['hyper_penalty_weight']:.3f}")
if "barrier_weight" in tuned_params:
    print(f"  barrier_weight: {tuned_params['barrier_weight']:.3f}")
if "zone_transition_smoothness" in tuned_params:
    print(f"  zone_transition_smoothness: {tuned_params['zone_transition_smoothness']:.3f}")
if "insulin_rate_penalty_weight" in tuned_params:
    print(f"  insulin_rate_penalty_weight: {tuned_params['insulin_rate_penalty_weight']:.3f}")
if "delta_u_asymmetry" in tuned_params:
    print(f"  delta_u_asymmetry: {tuned_params['delta_u_asymmetry']:.3f}")

# Run simulation
print("\n" + "-" * 70)
print("Running Simulation...")
print("-" * 70)

env = T1DSimEnv(patient, sensor, pump, scenario)
sim_obj = SimObj(env, nmpc_controller, timedelta(days=1), animate=False, path=path)
results = sim(sim_obj)

# Calculate statistics
bg_data = results['BG'].values
cgm_data = results['CGM'].values
insulin_data = results['insulin'].values if 'insulin' in results.columns else np.zeros(len(bg_data))

print("\n" + "=" * 70)
print("SIMULATION RESULTS")
print("=" * 70)

print(f"\nGlucose Control Performance:")
print(f"  Mean BG: {np.mean(bg_data):.2f} mg/dL")
print(f"  Std BG: {np.std(bg_data):.2f} mg/dL")
print(f"  Min BG: {np.min(bg_data):.2f} mg/dL")
print(f"  Max BG: {np.max(bg_data):.2f} mg/dL")
print(f"  Mean CGM: {np.mean(cgm_data):.2f} mg/dL")

# Time in range analysis
time_in_range_70_180 = np.sum((bg_data >= 70) & (bg_data <= 180)) / len(bg_data) * 100
time_below_70 = np.sum(bg_data < 70) / len(bg_data) * 100
time_above_180 = np.sum(bg_data > 180) / len(bg_data) * 100
time_in_range_70_140 = np.sum((bg_data >= 70) & (bg_data <= 140)) / len(bg_data) * 100
time_in_range_140_180 = np.sum((bg_data >= 140) & (bg_data <= 180)) / len(bg_data) * 100
time_below_54 = np.sum(bg_data < 54) / len(bg_data) * 100  # Level 2 hypoglycemia
time_above_250 = np.sum(bg_data > 250) / len(bg_data) * 100  # Level 2 hyperglycemia

print(f"\nTime in Range Analysis:")
print(f"  Time in Range (70-180 mg/dL): {time_in_range_70_180:.1f}%")
print(f"  Time Below Range (<70 mg/dL): {time_below_70:.1f}%")
print(f"  Time Above Range (>180 mg/dL): {time_above_180:.1f}%")
print(f"  Time in Tight Range (70-140 mg/dL): {time_in_range_70_140:.1f}%")
print(f"  Time in Upper Range (140-180 mg/dL): {time_in_range_140_180:.1f}%")
print(f"  Level 2 Hypoglycemia (<54 mg/dL): {time_below_54:.1f}%")
print(f"  Level 2 Hyperglycemia (>250 mg/dL): {time_above_250:.1f}%")

# Risk analysis
risk_data = results['Risk'].values if 'Risk' in results.columns else np.zeros(len(bg_data))
mean_risk = np.mean(risk_data)
max_risk = np.max(risk_data)
min_risk = np.min(risk_data)

print(f"\nRisk Analysis:")
print(f"  Mean Risk Index: {mean_risk:.4f}")
print(f"  Max Risk Index: {max_risk:.4f}")
print(f"  Min Risk Index: {min_risk:.4f}")

# Insulin usage
print(f"\nInsulin Delivery:")
print(f"  Mean Insulin Rate: {np.mean(insulin_data):.3f} U/min")
print(f"  Max Insulin Rate: {np.max(insulin_data):.3f} U/min")
print(f"  Total Insulin (24h): {np.sum(insulin_data) * 5 / 60:.2f} U")  # Convert to units

# Show sample data points
print(f"\nFirst 10 time steps:")
print(results[['BG', 'CGM', 'CHO', 'insulin']].head(10).to_string())
print(f"\nLast 10 time steps:")
print(results[['BG', 'CGM', 'CHO', 'insulin']].tail(10).to_string())

# Performance assessment
print("\n" + "=" * 70)
print("PERFORMANCE ASSESSMENT")
print("=" * 70)

# Time in range assessment
if time_in_range_70_180 >= 70:
    print("  ✅ Excellent: Time in range >= 70%")
elif time_in_range_70_180 >= 50:
    print("  ⚠️  Good: Time in range >= 50%")
else:
    print("  ❌ Needs improvement: Time in range < 50%")

# Hypoglycemia assessment
if time_below_70 < 5:
    print("  ✅ Excellent: Hypoglycemia < 5%")
elif time_below_70 < 10:
    print("  ⚠️  Acceptable: Hypoglycemia < 10%")
else:
    print("  ❌ High risk: Hypoglycemia >= 10%")

# Mean BG assessment
if 70 <= np.mean(bg_data) <= 180:
    print(f"  ✅ Mean BG in target range: {np.mean(bg_data):.1f} mg/dL")
else:
    print(f"  ⚠️  Mean BG outside target: {np.mean(bg_data):.1f} mg/dL")

# Risk assessment
if mean_risk < 1.0:
    print(f"  ✅ Low risk: Mean risk index {mean_risk:.2f}")
elif mean_risk < 5.0:
    print(f"  ⚠️  Moderate risk: Mean risk index {mean_risk:.2f}")
else:
    print(f"  ❌ High risk: Mean risk index {mean_risk:.2f}")

print("\n" + "=" * 70)
print("Simulation complete! Results saved to ./results/")
print("=" * 70)

# Additional statistics for paper
print("\n" + "=" * 70)
print("ADDITIONAL STATISTICS FOR ANALYSIS")
print("=" * 70)
print(f"  Coefficient of Variation (CV): {(np.std(bg_data) / np.mean(bg_data) * 100):.1f}%")
print(f"  Glucose Management Indicator (GMI): {3.31 + 0.02392 * np.mean(cgm_data):.1f}%")
print(f"  Time in Range (TIR) 70-180: {time_in_range_70_180:.1f}%")
print(f"  Time Below Range (TBR) <70: {time_below_70:.1f}%")
print(f"  Time Above Range (TAR) >180: {time_above_180:.1f}%")
print("=" * 70)

