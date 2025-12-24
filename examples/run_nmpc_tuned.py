#!/usr/bin/env python3
"""
Run NMPC controller with tuned parameters.

This script runs the NMPC controller with optimized parameters.
If best_nmpc_params.json exists (from RL tuning), it uses those.
Otherwise, it uses manually tuned reasonable defaults.
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
print("NMPC CONTROLLER WITH TUNED PARAMETERS")
print("=" * 70)

# Try to load tuned parameters from RL training
params_file = 'best_nmpc_params.json'
if os.path.exists(params_file):
    print(f"\nLoading tuned parameters from {params_file}...")
    with open(params_file, 'r') as f:
        tuned_params = json.load(f)
    print("Tuned Parameters:")
    for key, value in tuned_params.items():
        print(f"  {key}: {value}")
else:
    print("\nNo tuned parameters found. Using manually tuned defaults...")
    # Manually tuned reasonable parameters based on initial testing
    tuned_params = {
        'q_weight': 2.0,           # Increased tracking weight
        'r_weight': 0.05,          # Reduced insulin cost (less conservative)
        'prediction_horizon': 90,  # Longer prediction horizon
        'control_horizon': 45,     # Longer control horizon
        'opt_rate': 0.8            # Moderate learning rate
    }
    print("Default Parameters:")
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

print(f"\nPatient: {patient.name}")
print(f"Simulation duration: 1 day")
print(f"Meals: 7am (45g), 12pm (70g), 6pm (80g)")
print("\n" + "-" * 70)

# ========== NMPC CONTROLLER WITH TUNED PARAMETERS ==========
print("\nNMPC CONTROLLER (Tuned Parameters)")
print("-" * 70)

# Initialize NMPC controller with tuned parameters
nmpc_controller = NMPCController(
    target_bg=140.0,
    prediction_horizon=int(tuned_params['prediction_horizon']),
    control_horizon=int(tuned_params['control_horizon']),
    sample_time=5.0,
    q_weight=tuned_params['q_weight'],
    r_weight=tuned_params['r_weight'],
    bg_min=70.0,
    bg_max=180.0
)
# Set optimization rate
nmpc_controller.opt_rate = tuned_params['opt_rate']

print(f"\nController Configuration:")
print(f"  Prediction Horizon: {tuned_params['prediction_horizon']} minutes")
print(f"  Control Horizon: {tuned_params['control_horizon']} minutes")
print(f"  Q Weight (tracking): {tuned_params['q_weight']}")
print(f"  R Weight (control cost): {tuned_params['r_weight']}")
print(f"  Optimization Rate: {tuned_params['opt_rate']}")

env = T1DSimEnv(patient, sensor, pump, scenario)
sim_obj = SimObj(env, nmpc_controller, timedelta(days=1), animate=False, path=path)
results = sim(sim_obj)

# Calculate statistics
bg_data = results['BG'].values
cgm_data = results['CGM'].values

print(f"\nResults Summary:")
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

print(f"\nTime in Range Analysis:")
print(f"  Time in Range (70-180 mg/dL): {time_in_range_70_180:.1f}%")
print(f"  Time Below Range (<70 mg/dL): {time_below_70:.1f}%")
print(f"  Time Above Range (>180 mg/dL): {time_above_180:.1f}%")
print(f"  Time in Tight Range (70-140 mg/dL): {time_in_range_70_140:.1f}%")
print(f"  Time in Upper Range (140-180 mg/dL): {time_in_range_140_180:.1f}%")

# Risk analysis
risk_data = results['Risk'].values if 'Risk' in results.columns else np.zeros(len(bg_data))
mean_risk = np.mean(risk_data)
print(f"\nRisk Analysis:")
print(f"  Mean Risk Index: {mean_risk:.4f}")

# Show first 10 and last 10 data points
print(f"\nFirst 10 time steps:")
print(results[['BG', 'CGM', 'CHO', 'insulin']].head(10).to_string())
print(f"\nLast 10 time steps:")
print(results[['BG', 'CGM', 'CHO', 'insulin']].tail(10).to_string())

print("\n" + "=" * 70)
print("Simulation complete! Results saved to ./results/")
print("=" * 70)

# Performance assessment
print("\nPerformance Assessment:")
if time_in_range_70_180 >= 70:
    print("  ✅ Excellent: Time in range >= 70%")
elif time_in_range_70_180 >= 50:
    print("  ⚠️  Good: Time in range >= 50%")
else:
    print("  ❌ Needs improvement: Time in range < 50%")

if time_below_70 < 5:
    print("  ✅ Excellent: Hypoglycemia < 5%")
elif time_below_70 < 10:
    print("  ⚠️  Acceptable: Hypoglycemia < 10%")
else:
    print("  ❌ High risk: Hypoglycemia >= 10%")

if np.mean(bg_data) >= 70 and np.mean(bg_data) <= 180:
    print(f"  ✅ Mean BG in target range: {np.mean(bg_data):.1f} mg/dL")
else:
    print(f"  ⚠️  Mean BG outside target: {np.mean(bg_data):.1f} mg/dL")

print("=" * 70)

