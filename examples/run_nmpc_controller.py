#!/usr/bin/env python3
"""
Example script demonstrating the use of NMPC controller for glucose control.

Note: This example uses a placeholder implementation. You'll need to convert
your MATLAB NMPC solver code to Python to get full functionality.
"""
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from datetime import timedelta, datetime
import numpy as np

print("=" * 70)
print("NMPC CONTROLLER DEMONSTRATION")
print("=" * 70)

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

# ========== NMPC CONTROLLER ==========
print("\nNMPC CONTROLLER")
print("-" * 70)

# Initialize NMPC controller with custom parameters
nmpc_controller = NMPCController(
    target_bg=140.0,           # Target blood glucose (mg/dL)
    prediction_horizon=60,      # Prediction horizon (minutes)
    control_horizon=30,          # Control horizon (minutes)
    sample_time=5.0,            # Controller sample time (minutes)
    q_weight=1.0,               # Glucose tracking weight
    r_weight=0.1,               # Insulin cost weight
    bg_min=70.0,                # Minimum safe BG (mg/dL)
    bg_max=180.0                # Maximum safe BG (mg/dL)
)

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

print(f"\nTime in Range (70-180 mg/dL): {time_in_range_70_180:.1f}%")
print(f"Time Below Range (<70 mg/dL): {time_below_70:.1f}%")
print(f"Time Above Range (>180 mg/dL): {time_above_180:.1f}%")

print(f"\nFirst 10 time steps:")
print(results[['BG', 'CGM', 'CHO', 'insulin']].head(10).to_string())
print(f"\nLast 10 time steps:")
print(results[['BG', 'CGM', 'CHO', 'insulin']].tail(10).to_string())

print("\n" + "=" * 70)
print("Simulation complete! Results saved to ./results/")
print("=" * 70)
print("\nNOTE: This is using a placeholder NMPC implementation.")
print("To use your actual NMPC solver, convert your MATLAB code and")
print("implement the methods in NMPCController class:")
print("  - _solve_nmpc()")
print("  - _predict_glucose()")
print("  - _compute_objective()")
print("  - _compute_constraints()")
print("=" * 70)

