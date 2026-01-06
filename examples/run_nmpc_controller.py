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

# ========== NMPC CONTROLLER WITH CONSTRAINTS ==========
print("\nNMPC CONTROLLER WITH CONSTRAINTS")
print("-" * 70)

# Initialize NMPC controller with explicit constraint configuration
nmpc_controller = NMPCController(
    target_bg=140.0,           # Target blood glucose (mg/dL)
    prediction_horizon=60,     # Prediction horizon (minutes)
    control_horizon=30,        # Control horizon (minutes)
    sample_time=5.0,           # Controller sample time (minutes)
    q_weight=1.0,              # Glucose tracking weight
    r_weight=0.1,              # Insulin cost weight
    bg_min=70.0,               # Minimum safe BG constraint (mg/dL)
    bg_max=180.0,              # Maximum safe BG constraint (mg/dL)
    barrier_weight=100.0        # Constraint enforcement weight (higher = stricter)
)

# Display constraint configuration
print("\nConstraint Configuration:")
print(f"  Glucose Safety Bounds: [{nmpc_controller.bg_min}, {nmpc_controller.bg_max}] mg/dL")
print(f"  Insulin Bounds: [{nmpc_controller.insulin_min}, {nmpc_controller.insulin_max}] U/min")
print(f"  Barrier Function Weight: {nmpc_controller.barrier_weight}")
print(f"  Constraint Type: Control Barrier Function (penalty-based)")
print("\nConstraints enforced through:")
print("  - Glucose barrier function: J_G(t) = G(t) - G_max if G > G_max")
print("  - Glucose barrier function: J_G(t) = G(t) - G_min if G < G_min")
print("  - Insulin saturation: insulin clipped to [0, 10] U/min")

env = T1DSimEnv(patient, sensor, pump, scenario)
sim_obj = SimObj(env, nmpc_controller, timedelta(days=1), animate=False, path=path)
results = sim(sim_obj)

# Calculate statistics
bg_data = results['BG'].values
cgm_data = results['CGM'].values

print(f"\n" + "=" * 70)
print("SIMULATION RESULTS")
print("=" * 70)

print(f"\n📊 Blood Glucose Statistics:")
print(f"  Mean BG: {np.mean(bg_data):.2f} mg/dL")
print(f"  Std BG: {np.std(bg_data):.2f} mg/dL")
print(f"  Min BG: {np.min(bg_data):.2f} mg/dL")
print(f"  Max BG: {np.max(bg_data):.2f} mg/dL")
print(f"  Mean CGM: {np.mean(cgm_data):.2f} mg/dL")

# Time in range analysis
time_in_range_70_180 = np.sum((bg_data >= 70) & (bg_data <= 180)) / len(bg_data) * 100
time_below_70 = np.sum(bg_data < 70) / len(bg_data) * 100
time_above_180 = np.sum(bg_data > 180) / len(bg_data) * 100

print(f"\n📈 Time in Range Analysis:")
print(f"  Time in Range ({nmpc_controller.bg_min}-{nmpc_controller.bg_max} mg/dL): {time_in_range_70_180:.1f}%")
print(f"  Time Below Range (<{nmpc_controller.bg_min} mg/dL): {time_below_70:.1f}%")
print(f"  Time Above Range (>{nmpc_controller.bg_max} mg/dL): {time_above_180:.1f}%")

# Display sample data
print(f"\n📋 Sample Data (First 10 time steps):")
print(results[['BG', 'CGM', 'CHO', 'insulin']].head(10).to_string())
print(f"\n📋 Sample Data (Last 10 time steps):")
print(results[['BG', 'CGM', 'CHO', 'insulin']].tail(10).to_string())

# Constraint violation analysis
constraint_violations_below = np.sum(bg_data < nmpc_controller.bg_min)
constraint_violations_above = np.sum(bg_data > nmpc_controller.bg_max)
total_violations = constraint_violations_below + constraint_violations_above
violation_percentage = (total_violations / len(bg_data)) * 100

print("\n" + "=" * 70)
print("CONSTRAINT VIOLATION ANALYSIS")
print("=" * 70)
print(f"  Constraint violations below {nmpc_controller.bg_min} mg/dL: {constraint_violations_below} ({constraint_violations_below/len(bg_data)*100:.2f}%)")
print(f"  Constraint violations above {nmpc_controller.bg_max} mg/dL: {constraint_violations_above} ({constraint_violations_above/len(bg_data)*100:.2f}%)")
print(f"  Total constraint violations: {total_violations} ({violation_percentage:.2f}%)")
if violation_percentage < 5:
    print("  ✅ Excellent: Constraint violations < 5%")
elif violation_percentage < 10:
    print("  ⚠️  Good: Constraint violations < 10%")
else:
    print("  ❌ High violation rate: Consider increasing barrier_weight")

print("\n" + "=" * 70)
print("Simulation complete! Results saved to ./results/")
print("=" * 70)

