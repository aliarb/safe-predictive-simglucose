#!/usr/bin/env python3
"""
Demo script to run controllers and show results
"""
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

print("=" * 70)
print("SIMGLUCOSE CONTROLLER DEMONSTRATION")
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

# ========== BASAL-BOLUS CONTROLLER ==========
print("\n1. BASAL-BOLUS CONTROLLER")
print("-" * 70)

env1 = T1DSimEnv(patient, sensor, pump, scenario)
controller1 = BBController(target=140)
sim_obj1 = SimObj(env1, controller1, timedelta(days=1), animate=False, path=path)
results1 = sim(sim_obj1)

# Calculate statistics
bg_data = results1['BG'].values
cgm_data = results1['CGM'].values

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

# Show first 10 and last 10 data points
print(f"\nFirst 10 time steps:")
print(results1[['BG', 'CGM', 'CHO', 'insulin']].head(10).to_string())
print(f"\nLast 10 time steps:")
print(results1[['BG', 'CGM', 'CHO', 'insulin']].tail(10).to_string())

# ========== PID CONTROLLER ==========
print("\n\n" + "=" * 70)
print("2. PID CONTROLLER")
print("-" * 70)

env2 = T1DSimEnv(patient, sensor, pump, scenario)
controller2 = PIDController(P=0.001, I=0.00001, D=0.001, target=140)
sim_obj2 = SimObj(env2, controller2, timedelta(days=1), animate=False, path=path)
results2 = sim(sim_obj2)

# Calculate statistics
bg_data2 = results2['BG'].values
cgm_data2 = results2['CGM'].values

print(f"\nResults Summary:")
print(f"  Mean BG: {np.mean(bg_data2):.2f} mg/dL")
print(f"  Std BG: {np.std(bg_data2):.2f} mg/dL")
print(f"  Min BG: {np.min(bg_data2):.2f} mg/dL")
print(f"  Max BG: {np.max(bg_data2):.2f} mg/dL")
print(f"  Mean CGM: {np.mean(cgm_data2):.2f} mg/dL")

# Time in range analysis
time_in_range_70_180_2 = np.sum((bg_data2 >= 70) & (bg_data2 <= 180)) / len(bg_data2) * 100
time_below_70_2 = np.sum(bg_data2 < 70) / len(bg_data2) * 100
time_above_180_2 = np.sum(bg_data2 > 180) / len(bg_data2) * 100

print(f"\nTime in Range (70-180 mg/dL): {time_in_range_70_180_2:.1f}%")
print(f"Time Below Range (<70 mg/dL): {time_below_70_2:.1f}%")
print(f"Time Above Range (>180 mg/dL): {time_above_180_2:.1f}%")

print(f"\nFirst 10 time steps:")
print(results2[['BG', 'CGM', 'CHO', 'insulin']].head(10).to_string())
print(f"\nLast 10 time steps:")
print(results2[['BG', 'CGM', 'CHO', 'insulin']].tail(10).to_string())

# ========== COMPARISON ==========
print("\n\n" + "=" * 70)
print("COMPARISON")
print("-" * 70)
print(f"{'Metric':<30} {'Basal-Bolus':<20} {'PID':<20}")
print("-" * 70)
print(f"{'Mean BG (mg/dL)':<30} {np.mean(bg_data):<20.2f} {np.mean(bg_data2):<20.2f}")
print(f"{'Std BG (mg/dL)':<30} {np.std(bg_data):<20.2f} {np.std(bg_data2):<20.2f}")
print(f"{'Time in Range (%)':<30} {time_in_range_70_180:<20.1f} {time_in_range_70_180_2:<20.1f}")
print(f"{'Time Below Range (%)':<30} {time_below_70:<20.1f} {time_below_70_2:<20.1f}")
print(f"{'Time Above Range (%)':<30} {time_above_180:<20.1f} {time_above_180_2:<20.1f}")

print("\n" + "=" * 70)
print("Simulation complete! Results saved to ./results/")
print("=" * 70)

