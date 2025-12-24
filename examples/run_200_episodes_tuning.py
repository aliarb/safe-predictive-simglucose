#!/usr/bin/env python3
"""
Run RL tuning for 200 episodes and then test the tuned NMPC controller.

This script:
1. Trains RL agent for 200 episodes to find optimal NMPC parameters
2. Generates publication-quality plots
3. Tests the tuned controller on a full simulation
"""
import sys
sys.path.insert(0, '.')

from tune_nmpc_with_rl import (
    NMPCTuningEnv, SimpleRLAgent, train_rl_agent, 
    plot_training_history, save_training_history
)
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from datetime import datetime, timedelta
import numpy as np
import json

print("=" * 70)
print("RL-BASED NMPC PARAMETER TUNING (200 EPISODES)")
print("=" * 70)

print("\n" + "=" * 70)
print("REWARD FUNCTION EXPLANATION")
print("=" * 70)
print("""
The reward function evaluates glucose control performance at each time step:

1. TIME IN RANGE BONUS:
   - BG in 70-180 mg/dL: +1.0 (optimal range)
   - BG in 60-70 or 180-250 mg/dL: +0.5 (acceptable, near target)
   - BG outside these ranges: -1.0 (poor control)

2. RISK INDEX PENALTY:
   - Subtracts risk * 0.1 from reward
   - Risk index measures danger of hypo/hyperglycemia (lower is better)
   - Higher risk = more penalty

3. SEVERE PENALTIES:
   - BG < 50 mg/dL: -5.0 (severe hypoglycemia - dangerous)
   - BG > 300 mg/dL: -5.0 (severe hyperglycemia - dangerous)

4. REWARD NORMALIZATION:
   - Final reward clipped to [-10, 10] range

The cumulative reward over an episode reflects overall glucose control quality.
Higher cumulative reward = better glucose control performance.
""")
print("=" * 70)

# Create meal scenario
start_time = datetime(2025, 1, 1, 0, 0, 0)
scenario = CustomScenario(
    start_time=start_time,
    scenario=[(7, 45), (12, 70), (18, 80)]  # Meals at 7am, 12pm, 6pm
)

print("\n" + "=" * 70)
print("PHASE 1: RL TRAINING (200 EPISODES)")
print("=" * 70)

# Create environment
env = NMPCTuningEnv(
    patient_name='adolescent#001',
    custom_scenario=scenario,
    episode_length=288,  # Full day (1 day with 5-min sampling = 288 steps)
    seed=42
)

# Create RL agent
agent = SimpleRLAgent(learning_rate=0.01)

# Train agent for 200 episodes
num_episodes = 200
print(f"\nTraining for {num_episodes} episodes...")
print("This may take a while. Progress will be shown every 10 episodes.\n")

history, best_params = train_rl_agent(env, agent, num_episodes=num_episodes)

# Save training history
save_training_history(history)

# Plot results (publication quality)
print("\nGenerating publication-quality plots...")
plot_training_history(history, save_path='nmpc_rl_tuning_results_200ep.png', dpi=300)

# Save best parameters
with open('best_nmpc_params_200ep.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print("Best parameters saved to best_nmpc_params_200ep.json")

print("\n" + "=" * 70)
print("PHASE 2: TESTING TUNED CONTROLLER")
print("=" * 70)

# Create test environment
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
test_env = T1DSimEnv(patient, sensor, pump, scenario)

# Create controller with best parameters
print(f"\nUsing tuned parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

best_controller = NMPCController(
    target_bg=140.0,
    prediction_horizon=int(best_params['prediction_horizon']),
    control_horizon=int(best_params['control_horizon']),
    sample_time=5.0,
    q_weight=best_params['q_weight'],
    r_weight=best_params['r_weight'],
    bg_min=70.0,
    bg_max=180.0
)
best_controller.opt_rate = best_params['opt_rate']

# Run simulation
print("\nRunning simulation with tuned NMPC controller...")
sim_obj = SimObj(test_env, best_controller, timedelta(days=1), animate=False, path='./results')
results = sim(sim_obj)

# Calculate statistics
bg_data = results['BG'].values
cgm_data = results['CGM'].values
time_in_range = np.sum((bg_data >= 70) & (bg_data <= 180)) / len(bg_data) * 100
time_below_70 = np.sum(bg_data < 70) / len(bg_data) * 100
time_above_180 = np.sum(bg_data > 180) / len(bg_data) * 100
time_in_range_70_140 = np.sum((bg_data >= 70) & (bg_data <= 140)) / len(bg_data) * 100

# Risk analysis
risk_data = results['Risk'].values if 'Risk' in results.columns else np.zeros(len(bg_data))
mean_risk = np.mean(risk_data)

print("\n" + "=" * 70)
print("SIMULATION RESULTS WITH TUNED PARAMETERS")
print("=" * 70)
print(f"\nGlucose Control Performance:")
print(f"  Mean BG: {np.mean(bg_data):.2f} mg/dL")
print(f"  Std BG: {np.std(bg_data):.2f} mg/dL")
print(f"  Min BG: {np.min(bg_data):.2f} mg/dL")
print(f"  Max BG: {np.max(bg_data):.2f} mg/dL")
print(f"  Mean CGM: {np.mean(cgm_data):.2f} mg/dL")

print(f"\nTime in Range Analysis:")
print(f"  Time in Range (70-180 mg/dL): {time_in_range:.1f}%")
print(f"  Time Below Range (<70 mg/dL): {time_below_70:.1f}%")
print(f"  Time Above Range (>180 mg/dL): {time_above_180:.1f}%")
print(f"  Time in Tight Range (70-140 mg/dL): {time_in_range_70_140:.1f}%")

print(f"\nRisk Analysis:")
print(f"  Mean Risk Index: {mean_risk:.4f}")

# Performance assessment
print("\n" + "=" * 70)
print("PERFORMANCE ASSESSMENT")
print("=" * 70)
if time_in_range >= 70:
    print("  ✅ Excellent: Time in range >= 70%")
elif time_in_range >= 50:
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

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"Plots saved to: nmpc_rl_tuning_results_200ep.png")
print(f"Best parameters saved to: best_nmpc_params_200ep.json")
print(f"Simulation results saved to: ./results/")
print("=" * 70)

