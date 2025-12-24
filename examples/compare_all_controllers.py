#!/usr/bin/env python3
"""
Compare all controllers: NMPC, Tuned NMPC, PID, and Basal-Bolus (original)

This script runs simulations with all controllers and creates a comprehensive
comparison plot showing glucose trajectories and performance metrics.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta

from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim

print("=" * 70)
print("CONTROLLER COMPARISON: NMPC vs Tuned NMPC vs PID vs Basal-Bolus")
print("=" * 70)

# Setup simulation parameters
start_time = datetime(2025, 1, 1, 0, 0, 0)
path = './results'

# Create custom meal scenario
scenario = CustomScenario(
    start_time=start_time,
    scenario=[(7, 45), (12, 70), (18, 80)]  # Meals at 7am, 12pm, 6pm
)

# Create patient, sensor, and pump (same for all controllers)
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')

print(f"\nSimulation Setup:")
print(f"  Patient: {patient.name}")
print(f"  Duration: 1 day")
print(f"  Meals: 7am (45g), 12pm (70g), 6pm (80g)")
print("\n" + "=" * 70)

# Dictionary to store all results
all_results = {}

# ========== 1. BASAL-BOLUS CONTROLLER (Original) ==========
print("\n1. Running Basal-Bolus Controller (Original)...")
env1 = T1DSimEnv(patient, sensor, pump, scenario)
controller1 = BBController(target=140)
sim_obj1 = SimObj(env1, controller1, timedelta(days=1), animate=False, path=path)
results1 = sim(sim_obj1)
all_results['Basal-Bolus'] = results1
print("   ✓ Complete")

# ========== 2. PID CONTROLLER ==========
print("\n2. Running PID Controller...")
env2 = T1DSimEnv(patient, sensor, pump, scenario)
controller2 = PIDController(P=0.001, I=0.00001, D=0.001, target=140)
sim_obj2 = SimObj(env2, controller2, timedelta(days=1), animate=False, path=path)
results2 = sim(sim_obj2)
all_results['PID'] = results2
print("   ✓ Complete")

# ========== 3. NMPC CONTROLLER (Default Parameters) ==========
print("\n3. Running NMPC Controller (Default Parameters)...")
env3 = T1DSimEnv(patient, sensor, pump, scenario)
controller3 = NMPCController(
    target_bg=140.0,
    prediction_horizon=60,
    control_horizon=30,
    sample_time=5.0,
    q_weight=1.0,
    r_weight=0.1,
    bg_min=70.0,
    bg_max=180.0
)
sim_obj3 = SimObj(env3, controller3, timedelta(days=1), animate=False, path=path)
results3 = sim(sim_obj3)
all_results['NMPC (Default)'] = results3
print("   ✓ Complete")

# ========== 4. NMPC CONTROLLER (RL-Tuned Parameters) ==========
print("\n4. Running NMPC Controller (RL-Tuned Parameters)...")
# Load tuned parameters
params_file = 'best_nmpc_params_200ep.json'
if os.path.exists(params_file):
    with open(params_file, 'r') as f:
        tuned_params = json.load(f)
    print(f"   Using parameters from {params_file}")
else:
    print("   Warning: Tuned parameters not found, using defaults")
    tuned_params = {
        'q_weight': 1.0,
        'r_weight': 0.1,
        'prediction_horizon': 60,
        'control_horizon': 30,
        'opt_rate': 1.0
    }

env4 = T1DSimEnv(patient, sensor, pump, scenario)
controller4 = NMPCController(
    target_bg=140.0,
    prediction_horizon=int(tuned_params['prediction_horizon']),
    control_horizon=int(tuned_params['control_horizon']),
    sample_time=5.0,
    q_weight=tuned_params['q_weight'],
    r_weight=tuned_params['r_weight'],
    bg_min=70.0,
    bg_max=180.0
)
controller4.opt_rate = tuned_params['opt_rate']
sim_obj4 = SimObj(env4, controller4, timedelta(days=1), animate=False, path=path)
results4 = sim(sim_obj4)
all_results['NMPC (RL-Tuned)'] = results4
print("   ✓ Complete")

# ========== CALCULATE STATISTICS ==========
print("\n" + "=" * 70)
print("CALCULATING STATISTICS")
print("=" * 70)

stats = {}
for name, results in all_results.items():
    bg_data = results['BG'].values
    cgm_data = results['CGM'].values
    
    stats[name] = {
        'mean_bg': np.mean(bg_data),
        'std_bg': np.std(bg_data),
        'min_bg': np.min(bg_data),
        'max_bg': np.max(bg_data),
        'time_in_range': np.sum((bg_data >= 70) & (bg_data <= 180)) / len(bg_data) * 100,
        'time_below_70': np.sum(bg_data < 70) / len(bg_data) * 100,
        'time_above_180': np.sum(bg_data > 180) / len(bg_data) * 100,
        'mean_risk': np.mean(results['Risk'].values) if 'Risk' in results.columns else 0.0,
        'bg_data': bg_data,
        'cgm_data': cgm_data,
        'time': results.index
    }

# Print statistics table
print("\nPerformance Comparison:")
print(f"{'Controller':<20} {'Mean BG':<12} {'TIR (%)':<12} {'TBR (%)':<12} {'TAR (%)':<12} {'Risk':<12}")
print("-" * 80)
for name, s in stats.items():
    print(f"{name:<20} {s['mean_bg']:>10.2f}  {s['time_in_range']:>10.1f}  "
          f"{s['time_below_70']:>10.1f}  {s['time_above_180']:>10.1f}  {s['mean_risk']:>10.2f}")

# ========== CREATE COMPARISON PLOT ==========
print("\n" + "=" * 70)
print("GENERATING COMPARISON PLOT")
print("=" * 70)

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

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                      left=0.08, right=0.95, top=0.93, bottom=0.08)

# Color scheme
colors = {
    'Basal-Bolus': '#2E86AB',      # Blue
    'PID': '#A23B72',              # Purple
    'NMPC (Default)': '#F18F01',   # Orange
    'NMPC (RL-Tuned)': '#C73E1D'   # Red
}

# ========== Subplot 1: Glucose Trajectories (Full Day) ==========
ax1 = fig.add_subplot(gs[0, :])

for name, s in stats.items():
    # Convert time index to hours for plotting
    time_hours = [(t - s['time'][0]).total_seconds() / 3600 for t in s['time']]
    ax1.plot(time_hours, s['bg_data'], color=colors[name], label=name, linewidth=2, alpha=0.8)

# Add target range shading
ax1.axhspan(70, 180, alpha=0.2, color='green', label='Target Range (70-180 mg/dL)')
ax1.axhline(140, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Target (140 mg/dL)')

ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel('Blood Glucose (mg/dL)', fontweight='bold')
ax1.set_title('(a) Blood Glucose Trajectories Over 24 Hours', fontweight='bold', pad=10)
ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True, ncol=2)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 24)
ax1.set_ylim(0, 300)

# ========== Subplot 2: Time in Range Comparison ==========
ax2 = fig.add_subplot(gs[1, 0])

controllers = list(stats.keys())
tir_values = [stats[c]['time_in_range'] for c in controllers]
colors_list = [colors[c] for c in controllers]

bars = ax2.bar(controllers, tir_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax2.axhline(70, color='green', linestyle='--', linewidth=2, label='Target (70%)')
ax2.set_ylabel('Time in Range (%)', fontweight='bold')
ax2.set_title('(b) Time in Range (70-180 mg/dL)', fontweight='bold', pad=10)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.legend(loc='upper right')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, val in zip(bars, tir_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# ========== Subplot 3: Risk Index Comparison ==========
ax3 = fig.add_subplot(gs[1, 1])

risk_values = [stats[c]['mean_risk'] for c in controllers]
bars = ax3.bar(controllers, risk_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax3.set_ylabel('Mean Risk Index', fontweight='bold')
ax3.set_title('(c) Mean Risk Index (Lower is Better)', fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, val in zip(bars, risk_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# ========== Subplot 4: Hypoglycemia Comparison ==========
ax4 = fig.add_subplot(gs[2, 0])

tbr_values = [stats[c]['time_below_70'] for c in controllers]
bars = ax4.bar(controllers, tbr_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax4.axhline(5, color='red', linestyle='--', linewidth=2, label='Target (<5%)')
ax4.set_ylabel('Time Below Range (%)', fontweight='bold')
ax4.set_title('(d) Time Below Range (<70 mg/dL)', fontweight='bold', pad=10)
ax4.set_ylim(0, max(tbr_values) * 1.2)
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
ax4.legend(loc='upper right')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, val in zip(bars, tbr_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# ========== Subplot 5: Mean BG Comparison ==========
ax5 = fig.add_subplot(gs[2, 1])

mean_bg_values = [stats[c]['mean_bg'] for c in controllers]
bars = ax5.bar(controllers, mean_bg_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax5.axhspan(70, 180, alpha=0.2, color='green', label='Target Range')
ax5.axhline(140, color='black', linestyle='--', linewidth=2, label='Target (140 mg/dL)')
ax5.set_ylabel('Mean Blood Glucose (mg/dL)', fontweight='bold')
ax5.set_title('(e) Mean Blood Glucose', fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
ax5.legend(loc='best')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar, val in zip(bars, mean_bg_values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Add overall title
fig.suptitle('Controller Performance Comparison: NMPC vs Tuned NMPC vs PID vs Basal-Bolus',
            fontsize=16, fontweight='bold', y=0.98)

# Save figure
output_file = 'controller_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nComparison plot saved to: {output_file}")

# Also save as PDF
pdf_file = output_file.replace('.png', '.pdf')
plt.savefig(pdf_file, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"PDF version saved to: {pdf_file}")

plt.close()

print("\n" + "=" * 70)
print("COMPARISON COMPLETE!")
print("=" * 70)

