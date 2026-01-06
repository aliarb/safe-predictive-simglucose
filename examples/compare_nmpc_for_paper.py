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
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim

print("=" * 80)
print("NMPC WITH CONSTRAINTS vs OTHER CONTROLLERS - PAPER COMPARISON")
print("=" * 80)

# Setup simulation parameters
start_time = datetime(2025, 1, 1, 0, 0, 0)
base_path = './results'
comparison_path = './results/paper_comparison'
os.makedirs(comparison_path, exist_ok=True)

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
print(f"  Results will be saved to: {comparison_path}")
print("\n" + "=" * 80)

# Dictionary to store all results
all_results = {}
all_stats = {}

# ========== 1. BASAL-BOLUS CONTROLLER ==========
print("\n[1/4] Running Basal-Bolus Controller...")
env1 = T1DSimEnv(patient, sensor, pump, scenario)
controller1 = BBController(target=140)
sim_obj1 = SimObj(env1, controller1, timedelta(days=1), animate=False, path=comparison_path)
results1 = sim(sim_obj1)
all_results['Basal-Bolus'] = results1
print("   ✓ Complete")

# ========== 2. PID CONTROLLER ==========
print("\n[2/4] Running PID Controller...")
env2 = T1DSimEnv(patient, sensor, pump, scenario)
controller2 = PIDController(P=0.001, I=0.00001, D=0.001, target=140)
sim_obj2 = SimObj(env2, controller2, timedelta(days=1), animate=False, path=comparison_path)
results2 = sim(sim_obj2)
all_results['PID'] = results2
print("   ✓ Complete")

# ========== 3. NMPC CONTROLLER (WITH CONSTRAINTS) ==========
print("\n[3/4] Running NMPC Controller with Constraints...")
env3 = T1DSimEnv(patient, sensor, pump, scenario)
controller3 = NMPCController(
    target_bg=140.0,
    prediction_horizon=60,
    control_horizon=30,
    sample_time=5.0,
    q_weight=1.0,
    r_weight=0.1,
    bg_min=70.0,
    bg_max=180.0,
    barrier_weight=10.0  # Constraint enforcement weight
)
sim_obj3 = SimObj(env3, controller3, timedelta(days=1), animate=False, path=comparison_path)
results3 = sim(sim_obj3)
all_results['NMPC (Constrained)'] = results3
print("   ✓ Complete")
print(f"      Constraint bounds: [{controller3.bg_min}, {controller3.bg_max}] mg/dL")
print(f"      Barrier weight: {controller3.barrier_weight}")

# ========== 4. NMPC CONTROLLER (HIGH CONSTRAINT WEIGHT) ==========
print("\n[4/4] Running NMPC Controller with Strict Constraints...")
env4 = T1DSimEnv(patient, sensor, pump, scenario)
controller4 = NMPCController(
    target_bg=140.0,
    prediction_horizon=60,
    control_horizon=30,
    sample_time=5.0,
    q_weight=1.0,
    r_weight=0.1,
    bg_min=70.0,
    bg_max=180.0,
    barrier_weight=50.0  # Higher constraint enforcement
)
sim_obj4 = SimObj(env4, controller4, timedelta(days=1), animate=False, path=comparison_path)
results4 = sim(sim_obj4)
all_results['NMPC (Strict Constraints)'] = results4
print("   ✓ Complete")
print(f"      Constraint bounds: [{controller4.bg_min}, {controller4.bg_max}] mg/dL")
print(f"      Barrier weight: {controller4.barrier_weight}")

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
print(stats_df[['Mean BG (mg/dL)', 'Time in Range 70-180 (%)', 
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
    'Basal-Bolus': '#2E86AB',
    'PID': '#A23B72',
    'NMPC (Constrained)': '#F18F01',
    'NMPC (Strict Constraints)': '#C73E1D'
}

# Create comprehensive figure with more subplots for insulin
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3,
                      left=0.08, right=0.95, top=0.93, bottom=0.08)

# Subplot 1: Glucose Trajectories
ax1 = fig.add_subplot(gs[0, :])
for name, stats in all_stats.items():
    time_hours = [(t - stats['time'][0]).total_seconds() / 3600 for t in stats['time']]
    ax1.plot(time_hours, stats['bg_data'], color=colors[name], label=name, linewidth=2, alpha=0.8)

ax1.axhspan(70, 180, alpha=0.2, color='green', label='Target Range (70-180 mg/dL)')
ax1.axhline(140, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Target (140 mg/dL)')
ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel('Blood Glucose (mg/dL)', fontweight='bold')
ax1.set_title('(a) Blood Glucose Trajectories Over 24 Hours', fontweight='bold', pad=10)
ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True, ncol=2)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 24)
ax1.set_ylim(0, 300)

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
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, tir_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# Subplot 3: Constraint Violations
ax3 = fig.add_subplot(gs[1, 1])
violation_values = [all_stats[c]['Constraint Violation Rate (%)'] for c in controllers]
bars = ax3.bar(controllers, violation_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax3.axhline(5, color='red', linestyle='--', linewidth=2, label='Target (<5%)')
ax3.set_ylabel('Constraint Violation Rate (%)', fontweight='bold')
ax3.set_title('(c) Constraint Violation Rate', fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
ax3.legend(loc='upper right')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, violation_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

# Subplot 4: Risk Index
ax4 = fig.add_subplot(gs[2, 0])
risk_values = [all_stats[c]['Mean Risk Index'] for c in controllers]
bars = ax4.bar(controllers, risk_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax4.set_ylabel('Mean Risk Index', fontweight='bold')
ax4.set_title('(d) Mean Risk Index (Lower is Better)', fontweight='bold', pad=10)
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, risk_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# Subplot 5: Mean Insulin Rate
ax5 = fig.add_subplot(gs[2, 0])
mean_insulin_values = [all_stats[c]['Mean Insulin (U/min)'] for c in controllers]
bars = ax5.bar(controllers, mean_insulin_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax5.set_ylabel('Mean Insulin Rate (U/min)', fontweight='bold')
ax5.set_title('(e) Mean Insulin Injection Rate', fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, mean_insulin_values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Subplot 6: Insulin per Day
ax6 = fig.add_subplot(gs[2, 1])
insulin_per_day_values = [all_stats[c]['Insulin per Day (U/day)'] for c in controllers]
bars = ax6.bar(controllers, insulin_per_day_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax6.set_ylabel('Insulin per Day (U/day)', fontweight='bold')
ax6.set_title('(f) Total Daily Insulin', fontweight='bold', pad=10)
ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, insulin_per_day_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Subplot 7: Mean BG
ax7 = fig.add_subplot(gs[3, 0])
mean_bg_values = [all_stats[c]['Mean BG (mg/dL)'] for c in controllers]
bars = ax7.bar(controllers, mean_bg_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
ax7.axhspan(70, 180, alpha=0.2, color='green', label='Target Range')
ax7.axhline(140, color='black', linestyle='--', linewidth=2, label='Target (140 mg/dL)')
ax7.set_ylabel('Mean Blood Glucose (mg/dL)', fontweight='bold')
ax7.set_title('(g) Mean Blood Glucose', fontweight='bold', pad=10)
ax7.grid(True, alpha=0.3, linestyle='--', axis='y')
ax7.legend(loc='best')
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, mean_bg_values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Subplot 8: Insulin Trajectories
ax8 = fig.add_subplot(gs[3, 1])
for name, stats in all_stats.items():
    time_hours = [(t - stats['time'][0]).total_seconds() / 3600 for t in stats['time']]
    ax8.plot(time_hours, stats['insulin_data'], color=colors[name], label=name, linewidth=2, alpha=0.8)
ax8.set_xlabel('Time (hours)', fontweight='bold')
ax8.set_ylabel('Insulin Rate (U/min)', fontweight='bold')
ax8.set_title('(h) Insulin Injection Rate Over Time', fontweight='bold', pad=10)
ax8.legend(loc='best', frameon=True, fancybox=True, shadow=True)
ax8.grid(True, alpha=0.3, linestyle='--')
ax8.set_xlim(0, 24)

# Overall title
fig.suptitle('Controller Performance Comparison: NMPC with Constraints vs Baseline Methods',
            fontsize=16, fontweight='bold', y=0.98)

# Save figures
png_file = os.path.join(comparison_path, 'controller_comparison.png')
plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved PNG figure: {png_file}")

pdf_file = os.path.join(comparison_path, 'controller_comparison.pdf')
plt.savefig(pdf_file, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved PDF figure: {pdf_file}")

plt.close()

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

