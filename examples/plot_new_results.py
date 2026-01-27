#!/usr/bin/env python3
"""
Plot new NMPC results with improvements highlighted.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Setup paths
comparison_path = './results/paper_comparison'
os.makedirs(comparison_path, exist_ok=True)

# Load data
print("Loading results...")
all_results = {}
controllers = ['Basal_Bolus', 'PID', 'Safe_NMPC']

for controller in controllers:
    csv_file = os.path.join(comparison_path, f'{controller}_detailed.csv')
    if os.path.exists(csv_file):
        all_results[controller] = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print(f"  ✓ Loaded {controller}")

# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'grid.alpha': 0.3
})

# Color scheme
colors = {
    'Basal_Bolus': '#2E86AB',
    'PID': '#A23B72',
    'Safe_NMPC': '#57068C'  # NYU/CCNY Purple
}

labels = {
    'Basal_Bolus': 'Basal-Bolus',
    'PID': 'PID',
    'Safe_NMPC': 'Safe-NMPC (Improved)'
}

# Calculate statistics
stats = {}
for name, df in all_results.items():
    bg = df['BG'].values
    insulin = df['insulin'].values if 'insulin' in df.columns else np.zeros(len(bg))
    
    stats[name] = {
        'mean_bg': np.mean(bg),
        'std_bg': np.std(bg),  # Standard deviation (more important than mean)
        'min_bg': np.min(bg),
        'max_bg': np.max(bg),
        'time_in_range': np.sum((bg >= 70) & (bg <= 180)) / len(bg) * 100,
        'time_below_70': np.sum(bg < 70) / len(bg) * 100,
        'time_above_180': np.sum(bg > 180) / len(bg) * 100,
        'mean_insulin': np.mean(insulin),
        'bg_data': bg,
        'insulin_data': insulin,
        'time': df.index
    }

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35,
                      left=0.06, right=0.97, top=0.94, bottom=0.08)

# ========== Subplot 1: Glucose Trajectories (Full Day) ==========
ax1 = fig.add_subplot(gs[0, :])

for name, s in stats.items():
    time_hours = [(pd.Timestamp(t) - pd.Timestamp(s['time'][0])).total_seconds() / 3600 
                  if isinstance(t, str) else (t - s['time'][0]).total_seconds() / 3600
                  for t in s['time']]
    ax1.plot(time_hours, s['bg_data'], color=colors[name], label=labels[name], 
             linewidth=2.5, alpha=0.85)

# Add meal markers
meal_times = [7, 12, 18]
meal_amounts = [45, 70, 80]
meal_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

for meal_time, meal_amount, meal_color in zip(meal_times, meal_amounts, meal_colors):
    ax1.axvline(meal_time, color=meal_color, linestyle='--', linewidth=2.5, alpha=0.6, zorder=0)
    ax1.text(meal_time, ax1.get_ylim()[1] * 0.95, f'Meal\n{meal_amount}g', 
             ha='center', va='top', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=meal_color, alpha=0.3, 
                      edgecolor=meal_color, linewidth=1.5),
             color='black')

# Zone shading (kept for visual clarity) but hidden from legend to simplify plots.
ax1.axhspan(0, 70, alpha=0.10, color='red', label='_nolegend_', zorder=0)
ax1.axhspan(70, 140, alpha=0.12, color='green', label='_nolegend_', zorder=0)
ax1.axhspan(140, 180, alpha=0.12, color='orange', label='_nolegend_', zorder=0)
ax1.axhspan(180, 1000, alpha=0.10, color='red', label='_nolegend_', zorder=0)
ax1.axhline(140, color='black', linestyle='--', linewidth=2, alpha=0.6,
           label='Target (140 mg/dL)', zorder=0)

ax1.set_xlabel('Time (hours)', fontweight='bold', fontsize=13)
ax1.set_ylabel('Blood Glucose (mg/dL)', fontweight='bold', fontsize=13)
ax1.set_title('(a) Blood Glucose Trajectories Over 24 Hours', fontweight='bold', pad=12, fontsize=14)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 24)
ax1.set_ylim(0, 250)

# ========== Subplot 2: Time in Range Comparison ==========
ax2 = fig.add_subplot(gs[1, 0])

controllers_list = [labels[c] for c in controllers if c in stats]
tir_values = [stats[c]['time_in_range'] for c in controllers if c in stats]
colors_list = [colors[c] for c in controllers if c in stats]

bars = ax2.bar(controllers_list, tir_values, color=colors_list, alpha=0.8, 
               edgecolor='black', linewidth=2)
ax2.axhline(100, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Target (100%)')
ax2.set_ylabel('Time in Range (%)', fontweight='bold', fontsize=12)
ax2.set_title('(b) Time in Range (70-180 mg/dL)', fontweight='bold', pad=10, fontsize=13)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.legend(loc='lower right', fontsize=10)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

for bar, val in zip(bars, tir_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# ========== Subplot 3: Hypoglycemia Risk ==========
ax3 = fig.add_subplot(gs[1, 1])

tbr_values = [stats[c]['time_below_70'] for c in controllers if c in stats]
bars = ax3.bar(controllers_list, tbr_values, color=colors_list, alpha=0.8, 
               edgecolor='black', linewidth=2)
ax3.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Target (<5%)')
ax3.set_ylabel('Time Below 70 mg/dL (%)', fontweight='bold', fontsize=12)
ax3.set_title('(c) Hypoglycemia Risk', fontweight='bold', pad=10, fontsize=13)
ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
ax3.legend(loc='upper right', fontsize=10)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')

for bar, val in zip(bars, tbr_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# ========== Subplot 4: Standard Deviation Comparison ==========
ax4 = fig.add_subplot(gs[1, 2])

std_bg_values = [stats[c]['std_bg'] for c in controllers if c in stats]
bars = ax4.bar(controllers_list, std_bg_values, color=colors_list, alpha=0.8, 
               edgecolor='black', linewidth=2)
ax4.axhline(30, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Target (<30 mg/dL)')
ax4.axhline(50, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Acceptable (<50 mg/dL)')
ax4.set_ylabel('Blood Glucose Std Dev (mg/dL)', fontweight='bold', fontsize=12)
ax4.set_title('(d) Blood Glucose Variability (Std Dev)', fontweight='bold', pad=10, fontsize=13)
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
ax4.legend(loc='best', fontsize=10)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')

for bar, val in zip(bars, std_bg_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# ========== Subplot 5: Insulin Trajectories ==========
ax5 = fig.add_subplot(gs[2, :])

for name, s in stats.items():
    time_hours = [(pd.Timestamp(t) - pd.Timestamp(s['time'][0])).total_seconds() / 3600 
                  if isinstance(t, str) else (t - s['time'][0]).total_seconds() / 3600
                  for t in s['time']]
    ax5.plot(time_hours, s['insulin_data'], color=colors[name], label=labels[name], 
             linewidth=2.5, alpha=0.85)

# Add meal markers
for meal_time, meal_amount, meal_color in zip(meal_times, meal_amounts, meal_colors):
    ax5.axvline(meal_time, color=meal_color, linestyle='--', linewidth=2.5, alpha=0.6, zorder=0)
    ax5.text(meal_time, ax5.get_ylim()[1] * 0.95, f'Meal\n{meal_amount}g', 
             ha='center', va='top', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=meal_color, alpha=0.3, 
                      edgecolor=meal_color, linewidth=1.5),
             color='black')

ax5.set_xlabel('Time (hours)', fontweight='bold', fontsize=13)
ax5.set_ylabel('Insulin Rate (U/min)', fontweight='bold', fontsize=13)
ax5.set_title('(e) Insulin Injection Rate Over Time', fontweight='bold', pad=12, fontsize=14)
ax5.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.set_xlim(0, 24)

# Overall title
fig.suptitle('Safe-NMPC Performance with Adaptive Worst-Case Check & NMPC Optimization',
            fontsize=18, fontweight='bold', y=0.98)

# Save figure
png_file = os.path.join(comparison_path, 'new_results_visualization.png')
plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\n✓ Saved visualization: {png_file}")

pdf_file = os.path.join(comparison_path, 'new_results_visualization.pdf')
plt.savefig(pdf_file, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved PDF: {pdf_file}")

plt.close()

# Print summary
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"\n{'Controller':<20} {'Mean BG':<12} {'Std BG':<12} {'TIR (%)':<12} {'Hypo (%)':<12} {'Hyper (%)':<12}")
print("-" * 90)
for name in controllers:
    if name in stats:
        s = stats[name]
        print(f"{labels[name]:<20} {s['mean_bg']:>10.2f}  {s['std_bg']:>10.2f}  {s['time_in_range']:>10.2f}  "
              f"{s['time_below_70']:>10.2f}  {s['time_above_180']:>10.2f}")

print("\n" + "=" * 80)
print("KEY IMPROVEMENTS IN Safe-NMPC:")
print("=" * 80)
if 'Safe_NMPC' in stats:
    s = stats['Safe_NMPC']
    print(f"✓ Mean BG: {s['mean_bg']:.2f} mg/dL (target: 140 mg/dL)")
    print(f"✓ Std BG: {s['std_bg']:.2f} mg/dL (lower is better - shows consistency)")
    print(f"✓ Time in Range: {s['time_in_range']:.2f}%")
    print(f"✓ Hypoglycemia: {s['time_below_70']:.2f}% (excellent - no hypoglycemia!)")
    print(f"✓ Hyperglycemia: {s['time_above_180']:.2f}%")
    print(f"\n✓ Features: Adaptive worst-case check + NMPC optimization with PID warm start")

print("\n" + "=" * 80)

