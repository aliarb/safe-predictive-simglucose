#!/usr/bin/env python3
"""
Display NMPC controller simulation results with constraint analysis.

This script reads simulation results from CSV files and displays:
- Blood glucose statistics
- Time in range analysis
- Constraint violation analysis
- Performance metrics
"""
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import sys

def display_results(results_dir='./results', patient_name=None, constraint_bounds=(70, 180)):
    """
    Display simulation results with constraint analysis.
    
    Parameters
    ----------
    results_dir : str
        Directory containing simulation results
    patient_name : str, optional
        Specific patient to analyze (e.g., 'adolescent#001')
    constraint_bounds : tuple
        (bg_min, bg_max) constraint bounds in mg/dL
    """
    bg_min, bg_max = constraint_bounds
    
    print("=" * 70)
    print("NMPC CONTROLLER RESULTS WITH CONSTRAINT ANALYSIS")
    print("=" * 70)
    print(f"\nConstraint Configuration:")
    print(f"  Glucose Safety Bounds: [{bg_min}, {bg_max}] mg/dL")
    print(f"  Constraint Type: Control Barrier Function (penalty-based)")
    
    # Find result files
    result_files = []
    if patient_name:
        # Look for specific patient file
        pattern = os.path.join(results_dir, '**', f'{patient_name}.csv')
        result_files = glob.glob(pattern, recursive=True)
    else:
        # Look for all CSV files in results directory
        pattern = os.path.join(results_dir, '**', '*.csv')
        all_files = glob.glob(pattern, recursive=True)
        # Filter out summary files
        result_files = [f for f in all_files if 'performance_stats' not in f and 
                       'CVGA_stats' not in f and 'risk_trace' not in f]
    
    if not result_files:
        print(f"\n❌ No result files found in {results_dir}")
        print("   Run the simulation first using: python examples/run_nmpc_controller.py")
        return
    
    print(f"\nFound {len(result_files)} result file(s)")
    print("-" * 70)
    
    # Process each result file
    all_results = []
    for result_file in sorted(result_files)[:10]:  # Limit to first 10 for display
        try:
            df = pd.read_csv(result_file)
            
            # Extract patient name from filename
            filename = os.path.basename(result_file)
            patient = filename.replace('.csv', '')
            
            # Get BG data
            if 'BG' in df.columns:
                bg_data = df['BG'].values
            elif 'blood_glucose' in df.columns:
                bg_data = df['blood_glucose'].values
            else:
                print(f"⚠️  Skipping {filename}: No BG column found")
                continue
            
            # Calculate statistics
            stats = {
                'patient': patient,
                'mean_bg': np.mean(bg_data),
                'std_bg': np.std(bg_data),
                'min_bg': np.min(bg_data),
                'max_bg': np.max(bg_data),
                'time_in_range': np.sum((bg_data >= bg_min) & (bg_data <= bg_max)) / len(bg_data) * 100,
                'time_below': np.sum(bg_data < bg_min) / len(bg_data) * 100,
                'time_above': np.sum(bg_data > bg_max) / len(bg_data) * 100,
                'violations_below': np.sum(bg_data < bg_min),
                'violations_above': np.sum(bg_data > bg_max),
                'total_steps': len(bg_data)
            }
            
            all_results.append(stats)
            
            # Display individual patient results
            print(f"\n📊 Patient: {patient}")
            print(f"   Mean BG: {stats['mean_bg']:.2f} mg/dL")
            print(f"   Std BG: {stats['std_bg']:.2f} mg/dL")
            print(f"   Min BG: {stats['min_bg']:.2f} mg/dL")
            print(f"   Max BG: {stats['max_bg']:.2f} mg/dL")
            print(f"   Time in Range ({bg_min}-{bg_max} mg/dL): {stats['time_in_range']:.1f}%")
            print(f"   Time Below Range (<{bg_min} mg/dL): {stats['time_below']:.1f}%")
            print(f"   Time Above Range (>{bg_max} mg/dL): {stats['time_above']:.1f}%")
            
            # Constraint violation analysis
            violation_pct = (stats['violations_below'] + stats['violations_above']) / stats['total_steps'] * 100
            print(f"\n   🔒 Constraint Violations:")
            print(f"      Below {bg_min} mg/dL: {stats['violations_below']} ({stats['time_below']:.2f}%)")
            print(f"      Above {bg_max} mg/dL: {stats['violations_above']} ({stats['time_above']:.2f}%)")
            print(f"      Total violations: {stats['violations_below'] + stats['violations_above']} ({violation_pct:.2f}%)")
            
            if violation_pct < 5:
                print(f"      ✅ Excellent: Constraint violations < 5%")
            elif violation_pct < 10:
                print(f"      ⚠️  Good: Constraint violations < 10%")
            else:
                print(f"      ❌ High violation rate: Consider increasing barrier_weight")
            
        except Exception as e:
            print(f"⚠️  Error processing {result_file}: {e}")
            continue
    
    # Summary statistics across all patients
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS (Across All Patients)")
        print("=" * 70)
        
        avg_time_in_range = np.mean([r['time_in_range'] for r in all_results])
        avg_time_below = np.mean([r['time_below'] for r in all_results])
        avg_time_above = np.mean([r['time_above'] for r in all_results])
        avg_mean_bg = np.mean([r['mean_bg'] for r in all_results])
        total_violations = sum([r['violations_below'] + r['violations_above'] for r in all_results])
        total_steps = sum([r['total_steps'] for r in all_results])
        overall_violation_pct = (total_violations / total_steps) * 100
        
        print(f"\nAverage Time in Range ({bg_min}-{bg_max} mg/dL): {avg_time_in_range:.1f}%")
        print(f"Average Time Below Range (<{bg_min} mg/dL): {avg_time_below:.1f}%")
        print(f"Average Time Above Range (>{bg_max} mg/dL): {avg_time_above:.1f}%")
        print(f"Average Mean BG: {avg_mean_bg:.2f} mg/dL")
        print(f"\nOverall Constraint Violations: {total_violations}/{total_steps} ({overall_violation_pct:.2f}%)")
        
        if overall_violation_pct < 5:
            print("✅ Excellent constraint satisfaction!")
        elif overall_violation_pct < 10:
            print("⚠️  Good constraint satisfaction")
        else:
            print("❌ High violation rate - consider tuning barrier_weight")
    
    print("\n" + "=" * 70)
    print("Results display complete!")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Display NMPC controller simulation results')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing simulation results')
    parser.add_argument('--patient', type=str, default=None,
                       help='Specific patient to analyze (e.g., adolescent#001)')
    parser.add_argument('--bg-min', type=float, default=70.0,
                       help='Minimum safe BG constraint (mg/dL)')
    parser.add_argument('--bg-max', type=float, default=180.0,
                       help='Maximum safe BG constraint (mg/dL)')
    
    args = parser.parse_args()
    
    display_results(
        results_dir=args.results_dir,
        patient_name=args.patient,
        constraint_bounds=(args.bg_min, args.bg_max)
    )

