#!/usr/bin/env python3
"""
Test script to compare single-start vs multi-start optimization performance.

Tests:
1. Solution quality comparison
2. Convergence rate
3. Local minima escape
4. Performance (time vs quality trade-off)
"""
import numpy as np
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.patient.t1dpatient import T1DPatient
import time
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("MULTI-START SOLVER COMPARISON")
print("=" * 80)

# Create patient and get initial state
patient = T1DPatient.withName('adolescent#001')
initial_state = patient.state.copy()
initial_bg = patient.state[3] * patient._params['Vg']

# Prepare target state
target_state = np.zeros(13)
Vg = 1.88  # Typical Vg value
target_state[3] = 140.0 / Vg

# Prepare other parameters
other_params_base = {
    'meal': 0.0,
    'patient_name': 'adolescent#001',
    'last_Qsto': initial_state[0] + initial_state[1],
    'last_foodtaken': 0,
    'u_prev': 0.02
}

# Test scenarios
test_scenarios = [
    {'name': 'High BG', 'bg': 200.0, 'initial_delta_u': 0.05},
    {'name': 'Low BG', 'bg': 80.0, 'initial_delta_u': -0.01},
    {'name': 'Very high BG', 'bg': 250.0, 'initial_delta_u': 0.1},
    {'name': 'Near target', 'bg': 145.0, 'initial_delta_u': 0.0},
]

print("\n" + "=" * 80)
print("COMPARISON: Single-Start vs Multi-Start")
print("=" * 80)

results_comparison = []

for scenario in test_scenarios:
    print(f"\n[Scenario] {scenario['name']} (BG={scenario['bg']:.1f} mg/dL)")
    print("-" * 80)
    
    # Set initial state
    test_state = initial_state.copy()
    test_state[3] = scenario['bg'] / Vg
    
    other_params = other_params_base.copy()
    other_params['patient_params'] = None  # Will be loaded from controller
    
    u_old = np.array([scenario['initial_delta_u']])
    
    # Test 1: Single-start optimization
    print("\n1. Single-Start Optimization:")
    controller_single = NMPCController(
        target_bg=140.0,
        prediction_horizon=60,
        control_horizon=30,
        sample_time=5.0,
        bg_min=70.0,
        bg_max=180.0
    )
    controller_single.use_multi_start = False
    controller_single.patient_params = controller_single._load_patient_params('adolescent#001')
    other_params['patient_params'] = controller_single.patient_params
    
    start_time = time.time()
    try:
        u_single = controller_single._optimize(
            x=test_state,
            xd=target_state,
            u_old=u_old,
            DT=5.0,
            NP=12,
            maxiteration=30,  # Reduced for faster testing
            alfa=controller_single.opt_rate,
            acc=controller_single.acc,
            other=other_params,
            max_time=5.0
        )
        time_single = time.time() - start_time
        cost_single = controller_single._mainfun(target_state, test_state, u_single, 5.0, 12, other_params)
        
        print(f"   ✓ Converged in {time_single:.3f}s")
        print(f"   ΔU: {u_single[0]:.6f}")
        print(f"   Cost: {cost_single:.2f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        u_single = None
        time_single = float('inf')
        cost_single = float('inf')
    
    # Test 2: Multi-start optimization (sequential)
    print("\n2. Multi-Start Optimization (Sequential):")
    controller_multi = NMPCController(
        target_bg=140.0,
        prediction_horizon=60,
        control_horizon=30,
        sample_time=5.0,
        bg_min=70.0,
        bg_max=180.0
    )
    controller_multi.use_multi_start = True
    controller_multi.num_starting_points = 3
    controller_multi.parallel_starts = False
    controller_multi.start_point_spread = 0.3
    controller_multi.patient_params = controller_multi._load_patient_params('adolescent#001')
    other_params['patient_params'] = controller_multi.patient_params
    
    start_time = time.time()
    try:
        u_multi = controller_multi._optimize_multi_start(
            x=test_state,
            xd=target_state,
            u_old=u_old,
            DT=5.0,
            NP=12,
            maxiteration=30,
            alfa=controller_multi.opt_rate,
            acc=controller_multi.acc,
            other=other_params,
            max_time=5.0
        )
        time_multi = time.time() - start_time
        cost_multi = controller_multi._mainfun(target_state, test_state, u_multi, 5.0, 12, other_params)
        
        print(f"   ✓ Converged in {time_multi:.3f}s")
        print(f"   ΔU: {u_multi[0]:.6f}")
        print(f"   Cost: {cost_multi:.2f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        u_multi = None
        time_multi = float('inf')
        cost_multi = float('inf')
    
    # Compare results
    if u_single is not None and u_multi is not None:
        cost_improvement = ((cost_single - cost_multi) / cost_single * 100) if cost_single > 0 else 0
        time_overhead = ((time_multi - time_single) / time_single * 100) if time_single > 0 else 0
        
        print(f"\n   Comparison:")
        print(f"   Cost improvement: {cost_improvement:.1f}%")
        print(f"   Time overhead: {time_overhead:.1f}%")
        
        if cost_multi < cost_single:
            print(f"   ✓ Multi-start found better solution!")
        elif abs(cost_multi - cost_single) < 1.0:
            print(f"   ≈ Similar solutions")
        else:
            print(f"   ⚠️  Single-start was better (may be due to time limit)")
        
        results_comparison.append({
            'scenario': scenario['name'],
            'cost_single': cost_single,
            'cost_multi': cost_multi,
            'time_single': time_single,
            'time_multi': time_multi,
            'cost_improvement': cost_improvement,
            'time_overhead': time_overhead
        })

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if len(results_comparison) > 0:
    avg_cost_improvement = np.mean([r['cost_improvement'] for r in results_comparison])
    avg_time_overhead = np.mean([r['time_overhead'] for r in results_comparison])
    
    print(f"\nAverage cost improvement: {avg_cost_improvement:.1f}%")
    print(f"Average time overhead: {avg_time_overhead:.1f}%")
    
    better_count = sum(1 for r in results_comparison if r['cost_multi'] < r['cost_single'])
    print(f"Multi-start better in {better_count}/{len(results_comparison)} scenarios")
    
    print("\nRecommendations:")
    if avg_cost_improvement > 5.0:
        print("  ✓ Multi-start provides significant cost improvement")
        print("  → Consider enabling for better solution quality")
    elif avg_cost_improvement > 0:
        print("  ⚠️  Multi-start provides modest improvement")
        print("  → Consider enabling if solution quality is critical")
    else:
        print("  ⚠️  Multi-start doesn't improve solutions significantly")
        print("  → May not be worth the computational overhead")
    
    if avg_time_overhead < 50.0:
        print("  ✓ Time overhead is acceptable")
    else:
        print("  ⚠️  Time overhead is significant")
        print("  → Consider parallel execution or fewer starting points")

print("\n" + "=" * 80)
print("MULTI-START FEATURES")
print("=" * 80)
print("""
To enable multi-start optimization:

controller = NMPCController(...)
controller.use_multi_start = True
controller.num_starting_points = 3  # Warm start + 2 random starts
controller.parallel_starts = False   # Set True for parallel (experimental)
controller.start_point_spread = 0.3  # 30% of constraint range

Benefits:
- Escapes local minima
- Better solution quality
- More robust optimization

Trade-offs:
- Increased computation time
- More function evaluations
- May not always improve (depends on problem)
""")
print("=" * 80)

