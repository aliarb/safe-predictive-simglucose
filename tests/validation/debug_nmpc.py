#!/usr/bin/env python3
"""
Debug script for NMPC controller to identify why it's injecting excessive insulin.

This script runs a single step of NMPC and prints detailed debugging information.
"""
import numpy as np
import pandas as pd
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime, timedelta
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("NMPC CONTROLLER DEBUG SCRIPT")
print("=" * 80)

# Create patient, sensor, pump
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')

# Create scenario with a meal
scenario = CustomScenario(
    start_time=datetime(2025, 1, 1, 0, 0, 0),
    scenario=[(7, 45)]  # Single meal at 7am
)

# Create environment
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create NMPC controller with debugging enabled
nmpc = NMPCController(
    target_bg=140.0,
    prediction_horizon=60,
    control_horizon=30,
    sample_time=5.0,
    q_weight=1.0,
    r_weight=0.1,
    bg_min=70.0,
    bg_max=180.0,
    barrier_weight=10.0
)

# Reset environment and controller
obs, reward, done, info = env.reset()
nmpc.reset()

print("\n" + "=" * 80)
print("INITIAL STATE")
print("=" * 80)
print(f"Initial BG: {info.get('bg', obs.CGM):.2f} mg/dL")
print(f"CGM reading: {obs.CGM:.2f} mg/dL")
print(f"Patient state shape: {info.get('patient_state').shape if info.get('patient_state') is not None else 'None'}")

# Run a few steps and debug
print("\n" + "=" * 80)
print("RUNNING NMPC STEPS WITH DEBUGGING")
print("=" * 80)

for step in range(10):
    print(f"\n--- Step {step + 1} ---")
    
    # Get current state
    current_bg = info.get('bg', obs.CGM)
    patient_state = info.get('patient_state')
    meal = info.get('meal', 0.0)
    
    print(f"Current BG: {current_bg:.2f} mg/dL")
    print(f"Meal: {meal:.2f} g/min")
    print(f"Last action: {nmpc.last_action}")
    
    # Manually call _solve_nmpc to see what happens
    try:
        # Get PID suggestion first
        from simglucose.controller.base import Action as BaseAction
        pid_action = nmpc.pid_controller.policy(
            observation=type('obj', (object,), {'CGM': obs.CGM})(),
            reward=0.0,
            done=False,
            sample_time=info.get('sample_time', 5.0)
        )
        pid_insulin = pid_action.basal + pid_action.bolus
        print(f"PID suggestion: {pid_insulin:.6f} U/min")
        
        # Call optimization
        action = nmpc._solve_nmpc(
            current_state=patient_state,
            current_bg=current_bg,
            cgm_reading=obs.CGM,
            meal=meal,
            sample_time=info.get('sample_time', 5.0),
            patient_name=info.get('patient_name')
        )
        
        print(f"NMPC action: basal={action.basal:.6f}, bolus={action.bolus:.6f}")
        print(f"Total insulin: {action.basal + action.bolus:.6f} U/min")
        
        # Evaluate objective function at this action
        target_state = np.zeros(13)
        Vg = nmpc._get_param('Vg', 1.0)
        target_state[3] = nmpc.target_bg / Vg
        
        insulin_total = action.basal + action.bolus
        u_prev_debug = nmpc.last_action.basal + nmpc.last_action.bolus if nmpc.last_action else nmpc._compute_basal()
        delta_u_test = insulin_total - u_prev_debug
        
        other_params = {
            'meal': meal,
            'patient_params': nmpc.patient_params,
            'patient_name': info.get('patient_name'),
            'last_Qsto': patient_state[0] + patient_state[1] if patient_state is not None else 0,
            'last_foodtaken': 0,
            'u_prev': u_prev_debug
        }
        
        cost = nmpc._mainfun(
            xd=target_state,
            x=patient_state,
            delta_u=np.array([delta_u_test]),
            DelT=info.get('sample_time', 5.0),
            NP=nmpc.NP,
            other=other_params
        )
        
        print(f"Objective function value: {cost:.2f}")
        print(f"  ΔU used: {delta_u_test:.6f} U/min")
        print(f"  U_prev: {u_prev_debug:.6f} U/min")
        print(f"  U_total: {insulin_total:.6f} U/min")
        
        # Check what the optimization would return
        delta_u_initial = pid_insulin - u_prev_debug
        u_old = np.array([delta_u_initial])
        u_opt = nmpc._optimize(
            x=patient_state,
            xd=target_state,
            u_old=u_old,
            DT=info.get('sample_time', 5.0),
            NP=nmpc.NP,
            maxiteration=nmpc.Nopt,
            alfa=nmpc.opt_rate,
            acc=nmpc.acc,
            other=other_params,
            max_time=nmpc.max_time
        )
        
        u_opt_abs = u_prev_debug + u_opt[0]
        print(f"Optimization result: ΔU={u_opt[0]:.6f} U/min, U_abs={u_opt_abs:.6f} U/min")
        print(f"Optimization converged to: {'MAX' if abs(u_opt_abs - nmpc.insulin_max) < 0.01 else 'MIN' if abs(u_opt_abs - nmpc.insulin_min) < 0.01 else 'INTERMEDIATE'}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        break
    
    # Step environment
    obs, reward, done, info = env.step(action)
    
    if done:
        print("Simulation done!")
        break

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)

