#!/usr/bin/env python3
"""
Demo: Event-triggered injection on top of NMPC (supervisor/PID maintenance).

Runs a single patient with the standard meal scenario and prints trigger events.
Outputs a CSV in results/event_triggered_demo/ for quick inspection.
"""

from datetime import datetime, timedelta
import os

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.controller.event_triggered_nmpc import EventTriggeredNMPCController


def main():
    out_dir = "./results/event_triggered_demo"
    os.makedirs(out_dir, exist_ok=True)

    start_time = datetime(2025, 1, 1, 0, 0, 0)
    scenario = CustomScenario(start_time=start_time, scenario=[(7, 45), (12, 70), (18, 80)])

    patient = T1DPatient.withName("adolescent#001")
    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")

    env = T1DSimEnv(patient, sensor, pump, scenario)

    base = NMPCController(
        target_bg=140.0,
        prediction_horizon=60,
        control_horizon=30,
        sample_time=5.0,
        ode_time_step=1.0,
        bg_min=70.0,
        bg_max=180.0,
        use_optimization=False,  # supervisor mode (PID maintenance)
        verbose=False,
    )

    ctrl = EventTriggeredNMPCController(
        base,
        prediction_horizon_minutes=30.0,
        pulse_max_u_per_min=2.0,
        pulse_minutes=5.0,
        cooldown_minutes=30.0,
        suspend_minutes=15.0,
        verbose=True,
    )

    sim_obj = SimObj(env, ctrl, timedelta(hours=8), animate=False, path=out_dir)
    results = sim(sim_obj)
    print(f"Saved trace to: {out_dir}")
    print(results.tail())


if __name__ == "__main__":
    main()

