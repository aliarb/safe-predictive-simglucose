# How to Validate NMPC Simulator

This guide explains how to verify that the NMPC controller works correctly with the simulator.

## Quick Validation

Run the comprehensive validation script:

```bash
python validate_nmpc_simulator.py
```

This script performs 8 validation tests:

### Test 1: Patient Model Response
- **Purpose**: Verify patient model responds correctly to insulin and meals
- **Checks**: 
  - BG decreases with insulin
  - BG increases with meals
  - Model dynamics are physically reasonable

### Test 2: NMPC Prediction Accuracy
- **Purpose**: Verify prediction matches actual patient behavior
- **Checks**:
  - Mean prediction error < 200 mg/dL (for 60 min horizon)
  - Max prediction error < 300 mg/dL
- **Note**: Some error is expected due to model simplifications

### Test 3: Cost Function Computation
- **Purpose**: Verify cost function computes correctly
- **Checks**:
  - Cost is finite (no NaN/inf)
  - Cost is positive
  - Cost responds to different inputs

### Test 4: Safety Constraints
- **Purpose**: Verify barrier functions and safety mechanisms
- **Checks**:
  - Barrier function works for hypo/hyper/normal zones
  - Zone penalties are applied correctly
  - Constraints are continuous

### Test 5: Controller Behavior
- **Purpose**: Verify controller produces valid actions
- **Checks**:
  - Actions are finite
  - Actions are within bounds [0, insulin_max]
  - BG stays within reasonable range

### Test 6: Numerical Stability
- **Purpose**: Verify no numerical issues with extreme values
- **Checks**:
  - No NaN/inf in barrier functions
  - No NaN/inf in cost function
  - Handles extreme BG values gracefully

### Test 7: Consistency and Reproducibility
- **Purpose**: Verify deterministic behavior
- **Checks**:
  - Same inputs produce same outputs
  - Results are reproducible

### Test 8: Full Simulation
- **Purpose**: Verify end-to-end simulation works
- **Checks**:
  - Simulation completes without errors
  - Results contain valid data
  - BG values are reasonable

## Manual Validation Steps

### 1. Check Patient Model Integration

```python
from simglucose.patient.t1dpatient import T1DPatient, Action as PatientAction

patient = T1DPatient.withName('adolescent#001')
initial_bg = patient.state[3] * patient._params['Vg']

# Test insulin effect
action = PatientAction(insulin=0.05, CHO=0.0)
for _ in range(60):
    patient.step(action)

final_bg = patient.state[3] * patient._params['Vg']
print(f"BG change with insulin: {final_bg - initial_bg:.2f} mg/dL")
# Should be negative (BG decreases)
```

### 2. Check Prediction Accuracy

```python
from simglucose.controller.nmpc_ctrller import NMPCController

controller = NMPCController()
patient = T1DPatient.withName('adolescent#001')

# Get prediction
bg_predictions = controller._predict_glucose_trajectory(
    current_state=patient.state,
    insulin_rate=0.05,
    meal_rate=0.0,
    sample_time=5.0,
    other_params={...}
)

# Compare with actual simulation
# Prediction should be reasonably close to actual
```

### 3. Check Cost Function

```python
controller = NMPCController()

# Test cost with different inputs
cost1 = controller._mainfun(xd, x, delta_u=np.array([0.0]), ...)
cost2 = controller._mainfun(xd, x, delta_u=np.array([0.5]), ...)

# Verify costs are finite and positive
assert np.isfinite(cost1)
assert np.isfinite(cost2)
assert cost1 > 0
assert cost2 > 0
```

### 4. Check Safety Constraints

```python
controller = NMPCController(bg_min=70.0, bg_max=180.0)

# Test barrier function
barrier_low = controller._glucose_barrier_function(50.0)  # Should be negative
barrier_high = controller._glucose_barrier_function(200.0)  # Should be positive
barrier_normal = controller._glucose_barrier_function(100.0)  # Should be near zero

assert barrier_low < 0
assert barrier_high > 0
assert abs(barrier_normal) < 1.0
```

### 5. Run Short Simulation

```python
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from datetime import datetime, timedelta

# Setup
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
scenario = CustomScenario(
    start_time=datetime(2025, 1, 1, 0, 0, 0),
    scenario=[(7, 45)]  # Single meal
)
env = T1DSimEnv(patient, sensor, pump, scenario)
controller = NMPCController()

# Run short simulation (1 hour)
sim_obj = SimObj(env, controller, timedelta(hours=1), animate=False, path='./results/test')
results = sim(sim_obj)

# Check results
assert len(results) > 0
bg_data = results['BG'].values
assert np.all(np.isfinite(bg_data))
assert np.all(bg_data > 0)
assert np.all(bg_data < 1000)
```

## Common Issues and Solutions

### Issue: Prediction errors are large (> 200 mg/dL)
**Cause**: Model simplifications, worst-case scenarios, or integration errors
**Solution**: 
- Check ODE integration time step (`ode_time_step`)
- Verify patient parameters are loaded correctly
- Consider that worst-case scenarios intentionally overestimate

### Issue: Cost function returns NaN
**Cause**: Invalid inputs or numerical instability
**Solution**:
- Check for NaN in state vector
- Verify patient parameters are valid
- Check insulin values are within bounds

### Issue: Controller produces extreme insulin values
**Cause**: Cost function weights may be unbalanced
**Solution**:
- Adjust `r_weight` and `r_delta_weight`
- Check barrier function weights
- Verify PID warm start is working

### Issue: Simulation crashes or hangs
**Cause**: Infinite loop in optimization or invalid state
**Solution**:
- Check optimization convergence criteria
- Verify state vector is valid
- Add timeout to optimization

## Expected Behavior

### Normal Operation
- BG stays within reasonable range (50-500 mg/dL)
- Insulin rates are within bounds (0-10 U/min)
- No NaN/inf values in outputs
- Controller responds to BG changes
- Safety constraints are enforced

### Warning Signs
- BG values > 1000 mg/dL (unrealistic)
- BG values < 20 mg/dL (severe hypoglycemia)
- Insulin rates at maximum continuously
- NaN/inf in any outputs
- Optimization doesn't converge

## Debugging Tips

1. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check individual components**:
   - Patient model step
   - Prediction function
   - Cost function
   - Optimization loop

3. **Use debug script**:
   ```bash
   python debug_nmpc.py
   ```

4. **Check worst-case scenarios**:
   - Verify worst-case BG predictions are reasonable
   - Check that safety adjustments are working

5. **Compare with baseline controllers**:
   - Run PID controller for comparison
   - Verify NMPC performs similarly or better

## Validation Checklist

- [ ] Patient model responds correctly to insulin
- [ ] Patient model responds correctly to meals
- [ ] Prediction accuracy is acceptable (< 200 mg/dL error)
- [ ] Cost function computes correctly (finite, positive)
- [ ] Safety constraints are enforced
- [ ] Controller produces valid actions
- [ ] No numerical instability (NaN/inf)
- [ ] Results are reproducible
- [ ] Full simulation completes successfully
- [ ] BG values stay within reasonable range

## Next Steps

After validation:
1. Run comparison with other controllers
2. Tune cost function parameters
3. Test with different patients
4. Test with different scenarios
5. Optimize parameters using RL

