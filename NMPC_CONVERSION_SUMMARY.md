# NMPC Controller Conversion Summary

## Overview

The MATLAB NMPC controller code has been successfully converted to Python and integrated into the simglucose framework. The controller uses a gradient descent optimization method with conjugate gradient momentum to solve the nonlinear model predictive control problem.

## Key Conversions

### 1. Main NMPC Function → `_solve_nmpc()`

**MATLAB**: `NMPC(Xd,z,u,DT,other,ke,kp1,max_time)`
**Python**: `_solve_nmpc(current_state, current_bg, cgm_reading, meal, sample_time, patient_name)`

**Changes**:
- Robot state (x, y, theta, v) → Patient state (13-dimensional glucose model state)
- Robot control (steering, velocity) → Insulin control (basal, bolus)
- Removed feedback linearization (not needed for glucose control)
- Adapted target state to glucose tracking

### 2. Optimization Function → `_optimize()`

**MATLAB**: `optimize(x,xd,U_old,DT,NP,maxiteration,alfa,acc,other,max_time)`
**Python**: `_optimize(x, xd, u_old, DT, NP, maxiteration, alfa, acc, other, max_time)`

**Preserved**:
- Gradient descent with conjugate gradient momentum
- Adaptive learning rate (alfa)
- Convergence criteria (norm of gradient, max iterations, max time)
- Saturation constraints

### 3. Objective Function → `_mainfun()`

**MATLAB**: `mainfun(xd,x,u,DelT,NP,other)`
**Python**: `_mainfun(xd, x, u, DelT, NP, other)`

**Changes**:
- Robot position tracking → Blood glucose tracking
- Removed velocity/heading tracking (not applicable)
- Added terminal cost for final glucose state
- Adapted stability bound cost for glucose safety bounds

**Cost Function Structure**:
```python
J = sum(tracking_error^2) + terminal_cost + stability_cost + constraint_penalty
```

Where:
- `tracking_error`: (BG_predicted - BG_target)^2
- `terminal_cost`: Final state tracking
- `stability_cost`: Control barrier function for safety bounds
- `constraint_penalty`: Insulin bound violations

### 4. Gradient Computation → `_dot_fun()`

**MATLAB**: `dot_fun(X,n,fx,xd,x,DT,NP,other)`
**Python**: `_dot_fun(X, n, fx, xd, x, DT, NP, other)`

**Preserved**:
- Finite difference gradient computation
- Same epsilon (1e-3) for numerical differentiation

### 5. Patient Model → `_patient_model_step()`

**MATLAB**: `wheel_model(z(:,i),u,dz0)` (robot-specific)
**Python**: `_patient_model_step(x, u, dz0, other)` (glucose model)

**Changes**:
- Replaced robot wheel dynamics with patient glucose ODE model
- Uses `T1DPatient.model()` from simglucose
- Handles meal disturbances and insulin inputs
- Returns state derivative for integration

### 6. Safety Barrier Function → `_safety_barrier_function()`

**MATLAB**: `stability_bound_cost(x,other,bound)` (robot slip/yaw rate)
**Python**: `_safety_barrier_function(x, other, bound)` (glucose safety)

**Changes**:
- Robot stability (beta, yaw rate) → Glucose safety bounds
- Checks if BG is within safe range (70-180 mg/dL)
- Penalizes unsafe glucose levels

## Controller Parameters

### Optimization Parameters
- `NP = 15`: Prediction horizon steps (default: 60 minutes if sample_time=5)
- `Nopt = 20`: Maximum optimization iterations
- `opt_rate = 1.0`: Initial learning rate
- `acc = 1e-3`: Convergence tolerance
- `max_time = 0.1`: Maximum computation time (seconds)

### Control Parameters
- `target_bg = 140.0`: Target blood glucose (mg/dL)
- `bg_min = 70.0`: Minimum safe BG (mg/dL)
- `bg_max = 180.0`: Maximum safe BG (mg/dL)
- `insulin_max = 10.0`: Maximum insulin rate (U/min)
- `insulin_min = 0.0`: Minimum insulin rate (U/min)

## Usage

```python
from simglucose.controller.nmpc_ctrller import NMPCController

# Create controller
nmpc = NMPCController(
    target_bg=140.0,
    prediction_horizon=60,  # minutes
    control_horizon=30,     # minutes
    sample_time=5.0,       # minutes
    q_weight=1.0,         # Glucose tracking weight
    r_weight=0.1,         # Insulin cost weight
    bg_min=70.0,
    bg_max=180.0
)

# Use in simulation (same as other controllers)
from simglucose.simulation.user_interface import simulate
simulate(controller=nmpc)
```

## Testing

Run the example script:
```bash
python examples/run_nmpc_controller.py
```

## Notes

1. **Meal Prediction**: Currently uses constant meal rate over prediction horizon. Future enhancement: predict future meals from scenario.

2. **Control Horizon**: The current implementation uses a single control input over the prediction horizon. For multi-step control, modify `u` to be a sequence.

3. **Patient Parameters**: Automatically loaded from patient name. Ensure patient_params is a pandas Series (not dict) for model compatibility.

4. **Performance**: The gradient descent method may be slower than specialized NLP solvers (CasADi, IPOPT). Consider upgrading if performance is critical.

5. **Safety**: The controller includes control barrier functions through `_safety_barrier_function()` to ensure glucose stays within safe bounds.

## Future Enhancements

1. **Multi-step Control**: Extend to control sequence over control horizon
2. **Meal Prediction**: Incorporate meal schedule into prediction
3. **Solver Upgrade**: Consider CasADi/IPOPT for faster convergence
4. **Tuning**: Add automatic parameter tuning based on patient characteristics
5. **Robustness**: Add disturbance rejection and uncertainty handling

## Files Modified/Created

- `simglucose/controller/nmpc_ctrller.py`: Main controller implementation
- `examples/run_nmpc_controller.py`: Example usage script
- `NMPC_CONVERSION_GUIDE.md`: Detailed conversion guide
- `NMPC_CONVERSION_SUMMARY.md`: This file

