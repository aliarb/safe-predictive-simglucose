# NMPC Controller Conversion Guide and Summary

This document provides a comprehensive guide to the NMPC controller implementation, including the conversion from MATLAB to Python and integration with simglucose.

## Overview

The NMPC controller has been successfully implemented at `simglucose/controller/nmpc_ctrller.py`. The controller uses a gradient descent optimization method with conjugate gradient momentum to solve the nonlinear model predictive control problem for glucose regulation.

**Status**: ✅ **Conversion Complete** - The MATLAB code has been fully converted and integrated.

## Conversion Summary

The MATLAB NMPC controller code (originally for mobile robot control) has been successfully converted to Python and adapted for glucose control. Here's what was converted:

### 1. Main NMPC Function → `_solve_nmpc()`

**MATLAB**: `NMPC(Xd,z,u,DT,other,ke,kp1,max_time)`
**Python**: `_solve_nmpc(current_state, current_bg, cgm_reading, meal, sample_time, patient_name)`

**Changes**:
- ✅ Robot state (x, y, theta, v) → Patient state (13-dimensional glucose model state)
- ✅ Robot control (steering, velocity) → Insulin control (basal, bolus)
- ✅ Removed feedback linearization (not needed for glucose control)
- ✅ Adapted target state to glucose tracking

### 2. Optimization Function → `_optimize()`

**MATLAB**: `optimize(x,xd,U_old,DT,NP,maxiteration,alfa,acc,other,max_time)`
**Python**: `_optimize(x, xd, u_old, DT, NP, maxiteration, alfa, acc, other, max_time)`

**Preserved**:
- ✅ Gradient descent with conjugate gradient momentum
- ✅ Adaptive learning rate (alfa)
- ✅ Convergence criteria (norm of gradient, max iterations, max time)
- ✅ Saturation constraints

### 3. Objective Function → `_mainfun()`

**MATLAB**: `mainfun(xd,x,u,DelT,NP,other)`
**Python**: `_mainfun(xd, x, u, DelT, NP, other)`

**Changes**:
- ✅ Robot position tracking → Blood glucose tracking
- ✅ Removed velocity/heading tracking (not applicable)
- ✅ Added terminal cost for final glucose state
- ✅ Adapted safety barrier function for glucose safety bounds

**Cost Function Structure**:
```python
J = sum(tracking_error^2) + terminal_cost + safety_barrier_cost + constraint_penalty
```

Where:
- `tracking_error`: (BG_predicted - BG_target)^2
- `terminal_cost`: Final state tracking
- `safety_barrier_cost`: Control barrier function for safety bounds
- `constraint_penalty`: Insulin bound violations

### 4. Gradient Computation → `_dot_fun()`

**MATLAB**: `dot_fun(X,n,fx,xd,x,DT,NP,other)`
**Python**: `_dot_fun(X, n, fx, xd, x, DT, NP, other)`

**Preserved**:
- ✅ Finite difference gradient computation
- ✅ Same epsilon (1e-3) for numerical differentiation

### 5. Patient Model → `_patient_model_step()`

**MATLAB**: `wheel_model(z(:,i),u,dz0)` (robot-specific)
**Python**: `_patient_model_step(x, u, dz0, other)` (glucose model)

**Changes**:
- ✅ Replaced robot wheel dynamics with patient glucose ODE model
- ✅ Uses `T1DPatient.model()` from simglucose
- ✅ Handles meal disturbances and insulin inputs
- ✅ Returns state derivative for integration

### 6. Safety Barrier Function → `_safety_barrier_function()`

**MATLAB**: `stability_bound_cost(x,other,bound)` (robot slip/yaw rate)
**Python**: `_safety_barrier_function(x, other, bound)` (glucose safety)

**Changes**:
- ✅ Robot stability (beta, yaw rate) → Glucose safety bounds
- ✅ Checks if BG is within safe range (70-180 mg/dL)
- ✅ Penalizes unsafe glucose levels

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

## Detailed Conversion Guide

### Key Methods Implementation

#### 1. `_solve_nmpc()` - Main Optimization Solver

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~169

**What it does**: Solves the NMPC optimization problem at each time step using gradient descent.

**Implementation Status**: ✅ Complete

**Key Features**:
- Sets up target state from desired blood glucose
- Uses previous action as initial guess
- Calls `_optimize()` to solve the problem
- Splits result into basal and bolus components

#### 2. `_optimize()` - Gradient Descent Optimizer

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~220

**What it does**: Implements gradient descent with conjugate gradient momentum.

**Implementation Status**: ✅ Complete

**Algorithm**:
1. Initialize with previous control input
2. Compute gradient using finite differences
3. Update with momentum (conjugate gradient)
4. Adaptive learning rate based on cost improvement
5. Apply saturation constraints
6. Repeat until convergence or max iterations

#### 3. `_mainfun()` - Objective Function

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~330

**What it does**: Computes the NMPC objective function value over prediction horizon.

**Implementation Status**: ✅ Complete

**Cost Components**:
- Tracking cost: `(BG_predicted - BG_target)^2` at each step
- Terminal cost: Final state tracking error
- Safety barrier cost: Penalty for unsafe glucose levels
- Constraint penalty: Insulin bound violations

#### 4. `_patient_model_step()` - Model Prediction

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~405

**What it does**: Computes one step of patient model dynamics.

**Implementation Status**: ✅ Complete

**Details**:
- Wraps `T1DPatient.model()` from simglucose
- Handles insulin and meal inputs
- Returns state derivative for integration
- Maintains compatibility with optimization loop

#### 5. `_safety_barrier_function()` - Control Barrier Function

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~492

**What it does**: Implements safety barrier function to ensure glucose stays within safe bounds.

**Implementation Status**: ✅ Complete

**Safety Logic**:
- Checks if BG is within safe range (70-180 mg/dL)
- Penalizes states outside safe bounds
- Returns cost proportional to violation severity

## Patient Model Reference

The patient model is a 13-dimensional ODE system. Key states:
- `x[3]`: Blood glucose (G) in mg/kg
- `x[12]`: Subcutaneous glucose (Gsub) in mg/kg

The model is defined in `simglucose/patient/t1dpatient.py`:
- Method: `T1DPatient.model(t, x, action, params, last_Qsto, last_foodtaken)`
- Inputs: `action.insulin` (U/min), `action.CHO` (g/min)
- Returns: `dxdt` (13-dimensional state derivative)

The NMPC controller uses this model directly for prediction.

## Control Barrier Functions (CBF)

The controller implements safety barrier functions to ensure glucose stays within safe bounds:

```python
def _safety_barrier_function(self, x, other, bound):
    """
    Safety barrier function (control barrier function).
    Implements control barrier functions to ensure glucose stays within safe bounds.
    Penalizes states that violate safety bounds.
    """
    Vg = self._get_param('Vg', 1.0)
    bg = x[3] * Vg  # Blood glucose in mg/dL
    
    # Penalty if outside safe bounds
    if bg > self.bg_max or bg < self.bg_min:
        j = 1.0 * ((bg - self.target_bg)**2)
    else:
        j = 0.0
    
    return j
```

## Testing

### Run the Example Script

```bash
python examples/run_nmpc_controller.py
```

### Unit Testing

Test individual methods:
```python
from simglucose.controller.nmpc_ctrller import NMPCController
import numpy as np

controller = NMPCController()
# Test prediction
x0 = np.zeros(13)
x0[3] = 140.0 / 1.0  # Initial BG
u_seq = np.array([0.02])  # Basal insulin
meal_seq = np.array([0.0])
predicted = controller._predict_glucose(x0, u_seq, meal_seq, horizon=60)
```

## Common Issues and Solutions

### Issue: Solver convergence problems
- **Solution**: Check initial guess, constraint feasibility, solver tolerances
- **Current Implementation**: Uses adaptive learning rate and fallback to basal insulin

### Issue: Prediction doesn't match MATLAB
- **Solution**: Verify ODE integration method and step size match MATLAB
- **Current Implementation**: Uses same patient model as simglucose (FDA-approved UVa/Padova)

### Issue: Performance too slow
- **Solution**: Use compiled solvers (CasADi), optimize prediction code, reduce horizon
- **Current Implementation**: Gradient descent may be slower than specialized NLP solvers
- **Future Enhancement**: Consider upgrading to CasADi/IPOPT

### Issue: Control actions seem wrong
- **Solution**: Check units (U/min vs U), verify action extraction from solution
- **Current Implementation**: Properly converts to basal/bolus format

## Implementation Notes

1. **Meal Prediction**: Currently uses constant meal rate over prediction horizon. Future enhancement: predict future meals from scenario.

2. **Control Horizon**: The current implementation uses a single control input over the prediction horizon. For multi-step control, modify `u` to be a sequence.

3. **Patient Parameters**: Automatically loaded from patient name. Patient_params is stored as pandas Series (as expected by patient model).

4. **Performance**: The gradient descent method may be slower than specialized NLP solvers (CasADi, IPOPT). Consider upgrading if performance is critical.

5. **Safety**: The controller includes control barrier functions through `_safety_barrier_function()` to ensure glucose stays within safe bounds.

## Future Enhancements

1. **Multi-step Control**: Extend to control sequence over control horizon
2. **Meal Prediction**: Incorporate meal schedule into prediction
3. **Solver Upgrade**: Consider CasADi/IPOPT for faster convergence
4. **Tuning**: Add automatic parameter tuning based on patient characteristics
5. **Robustness**: Add disturbance rejection and uncertainty handling

## Alternative Solver: CasADi NMPC Setup

For improved performance, consider upgrading to CasADi:

```python
import casadi as ca

# Decision variables
u = ca.MX.sym('u', control_horizon)

# Objective
J = 0
x_pred = initial_state
for k in range(control_horizon):
    # Predict one step
    x_pred = self._predict_one_step(x_pred, u[k], meal[k])
    bg_pred = x_pred[3] * params['Vg']  # Convert to mg/dL
    
    # Add to objective
    J += q_weight * (bg_pred - target_bg)**2 + r_weight * u[k]**2

# Constraints
g = []
for k in range(prediction_horizon):
    bg_pred = x_pred_traj[k, 3] * params['Vg']
    g.append(bg_pred - bg_min)  # >= 0
    g.append(bg_max - bg_pred)   # >= 0

# NLP problem
nlp = {'x': u, 'f': J, 'g': ca.vertcat(*g)}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# Solve
result = solver(x0=u0, lbg=0, ubg=ca.inf)
u_opt = result['x']
```

## Files Created

- `simglucose/controller/nmpc_ctrller.py`: Main controller implementation (797 lines)
- `examples/run_nmpc_controller.py`: Example usage script
- `NMPC_CONVERSION_GUIDE.md`: This comprehensive guide

## Resources

- [CasADi Python Documentation](https://web.casadi.org/python-api/)
- [scipy.optimize Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [simglucose Patient Model](simglucose/patient/t1dpatient.py)
- [Example Controllers](simglucose/controller/)
- [FDA-Approved UVa/Padova Simulator](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/)
