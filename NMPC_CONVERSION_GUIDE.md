# NMPC Controller Conversion Guide

This guide will help you convert your MATLAB NMPC controller code to Python for integration with simglucose.

## Overview

The NMPC controller skeleton has been created at `simglucose/controller/nmpc_ctrller.py`. This file contains:
- A complete controller interface that integrates with simglucose
- Placeholder methods that need to be filled with your converted MATLAB code
- Helper methods for patient parameter loading and fallback actions

## Key Methods to Convert

### 1. `_solve_nmpc()` - Main Optimization Solver

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~150

**What it does**: Solves the NMPC optimization problem at each time step.

**MATLAB to Python conversion checklist**:
- [ ] Set up optimization variables (insulin sequence over control horizon)
- [ ] Define objective function (call `_compute_objective()`)
- [ ] Define constraints (call `_compute_constraints()`)
- [ ] Set up solver (CasADi, scipy.optimize.minimize, etc.)
- [ ] Solve optimization problem
- [ ] Extract first control action from solution
- [ ] Return `Action(basal=..., bolus=...)`

**Common MATLAB → Python conversions**:
```matlab
% MATLAB
fmincon(@(u) objective(u, x0, params), u0, A, b, Aeq, beq, lb, ub, @(u) constraints(u, x0, params))
```

```python
# Python (using scipy)
from scipy.optimize import minimize
result = minimize(
    fun=lambda u: self._compute_objective(...),
    x0=u0,
    method='SLSQP',
    bounds=[(0, 10)] * len(u0),
    constraints={'type': 'ineq', 'fun': lambda u: self._compute_constraints(...)}
)
```

**Recommended Python optimization libraries**:
- **CasADi**: Best for NMPC (similar to MATLAB's fmincon), supports automatic differentiation
- **scipy.optimize**: Built-in Python, good for simpler problems
- **cvxpy**: Good for convex optimization problems
- **pyomo**: For more complex optimization modeling

### 2. `_predict_glucose()` - Model Prediction

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~200

**What it does**: Simulates the patient ODE model forward to predict future glucose.

**MATLAB to Python conversion checklist**:
- [ ] Convert ODE integration (MATLAB `ode45` → Python `scipy.integrate.ode` or `scipy.integrate.solve_ivp`)
- [ ] Handle meal disturbances over prediction horizon
- [ ] Apply insulin sequence
- [ ] Return predicted state trajectory

**Common MATLAB → Python conversions**:
```matlab
% MATLAB
[t, x] = ode45(@(t, x) patient_model(t, x, u, params), [0 horizon], x0);
```

```python
# Python
from scipy.integrate import ode
solver = ode(self._patient_ode_model)
solver.set_integrator('dopri5')  # Runge-Kutta method (similar to ode45)
solver.set_initial_value(x0, 0)
solver.set_f_params(u, params)
while solver.t < horizon:
    solver.integrate(solver.t + dt)
    x_pred.append(solver.y)
```

**Note**: The patient model is already implemented in `simglucose/patient/t1dpatient.py` (method `model()`). You can reference this or use it directly.

### 3. `_compute_objective()` - Cost Function

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~240

**What it does**: Computes the NMPC objective function value.

**MATLAB to Python conversion checklist**:
- [ ] Convert tracking cost: `sum((BG - target)^2)`
- [ ] Convert control cost: `sum(insulin^2)`
- [ ] Add terminal cost if applicable
- [ ] Add control barrier function penalties if applicable

**Common MATLAB → Python conversions**:
```matlab
% MATLAB
J = sum(q * (BG_pred - target).^2) + sum(r * u.^2);
```

```python
# Python
J = np.sum(self.q_weight * (bg_pred - self.target_bg)**2) + \
    np.sum(self.r_weight * insulin_seq**2)
```

### 4. `_compute_constraints()` - Constraint Functions

**Location**: `simglucose/controller/nmpc_ctrller.py`, line ~270

**What it does**: Computes constraint values for the optimization problem.

**MATLAB to Python conversion checklist**:
- [ ] Convert safety bounds: `bg_min <= BG <= bg_max`
- [ ] Convert control bounds: `0 <= insulin <= insulin_max`
- [ ] Convert control barrier function constraints
- [ ] Return constraints in format expected by your solver

**Common MATLAB → Python conversions**:
```matlab
% MATLAB
c = [BG - bg_max; bg_min - BG; -u; u - u_max];
```

```python
# Python
constraints = {
    'bg_upper': self.bg_max - bg_pred,  # >= 0
    'bg_lower': bg_pred - self.bg_min,   # >= 0
    'insulin_upper': insulin_max - insulin_seq,  # >= 0
    'insulin_lower': insulin_seq  # >= 0
}
```

## Patient Model Reference

The patient model is a 13-dimensional ODE system. Key states:
- `x[3]`: Blood glucose (G) in mg/kg
- `x[12]`: Subcutaneous glucose (Gsub) in mg/kg

The model is defined in `simglucose/patient/t1dpatient.py`:
- Method: `T1DPatient.model(t, x, action, params, last_Qsto, last_foodtaken)`
- Inputs: `action.insulin` (U/min), `action.CHO` (g/min)
- Returns: `dxdt` (13-dimensional state derivative)

You can use this model directly in your prediction function.

## Control Barrier Functions (CBF)

If your MATLAB code includes control barrier functions for safety, convert them here:

**Typical CBF constraint**:
```python
# CBF constraint: h_dot + alpha * h >= 0
# where h is the barrier function (e.g., h = BG - bg_min for lower bound)
def cbf_constraint(predicted_bg, bg_min, alpha=1.0):
    h = predicted_bg - bg_min  # Barrier function
    h_dot = np.diff(predicted_bg)  # Approximate derivative
    return h_dot[1:] + alpha * h[1:]  # Must be >= 0
```

## Step-by-Step Conversion Process

1. **Start with the solver**: Convert your MATLAB optimization solver first
   - Choose Python equivalent (CasADi recommended for NMPC)
   - Set up optimization problem structure
   - Test with simple objective/constraints

2. **Convert prediction function**: 
   - Use `scipy.integrate` or reference `T1DPatient.model()`
   - Test prediction with known inputs

3. **Convert objective function**:
   - Straightforward conversion (mostly numpy operations)
   - Test with known values

4. **Convert constraints**:
   - Include CBF constraints if applicable
   - Test constraint evaluation

5. **Integration**:
   - Connect all pieces in `_solve_nmpc()`
   - Test with simglucose simulation
   - Debug and tune parameters

## Testing Your Implementation

1. **Unit tests**: Test each method independently
   ```python
   # Test prediction
   predicted = controller._predict_glucose(x0, u_seq, meal_seq, horizon=60)
   assert predicted.shape == (60, 13)
   ```

2. **Integration test**: Run with simglucose
   ```python
   python examples/run_nmpc_controller.py
   ```

3. **Compare with MATLAB**: Run same scenario in MATLAB and Python, compare results

## Common Issues and Solutions

### Issue: Solver convergence problems
- **Solution**: Check initial guess, constraint feasibility, solver tolerances

### Issue: Prediction doesn't match MATLAB
- **Solution**: Verify ODE integration method and step size match MATLAB

### Issue: Performance too slow
- **Solution**: Use compiled solvers (CasADi), optimize prediction code, reduce horizon

### Issue: Control actions seem wrong
- **Solution**: Check units (U/min vs U), verify action extraction from solution

## Example: CasADi NMPC Setup

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

## Next Steps

1. Share your MATLAB NMPC code and solver
2. We'll convert it step by step, starting with the solver setup
3. Test each component as we convert
4. Integrate and tune parameters

## Resources

- [CasADi Python Documentation](https://web.casadi.org/python-api/)
- [scipy.optimize Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [simglucose Patient Model](simglucose/patient/t1dpatient.py)
- [Example Controllers](simglucose/controller/)

