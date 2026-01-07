# NMPC Cost Function Design

## Overview

The NMPC controller now uses a comprehensive, continuous, and parameterized cost function designed for:
- **Continuous differentiability**: All functions use smooth transitions (sigmoid-based)
- **RL optimization**: Fully parameterized for reinforcement learning tuning
- **Safety guarantees**: Multiple barrier functions ensure glucose stays within safe bounds
- **MPC best practices**: Based on infinite-horizon MPC principles

## Cost Function Components

The total cost `J` consists of 10 components:

### 1. Tracking Cost
**Purpose**: Minimize deviation from target glucose
```
J_tracking = q_weight * (BG_pred - BG_target)²
```
- **Parameter**: `q_weight` (default: 1.0)
- **Tunable**: Yes

### 2. Control Effort Cost
**Purpose**: Penalize rate of change (ΔU) with asymmetry
```
J_control = r_delta_weight * [asymmetry_factor * (ΔU)² if ΔU > 0 else (ΔU)²]
```
- **Parameters**: 
  - `r_delta_weight` (default: 0.5)
  - `delta_u_asymmetry` (default: 2.0) - increases penalized more than decreases
- **Tunable**: Yes

### 3. Barrier Function Penalty
**Purpose**: Smooth CBF penalty for safety bounds
```
J_barrier = barrier_weight * (barrier_function(BG))²
```
- **Parameters**: 
  - `barrier_weight` (default: 10.0)
  - `zone_transition_smoothness` (default: 5.0)
- **Tunable**: Yes
- **Continuous**: Uses smooth sigmoid transitions

### 4. Zone-Based Penalty
**Purpose**: Different penalties for different glucose zones
```
J_zone = hypo_penalty_weight * penalty_hypo(BG) + hyper_penalty_weight * penalty_hyper(BG)
```
- **Parameters**:
  - `hypo_penalty_weight` (default: 50.0)
  - `hyper_penalty_weight` (default: 20.0)
- **Tunable**: Yes
- **Continuous**: Smooth transitions between zones

### 5. Predictive CBF Penalty
**Purpose**: Context-aware control cost adjustment
- Adjusts control cost based on predicted BG zone
- Strong penalty for increasing insulin when hypoglycemic
- Moderate penalty for decreasing insulin when hyperglycemic
- **Continuous**: Uses smooth sigmoid transitions

### 6. Insulin Rate Constraint Penalty
**Purpose**: Smooth penalty for insulin bound violations
```
J_constraint = insulin_rate_penalty_weight * smooth_penalty(violation)
```
- **Parameters**: `insulin_rate_penalty_weight` (default: 100.0)
- **Tunable**: Yes
- **Continuous**: Smooth sigmoid-weighted quadratic penalty

### 7. Rate of Change Constraint Penalty
**Purpose**: Adaptive, smooth ΔU limits based on predicted BG
- Adaptive bounds: larger changes allowed when hyperglycemic
- Smooth transitions between zones
- **Continuous**: Weighted average of zone-specific bounds

### 8. Terminal Tracking Cost
**Purpose**: Final state tracking error
```
J_terminal_tracking = q_terminal_weight * (BG_final - BG_target)²
```
- **Parameters**: `q_terminal_weight` (default: 2.0)
- **Tunable**: Yes

### 9. Terminal Barrier Penalty
**Purpose**: Final state safety penalty
```
J_terminal_barrier = barrier_weight * (barrier_function(BG_final))²
```
- **Continuous**: Same smooth barrier function as step cost

### 10. Terminal Zone Penalty
**Purpose**: Final state zone penalty
- Same zone-based penalty applied to final state
- **Continuous**: Smooth transitions

## Continuous Functions

All functions use smooth approximations for continuity:

### Smooth Barrier Function
```python
barrier(G) = (G - G_max) * sigmoid(G - G_max) + (G - G_min) * sigmoid(G_min - G)
```
- Uses `tanh` for smooth sigmoid transitions
- Continuously differentiable everywhere

### Smooth Zone Transitions
```python
zone_factor = 0.5 * (1 + tanh(alpha * (G - threshold)))
```
- `alpha` = `zone_transition_smoothness` controls transition sharpness
- Higher alpha = sharper transitions
- Lower alpha = smoother transitions

## Tunable Parameters for RL

All cost function parameters can be tuned via RL:

```python
controller = NMPCController(
    # Basic parameters
    q_weight=1.0,                    # Tracking cost weight
    r_weight=0.1,                    # Control cost weight (legacy)
    barrier_weight=10.0,             # Barrier function weight
    
    # New tunable parameters
    q_terminal_weight=2.0,           # Terminal tracking weight
    r_delta_weight=0.5,              # Rate of change penalty weight
    hypo_penalty_weight=50.0,        # Hypoglycemia penalty weight
    hyper_penalty_weight=20.0,       # Hyperglycemia penalty weight
    zone_transition_smoothness=5.0,  # Zone transition smoothness
    insulin_rate_penalty_weight=100.0, # Constraint violation penalty
    delta_u_asymmetry=2.0            # Asymmetry factor for ΔU penalty
)
```

## Parameter Ranges for RL Tuning

Recommended ranges for RL optimization:

- `q_weight`: [0.1, 10.0] - Tracking importance
- `q_terminal_weight`: [0.5, 5.0] - Terminal state importance
- `r_delta_weight`: [0.01, 2.0] - Control effort penalty
- `hypo_penalty_weight`: [10.0, 200.0] - Hypoglycemia safety
- `hyper_penalty_weight`: [5.0, 100.0] - Hyperglycemia penalty
- `barrier_weight`: [1.0, 50.0] - Overall safety enforcement
- `zone_transition_smoothness`: [1.0, 20.0] - Transition sharpness
- `insulin_rate_penalty_weight`: [10.0, 500.0] - Constraint enforcement
- `delta_u_asymmetry`: [1.0, 5.0] - Control asymmetry

## Usage Example

```python
from simglucose.controller.nmpc_ctrller import NMPCController

# Default parameters
controller = NMPCController()

# Custom tuned parameters (from RL optimization)
controller = NMPCController(
    q_weight=2.5,
    q_terminal_weight=3.0,
    r_delta_weight=0.3,
    hypo_penalty_weight=75.0,
    hyper_penalty_weight=25.0,
    barrier_weight=15.0,
    zone_transition_smoothness=7.0,
    insulin_rate_penalty_weight=150.0,
    delta_u_asymmetry=2.5
)
```

## Benefits

1. **Continuous**: All functions are continuously differentiable
   - Enables gradient-based optimization
   - No numerical issues from discontinuities

2. **Parameterized**: Fully tunable via RL
   - 9 tunable parameters
   - Easy to optimize for different objectives

3. **Safe**: Multiple safety mechanisms
   - Barrier functions
   - Zone penalties
   - Constraint penalties

4. **Flexible**: Adaptive behavior
   - Context-aware penalties
   - Adaptive rate limits
   - Smooth zone transitions

## Notes

- All sigmoid functions use `tanh` for smooth transitions
- Default parameters are conservative (prioritize safety)
- RL can tune parameters to balance performance vs. safety
- Cost function is designed to be convex near the solution for fast convergence

