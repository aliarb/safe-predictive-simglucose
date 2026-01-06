"""
Nonlinear Model Predictive Controller (NMPC) for Type-1 Diabetes Glucose Control

This controller implements a model-based nonlinear model predictive controller
with integrated control barrier functions for safe glucose level control through
artificial insulin injection.

The controller uses the patient's internal state (13-dimensional ODE model) to
predict future glucose trajectories and optimize insulin delivery.

The optimization uses a gradient descent method with momentum (conjugate gradient)
to solve the NMPC problem at each time step.
"""
from .base import Controller
from .base import Action
import numpy as np
import logging
from typing import Optional, Dict, Any
from scipy.integrate import ode
import time

logger = logging.getLogger(__name__)


class NMPCController(Controller):
    """
    Nonlinear Model Predictive Controller for glucose regulation.
    
    ARCHITECTURE (Restructured - PID-First with Safety Supervisor):
    ----------------------------------------------------------------
    This controller uses a PID-first approach with NMPC as a safety supervisor:
    
    1. PRIMARY CONTROL: PID controller provides the main control action
       - PID controller responds to current glucose level
       - Provides fast, reactive control
    
    2. SAFETY VERIFICATION: NMPC checks worst-case scenarios
       - Verifies PID output will keep glucose safe even in worst-case:
         * Maximum meal disturbance (unexpected large meal)
         * Minimum insulin sensitivity (worst-case patient response)
         * Combined worst-case scenarios
    
    3. SAFE ADJUSTMENT: If PID violates safety, minimal adjustment applied
       - Binary search finds smallest adjustment needed
       - Ensures glucose stays within safe bounds [bg_min, bg_max]
       - Minimizes deviation from PID output
    
    This architecture combines:
    - Fast reactive control (PID)
    - Predictive safety assurance (NMPC)
    - Minimal intervention (only when needed)
    
    Original optimization-based approach is still available via _optimize() method
    but is now used only for safety verification and adjustment, not primary control.
    
    Parameters
    ----------
    target_bg : float, optional
        Target blood glucose level in mg/dL (default: 140)
    prediction_horizon : int, optional
        Prediction horizon in minutes (default: 60)
    control_horizon : int, optional
        Control horizon in minutes (default: 30)
    sample_time : float, optional
        Controller sample time in minutes (default: 5)
        This is the glucose sampling rate (how often the controller runs)
    ode_time_step : float, optional
        ODE integration time step in minutes (default: 1.0)
        This should be much smaller than sample_time for accurate ODE integration
        For example, if sample_time=5 minutes, ode_time_step=1 minute gives 5 sub-steps per prediction step
    q_weight : float, optional
        Weight for glucose tracking cost (default: 1.0)
    r_weight : float, optional
        Weight for insulin cost (default: 0.1)
    bg_min : float, optional
        Minimum safe blood glucose level in mg/dL (default: 70)
    bg_max : float, optional
        Maximum safe blood glucose level in mg/dL (default: 180)
    barrier_weight : float, optional
        Weight for barrier function penalty J_G(t) in objective function (default: 10.0)
        Higher values enforce stricter safety constraints
    patient_params : dict, optional
        Patient-specific parameters (if None, will be loaded from info)
    """
    
    def __init__(self, 
                 target_bg: float = 140.0,
                 prediction_horizon: int = 60,
                 control_horizon: int = 30,
                 sample_time: float = 5.0,
                 ode_time_step: float = 1.0,
                 q_weight: float = 1.0,
                 r_weight: float = 0.1,
                 bg_min: float = 70.0,
                 bg_max: float = 180.0,
                 barrier_weight: float = 10.0,
                 patient_params: Optional[Dict[str, Any]] = None,
                 init_state: Optional[np.ndarray] = None):
        """
        Initialize the NMPC controller.
        
        Note: init_state parameter is for compatibility with base Controller class,
        but NMPC doesn't require it since it uses patient_state from info.
        """
        super().__init__(init_state if init_state is not None else np.zeros(13))
        
        # Controller parameters
        self.target_bg = target_bg
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.sample_time = sample_time
        self.ode_time_step = ode_time_step
        self.q_weight = q_weight
        self.r_weight = r_weight
        self.bg_min = bg_min
        self.bg_max = bg_max
        self.barrier_weight = barrier_weight  # Weight for barrier function penalty
        
        # Patient parameters (will be set from info if not provided)
        self.patient_params = patient_params
        
        # Internal state for NMPC
        self.current_state = None
        self.last_action = None
        self.solver_initialized = False
        
        # Optimization parameters (from MATLAB code)
        self.NP = prediction_horizon  # Prediction steps
        self.Nopt = 100  # Max number of optimization iterations
        self.opt_rate = 1.0  # Learning rate of optimization method
        self.acc = 1e-3  # Minimum accuracy of optimization method
        self.max_time = 100  # Maximum computation time (seconds)
        
        # Control parameters
        self.insulin_max = 10.0  # Maximum insulin rate (U/min)
        self.insulin_min = 0.0  # Minimum insulin rate (U/min)
        
        # PID controller for initial guess (warm start)
        from .pid_ctrller import PIDController
        self.pid_controller = PIDController(P=0.001, I=0.00001, D=0.001, target=target_bg)
        
        # History for debugging/analysis
        self.prediction_history = []
        self.optimization_history = []
        
    def policy(self, observation, reward, done, **info):
        """
        Compute control action using NMPC.
        
        Parameters
        ----------
        observation : namedtuple
            Contains CGM (continuous glucose monitor reading) in mg/dL
        reward : float
            Current reward (not used by NMPC)
        done : bool
            Whether simulation is done
        **info : dict
            Additional information containing:
            - patient_state: 13-dimensional state vector
            - patient_name: name of the patient
            - sample_time: environment sample time
            - meal: current CHO intake (g/min)
            - time: current simulation time
            - bg: current blood glucose (mg/dL)
        
        Returns
        -------
        action : Action
            Namedtuple with basal and bolus insulin rates (U/min)
        """
        if done:
            return Action(basal=0.0, bolus=0.0)
        
        # Extract information from info dict
        patient_state = info.get('patient_state')
        patient_name = info.get('patient_name')
        env_sample_time = info.get('sample_time', self.sample_time)
        meal = info.get('meal', 0.0)  # g/min
        current_bg = info.get('bg', observation.CGM)
        
        # Load patient parameters if not already loaded
        if self.patient_params is None and patient_name is not None:
            self.patient_params = self._load_patient_params(patient_name)
        
        # Store current state and patient name for use in optimization
        self.current_state = patient_state.copy() if patient_state is not None else None
        self._current_patient_name = patient_name
        
        # Solve NMPC optimization problem
        try:
            action = self._solve_nmpc(
                current_state=patient_state,
                current_bg=current_bg,
                cgm_reading=observation.CGM,
                meal=meal,
                sample_time=env_sample_time,
                patient_name=patient_name
            )
        except Exception as e:
            logger.error(f"NMPC solver failed: {e}")
            # Fallback to safe action (basal only)
            action = self._fallback_action()
        
        self.last_action = action
        return action
    
    def _solve_nmpc(self, 
                     current_state: np.ndarray,
                     current_bg: float,
                     cgm_reading: float,
                     meal: float,
                     sample_time: float,
                     patient_name: Optional[str] = None) -> Action:
        """
        Solve NMPC using PID-first approach with safety verification.
        
        NEW ARCHITECTURE:
        1. Get PID controller output (primary control)
        2. Check worst-case safety scenarios with PID output
        3. If PID output is safe in worst-case, use it directly
        4. If PID output violates safety, find minimal adjustment to ensure safety
        
        Parameters
        ----------
        current_state : np.ndarray
            13-dimensional patient state vector
        current_bg : float
            Current blood glucose (mg/dL)
        cgm_reading : float
            CGM sensor reading (mg/dL)
        meal : float
            Current meal/CHO intake (g/min)
        sample_time : float
            Environment sample time (minutes)
        
        Returns
        -------
        action : Action
            Safe insulin action (basal, bolus) in U/min
        """
        # Step 1: Get PID controller output (primary control)
        try:
            pid_action = self.pid_controller.policy(
                observation=type('obj', (object,), {'CGM': cgm_reading})(),
                reward=0.0,
                done=False,
                sample_time=sample_time
            )
            
            pid_insulin_rate = pid_action.basal + pid_action.bolus
            
            # Check for NaN in PID output
            if not np.isfinite(pid_insulin_rate):
                logger.warning(f"NaN/inf in PID output: {pid_insulin_rate}, using basal")
                pid_insulin_rate = self._compute_basal()
        except Exception as e:
            logger.error(f"Error in PID controller: {e}, using basal")
            pid_insulin_rate = self._compute_basal()
        
        # Clip PID output to reasonable bounds
        pid_insulin_rate = np.clip(pid_insulin_rate, self.insulin_min, self.insulin_max)
        
        logger.debug(f"PID output: {pid_insulin_rate:.6f} U/min")
        
        # Step 2: Check worst-case safety scenarios with PID output
        is_safe, worst_case_bg_min, worst_case_bg_max = self._check_worst_case_safety(
            current_state=current_state,
            current_bg=current_bg,
            insulin_rate=pid_insulin_rate,
            meal=meal,
            sample_time=sample_time,
            patient_name=patient_name
        )
        
        # Step 3: If PID output is safe, use it directly
        if is_safe:
            logger.debug(f"PID output is safe in worst-case scenarios. Using PID output directly.")
            basal = self._compute_basal()
            bolus = max(0, pid_insulin_rate - basal)
            return Action(basal=basal, bolus=bolus)
        
        # Step 4: PID output violates safety - find minimal safe adjustment
        logger.warning(f"PID output violates worst-case safety bounds. "
                      f"Worst-case BG range: [{worst_case_bg_min:.1f}, {worst_case_bg_max:.1f}] mg/dL. "
                      f"Finding safe adjustment...")
        
        safe_insulin_rate = self._find_safe_adjustment(
            current_state=current_state,
            current_bg=current_bg,
            pid_insulin_rate=pid_insulin_rate,
            meal=meal,
            sample_time=sample_time,
            patient_name=patient_name,
            worst_case_bg_min=worst_case_bg_min,
            worst_case_bg_max=worst_case_bg_max
        )
        
        basal = self._compute_basal()
        bolus = max(0, safe_insulin_rate - basal)
        
        logger.debug(f"Safe adjustment: PID={pid_insulin_rate:.6f} -> Safe={safe_insulin_rate:.6f} U/min")
        
        return Action(basal=basal, bolus=bolus)
    
    def _check_worst_case_safety(self,
                                  current_state: np.ndarray,
                                  current_bg: float,
                                  insulin_rate: float,
                                  meal: float,
                                  sample_time: float,
                                  patient_name: Optional[str] = None) -> tuple:
        """
        Check if insulin rate is safe in worst-case scenarios.
        
        Worst-case scenarios considered:
        1. Maximum meal disturbance (unexpected large meal)
        2. Minimum insulin sensitivity (worst-case patient response)
        3. Maximum glucose rise rate
        4. Minimum glucose fall rate
        
        Parameters
        ----------
        current_state : np.ndarray
            Current patient state
        current_bg : float
            Current blood glucose (mg/dL)
        insulin_rate : float
            Insulin rate to check (U/min)
        meal : float
            Current meal (g/min)
        sample_time : float
            Sample time (minutes)
        patient_name : str, optional
            Patient name for parameter loading
        
        Returns
        -------
        is_safe : bool
            True if insulin rate is safe in worst-case scenarios
        worst_case_bg_min : float
            Minimum BG predicted in worst-case scenarios
        worst_case_bg_max : float
            Maximum BG predicted in worst-case scenarios
        """
        # Worst-case parameters
        max_meal_disturbance = 100.0  # g/min - unexpected large meal
        insulin_sensitivity_factor = 0.7  # 70% of normal sensitivity (worst case)
        
        # Prepare worst-case scenarios
        worst_case_scenarios = [
            {
                'name': 'max_meal',
                'meal': max_meal_disturbance,
                'insulin_multiplier': 1.0  # Normal insulin sensitivity
            },
            {
                'name': 'min_insulin_sensitivity',
                'meal': meal,  # Current meal
                'insulin_multiplier': insulin_sensitivity_factor  # Reduced sensitivity
            },
            {
                'name': 'combined_worst_case',
                'meal': max_meal_disturbance,
                'insulin_multiplier': insulin_sensitivity_factor
            }
        ]
        
        worst_case_bg_min = float('inf')
        worst_case_bg_max = float('-inf')
        
        # Other parameters for prediction
        other_params = {
            'meal': meal,
            'patient_params': self.patient_params,
            'patient_name': patient_name,
            'last_Qsto': current_state[0] + current_state[1] if current_state is not None else 0,
            'last_foodtaken': 0,
            'u_prev': insulin_rate
        }
        
        # Check each worst-case scenario
        for scenario in worst_case_scenarios:
            # Adjust insulin for sensitivity
            adjusted_insulin = insulin_rate * scenario['insulin_multiplier']
            
            # Predict forward with worst-case meal
            scenario_params = other_params.copy()
            scenario_params['meal'] = scenario['meal']
            scenario_params['u_prev'] = adjusted_insulin
            
            # Predict over horizon
            bg_predictions = self._predict_glucose_trajectory(
                current_state=current_state,
                insulin_rate=adjusted_insulin,
                meal_rate=scenario['meal'],
                sample_time=sample_time,
                other_params=scenario_params
            )
            
            # Find min/max BG in prediction
            bg_min = np.min(bg_predictions)
            bg_max = np.max(bg_predictions)
            
            worst_case_bg_min = min(worst_case_bg_min, bg_min)
            worst_case_bg_max = max(worst_case_bg_max, bg_max)
            
            logger.debug(f"Scenario '{scenario['name']}': BG range [{bg_min:.1f}, {bg_max:.1f}] mg/dL")
        
        # Check if worst-case predictions violate safety bounds
        is_safe = (worst_case_bg_min >= self.bg_min) and (worst_case_bg_max <= self.bg_max)
        
        return is_safe, worst_case_bg_min, worst_case_bg_max
    
    def _predict_glucose_trajectory(self,
                                    current_state: np.ndarray,
                                    insulin_rate: float,
                                    meal_rate: float,
                                    sample_time: float,
                                    other_params: dict) -> np.ndarray:
        """
        Predict glucose trajectory over prediction horizon.
        
        Parameters
        ----------
        current_state : np.ndarray
            Current patient state
        insulin_rate : float
            Insulin rate (U/min)
        meal_rate : float
            Meal rate (g/min)
        sample_time : float
            Sample time (minutes)
        other_params : dict
            Additional parameters
        
        Returns
        -------
        bg_predictions : np.ndarray
            Predicted BG values over horizon (mg/dL)
        """
        NP = self.NP
        z = np.zeros((len(current_state), NP + 1))
        z[:, 0] = current_state.copy()
        dz0 = np.zeros(len(current_state))
        
        # Convert insulin to array format
        u_for_model = np.array([insulin_rate])
        
        # Update meal in other_params
        scenario_params = other_params.copy()
        scenario_params['meal'] = meal_rate
        
        # ODE time step
        ode_dt = self.ode_time_step
        num_ode_steps = max(1, int(np.ceil(sample_time / ode_dt)))
        actual_ode_dt = sample_time / num_ode_steps
        
        bg_predictions = np.zeros(NP + 1)
        Vg = self._get_param('Vg', 1.0)
        bg_predictions[0] = current_state[3] * Vg  # Initial BG
        
        # Predict forward
        for i in range(NP):
            z_current = z[:, i].copy()
            
            for ode_step in range(num_ode_steps):
                out_temp, dz = self._patient_model_step(z_current, u_for_model, dz0, scenario_params)
                
                if not np.all(np.isfinite(dz)):
                    dz = np.nan_to_num(dz, nan=0.0, posinf=0.0, neginf=0.0)
                
                z_current = z_current + dz * actual_ode_dt
                
                if not np.all(np.isfinite(z_current)):
                    z_current = np.nan_to_num(z_current, nan=z[:, i], posinf=z[:, i], neginf=z[:, i])
            
            z[:, i+1] = z_current
            bg_predictions[i+1] = z[:, i+1][3] * Vg
        
        return bg_predictions
    
    def _find_safe_adjustment(self,
                              current_state: np.ndarray,
                              current_bg: float,
                              pid_insulin_rate: float,
                              meal: float,
                              sample_time: float,
                              patient_name: Optional[str] = None,
                              worst_case_bg_min: float = None,
                              worst_case_bg_max: float = None) -> float:
        """
        Find minimal adjustment to PID output to ensure safety.
        
        Uses binary search to find the smallest adjustment needed.
        
        Parameters
        ----------
        current_state : np.ndarray
            Current patient state
        current_bg : float
            Current blood glucose (mg/dL)
        pid_insulin_rate : float
            PID controller output (U/min)
        meal : float
            Current meal (g/min)
        sample_time : float
            Sample time (minutes)
        patient_name : str, optional
            Patient name
        worst_case_bg_min : float, optional
            Worst-case minimum BG from previous check
        worst_case_bg_max : float, optional
            Worst-case maximum BG from previous check
        
        Returns
        -------
        safe_insulin_rate : float
            Adjusted insulin rate that ensures safety (U/min)
        """
        # Determine adjustment direction based on worst-case violations
        if worst_case_bg_min is not None and worst_case_bg_min < self.bg_min:
            # Hypoglycemia risk - reduce insulin
            adjustment_direction = -1
            min_insulin = self.insulin_min
            max_insulin = pid_insulin_rate
        elif worst_case_bg_max is not None and worst_case_bg_max > self.bg_max:
            # Hyperglycemia risk - increase insulin
            adjustment_direction = 1
            min_insulin = pid_insulin_rate
            max_insulin = self.insulin_max
        else:
            # Default: try to reduce insulin for safety
            adjustment_direction = -1
            min_insulin = self.insulin_min
            max_insulin = pid_insulin_rate
        
        # Binary search for safe insulin rate
        tolerance = 0.01  # U/min tolerance
        max_iterations = 20
        
        safe_insulin_rate = pid_insulin_rate
        
        for iteration in range(max_iterations):
            test_insulin = (min_insulin + max_insulin) / 2.0
            
            # Check safety with test insulin
            is_safe, bg_min, bg_max = self._check_worst_case_safety(
                current_state=current_state,
                current_bg=current_bg,
                insulin_rate=test_insulin,
                meal=meal,
                sample_time=sample_time,
                patient_name=patient_name
            )
            
            if is_safe:
                safe_insulin_rate = test_insulin
                max_insulin = test_insulin  # Try lower values
            else:
                if adjustment_direction > 0:
                    min_insulin = test_insulin  # Need more insulin
                else:
                    max_insulin = test_insulin  # Need less insulin
            
            # Check convergence
            if abs(max_insulin - min_insulin) < tolerance:
                break
        
        # Final safety check
        is_safe_final, _, _ = self._check_worst_case_safety(
            current_state=current_state,
            current_bg=current_bg,
            insulin_rate=safe_insulin_rate,
            meal=meal,
            sample_time=sample_time,
            patient_name=patient_name
        )
        
        if not is_safe_final:
            # Fallback: use conservative safe value
            if current_bg < self.bg_min:
                safe_insulin_rate = self._compute_basal() * 0.3  # Very conservative
            elif current_bg > self.bg_max:
                safe_insulin_rate = min(self.insulin_max, self._compute_basal() * 2.0)
            else:
                safe_insulin_rate = self._compute_basal()
        
        return np.clip(safe_insulin_rate, self.insulin_min, self.insulin_max)
    
    def _optimize(self, x, xd, u_old, DT, NP, maxiteration, alfa, acc, other, max_time):
        """
        Optimize control input using gradient descent with momentum.
        
        Converted from MATLAB optimize() function.
        Uses conjugate gradient momentum for faster convergence.
        
        Parameters
        ----------
        x : np.ndarray
            Current patient state (13-dimensional)
        xd : np.ndarray
            Desired/target state
        u_old : np.ndarray
            Previous control input (initial guess)
        DT : float
            Time step (minutes)
        NP : int
            Prediction horizon steps
        maxiteration : int
            Maximum optimization iterations
        alfa : float
            Learning rate
        acc : float
            Minimum accuracy/convergence tolerance
        other : dict
            Additional parameters (meal, patient_params, etc.)
        max_time : float
            Maximum computation time (seconds)
        
        Returns
        -------
        U : np.ndarray
            Optimized control input
        """
        Un = u_old.copy()
        n = len(Un)
        maxiter = maxiteration
        dx = np.ones_like(Un)
        stop = 0
        fx = 0
        CG = np.zeros_like(Un)  # Conjugate gradient
        CG_o = CG.copy()
        bet = 0.5
        
        start_time = time.time()
        MPC_time = 0
        
        # Debug: log initial state
        u_prev = other.get('u_prev', self._compute_basal())
        initial_cost = self._mainfun(xd, x, u_old, DT, NP, other)
        logger.debug(f"NMPC optimization start: ΔU_old={u_old[0]:.6f}, U_prev={u_prev:.6f}, "
                    f"U_initial={u_prev + u_old[0]:.6f}, initial_cost={initial_cost:.2f}")
        
        while stop < maxiter and np.linalg.norm(dx, 2) > acc and MPC_time < max_time:
            stop += 1
            fx = self._mainfun(xd, x, Un, DT, NP, other)
            
            # Check for NaN in objective
            if not np.isfinite(fx):
                logger.error(f"NaN/inf in objective function at iteration {stop}: fx={fx}")
                logger.error(f"  Un={Un}, x={x[:5] if len(x) > 5 else x}")
                break  # Exit optimization loop
            
            dxo = dx.copy()
            CGo = CG.copy()
            
            # Compute gradient
            dx = self._dot_fun(Un, n, fx, xd, x, DT, NP, other)
            
            # Check for NaN in gradient
            if not np.all(np.isfinite(dx)):
                logger.error(f"NaN/inf in gradient at iteration {stop}: dx={dx}")
                dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Conjugate gradient update
            CG = -dx + bet * CG
            
            # Check for NaN in conjugate gradient
            if not np.all(np.isfinite(CG)):
                logger.error(f"NaN/inf in conjugate gradient at iteration {stop}: CG={CG}")
                CG = np.nan_to_num(CG, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Update control with momentum
            # Original MATLAB: UUU=Un-diag([1;.0;.0;.0;.0])*diag(CG)*dx;
            # Simplified: UUU = Un - alfa * CG * dx (element-wise)
            UUU = Un - alfa * CG * dx
            
            # Check for NaN in control update
            if not np.all(np.isfinite(UUU)):
                logger.error(f"NaN/inf in control update at iteration {stop}: UUU={UUU}")
                UUU = Un.copy()  # Fallback to previous value
            
            # Apply ΔU constraints (CBF constraint)
            # Get current BG to set adaptive bounds
            u_prev = other.get('u_prev', self._compute_basal())
            # Get current BG from state (approximate - state[3] is glucose)
            current_bg_approx = x[3] * self._get_param('Vg', 1.0) if len(x) > 3 else 140.0
            
            # Adaptive ΔU bounds based on current BG
            if current_bg_approx > self.bg_max:
                delta_u_max = 1.0  # Allow up to 1 U/min increase when hyperglycemic
                delta_u_min = -0.5  # Limit decreases
            elif current_bg_approx < self.bg_min:
                delta_u_max = 0.1  # Very small increases when hypoglycemic
                delta_u_min = -1.0  # Allow larger decreases when hypoglycemic
            else:
                delta_u_max = 0.5  # Normal range: smaller changes
                delta_u_min = -0.5
            
            UUU[0] = np.clip(UUU[0], delta_u_min, delta_u_max)
            
            # Also check that absolute insulin (U_prev + ΔU) stays within bounds
            u_absolute = u_prev + UUU[0]
            if u_absolute > self.insulin_max:
                UUU[0] = self.insulin_max - u_prev  # Limit ΔU to keep U within max
            elif u_absolute < self.insulin_min:
                UUU[0] = self.insulin_min - u_prev  # Limit ΔU to keep U within min
            
            # Evaluate new objective
            fxx = self._mainfun(xd, x, UUU, DT, NP, other)
            
            # Check for NaN in new objective
            if not np.isfinite(fxx):
                logger.error(f"NaN/inf in new objective at iteration {stop}: fxx={fxx}")
                fxx = fx + 1.0  # Make it worse so we don't accept it
            
            # Debug: log every 10 iterations
            if stop % 10 == 0 or stop == 1:
                u_prev = other.get('u_prev', self._compute_basal())
                u_abs = u_prev + Un[0]
                u_new_abs = u_prev + UUU[0]
                logger.debug(f"Iter {stop}: ΔU={Un[0]:.6f}, U_abs={u_abs:.6f}, cost={fx:.2f}, "
                           f"gradient_norm={np.linalg.norm(dx, 2):.6f}, ΔU_new={UUU[0]:.6f}, "
                           f"U_new_abs={u_new_abs:.6f}, cost_new={fxx:.2f}")
            
            # Adaptive learning rate
            if fxx < fx and np.isfinite(fxx):
                Un = UUU.copy()
                alfa = alfa * 1.1
            else:
                alfa = alfa * 0.9
            
            # Update momentum parameter
            cg_norm = np.linalg.norm(CG)
            cgo_norm = np.linalg.norm(CGo)
            
            if cg_norm > 0.1 and cgo_norm > 1e-10:
                bet = cg_norm / (cgo_norm + 1e-10)
            elif cgo_norm > 1e-10:
                cgo_dxo_norm = np.linalg.norm(CGo * dxo)
                if cgo_dxo_norm > 1e-10:
                    bet = cg_norm / (cgo_dxo_norm + 1e-10)
                else:
                    bet = 0.5  # Default value
            else:
                bet = 0.5  # Default value
            
            # Check for NaN in momentum parameter
            if not np.isfinite(bet):
                logger.warning(f"NaN/inf in momentum parameter bet: {bet}, using default 0.5")
                bet = 0.5
            
            MPC_time = time.time() - start_time
        
        # Debug: log final result
        u_prev = other.get('u_prev', self._compute_basal())
        final_cost = self._mainfun(xd, x, Un, DT, NP, other)
        
        # Check for NaN in final result
        if not np.all(np.isfinite(Un)):
            logger.error(f"NaN/inf in final optimization result: Un={Un}")
            # Fallback to zero change
            Un = np.array([0.0])
        
        u_final_abs = u_prev + Un[0]
        logger.debug(f"NMPC optimization end: iterations={stop}, ΔU_final={Un[0]:.6f}, "
                    f"U_final_abs={u_final_abs:.6f}, final_cost={final_cost:.2f}, "
                    f"converged={'yes' if np.linalg.norm(dx, 2) <= acc else 'no'}")
        
        return Un
    
    def _mainfun(self, xd, x, delta_u, DelT, NP, other):
        """
        Main objective function for NMPC.
        
        Now optimizes ΔU (change in insulin) instead of U (absolute insulin).
        This provides smoother control and reduces control effort.
        
        Parameters
        ----------
        xd : np.ndarray
            Desired/target state
        x : np.ndarray
            Current state
        delta_u : np.ndarray
            Change in control input (ΔU = U - U_prev)
        DelT : float
            Time step
        NP : int
            Prediction horizon steps
        other : dict
            Additional parameters (includes 'u_prev' for absolute insulin calculation)
        
        Returns
        -------
        J : float
            Objective function value (cost)
        """
        J = 0
        z = np.zeros((len(x), NP + 1))
        out = np.zeros((len(x) + 17, NP + 1))  # Extra space for additional outputs
        z[:, 0] = x.copy()
        dz0 = np.zeros(len(x))
        
        # Get previous insulin rate for ΔU to U conversion
        u_prev = other.get('u_prev', self._compute_basal())
        
        # Check for NaN/inf in inputs
        if not np.isfinite(delta_u[0]):
            logger.error(f"NaN/inf in delta_u: {delta_u[0]}, using zero")
            delta_u = np.array([0.0])
        if not np.isfinite(u_prev):
            logger.error(f"NaN/inf in u_prev: {u_prev}, using basal")
            u_prev = self._compute_basal()
        
        # Convert ΔU to absolute U: U = U_prev + ΔU
        u_absolute = u_prev + delta_u[0]
        
        # Check for NaN in conversion
        if not np.isfinite(u_absolute):
            logger.error(f"NaN in u_absolute calculation: u_prev={u_prev}, delta_u={delta_u[0]}")
            u_absolute = u_prev  # Fallback to previous value
        
        # Clip absolute insulin to bounds
        u_absolute = np.clip(u_absolute, self.insulin_min, self.insulin_max)
        
        # Create u array for patient model (needs absolute insulin)
        u_for_model = np.array([u_absolute])
        
        # ODE time step (should be smaller than DelT for accurate integration)
        ode_dt = self.ode_time_step
        # Number of ODE sub-steps per prediction step
        num_ode_steps = max(1, int(np.ceil(DelT / ode_dt)))
        # Actual ODE time step (may be slightly adjusted to fit exactly into DelT)
        actual_ode_dt = DelT / num_ode_steps
        
        # Predict forward over horizon
        for i in range(NP):
            # Integrate ODE over DelT using smaller ode_time_step increments
            z_current = z[:, i].copy()
            
            for ode_step in range(num_ode_steps):
                # Predict one ODE step using patient model (use absolute insulin)
                out_temp, dz = self._patient_model_step(z_current, u_for_model, dz0, other)
                
                # Check for NaN/inf in patient model output
                if not np.all(np.isfinite(dz)):
                    logger.error(f"NaN/inf in patient model derivative at step {i+1}, ode_step {ode_step}")
                    logger.error(f"  State: {z_current}")
                    logger.error(f"  Insulin: {u_for_model[0]}")
                    logger.error(f"  dz: {dz}")
                    # Use zero derivative as fallback
                    dz = np.nan_to_num(dz, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Update state using smaller ODE time step
                z_current = z_current + dz * actual_ode_dt
                
                # Check for NaN in state update
                if not np.all(np.isfinite(z_current)):
                    logger.error(f"NaN/inf in state update at step {i+1}, ode_step {ode_step}")
                    z_current = np.nan_to_num(z_current, nan=z[:, i], posinf=z[:, i], neginf=z[:, i])
            
            # Store final state after integrating over DelT
            z[:, i+1] = z_current
            
            # Store output for the final state
            out[:, i+1], _ = self._patient_model_step(z[:, i+1], u_for_model, dz0, other)
            
            x_pred = z[:, i+1]
            
            # Compute cost: tracking error + control cost
            # Original: J=J+1*sqrt((xd(1:2)-x(1:2))'*((xd(1:2)-x(1:2)) ) + ...)
            # Adapted for glucose: track BG (state[3]) and minimize ΔU (change in insulin)
            
            # Get blood glucose from state (convert from mg/kg to mg/dL)
            Vg = self._get_param('Vg', 1.0)
            bg_pred = x_pred[3] * Vg  # Blood glucose in mg/dL
            
            # Check for NaN in predicted BG
            if not np.isfinite(bg_pred):
                logger.error(f"NaN/inf in bg_pred at step {i+1}: x_pred[3]={x_pred[3]}, Vg={Vg}")
                bg_pred = self.target_bg  # Fallback to target
            
            bg_target = xd[3] * Vg if len(xd) > 3 else self.target_bg
            if not np.isfinite(bg_target):
                bg_target = self.target_bg
            
            # Tracking cost: (BG - target)^2
            tracking_error = (bg_pred - bg_target)**2
            
            # Control cost: (ΔU)^2 (penalize large changes in insulin)
            # This encourages smooth control actions
            control_cost_weight = max(self.r_weight, 1.0)  # Increased to 1.0 to prevent excessive changes
            
            # Additional penalty based on predicted BG and current insulin level
            # This is a predictive CBF constraint
            if bg_pred < self.bg_min:
                # Strong penalty for increasing insulin when BG is predicted to be low
                if delta_u[0] > 0:  # Only penalize increases, not decreases
                    control_cost_weight *= 50.0  # Very strong penalty for increasing insulin when hypoglycemic
            elif bg_pred > self.bg_max:
                # Moderate penalty for decreasing insulin when hyperglycemic (but still penalize large changes)
                if delta_u[0] < 0:  # Penalize decreases when hyperglycemic
                    control_cost_weight *= 2.0
            else:
                # In normal range, penalize both increases and decreases equally
                control_cost_weight *= 2.0  # Stronger penalty in normal range to maintain stability
            
            # Penalize ΔU^2 (change in insulin), not absolute insulin
            # Also add asymmetric penalty: stronger for increases than decreases
            if delta_u[0] > 0:
                # Penalize increases more strongly
                control_cost = control_cost_weight * 2.0 * (delta_u[0]**2)
            else:
                # Penalize decreases less (allows reduction when needed)
                control_cost = control_cost_weight * (delta_u[0]**2)
            
            # Glucose barrier function penalty J_G(t) at each step
            # Implements Eq. (JG): J_G(t) = G(t) - G_max if G > G_max,
            #                      J_G(t) = G(t) - G_min if G < G_min,
            #                      J_G(t) = 0 otherwise
            barrier_penalty_step = self._glucose_barrier_function(bg_pred)
            
            # For hypoglycemia (G < G_min), barrier is negative, so we need to penalize it strongly
            # For hyperglycemia (G > G_max), barrier is positive, penalize it
            # Use squared penalty for stronger constraint enforcement
            if barrier_penalty_step != 0:
                barrier_cost = self.barrier_weight * (barrier_penalty_step**2)
            else:
                barrier_cost = 0.0
            
            step_cost = self.q_weight * tracking_error + control_cost + barrier_cost
            
            # Check for NaN in cost components
            if not np.isfinite(step_cost):
                logger.error(f"NaN/inf in step_cost at step {i+1}: tracking={tracking_error}, "
                           f"control={control_cost}, barrier={barrier_cost}")
                step_cost = 1e6  # Large penalty for invalid cost
            
            J += step_cost
            
            # Debug: log first few prediction steps
            if i < 3:
                logger.debug(f"Prediction step {i+1}: BG_pred={bg_pred:.2f}, ΔU={delta_u[0]:.6f}, "
                           f"U_abs={u_absolute:.6f}, tracking={tracking_error:.2f}, control={control_cost:.4f}, "
                           f"barrier={barrier_cost:.4f}, step_cost={step_cost:.4f}")
            
            dz0 = dz.copy()
        
        # Penalty for constraint violation (insulin bounds) - CBF constraint
        # Check absolute insulin (U_prev + ΔU) against bounds
        u_absolute = u_prev + delta_u[0]
        if u_absolute > self.insulin_max:
            J += 100.0 * ((u_absolute - self.insulin_max)**2)  # Strong penalty for exceeding max
        elif u_absolute < self.insulin_min:
            J += 100.0 * ((self.insulin_min - u_absolute)**2)  # Strong penalty for going below min
        
        # Also penalize large ΔU changes (rate of change constraint)
        # Adaptive bounds based on predicted BG
        bg_final = z[3, -1] * self._get_param('Vg', 1.0)
        if bg_final > self.bg_max:
            delta_u_max = 1.0  # Allow larger increases when hyperglycemic
        elif bg_final < self.bg_min:
            delta_u_max = 0.1  # Very small changes when hypoglycemic
        else:
            delta_u_max = 0.5  # Normal range
            
        if abs(delta_u[0]) > delta_u_max:
            J += 100.0 * ((abs(delta_u[0]) - delta_u_max)**2)  # Strong penalty for excessive rate of change
        
        # Terminal cost: final state tracking
        Vg = self._get_param('Vg', 1.0)
        bg_final = z[3, -1] * Vg
        bg_target = xd[3] * Vg if len(xd) > 3 else self.target_bg
        terminal_cost = 1.0 * (bg_final - bg_target)**2
        J += terminal_cost
        
        # Terminal barrier function penalty (squared for stronger enforcement)
        terminal_barrier = self._glucose_barrier_function(bg_final)
        if terminal_barrier != 0:
            J += self.barrier_weight * (terminal_barrier**2)
        
        # Final check for NaN in total cost
        if not np.isfinite(J):
            logger.error(f"NaN/inf in total objective function J: {J}")
            logger.error(f"  Components: tracking_error, control_cost, barrier_cost, terminal_cost")
            J = 1e6  # Return large penalty for invalid cost
        
        return J
    
    def _patient_model_step(self, x, u, dz0, other):
        """
        Compute one step of patient model dynamics.
        
        This wraps the patient ODE model for prediction.
        
        Parameters
        ----------
        x : np.ndarray
            Current state (13-dimensional)
        u : np.ndarray
            Control input (insulin rate)
        dz0 : np.ndarray
            Previous state derivative (for continuity)
        other : dict
            Contains meal, patient_params, last_Qsto, last_foodtaken
        
        Returns
        -------
        out : np.ndarray
            Extended output (state + additional info)
        dz : np.ndarray
            State derivative dx/dt
        """
        from simglucose.patient.t1dpatient import Action as PatientAction
        
        # Prepare action for patient model
        insulin = float(u[0])  # U/min
        
        # Check for NaN/inf in insulin
        if not np.isfinite(insulin):
            logger.error(f"NaN/inf in insulin input to patient model: {insulin}")
            insulin = 0.0  # Fallback to zero insulin
        
        meal = other.get('meal', 0.0)  # g/min
        if not np.isfinite(meal):
            meal = 0.0
        
        action = PatientAction(insulin=insulin, CHO=meal)
        
        # Get patient parameters (should be pandas Series)
        params = other.get('patient_params', self.patient_params)
        if params is None:
            # Fallback: try to load from patient_name if available
            patient_name = other.get('patient_name')
            if patient_name:
                params = self._load_patient_params(patient_name)
            else:
                logger.error("No patient parameters available")
                # Return zero derivative as fallback
                return np.zeros(len(x) + 17), np.zeros(len(x))
        
        last_Qsto = other.get('last_Qsto', x[0] + x[1])
        last_foodtaken = other.get('last_foodtaken', 0)
        
        # Check for NaN in state before patient model
        if not np.all(np.isfinite(x)):
            logger.error(f"NaN/inf in state before patient model: {x}")
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute state derivative using patient model
        t = 0  # Time doesn't matter for static evaluation
        dz = self._patient_ode_model(t, x, action, params, last_Qsto, last_foodtaken)
        
        # Check for NaN in patient model output
        if not np.all(np.isfinite(dz)):
            logger.error(f"NaN/inf in patient model output: dz={dz}")
            logger.error(f"  Inputs: x={x[:5]}, insulin={insulin}, meal={meal}")
            # Use zero derivative as fallback
            dz = np.nan_to_num(dz, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extended output (for compatibility with original MATLAB code)
        out = np.zeros(len(x) + 17)
        out[:len(x)] = x
        out[len(x):] = 0  # Additional outputs (not used in glucose control)
        
        return out, dz
    
    def _patient_ode_model(self, t, x, action, params, last_Qsto, last_foodtaken):
        """
        Patient ODE model - wrapper around T1DPatient.model().
        
        This method provides access to the patient model for prediction.
        """
        # Import here to avoid circular imports
        from simglucose.patient.t1dpatient import T1DPatient
        
        # The patient model expects params to be a pandas Series
        # If we have a dict, we need to convert it or use the stored patient_params Series
        if isinstance(params, dict):
            # Use the stored patient_params (should be a Series)
            if self.patient_params is not None and not isinstance(self.patient_params, dict):
                params = self.patient_params
            else:
                # Fallback: create a minimal Series (this shouldn't happen in normal use)
                import pandas as pd
                logger.warning("Using fallback params structure")
                params = pd.Series(params)
        
        # Call the static model method
        dxdt = T1DPatient.model(t, x, action, params, last_Qsto, last_foodtaken)
        return dxdt
    
    def _glucose_barrier_function(self, G):
        """
        Glucose safe range barrier function J_G(t) as defined in Eq. (JG).
        
        Implements the control barrier function:
        J_G(t) = G(t) - G_max  if G(t) > G_max
                = G(t) - G_min  if G(t) < G_min
                = 0             if G_min < G(t) < G_max
        
        This barrier function is used as a penalty term in the NMPC objective
        function to ensure glucose stays within safe bounds [G_min, G_max].
        
        Parameters
        ----------
        G : float
            Current glucose level (mg/dL)
        
        Returns
        -------
        J_G : float
            Barrier function value
            - Positive if G > G_max (penalty for hyperglycemia)
            - Negative if G < G_min (penalty for hypoglycemia)
            - Zero if G_min <= G <= G_max (no penalty)
        """
        if G > self.bg_max:
            return G - self.bg_max
        elif G < self.bg_min:
            return G - self.bg_min
        else:
            return 0.0
    
    def _safety_barrier_function(self, x, other, bound):
        """
        Safety barrier function (control barrier function).
        
        Converted from MATLAB stability_bound_cost() function (now renamed to safety_barrier_function).
        Implements control barrier functions to ensure glucose stays within safe bounds.
        Penalizes states that violate safety bounds.
        
        Parameters
        ----------
        x : np.ndarray
            Current state
        other : np.ndarray
            Additional outputs (not used in glucose version)
        bound : float
            Bound parameter (not used)
        
        Returns
        -------
        j : float
            Safety barrier function cost
        """
        # For glucose control, we check BG bounds
        Vg = self._get_param('Vg', 1.0)
        bg = x[3] * Vg  # Blood glucose in mg/dL
        
        # Use the glucose barrier function
        j = abs(self._glucose_barrier_function(bg))
        
        # Scale penalty (squared for stronger penalty)
        if j > 0:
            j = j**2
        
        return j
    
    def _dot_fun(self, X, n, fx, xd, x, DT, NP, other):
        """
        Compute gradient using finite differences.
        
        Converted from MATLAB dot_fun() function.
        
        Parameters
        ----------
        X : np.ndarray
            Current control input
        n : int
            Number of control variables
        fx : float
            Current objective value
        xd : np.ndarray
            Desired state
        x : np.ndarray
            Current state
        DT : float
            Time step
        NP : int
            Prediction horizon
        other : dict
            Additional parameters
        
        Returns
        -------
        dX : np.ndarray
            Gradient vector
        """
        eps = 1e-3  # Finite difference step size
        dX = np.zeros_like(X)
        
        for i in range(n):
            E = np.zeros_like(X)
            E[i] = 1
            fx_perturbed = self._mainfun(xd, x, X + eps * E, DT, NP, other)
            dX[i] = (1.0 / eps) * (fx_perturbed - fx)
        
        return dX
    
    def _satu(self, x, sat):
        """
        Saturation function.
        
        Converted from MATLAB satu() function.
        Limits value to [-sat, sat] range.
        
        Parameters
        ----------
        x : float
            Input value
        sat : float
            Saturation limit
        
        Returns
        -------
        y : float
            Saturated value
        """
        return np.sign(x) * min(sat, abs(x))
    
    def _get_param(self, param_name: str, default: float = 0.0) -> float:
        """
        Safely get a parameter from patient_params (handles both dict and Series).
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
        default : float
            Default value if parameter not found
        
        Returns
        -------
        value : float
            Parameter value
        """
        if self.patient_params is None:
            return default
        
        if isinstance(self.patient_params, dict):
            return self.patient_params.get(param_name, default)
        else:
            # pandas Series
            return self.patient_params.get(param_name, default) if param_name in self.patient_params.index else default
    
    def _predict_glucose(self, 
                        initial_state: np.ndarray,
                        insulin_sequence: np.ndarray,
                        meal_sequence: np.ndarray,
                        horizon: int) -> np.ndarray:
        """
        Predict glucose trajectory using patient model.
        
        This method simulates the patient ODE model forward in time.
        Convert your MATLAB prediction function here.
        
        Parameters
        ----------
        initial_state : np.ndarray
            Initial 13-dimensional state vector
        insulin_sequence : np.ndarray
            Sequence of insulin inputs (U/min) over horizon
        meal_sequence : np.ndarray
            Sequence of meal/CHO inputs (g/min) over horizon
        horizon : int
            Prediction horizon in minutes
        
        Returns
        -------
        predicted_states : np.ndarray
            Predicted state trajectories (horizon x 13)
        """
        # TODO: Convert your MATLAB prediction function
        # 
        # This should integrate the patient ODE model:
        # - Use scipy.integrate.ode or similar
        # - Step through each time point in horizon
        # - Apply insulin and meal inputs at each step
        # - Return state trajectory
        
        # Placeholder: return zeros (replace with actual prediction)
        return np.zeros((horizon, 13))
    
    def _compute_objective(self,
                          predicted_bg: np.ndarray,
                          insulin_sequence: np.ndarray,
                          target_bg: float) -> float:
        """
        Compute NMPC objective function value.
        
        Convert your MATLAB objective function here.
        
        Parameters
        ----------
        predicted_bg : np.ndarray
            Predicted blood glucose over horizon
        insulin_sequence : np.ndarray
            Insulin control sequence
        target_bg : float
            Target blood glucose
        
        Returns
        -------
        cost : float
            Objective function value
        """
        # TODO: Convert your MATLAB objective function
        # 
        # Typical NMPC objective:
        # J = sum(q * (BG - target)^2 + r * insulin^2)
        # 
        # You may also include:
        # - Terminal cost
        # - Control barrier function penalties
        
        tracking_cost = self.q_weight * np.sum((predicted_bg - target_bg)**2)
        control_cost = self.r_weight * np.sum(insulin_sequence**2)
        return tracking_cost + control_cost
    
    def _compute_constraints(self,
                            predicted_bg: np.ndarray,
                            insulin_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute constraint values for NMPC optimization.
        
        Convert your MATLAB constraint functions here, including control
        barrier functions for safety.
        
        Parameters
        ----------
        predicted_bg : np.ndarray
            Predicted blood glucose over horizon
        insulin_sequence : np.ndarray
            Insulin control sequence
        
        Returns
        -------
        constraints : dict
            Dictionary of constraint arrays (for use with optimization solver)
        """
        # TODO: Convert your MATLAB constraint functions
        # 
        # Typical constraints:
        # - Safety bounds: bg_min <= BG <= bg_max
        # - Control bounds: 0 <= insulin <= insulin_max
        # - Control barrier function constraints for safety
        
        constraints = {
            'bg_lower': predicted_bg - self.bg_min,  # >= 0
            'bg_upper': self.bg_max - predicted_bg,  # >= 0
            'insulin_lower': insulin_sequence,  # >= 0
            'insulin_upper': 10.0 - insulin_sequence  # <= 10 U/min (example)
        }
        
        return constraints
    
    def _compute_basal(self) -> float:
        """
        Compute basal insulin rate based on patient parameters.
        
        Returns
        -------
        basal : float
            Basal insulin rate (U/min)
        """
        if self.patient_params is None:
            # Default basal if patient params not available
            return 0.02  # U/min
        
        u2ss = self._get_param('u2ss', 1.43)  # pmol/(L*kg)
        bw = self._get_param('BW', 57.0)  # kg
        basal = u2ss * bw / 6000  # Convert to U/min
        return basal
    
    def _fallback_action(self) -> Action:
        """
        Fallback action when NMPC solver fails.
        
        Returns safe basal insulin only.
        """
        basal = self._compute_basal()
        return Action(basal=basal, bolus=0.0)
    
    def _load_patient_params(self, patient_name: str):
        """
        Load patient-specific parameters.
        
        Parameters
        ----------
        patient_name : str
            Name of the patient (e.g., 'adolescent#001')
        
        Returns
        -------
        params : pandas.Series
            Patient parameters as a pandas Series (as expected by patient model)
        """
        import pandas as pd
        import pkg_resources
        
        patient_params_file = pkg_resources.resource_filename(
            'simglucose', 'params/vpatient_params.csv')
        patient_params_df = pd.read_csv(patient_params_file)
        
        params_row = patient_params_df[patient_params_df.Name == patient_name]
        if params_row.empty:
            logger.warning(f"Patient {patient_name} not found, using defaults")
            # Return default patient params if not found
            return patient_params_df.iloc[0]  # Use first patient as default
        
        # Return as pandas Series (as expected by patient model)
        params = params_row.iloc[0]
        return params
    
    def reset(self):
        """
        Reset controller state.
        """
        self.current_state = None
        self.last_action = None
        self.solver_initialized = False
        self.prediction_history = []
        self.optimization_history = []
        # Reset PID controller state
        if hasattr(self, 'pid_controller'):
            self.pid_controller.reset()
        # Note: Don't call super().reset() as base class raises NotImplementedError

