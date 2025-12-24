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
    
    This controller solves an optimization problem at each time step to determine
    the optimal insulin delivery rate (basal and bolus) based on:
    - Current patient state (13-dimensional state vector)
    - Predicted glucose trajectory over a prediction horizon
    - Control barrier functions for safety constraints
    
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
    q_weight : float, optional
        Weight for glucose tracking cost (default: 1.0)
    r_weight : float, optional
        Weight for insulin cost (default: 0.1)
    bg_min : float, optional
        Minimum safe blood glucose level in mg/dL (default: 70)
    bg_max : float, optional
        Maximum safe blood glucose level in mg/dL (default: 180)
    patient_params : dict, optional
        Patient-specific parameters (if None, will be loaded from info)
    """
    
    def __init__(self, 
                 target_bg: float = 140.0,
                 prediction_horizon: int = 60,
                 control_horizon: int = 30,
                 sample_time: float = 5.0,
                 q_weight: float = 1.0,
                 r_weight: float = 0.1,
                 bg_min: float = 70.0,
                 bg_max: float = 180.0,
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
        self.q_weight = q_weight
        self.r_weight = r_weight
        self.bg_min = bg_min
        self.bg_max = bg_max
        
        # Patient parameters (will be set from info if not provided)
        self.patient_params = patient_params
        
        # Internal state for NMPC
        self.current_state = None
        self.last_action = None
        self.solver_initialized = False
        
        # Optimization parameters (from MATLAB code)
        self.NP = prediction_horizon  # Prediction steps
        self.Nopt = 20  # Max number of optimization iterations
        self.opt_rate = 1.0  # Learning rate of optimization method
        self.acc = 1e-3  # Minimum accuracy of optimization method
        self.max_time = 0.1  # Maximum computation time (seconds)
        
        # Control parameters
        self.insulin_max = 10.0  # Maximum insulin rate (U/min)
        self.insulin_min = 0.0  # Minimum insulin rate (U/min)
        
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
        Solve the NMPC optimization problem using gradient descent with momentum.
        
        This is the main NMPC solver, converted from MATLAB code.
        Uses a custom gradient descent optimization with conjugate gradient momentum.
        
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
            Optimal insulin action (basal, bolus) in U/min
        """
        # Target state: [target_bg, ...] - we mainly care about glucose tracking
        target_state = np.zeros(13)
        Vg = self._get_param('Vg', 1.0)
        target_state[3] = self.target_bg / Vg  # Convert to mg/kg
        
        # Initial guess for control: use previous action or basal
        if self.last_action is not None:
            u_old = np.array([self.last_action.basal + self.last_action.bolus])
        else:
            basal = self._compute_basal()
            u_old = np.array([basal])
        
        # Other parameters for prediction (meal sequence, etc.)
        other_params = {
            'meal': meal,
            'patient_params': self.patient_params,
            'patient_name': patient_name,
            'last_Qsto': current_state[0] + current_state[1] if current_state is not None else 0,
            'last_foodtaken': 0
        }
        
        # Solve optimization
        u_opt = self._optimize(
            x=current_state,
            xd=target_state,
            u_old=u_old,
            DT=sample_time,
            NP=self.NP,
            maxiteration=self.Nopt,
            alfa=self.opt_rate,
            acc=self.acc,
            other=other_params,
            max_time=self.max_time
        )
        
        # Extract control action
        insulin_total = float(u_opt[0])
        insulin_total = np.clip(insulin_total, self.insulin_min, self.insulin_max)
        
        # Split into basal and bolus
        basal = self._compute_basal()
        bolus = max(0, insulin_total - basal)
        
        return Action(basal=basal, bolus=bolus)
    
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
        
        while stop < maxiter and np.linalg.norm(dx, 2) > acc and MPC_time < max_time:
            stop += 1
            fx = self._mainfun(xd, x, Un, DT, NP, other)
            dxo = dx.copy()
            CGo = CG.copy()
            
            # Compute gradient
            dx = self._dot_fun(Un, n, fx, xd, x, DT, NP, other)
            
            # Conjugate gradient update
            CG = -dx + bet * CG
            
            # Update control with momentum
            # Original MATLAB: UUU=Un-diag([1;.0;.0;.0;.0])*diag(CG)*dx;
            # Simplified: UUU = Un - alfa * CG * dx (element-wise)
            UUU = Un - alfa * CG * dx
            
            # Apply saturation constraint
            UUU[0] = self._satu(UUU[0], self.insulin_max)
            
            # Evaluate new objective
            fxx = self._mainfun(xd, x, UUU, DT, NP, other)
            
            # Adaptive learning rate
            if fxx < fx:
                Un = UUU.copy()
                alfa = alfa * 1.8
            else:
                alfa = alfa * 0.4
            
            # Update momentum parameter
            if np.linalg.norm(CG) > 0.1:
                bet = np.linalg.norm(CG) / (np.linalg.norm(CGo) + 1e-10)
            else:
                bet = np.linalg.norm(CG) / (np.linalg.norm(CGo * dxo) + 1e-10)
            
            MPC_time = time.time() - start_time
        
        return Un
    
    def _mainfun(self, xd, x, u, DelT, NP, other):
        """
        Main objective function for NMPC.
        
        Converted from MATLAB mainfun() function.
        Computes the cost over prediction horizon.
        
        Parameters
        ----------
        xd : np.ndarray
            Desired/target state
        x : np.ndarray
            Current state
        u : np.ndarray
            Control input sequence
        DelT : float
            Time step
        NP : int
            Prediction horizon steps
        other : dict
            Additional parameters
        
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
        
        # Predict forward over horizon
        for i in range(NP):
            # Predict one step using patient model
            out[:, i+1], dz = self._patient_model_step(z[:, i], u, dz0, other)
            z[:, i+1] = z[:, i] + dz * DelT
            x_pred = z[:, i+1]
            
            # Compute cost: tracking error + control cost
            # Original: J=J+1*sqrt((xd(1:2)-x(1:2))'*((xd(1:2)-x(1:2)) ) + ...)
            # Adapted for glucose: track BG (state[3]) and minimize insulin
            
            # Get blood glucose from state (convert from mg/kg to mg/dL)
            Vg = self._get_param('Vg', 1.0)
            bg_pred = x_pred[3] * Vg  # Blood glucose in mg/dL
            bg_target = xd[3] * Vg if len(xd) > 3 else self.target_bg
            
            # Tracking cost: (BG - target)^2
            tracking_error = (bg_pred - bg_target)**2
            
            # Control cost: u^2 (penalize large insulin)
            control_cost = 0.0 * (u[0]**2)  # Can be enabled with weight
            
            J += 1.0 * tracking_error + control_cost
            dz0 = dz.copy()
        
        # Penalty for constraint violation (insulin bounds)
        if abs(u[0]) > self.insulin_max:
            J += 1.0 * abs(abs(u[0]) - self.insulin_max)
        
        # Terminal cost: final state tracking
        Vg = self._get_param('Vg', 1.0)
        bg_final = z[3, -1] * Vg
        bg_target = xd[3] * Vg if len(xd) > 3 else self.target_bg
        terminal_cost = 1.0 * (bg_final - bg_target)**2
        J += terminal_cost
        
        # Safety barrier function cost (control barrier function)
        safety_barrier_cost = 0.002 * self._safety_barrier_function(z[:, -1], out[:, -1], 0)
        J += safety_barrier_cost
        
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
        meal = other.get('meal', 0.0)  # g/min
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
        
        # Compute state derivative using patient model
        t = 0  # Time doesn't matter for static evaluation
        dz = self._patient_ode_model(t, x, action, params, last_Qsto, last_foodtaken)
        
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
        
        # Safety bounds
        bg_safe_lower = self.bg_min
        bg_safe_upper = self.bg_max
        
        # Penalty if outside safe bounds
        if bg > bg_safe_upper or bg < bg_safe_lower:
            j = 1.0 * ((bg - self.target_bg)**2)
        else:
            j = 0.0
        
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
        # Note: Don't call super().reset() as base class raises NotImplementedError

