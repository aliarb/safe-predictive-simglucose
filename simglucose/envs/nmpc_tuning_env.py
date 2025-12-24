"""
Reinforcement Learning Environment for Tuning NMPC Controller Parameters

This environment wraps the glucose simulation and treats NMPC hyperparameters
as actions. The RL agent learns optimal NMPC parameters (q_weight, r_weight,
prediction_horizon, etc.) to maximize glucose control performance.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.analysis.risk import risk_index


class NMPCTuningEnv(gym.Env):
    """
    RL Environment for tuning NMPC controller parameters.
    
    Action Space: NMPC hyperparameters to tune
    - q_weight: Glucose tracking weight [0.1, 10.0]
    - r_weight: Insulin cost weight [0.01, 1.0]
    - prediction_horizon: Prediction steps [30, 120] minutes
    - control_horizon: Control steps [15, 60] minutes
    - opt_rate: Learning rate [0.1, 2.0]
    
    Observation Space: Current glucose state and performance metrics
    - Current CGM reading
    - Recent BG history (last 6 readings)
    - Time in range percentage
    - Risk index
    
    Reward: Based on glucose control performance
    - Time in range (70-180 mg/dL)
    - Risk index (lower is better)
    - Hypoglycemia penalty
    - Hyperglycemia penalty
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(
        self,
        patient_name: Optional[str] = None,
        custom_scenario: Optional[CustomScenario] = None,
        episode_length: int = 288,  # 1 day with 5-min sampling (288 steps)
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize NMPC tuning environment.
        
        Parameters
        ----------
        patient_name : str, optional
            Patient name (e.g., 'adolescent#001'). If None, uses random patient.
        custom_scenario : CustomScenario, optional
            Custom meal scenario. If None, uses RandomScenario.
        episode_length : int
            Number of steps per episode (default: 288 = 1 day)
        seed : int, optional
            Random seed
        render_mode : str, optional
            Rendering mode
        """
        super().__init__()
        
        self.patient_name = patient_name
        self.custom_scenario = custom_scenario
        self.episode_length = episode_length
        self.render_mode = render_mode
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Action space: NMPC parameters to tune
        # [q_weight, r_weight, prediction_horizon, control_horizon, opt_rate]
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.01, 30.0, 15.0, 0.1], dtype=np.float32),
            high=np.array([10.0, 1.0, 120.0, 60.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: Current state + performance metrics
        # [CGM, BG_history(6), time_in_range, risk_index]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(9,),  # 1 CGM + 6 BG history + 1 TIR + 1 risk
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.bg_history = []
        self.cgm_history = []
        self.risk_history = []
        self.env = None
        self.controller = None
        self.episode_rewards = []
        self.current_obs = None
        self.current_info = None
        
        # Store current parameters
        self.current_q_weight = 1.0
        self.current_r_weight = 0.1
        self.current_prediction_horizon = 60
        self.current_control_horizon = 30
        self.current_opt_rate = 1.0
        
        # Create initial environment
        self._create_env()
    
    def _create_env(self):
        """Create or recreate the glucose simulation environment."""
        # Generate random seed for patient/scenario
        seed2 = self.np_random.randint(0, 2**31)
        seed3 = self.np_random.randint(0, 2**31)
        seed4 = self.np_random.randint(0, 2**31)
        
        # Random start time
        hour = self.np_random.randint(0, 24)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        
        # Select patient
        if self.patient_name is None:
            # Random patient selection
            patient_names = [
                f"adolescent#{i:03d}" for i in range(1, 11)
            ] + [f"adult#{i:03d}" for i in range(1, 11)] + [f"child#{i:03d}" for i in range(1, 11)]
            patient_name = self.np_random.choice(patient_names)
        else:
            patient_name = self.patient_name
        
        patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        
        # Select scenario
        if self.custom_scenario is None:
            scenario = RandomScenario(start_time=start_time, seed=seed3)
        else:
            scenario = self.custom_scenario
        
        # Create sensor and pump
        sensor = CGMSensor.withName("Dexcom", seed=seed2)
        pump = InsulinPump.withName("Insulet")
        
        # Create environment
        self.env = _T1DSimEnv(patient, sensor, pump, scenario)
        
        # Reset histories
        self.bg_history = []
        self.cgm_history = []
        self.risk_history = []
        self.episode_rewards = []
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Current CGM
        current_cgm = self.cgm_history[-1] if self.cgm_history else 0.0
        
        # BG history (pad with zeros if not enough history)
        bg_hist = self.bg_history[-6:] if len(self.bg_history) >= 6 else self.bg_history + [0.0] * (6 - len(self.bg_history))
        bg_hist = bg_hist[:6]  # Ensure exactly 6 values
        
        # Time in range percentage
        if len(self.bg_history) > 0:
            bg_array = np.array(self.bg_history)
            time_in_range = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100.0
        else:
            time_in_range = 0.0
        
        # Risk index (average of recent risks)
        if len(self.risk_history) > 0:
            avg_risk = np.mean(self.risk_history[-10:])  # Last 10 steps
        else:
            avg_risk = 0.0
        
        obs = np.array([
            current_cgm,
            bg_hist[0], bg_hist[1], bg_hist[2], bg_hist[3], bg_hist[4], bg_hist[5],
            time_in_range,
            avg_risk
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, bg: float, cgm: float, risk: float) -> float:
        """
        Compute reward based on glucose control performance.
        
        Reward components:
        - Time in range bonus (70-180 mg/dL)
        - Risk index penalty (lower is better)
        - Hypoglycemia penalty (<70 mg/dL)
        - Hyperglycemia penalty (>180 mg/dL)
        """
        reward = 0.0
        
        # Time in range bonus
        if 70 <= bg <= 180:
            reward += 1.0
        elif 60 <= bg < 70 or 180 < bg <= 250:
            reward += 0.5  # Near target
        else:
            reward -= 1.0  # Out of range
        
        # Risk index penalty (scaled)
        reward -= risk * 0.1
        
        # Severe penalties for extreme values
        if bg < 50:
            reward -= 5.0  # Severe hypoglycemia
        elif bg > 300:
            reward -= 5.0  # Severe hyperglycemia
        
        # Normalize reward to reasonable range
        reward = np.clip(reward, -10.0, 10.0)
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step the environment with NMPC parameter action.
        
        Parameters
        ----------
        action : np.ndarray
            NMPC parameters [q_weight, r_weight, prediction_horizon, control_horizon, opt_rate]
        
        Returns
        -------
        observation : np.ndarray
            Current observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode terminated (patient died)
        truncated : bool
            Whether episode truncated (max steps reached)
        info : dict
            Additional information
        """
        # Create/update controller with new parameters (only on first step or if not created)
        if self.current_step == 0 or self.controller is None:
            self.current_q_weight = float(action[0])
            self.current_r_weight = float(action[1])
            self.current_prediction_horizon = int(action[2])
            self.current_control_horizon = int(action[3])
            self.current_opt_rate = float(action[4])
            
            self.controller = NMPCController(
                target_bg=140.0,
                prediction_horizon=self.current_prediction_horizon,
                control_horizon=self.current_control_horizon,
                sample_time=5.0,
                q_weight=self.current_q_weight,
                r_weight=self.current_r_weight,
                bg_min=70.0,
                bg_max=180.0
            )
            self.controller.opt_rate = self.current_opt_rate
        
        # Get current observation and step environment
        if self.current_step == 0:
            # First step: use observation from reset()
            obs = self.current_obs
            info = self.current_info
            cgm = obs.CGM
            bg = info.get('bg', cgm)
            risk = info.get('risk', 0.0)
            done = False
        else:
            # Subsequent steps: use controller to get action and step environment
            # Get controller action
            controller_action = self.controller.policy(self.current_obs, 0, False, **self.current_info)
            
            # Step environment
            step_result = self.env.step(controller_action)
            obs = step_result.observation
            reward_env = step_result.reward
            done = step_result.done
            
            # Extract info from step result
            if hasattr(step_result, 'info'):
                info = step_result.info
            else:
                # Fallback: construct info from step result attributes
                info = {
                    'patient_state': self.env.patient.state,
                    'patient_name': self.env.patient.name,
                    'sample_time': self.env.sample_time,
                    'meal': self.env.CHO_hist[-1] if self.env.CHO_hist else 0.0,
                    'bg': self.env.BG_hist[-1] if self.env.BG_hist else 0.0,
                    'risk': self.env.risk_hist[-1] if self.env.risk_hist else 0.0,
                }
            
            # Update current observation and info for next step
            self.current_obs = obs
            self.current_info = info
            
            cgm = obs.CGM
            bg = info.get('bg', cgm)
            risk = info.get('risk', 0.0)
        
        # Update histories
        self.bg_history.append(bg)
        self.cgm_history.append(cgm)
        self.risk_history.append(risk)
        
        # Compute reward
        reward = self._compute_reward(bg, cgm, risk)
        self.episode_rewards.append(reward)
        
        # Check termination
        terminated = done  # Patient died or BG out of bounds
        truncated = (self.current_step >= self.episode_length - 1)
        
        # Update step counter
        self.current_step += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Info dict
        info_dict = {
            'bg': bg,
            'cgm': cgm,
            'risk': risk,
            'step': self.current_step,
            'q_weight': self.current_q_weight,
            'r_weight': self.current_r_weight,
            'prediction_horizon': self.current_prediction_horizon,
            'control_horizon': self.current_control_horizon,
            'opt_rate': self.current_opt_rate,
        }
        
        return observation, reward, terminated, truncated, info_dict
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Recreate environment
        self._create_env()
        
        # Reset controller (will be created on first step)
        self.controller = None
        
        # Get initial observation
        obs, _, _, info = self.env.reset()
        
        # Store current observation and info for step() to use
        self.current_obs = obs
        self.current_info = info if hasattr(info, 'get') else {
            'patient_state': self.env.patient.state,
            'patient_name': self.env.patient.name,
            'sample_time': self.env.sample_time,
            'meal': 0.0,
            'bg': self.env.BG_hist[-1] if self.env.BG_hist else obs.CGM,
            'risk': self.env.risk_hist[-1] if self.env.risk_hist else 0.0,
        }
        
        cgm = obs.CGM
        bg = self.current_info.get('bg', cgm)
        
        # Initialize histories
        self.bg_history = [bg]
        self.cgm_history = [cgm]
        self.risk_history = [self.current_info.get('risk', 0.0)]
        self.episode_rewards = []
        
        # Get observation
        observation = self._get_observation()
        
        return observation, {'bg': bg, 'cgm': cgm}
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.env.render()
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env._close_viewer()

