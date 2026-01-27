#!/usr/bin/env python3
"""
Enhanced RL Tuning for NMPC to Achieve Optimal Performance:
- 0% violation rate
- 100% time in range (70-180 mg/dL)
- Mean BG around 140 mg/dL

Tunes all cost function parameters including the new parameterized cost function.
"""
import numpy as np
import gymnasium as gym
from typing import Optional
from simglucose.envs.nmpc_tuning_env import NMPCTuningEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.nmpc_ctrller import NMPCController
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class EnhancedNMPCTuningEnv(gym.Env):
    """
    Enhanced RL Environment for tuning ALL NMPC cost function parameters.
    
    Action Space: Extended NMPC parameters
    - q_weight: Glucose tracking weight [0.1, 10.0]
    - q_terminal_weight: Terminal tracking weight [0.5, 5.0]
    - r_delta_weight: Rate of change penalty [0.01, 2.0]
    - hypo_penalty_weight: Hypoglycemia penalty [10.0, 200.0]
    - hyper_penalty_weight: Hyperglycemia penalty [5.0, 100.0]
    - barrier_weight: Barrier function weight [1.0, 50.0]
    - zone_transition_smoothness: [1.0, 20.0]
    - insulin_rate_penalty_weight: [10.0, 500.0]
    - delta_u_asymmetry: [1.0, 5.0]
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(
        self,
        patient_name: str = 'adolescent#001',
        custom_scenario: CustomScenario = None,
        episode_length: int = 288,  # 1 day with 5-min sampling
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.patient_name = patient_name
        self.custom_scenario = custom_scenario
        self.episode_length = episode_length
        self.np_random = np.random.RandomState(seed)
        
        # Extended action space: 9 parameters
        self.action_space = gym.spaces.Box(
            low=np.array([0.1, 0.5, 0.01, 10.0, 5.0, 1.0, 1.0, 10.0, 1.0], dtype=np.float32),
            high=np.array([10.0, 5.0, 2.0, 200.0, 100.0, 50.0, 20.0, 500.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: Current state + performance metrics
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(9,),
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.bg_history = []
        self.cgm_history = []
        self.env = None
        self.controller = None
        self.current_obs = None
        self.current_info = None
        
        # Create environment
        self._create_env()
    
    def _create_env(self):
        """Create glucose simulation environment."""
        patient = T1DPatient.withName(self.patient_name)
        sensor = CGMSensor.withName("Dexcom", seed=1)
        pump = InsulinPump.withName("Insulet")
        
        if self.custom_scenario is None:
            start_time = datetime(2025, 1, 1, 0, 0, 0)
            scenario = CustomScenario(
                start_time=start_time,
                scenario=[(7, 45), (12, 70), (18, 80)]
            )
        else:
            scenario = self.custom_scenario
        
        self.env = T1DSimEnv(patient, sensor, pump, scenario)
        self.bg_history = []
        self.cgm_history = []
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        current_cgm = self.cgm_history[-1] if self.cgm_history else 140.0
        bg_hist = self.bg_history[-6:] if len(self.bg_history) >= 6 else self.bg_history + [140.0] * (6 - len(self.bg_history))
        bg_hist = bg_hist[:6]
        
        if len(self.bg_history) > 0:
            bg_array = np.array(self.bg_history)
            time_in_range = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100.0
            mean_bg = np.mean(bg_array)
        else:
            time_in_range = 100.0
            mean_bg = 140.0
        
        obs = np.array([
            current_cgm,
            bg_hist[0], bg_hist[1], bg_hist[2], bg_hist[3], bg_hist[4], bg_hist[5],
            time_in_range,
            mean_bg
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, bg: float, episode_stats: dict) -> float:
        """
        Compute reward targeting:
        - 0% violation (heavy penalty)
        - 100% time in range (high reward)
        - Mean BG around 140 (reward for proximity)
        
        Balanced reward function that provides learning signal even for imperfect episodes.
        """
        reward = 0.0
        
        # 1. Time in range reward (most important) - per step reward
        if 70 <= bg <= 180:
            reward += 1.0  # Reward for being in range
        elif 60 <= bg < 70 or 180 < bg <= 250:
            reward += 0.2  # Small reward for near range
        else:
            reward -= 2.0  # Penalty for out of range
        
        # 2. Mean BG targeting (secondary) - only computed at end of episode
        mean_bg = episode_stats.get('mean_bg', 140.0)
        bg_error = abs(mean_bg - 140.0)
        # Scale by episode length to make it comparable
        if bg_error < 5:
            reward += 0.5  # Perfect mean BG
        elif bg_error < 10:
            reward += 0.2  # Good mean BG
        elif bg_error < 20:
            reward += 0.05  # Acceptable mean BG
        else:
            reward -= 0.1 * (bg_error - 20) / 20  # Gradual penalty for poor mean BG
        
        # 3. Violation penalty (critical) - scaled by episode length
        violations = episode_stats.get('violations', 0)
        total_steps = episode_stats.get('total_steps', 288)
        violation_rate = violations / total_steps if total_steps > 0 else 0.0
        
        # Penalty proportional to violation rate
        if violation_rate == 0:
            reward += 5.0  # Bonus for zero violations
        else:
            reward -= 10.0 * violation_rate  # Penalty proportional to violation rate
        
        # 4. Time in range bonus (episode-level)
        time_in_range = episode_stats.get('time_in_range', 0.0)
        if time_in_range >= 100.0:
            reward += 10.0  # Perfect TIR bonus
        elif time_in_range >= 95.0:
            reward += 5.0  # Excellent TIR bonus
        elif time_in_range >= 90.0:
            reward += 2.0  # Good TIR bonus
        elif time_in_range >= 80.0:
            reward += 0.5  # Acceptable TIR bonus
        
        # 5. Severe hypoglycemia/hyperglycemia penalty (per step)
        if bg < 50:
            reward -= 5.0  # Severe hypoglycemia
        elif bg > 300:
            reward -= 5.0  # Severe hyperglycemia
        
        return reward
    
    def step(self, action: np.ndarray):
        """Step environment with NMPC parameters."""
        # Create controller with new parameters on first step
        if self.current_step == 0:
            params = {
                'q_weight': float(action[0]),
                'q_terminal_weight': float(action[1]),
                'r_delta_weight': float(action[2]),
                'hypo_penalty_weight': float(action[3]),
                'hyper_penalty_weight': float(action[4]),
                'barrier_weight': float(action[5]),
                'zone_transition_smoothness': float(action[6]),
                'insulin_rate_penalty_weight': float(action[7]),
                'delta_u_asymmetry': float(action[8]),
            }
            
            self.controller = NMPCController(
                target_bg=140.0,
                prediction_horizon=60,
                control_horizon=30,
                sample_time=5.0,
                q_weight=params['q_weight'],
                r_weight=0.1,  # Legacy parameter
                bg_min=70.0,
                bg_max=180.0,
                barrier_weight=params['barrier_weight'],
                q_terminal_weight=params['q_terminal_weight'],
                r_delta_weight=params['r_delta_weight'],
                hypo_penalty_weight=params['hypo_penalty_weight'],
                hyper_penalty_weight=params['hyper_penalty_weight'],
                zone_transition_smoothness=params['zone_transition_smoothness'],
                insulin_rate_penalty_weight=params['insulin_rate_penalty_weight'],
                delta_u_asymmetry=params['delta_u_asymmetry'],
            )
        
        # Get controller action and step environment
        if self.current_step == 0:
            obs = self.current_obs
            info = self.current_info
        else:
            controller_action = self.controller.policy(self.current_obs, 0, False, **self.current_info)
            step_result = self.env.step(controller_action)
            obs = step_result.observation
            info = step_result.info if hasattr(step_result, 'info') else {
                'patient_state': self.env.patient.state,
                'patient_name': self.env.patient.name,
                'sample_time': self.env.sample_time,
                'meal': self.env.CHO_hist[-1] if self.env.CHO_hist else 0.0,
                'bg': self.env.BG_hist[-1] if self.env.BG_hist else obs.CGM,
            }
            self.current_obs = obs
            self.current_info = info
        
        bg = info.get('bg', obs.CGM)
        self.bg_history.append(bg)
        self.cgm_history.append(obs.CGM)
        
        # Check termination
        terminated = bg < 10 or bg > 600
        truncated = (self.current_step >= self.episode_length - 1)
        
        # Compute episode statistics
        bg_array = np.array(self.bg_history)
        violations = np.sum((bg_array < 70) | (bg_array > 180))
        mean_bg = np.mean(bg_array)
        time_in_range = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100.0 if len(bg_array) > 0 else 0.0
        
        # Compute reward (only at end of episode for episode-level stats)
        if truncated or terminated:
            # Final episode reward includes episode-level bonuses
            episode_stats = {
                'violations': violations,
                'mean_bg': mean_bg,
                'time_in_range': time_in_range,
                'total_steps': len(bg_array)
            }
            reward = self._compute_reward(bg, episode_stats)
        else:
            # Step reward (simpler, per-step)
            if 70 <= bg <= 180:
                reward = 1.0
            elif 60 <= bg < 70 or 180 < bg <= 250:
                reward = 0.2
            else:
                reward = -2.0
            
            # Severe penalties
            if bg < 50 or bg > 300:
                reward -= 5.0
        
        self.current_step += 1
        
        observation = self._get_observation()
        info_dict = {
            'bg': bg,
            'violations': violations,
            'mean_bg': mean_bg,
            'time_in_range': time_in_range,
            'step': self.current_step,
            'total_steps': len(bg_array),
        }
        
        return observation, reward, terminated, truncated, info_dict
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        self.current_step = 0
        self._create_env()
        self.controller = None
        
        obs, _, _, info = self.env.reset()
        self.current_obs = obs
        self.current_info = info if hasattr(info, 'get') else {
            'patient_state': self.env.patient.state,
            'patient_name': self.env.patient.name,
            'sample_time': self.env.sample_time,
            'meal': 0.0,
            'bg': obs.CGM,
        }
        
        bg = self.current_info.get('bg', obs.CGM)
        self.bg_history = [bg]
        self.cgm_history = [obs.CGM]
        
        observation = self._get_observation()
        return observation, {'bg': bg}


class OptimizedRLAgent:
    """RL Agent optimized for finding parameters that achieve 0% violation and 100% TIR."""
    
    def __init__(self, action_dim=9, learning_rate=0.05):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Initialize with reasonable defaults (from NMPC_COST_FUNCTION.md)
        self.policy_mean = np.array([
            2.0,   # q_weight
            2.0,   # q_terminal_weight
            0.5,   # r_delta_weight
            75.0,  # hypo_penalty_weight
            30.0,  # hyper_penalty_weight
            15.0,  # barrier_weight
            7.0,   # zone_transition_smoothness
            150.0, # insulin_rate_penalty_weight
            2.5,   # delta_u_asymmetry
        ], dtype=np.float32)
        
        self.policy_std = np.array([
            1.0,   # q_weight
            1.0,   # q_terminal_weight
            0.3,   # r_delta_weight
            30.0,  # hypo_penalty_weight
            20.0,  # hyper_penalty_weight
            10.0,  # barrier_weight
            3.0,   # zone_transition_smoothness
            100.0, # insulin_rate_penalty_weight
            1.0,   # delta_u_asymmetry
        ], dtype=np.float32)
        
        self.action_low = np.array([0.1, 0.5, 0.01, 10.0, 5.0, 1.0, 1.0, 10.0, 1.0])
        self.action_high = np.array([10.0, 5.0, 2.0, 200.0, 100.0, 50.0, 20.0, 500.0, 5.0])
        
        self.best_reward = -np.inf
        self.best_params = None
    
    def select_action(self):
        """Select action from policy."""
        action = np.random.normal(self.policy_mean, self.policy_std)
        action = np.clip(action, self.action_low, self.action_high)
        return action.astype(np.float32)
    
    def update_policy(self, episode_reward, episode_actions, episode_stats):
        """
        Update policy based on episode performance using REINFORCE-style update.
        
        Always updates, but with different weights based on performance quality.
        """
        violations = episode_stats.get('violations', float('inf'))
        time_in_range = episode_stats.get('time_in_range', 0.0)
        mean_bg = episode_stats.get('mean_bg', 140.0)
        
        # Normalize reward to [0, 1] range for stable learning
        # Expected reward range: roughly [-100, 100]
        normalized_reward = (episode_reward + 100) / 200.0
        normalized_reward = np.clip(normalized_reward, 0.0, 1.0)
        
        # Compute performance score (0 to 1)
        performance_score = 0.0
        
        # Violation component (40% weight)
        if violations == 0:
            performance_score += 0.4
        else:
            violation_rate = violations / episode_stats.get('total_steps', 288)
            performance_score += 0.4 * max(0, 1.0 - violation_rate * 10)  # Penalize violations
        
        # Time in range component (40% weight)
        performance_score += 0.4 * (time_in_range / 100.0)
        
        # Mean BG component (20% weight)
        bg_error = abs(mean_bg - 140.0)
        bg_score = max(0, 1.0 - bg_error / 50.0)  # Perfect at 140, decreases with error
        performance_score += 0.2 * bg_score
        
        # Combine normalized reward and performance score
        update_weight = self.learning_rate * (normalized_reward * 0.5 + performance_score * 0.5)
        update_weight = np.clip(update_weight, 0.001, 0.1)  # Clamp update weight
        
        # Update policy mean towards actions that led to this reward
        action_mean = np.mean(episode_actions, axis=0)
        
        # REINFORCE-style update: move towards actions that gave good reward
        if episode_reward > 0 or performance_score > 0.5:
            # Positive update: move towards these actions
            self.policy_mean = (1 - update_weight) * self.policy_mean + update_weight * action_mean
        else:
            # Negative update: move away from these actions (smaller step)
            self.policy_mean = (1 - update_weight * 0.5) * self.policy_mean - update_weight * 0.5 * (action_mean - self.policy_mean)
        
        self.policy_mean = np.clip(self.policy_mean, self.action_low, self.action_high)
        
        # Adaptive exploration: reduce std if performance is good, increase if poor
        if performance_score > 0.7:
            self.policy_std *= 0.998  # Reduce exploration (exploit)
        elif performance_score < 0.3:
            self.policy_std *= 1.002  # Increase exploration (explore more)
        else:
            self.policy_std *= 0.999  # Slow decay
        
        self.policy_std = np.clip(self.policy_std, 0.1, self.policy_std.max())
        
        # Track best
        if episode_reward > self.best_reward or (violations == 0 and time_in_range > 95.0):
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
            self.best_params = {
                'q_weight': float(episode_actions[0][0]),
                'q_terminal_weight': float(episode_actions[0][1]),
                'r_delta_weight': float(episode_actions[0][2]),
                'hypo_penalty_weight': float(episode_actions[0][3]),
                'hyper_penalty_weight': float(episode_actions[0][4]),
                'barrier_weight': float(episode_actions[0][5]),
                'zone_transition_smoothness': float(episode_actions[0][6]),
                'insulin_rate_penalty_weight': float(episode_actions[0][7]),
                'delta_u_asymmetry': float(episode_actions[0][8]),
            }
    
    def get_best_params(self):
        """Get best parameters found."""
        if self.best_params is None:
            return {
                'q_weight': float(self.policy_mean[0]),
                'q_terminal_weight': float(self.policy_mean[1]),
                'r_delta_weight': float(self.policy_mean[2]),
                'hypo_penalty_weight': float(self.policy_mean[3]),
                'hyper_penalty_weight': float(self.policy_mean[4]),
                'barrier_weight': float(self.policy_mean[5]),
                'zone_transition_smoothness': float(self.policy_mean[6]),
                'insulin_rate_penalty_weight': float(self.policy_mean[7]),
                'delta_u_asymmetry': float(self.policy_mean[8]),
            }
        return self.best_params


def train_optimal_nmpc(env, agent, num_episodes=100):
    """Train RL agent to find optimal NMPC parameters."""
    history = {
        'episode_rewards': [],
        'violations': [],
        'time_in_range': [],
        'mean_bg': [],
        'best_params': [],
    }
    
    print("=" * 80)
    print("RL TUNING FOR OPTIMAL NMPC PERFORMANCE")
    print("=" * 80)
    print("Target: 0% violation, 100% time in range, Mean BG ≈ 140 mg/dL")
    print(f"Training for {num_episodes} episodes...\n")
    
    best_performance = {'violations': float('inf'), 'time_in_range': 0.0, 'mean_bg': 140.0}
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_actions = []
        episode_stats = {}
        
        action = agent.select_action()
        episode_actions.append(action)
        
        done = False
        truncated = False
        
        while not (done or truncated):
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            done = terminated
        
        # Get final episode stats from last step info
        episode_stats = {
            'violations': step_info.get('violations', 0),
            'time_in_range': step_info.get('time_in_range', 0.0),
            'mean_bg': step_info.get('mean_bg', 140.0),
            'total_steps': step_info.get('total_steps', step_info.get('step', 288)),
        }
        
        # Update agent (always update, but with different weights)
        agent.update_policy(episode_reward, np.array(episode_actions), episode_stats)
        
        # Record history
        history['episode_rewards'].append(episode_reward)
        history['violations'].append(episode_stats.get('violations', 0))
        history['time_in_range'].append(episode_stats.get('time_in_range', 0.0))
        history['mean_bg'].append(episode_stats.get('mean_bg', 140.0))
        history['best_params'].append(agent.get_best_params())
        
        # Track best performance
        if (episode_stats.get('violations', float('inf')) < best_performance['violations'] or
            (episode_stats.get('violations', float('inf')) == best_performance['violations'] and
             episode_stats.get('time_in_range', 0.0) > best_performance['time_in_range'])):
            best_performance = episode_stats.copy()
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_violations = np.mean(history['violations'][-10:])
            recent_tir = np.mean(history['time_in_range'][-10:])
            recent_mean_bg = np.mean(history['mean_bg'][-10:])
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Recent Avg: Violations={recent_violations:.1f}, TIR={recent_tir:.1f}%, Mean BG={recent_mean_bg:.1f} mg/dL")
            print(f"  Best So Far: Violations={best_performance['violations']:.0f}, "
                  f"TIR={best_performance['time_in_range']:.1f}%, "
                  f"Mean BG={best_performance['mean_bg']:.1f} mg/dL")
            print()
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Performance:")
    print(f"  Violations: {best_performance['violations']:.0f}")
    print(f"  Time in Range: {best_performance['time_in_range']:.1f}%")
    print(f"  Mean BG: {best_performance['mean_bg']:.1f} mg/dL")
    print(f"\nBest Parameters:")
    best_params = agent.get_best_params()
    for key, value in best_params.items():
        print(f"  {key}: {value:.3f}")
    print()
    
    return history, best_params


def test_optimal_params(best_params, patient_name='adolescent#001', days=1):
    """Test optimal parameters on full simulation."""
    print("=" * 80)
    print("TESTING OPTIMAL PARAMETERS")
    print("=" * 80)
    
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    scenario = CustomScenario(
        start_time=start_time,
        scenario=[(7, 45), (12, 70), (18, 80)]
    )
    
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    env = T1DSimEnv(patient, sensor, pump, scenario)
    
    controller = NMPCController(
        target_bg=140.0,
        prediction_horizon=60,
        control_horizon=30,
        sample_time=5.0,
        q_weight=best_params['q_weight'],
        r_weight=0.1,
        bg_min=70.0,
        bg_max=180.0,
        barrier_weight=best_params['barrier_weight'],
        q_terminal_weight=best_params['q_terminal_weight'],
        r_delta_weight=best_params['r_delta_weight'],
        hypo_penalty_weight=best_params['hypo_penalty_weight'],
        hyper_penalty_weight=best_params['hyper_penalty_weight'],
        zone_transition_smoothness=best_params['zone_transition_smoothness'],
        insulin_rate_penalty_weight=best_params['insulin_rate_penalty_weight'],
        delta_u_asymmetry=best_params['delta_u_asymmetry'],
    )
    
    sim_obj = SimObj(env, controller, timedelta(days=days), animate=False, path='./results/optimal_nmpc')
    results = sim(sim_obj)
    
    # Calculate statistics
    bg_data = results['BG'].values
    violations = np.sum((bg_data < 70) | (bg_data > 180))
    violation_rate = violations / len(bg_data) * 100.0
    time_in_range = np.sum((bg_data >= 70) & (bg_data <= 180)) / len(bg_data) * 100.0
    mean_bg = np.mean(bg_data)
    std_bg = np.std(bg_data)
    
    print(f"\nTest Results:")
    print(f"  Mean BG: {mean_bg:.2f} mg/dL")
    print(f"  Std BG: {std_bg:.2f} mg/dL")
    print(f"  Time in Range: {time_in_range:.2f}%")
    print(f"  Violations: {violations} ({violation_rate:.2f}%)")
    print(f"  Min BG: {np.min(bg_data):.2f} mg/dL")
    print(f"  Max BG: {np.max(bg_data):.2f} mg/dL")
    print()
    
    return {
        'mean_bg': mean_bg,
        'time_in_range': time_in_range,
        'violations': violations,
        'violation_rate': violation_rate,
    }


if __name__ == "__main__":
    # Create scenario
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    scenario = CustomScenario(
        start_time=start_time,
        scenario=[(7, 45), (12, 70), (18, 80)]
    )
    
    # Create environment
    env = EnhancedNMPCTuningEnv(
        patient_name='adolescent#001',
        custom_scenario=scenario,
        episode_length=288,  # 1 day
        seed=42
    )
    
    # Create RL agent
    agent = OptimizedRLAgent(learning_rate=0.05)
    
    # Train
    history, best_params = train_optimal_nmpc(env, agent, num_episodes=100)
    
    # Save results
    os.makedirs('./results/optimal_nmpc', exist_ok=True)
    with open('./results/optimal_nmpc/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    with open('./results/optimal_nmpc/training_history.json', 'w') as f:
        json.dump({
            'episode_rewards': [float(r) for r in history['episode_rewards']],
            'violations': [int(v) for v in history['violations']],
            'time_in_range': [float(t) for t in history['time_in_range']],
            'mean_bg': [float(m) for m in history['mean_bg']],
        }, f, indent=2)
    
    print("Results saved to ./results/optimal_nmpc/")
    
    # Test optimal parameters
    test_results = test_optimal_params(best_params, days=1)
    
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Final Performance:")
    print(f"  Violation Rate: {test_results['violation_rate']:.2f}%")
    print(f"  Time in Range: {test_results['time_in_range']:.2f}%")
    print(f"  Mean BG: {test_results['mean_bg']:.2f} mg/dL")
    print(f"\nOptimal parameters saved to: ./results/optimal_nmpc/best_params.json")

