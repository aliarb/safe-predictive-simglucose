#!/usr/bin/env python3
"""
RL Tuning for NMPC starting from fine-tuned parameters.

Uses current fine-tuned parameters as baseline and optimizes further:
- q_weight: 2.0 (baseline)
- q_terminal_weight: 3.0 (baseline)
- r_delta_weight: 0.3 (baseline)
- hypo_penalty_weight: 100.0 (baseline)
- hyper_penalty_weight: 15.0 (baseline)
- zone_transition_smoothness: 5.0 (baseline)
- insulin_rate_penalty_weight: 100.0 (baseline)
- delta_u_asymmetry: 2.0 (baseline)

Goal: Further improve performance while maintaining 0% hypoglycemia.
"""
import numpy as np
import gymnasium as gym
from typing import Optional
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


# Fine-tuned baseline parameters
BASELINE_PARAMS = {
    'q_weight': 2.0,
    'q_terminal_weight': 3.0,
    'r_delta_weight': 0.3,
    'hypo_penalty_weight': 100.0,
    'hyper_penalty_weight': 15.0,
    'zone_transition_smoothness': 5.0,
    'insulin_rate_penalty_weight': 100.0,
    'delta_u_asymmetry': 2.0
}


class FineTunedNMPCEnv(gym.Env):
    """
    RL Environment for fine-tuning NMPC parameters around baseline.
    
    Action Space: Relative adjustments to baseline parameters (multiplicative factors)
    - Each action is a factor [0.5, 2.0] to multiply baseline parameter
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
        
        # Action space: multiplicative factors for baseline parameters
        # [q_weight, q_terminal, r_delta, hypo_penalty, hyper_penalty, 
        #  zone_smoothness, insulin_penalty, delta_u_asymmetry]
        self.action_space = gym.spaces.Box(
            low=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            high=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: CGM, BG history, performance metrics
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(10,),
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
    
    def reset(self, seed=None, options=None):
        """Reset environment with new parameter set."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.bg_history = []
        self.cgm_history = []
        
        # Reset environment (returns Step object)
        step_result = self.env.reset()
        # Step object has observation, reward, done, and info attributes
        obs = step_result.observation
        
        # Get patient_state safely - use env.patient.state if not available in step_result
        if hasattr(step_result, 'patient_state') and step_result.patient_state is not None:
            patient_state = step_result.patient_state
        else:
            patient_state = self.env.patient.state
        
        info = {
            'patient_state': patient_state,
            'patient_name': step_result.patient_name if hasattr(step_result, 'patient_name') else self.patient_name,
            'sample_time': step_result.sample_time if hasattr(step_result, 'sample_time') else 5.0,
            'meal': step_result.meal if hasattr(step_result, 'meal') else 0.0,
            'bg': step_result.bg if hasattr(step_result, 'bg') else obs.CGM,
            'risk': step_result.risk if hasattr(step_result, 'risk') else 0.0,
        }
        self.current_obs = obs
        self.current_info = info
        
        # Create observation vector
        cgm = obs.CGM
        bg = info.get('bg', cgm)
        
        self.bg_history.append(bg)
        self.cgm_history.append(cgm)
        
        # Initial observation: CGM, BG, zeros for metrics
        observation = np.array([
            cgm,
            bg,
            0.0,  # time_in_range
            0.0,  # violation_rate
            0.0,  # std_bg (standard deviation)
            0.0,  # hypo_count
            0.0,  # hyper_count
            0.0,  # risk_index
            0.0,  # insulin_total
            0.0   # step_count
        ], dtype=np.float32)
        
        return observation, info
    
    def step(self, action):
        """Step environment with parameter adjustment."""
        # Convert action (multiplicative factors) to actual parameters
        params = self._action_to_params(action)
        
        # Create controller with adjusted parameters
        self.controller = NMPCController(
            target_bg=140.0,
            bg_min=70.0,
            bg_max=180.0,
            **params
        )
        
        # Get controller action - ensure patient_state is passed correctly
        # Use proper None check for numpy arrays (can't use 'or' operator)
        patient_state = self.current_info.get('patient_state')
        if patient_state is None:
            patient_state = self.env.patient.state
        
        # Get other info with proper None checks
        patient_name = self.current_info.get('patient_name')
        if patient_name is None:
            patient_name = self.patient_name
        
        sample_time = self.current_info.get('sample_time')
        if sample_time is None:
            sample_time = 5.0
        
        meal = self.current_info.get('meal')
        if meal is None:
            meal = 0.0
        
        bg = self.current_info.get('bg')
        if bg is None:
            bg = self.current_obs.CGM
        
        policy_info = {
            'patient_state': patient_state,
            'patient_name': patient_name,
            'sample_time': sample_time,
            'meal': meal,
            'bg': bg,
        }
        
        controller_action = self.controller.policy(
            self.current_obs,
            reward=0.0,
            done=False,
            **policy_info
        )
        
        # Run simulation for one step
        step_result = self.env.step(controller_action)
        # Handle Step object
        obs = step_result.observation
        reward = step_result.reward
        done = step_result.done
        truncated = False  # T1DSimEnv doesn't use truncated
        
        # Ensure patient_state is available from env (proper None check for numpy arrays)
        if hasattr(step_result, 'patient_state') and step_result.patient_state is not None:
            patient_state = step_result.patient_state
        else:
            patient_state = self.env.patient.state
        
        # Get other attributes with proper None checks
        patient_name = step_result.patient_name if hasattr(step_result, 'patient_name') and step_result.patient_name is not None else self.patient_name
        sample_time = step_result.sample_time if hasattr(step_result, 'sample_time') and step_result.sample_time is not None else 5.0
        meal = step_result.meal if hasattr(step_result, 'meal') and step_result.meal is not None else 0.0
        bg = step_result.bg if hasattr(step_result, 'bg') and step_result.bg is not None else obs.CGM
        risk = step_result.risk if hasattr(step_result, 'risk') and step_result.risk is not None else 0.0
        insulin = step_result.insulin if hasattr(step_result, 'insulin') and step_result.insulin is not None else 0.0
        
        info = {
            'patient_state': patient_state,
            'patient_name': patient_name,
            'sample_time': sample_time,
            'meal': meal,
            'bg': bg,
            'risk': risk,
            'insulin': insulin,
        }
        
        self.current_step += 1
        self.current_obs = obs
        self.current_info = info
        
        # Update history
        cgm = obs.CGM
        bg = info.get('bg', cgm)
        self.bg_history.append(bg)
        self.cgm_history.append(cgm)
        
        # Calculate performance metrics
        bg_array = np.array(self.bg_history)
        time_in_range = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100
        violation_rate = np.sum((bg_array < 70) | (bg_array > 180)) / len(bg_array) * 100
        std_bg = np.std(bg_array) if len(bg_array) > 1 else 0.0  # Standard deviation instead of mean
        hypo_count = np.sum(bg_array < 70)
        hyper_count = np.sum(bg_array > 180)
        risk_index = info.get('Risk', 0.0)
        insulin_total = info.get('insulin', 0.0)
        
        # Create observation
        observation = np.array([
            cgm,
            bg,
            time_in_range,
            violation_rate,
            std_bg,  # Standard deviation instead of mean
            hypo_count,
            hyper_count,
            risk_index,
            insulin_total,
            self.current_step
        ], dtype=np.float32)
        
        # Calculate reward
        reward = self._calculate_reward(bg_array, time_in_range, violation_rate, 
                                       std_bg, hypo_count, hyper_count)
        
        # Check if episode is done
        if self.current_step >= self.episode_length:
            done = True
        
        return observation, reward, done, truncated, info
    
    def _action_to_params(self, action):
        """Convert action (multiplicative factors) to NMPC parameters."""
        return {
            'q_weight': BASELINE_PARAMS['q_weight'] * action[0],
            'q_terminal_weight': BASELINE_PARAMS['q_terminal_weight'] * action[1],
            'r_delta_weight': BASELINE_PARAMS['r_delta_weight'] * action[2],
            'hypo_penalty_weight': BASELINE_PARAMS['hypo_penalty_weight'] * action[3],
            'hyper_penalty_weight': BASELINE_PARAMS['hyper_penalty_weight'] * action[4],
            'zone_transition_smoothness': BASELINE_PARAMS['zone_transition_smoothness'] * action[5],
            'insulin_rate_penalty_weight': BASELINE_PARAMS['insulin_rate_penalty_weight'] * action[6],
            'delta_u_asymmetry': BASELINE_PARAMS['delta_u_asymmetry'] * action[7]
        }
    
    def _calculate_reward(self, bg_array, time_in_range, violation_rate, 
                         std_bg, hypo_count, hyper_count):
        """Calculate reward based on performance."""
        # Per-step reward
        if len(bg_array) > 0:
            current_bg = bg_array[-1]
            if 70 <= current_bg <= 180:
                step_reward = 1.0  # In range
            elif 60 <= current_bg < 70 or 180 < current_bg <= 200:
                step_reward = 0.2  # Near range
            else:
                step_reward = -2.0  # Out of range
        else:
            step_reward = 0.0
        
        # Episode-level bonuses/penalties (scaled by step)
        episode_bonus = 0.0
        if len(bg_array) >= self.episode_length:
            # Bonus for 100% time in range
            if time_in_range >= 100.0:
                episode_bonus += 10.0
            
            # Bonus for 0% violations
            if violation_rate <= 0.0:
                episode_bonus += 5.0
            
            # Penalty for violations
            episode_bonus -= violation_rate * 10.0
            
            # Bonus for low standard deviation (consistent control)
            # Lower std is better - target std < 30 mg/dL
            if std_bg < 20.0:
                episode_bonus += 5.0
            elif std_bg < 30.0:
                episode_bonus += 2.0
            elif std_bg > 50.0:
                episode_bonus -= (std_bg - 50.0) * 0.2  # Penalty for high variability
        
        # Severe penalty for hypoglycemia
        hypo_penalty = -50.0 * hypo_count
        
        return step_reward + episode_bonus / self.episode_length + hypo_penalty / len(bg_array) if len(bg_array) > 0 else step_reward


class SimpleRLAgent:
    """
    Simple RL agent with adaptive policy updates.
    """
    
    def __init__(self, action_dim=8, learning_rate=0.1, exploration_rate=0.3):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Policy: mean and std for each action dimension
        # Start with 1.0 (no change from baseline)
        self.policy_mean = np.ones(action_dim, dtype=np.float32)
        self.policy_std = np.ones(action_dim, dtype=np.float32) * 0.2
        
        # Performance tracking
        self.best_performance = -np.inf
        self.best_params = None
        self.performance_history = []
    
    def select_action(self, observation):
        """Select action using policy."""
        # Sample from Gaussian distribution
        action = np.random.normal(self.policy_mean, self.policy_std)
        
        # Clip to action space bounds
        action = np.clip(action, 0.5, 2.0)
        
        return action.astype(np.float32)
    
    def update_policy(self, episode_reward, episode_metrics):
        """Update policy based on episode performance."""
        # Calculate performance score (using std_bg instead of mean_bg)
        # Lower std is better, so invert it: max(0, 100 - std_bg)
        std_score = max(0, 100 - episode_metrics.get('std_bg', 50))
        performance_score = (
            episode_metrics.get('time_in_range', 0) * 0.4 +
            (100 - episode_metrics.get('violation_rate', 100)) * 0.3 +
            std_score * 0.2 +
            episode_reward * 0.1
        )
        
        self.performance_history.append(performance_score)
        
        # Update policy if performance improved
        if performance_score > self.best_performance:
            self.best_performance = performance_score
            self.best_params = self.policy_mean.copy()
            
            # Reduce exploration (we're doing well)
            self.policy_std *= 0.95
            self.policy_std = np.clip(self.policy_std, 0.05, 0.5)
        else:
            # Increase exploration (we're not improving)
            self.policy_std *= 1.05
            self.policy_std = np.clip(self.policy_std, 0.05, 0.5)
        
        # Update policy mean towards better actions
        # Simple gradient-free approach: move mean towards actions that led to better performance
        if len(self.performance_history) > 1:
            if performance_score > np.mean(self.performance_history[-10:]):
                # Recent performance is good, keep current mean
                pass
            else:
                # Recent performance is poor, explore more
                self.policy_mean += np.random.normal(0, 0.1, self.action_dim)
                self.policy_mean = np.clip(self.policy_mean, 0.5, 2.0)


def train_rl_agent(env, agent, num_episodes=50):
    """Train RL agent to optimize NMPC parameters."""
    print(f"\nTraining RL agent for {num_episodes} episodes...")
    print("=" * 80)
    
    history = {
        'episode': [],
        'reward': [],
        'time_in_range': [],
        'violation_rate': [],
        'std_bg': [],  # Standard deviation instead of mean
        'hypo_count': [],
        'hyper_count': [],
        'params': []
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_bg = []
        
        done = False
        truncated = False
        while not done and not truncated:
            action = agent.select_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            bg = info.get('BG', obs[1])
            episode_bg.append(bg)
        
        # Calculate episode metrics
        bg_array = np.array(episode_bg)
        time_in_range = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100
        violation_rate = np.sum((bg_array < 70) | (bg_array > 180)) / len(bg_array) * 100
        std_bg = np.std(bg_array) if len(bg_array) > 1 else 0.0  # Standard deviation instead of mean
        hypo_count = np.sum(bg_array < 70)
        hyper_count = np.sum(bg_array > 180)
        
        episode_metrics = {
            'time_in_range': time_in_range,
            'violation_rate': violation_rate,
            'std_bg': std_bg,  # Standard deviation instead of mean
            'hypo_count': hypo_count,
            'hyper_count': hyper_count
        }
        
        # Update policy
        agent.update_policy(episode_reward, episode_metrics)
        
        # Store history
        history['episode'].append(episode + 1)
        history['reward'].append(episode_reward)
        history['time_in_range'].append(time_in_range)
        history['violation_rate'].append(violation_rate)
        history['std_bg'].append(std_bg)  # Standard deviation instead of mean
        history['hypo_count'].append(hypo_count)
        history['hyper_count'].append(hyper_count)
        
        # Get current parameters
        current_params = env._action_to_params(agent.policy_mean)
        history['params'].append(current_params.copy())
        
        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episode {episode + 1:3d}/{num_episodes}: "
                  f"TIR={time_in_range:5.1f}%, Viol={violation_rate:5.1f}%, "
                  f"Std BG={std_bg:5.1f}, Hypo={hypo_count:3d}, "
                  f"Reward={episode_reward:7.2f}")
    
    return history, agent


def test_optimal_params(env, optimal_params, num_tests=3):
    """Test optimal parameters multiple times."""
    print(f"\nTesting optimal parameters {num_tests} times...")
    print("=" * 80)
    
    results = []
    
    for test in range(num_tests):
        obs, info = env.reset()
        episode_bg = []
        episode_insulin = []
        
        # Create controller with optimal parameters
        controller = NMPCController(
            target_bg=140.0,
            bg_min=70.0,
            bg_max=180.0,
            **optimal_params
        )
        
        done = False
        truncated = False
        step = 0
        while not done and not truncated:
            action = controller.policy(obs, reward=0.0, done=False, **info)
            obs, reward, done, truncated, info = env.step(controller)
            
            bg = info.get('BG', obs.CGM if hasattr(obs, 'CGM') else 140.0)
            insulin = info.get('insulin', 0.0)
            episode_bg.append(bg)
            episode_insulin.append(insulin)
            step += 1
        
        bg_array = np.array(episode_bg)
        time_in_range = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100
        violation_rate = np.sum((bg_array < 70) | (bg_array > 180)) / len(bg_array) * 100
        std_bg = np.std(bg_array) if len(bg_array) > 1 else 0.0  # Standard deviation instead of mean
        hypo_count = np.sum(bg_array < 70)
        hyper_count = np.sum(bg_array > 180)
        
        results.append({
            'time_in_range': time_in_range,
            'violation_rate': violation_rate,
            'std_bg': std_bg,  # Standard deviation instead of mean
            'hypo_count': hypo_count,
            'hyper_count': hyper_count
        })
        
        print(f"Test {test + 1}: TIR={time_in_range:5.1f}%, Viol={violation_rate:5.1f}%, "
              f"Std BG={std_bg:5.1f}, Hypo={hypo_count:3d}, Hyper={hyper_count:3d}")
    
    # Average results
    avg_results = {
        'time_in_range': np.mean([r['time_in_range'] for r in results]),
        'violation_rate': np.mean([r['violation_rate'] for r in results]),
        'std_bg': np.mean([r['std_bg'] for r in results]),  # Standard deviation instead of mean
        'hypo_count': np.mean([r['hypo_count'] for r in results]),
        'hyper_count': np.mean([r['hyper_count'] for r in results])
    }
    
    print(f"\nAverage: TIR={avg_results['time_in_range']:5.1f}%, "
          f"Viol={avg_results['violation_rate']:5.1f}%, "
          f"Std BG={avg_results['std_bg']:5.1f}")
    
    return avg_results


def main():
    """Main training loop."""
    print("=" * 80)
    print("RL FINE-TUNING OF NMPC PARAMETERS")
    print("=" * 80)
    print(f"\nBaseline Parameters:")
    for key, value in BASELINE_PARAMS.items():
        print(f"  {key}: {value}")
    
    # Create environment
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    scenario = CustomScenario(
        start_time=start_time,
        scenario=[(7, 45), (12, 70), (18, 80)]
    )
    
    env = FineTunedNMPCEnv(
        patient_name='adolescent#001',
        custom_scenario=scenario,
        episode_length=288
    )
    
    # Create RL agent
    agent = SimpleRLAgent(action_dim=8, learning_rate=0.1, exploration_rate=0.3)
    
    # Train agent
    history, trained_agent = train_rl_agent(env, agent, num_episodes=50)
    
    # Get best parameters
    best_idx = np.argmax([h['time_in_range'] - h['violation_rate'] for h in 
                         zip(history['time_in_range'], history['violation_rate'])])
    optimal_params = history['params'][best_idx]
    
    print(f"\n{'=' * 80}")
    print("OPTIMAL PARAMETERS FOUND:")
    print("=" * 80)
    for key, value in optimal_params.items():
        baseline = BASELINE_PARAMS[key]
        change = (value / baseline - 1) * 100
        print(f"  {key:30s}: {value:8.3f} (baseline: {baseline:6.2f}, change: {change:+6.1f}%)")
    
    # Test optimal parameters
    test_results = test_optimal_params(env, optimal_params, num_tests=3)
    
    # Save results
    output_dir = './results/rl_finetuning'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parameters
    params_file = os.path.join(output_dir, 'optimal_params.json')
    with open(params_file, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    print(f"\n✓ Saved optimal parameters: {params_file}")
    
    # Save training history
    history_file = os.path.join(output_dir, 'training_history.json')
    # Convert numpy arrays to lists for JSON
    history_json = {k: [float(v) if isinstance(v, (np.integer, np.floating)) else v 
                        for v in vals] if isinstance(vals, list) else vals
                    for k, vals in history.items()}
    with open(history_file, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f"✓ Saved training history: {history_file}")
    
    # Plot training history
    plot_file = os.path.join(output_dir, 'training_history.png')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['episode'], history['time_in_range'], 'b-', linewidth=2)
    axes[0, 0].axhline(100, color='g', linestyle='--', label='Target (100%)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Time in Range (%)')
    axes[0, 0].set_title('Time in Range Over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['episode'], history['violation_rate'], 'r-', linewidth=2)
    axes[0, 1].axhline(0, color='g', linestyle='--', label='Target (0%)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Violation Rate (%)')
    axes[0, 1].set_title('Violation Rate Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['episode'], history['std_bg'], 'g-', linewidth=2)
    axes[1, 0].axhline(30, color='k', linestyle='--', label='Target (<30 mg/dL)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Std BG (mg/dL)')
    axes[1, 0].set_title('Blood Glucose Standard Deviation Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['episode'], history['reward'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Episode Reward')
    axes[1, 1].set_title('Episode Reward Over Training')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training plots: {plot_file}")
    
    print(f"\n{'=' * 80}")
    print("RL FINE-TUNING COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()

