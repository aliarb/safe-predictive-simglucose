#!/usr/bin/env python3
"""
Reinforcement Learning for Tuning NMPC Controller Parameters

This script uses RL to learn optimal NMPC hyperparameters (q_weight, r_weight,
prediction_horizon, control_horizon, opt_rate) for glucose control.

The RL agent learns which parameter combinations lead to better glucose control
performance (higher time in range, lower risk index).
"""
import numpy as np
import gymnasium as gym
from simglucose.envs.nmpc_tuning_env import NMPCTuningEnv
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import json

class SimpleRLAgent:
    """
    Simple RL agent using policy gradient method for parameter tuning.
    
    Uses a simple Gaussian policy to explore parameter space.
    """
    def __init__(self, action_dim=5, learning_rate=0.01):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Policy parameters (mean and std for each action dimension)
        # Initialize with reasonable defaults
        self.policy_mean = np.array([1.0, 0.1, 60.0, 30.0, 1.0], dtype=np.float32)
        self.policy_std = np.array([2.0, 0.2, 20.0, 10.0, 0.5], dtype=np.float32)
        
        # Action bounds
        self.action_low = np.array([0.1, 0.01, 30.0, 15.0, 0.1])
        self.action_high = np.array([10.0, 1.0, 120.0, 60.0, 2.0])
        
        # Episode history
        self.episode_rewards = []
        self.episode_actions = []
        
    def select_action(self):
        """Select action from policy (Gaussian distribution)."""
        action = np.random.normal(self.policy_mean, self.policy_std)
        # Clip to bounds
        action = np.clip(action, self.action_low, self.action_high)
        return action.astype(np.float32)
    
    def update_policy(self, episode_reward, episode_actions):
        """
        Update policy using simple policy gradient.
        
        If episode was good (high reward), move policy mean towards those actions.
        """
        if episode_reward > 0:  # Only update if episode was successful
            # Simple gradient: move mean towards actions that led to good reward
            weight = self.learning_rate * episode_reward
            self.policy_mean = (1 - weight) * self.policy_mean + weight * np.mean(episode_actions, axis=0)
            self.policy_mean = np.clip(self.policy_mean, self.action_low, self.action_high)
            
            # Reduce exploration over time
            self.policy_std *= 0.99
            self.policy_std = np.clip(self.policy_std, 0.1, self.policy_std.max())
    
    def get_best_params(self):
        """Get current best parameter estimate."""
        return {
            'q_weight': float(self.policy_mean[0]),
            'r_weight': float(self.policy_mean[1]),
            'prediction_horizon': int(self.policy_mean[2]),
            'control_horizon': int(self.policy_mean[3]),
            'opt_rate': float(self.policy_mean[4]),
        }


def train_rl_agent(env, agent, num_episodes=50):
    """
    Train RL agent to tune NMPC parameters.
    
    Parameters
    ----------
    env : NMPCTuningEnv
        The tuning environment
    agent : SimpleRLAgent
        The RL agent
    num_episodes : int
        Number of training episodes
    
    Returns
    -------
    history : dict
        Training history with rewards and parameters
    """
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'best_params': [],
        'mean_rewards': []
    }
    
    print("=" * 70)
    print("RL-BASED NMPC PARAMETER TUNING")
    print("=" * 70)
    print(f"Training for {num_episodes} episodes...")
    print()
    
    best_reward = -np.inf
    best_params = None
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        
        # Select parameters for this episode (action is chosen once per episode)
        action = agent.select_action()
        episode_actions.append(action)
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Use same parameters for entire episode
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated
        
        # Update agent
        agent.update_policy(episode_reward, np.array(episode_actions))
        
        # Record history
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['best_params'].append(agent.get_best_params())
        
        # Track best
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_params = agent.get_best_params()
        
        # Print progress
        if (episode + 1) % 10 == 0:
            mean_reward = np.mean(history['episode_rewards'][-10:])
            history['mean_rewards'].append(mean_reward)
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Episode Reward: {episode_reward:.2f}")
            print(f"  Mean Reward (last 10): {mean_reward:.2f}")
            print(f"  Current Params: q={agent.policy_mean[0]:.2f}, r={agent.policy_mean[1]:.3f}, "
                  f"PH={agent.policy_mean[2]:.0f}, CH={agent.policy_mean[3]:.0f}, LR={agent.policy_mean[4]:.2f}")
            print()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Episode Reward: {best_reward:.2f}")
    print(f"Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()
    
    return history, best_params


def plot_training_history(history):
    """Plot training progress."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Episode rewards
    axes[0].plot(history['episode_rewards'], alpha=0.5, label='Episode Reward')
    if history['mean_rewards']:
        axes[0].plot(range(9, len(history['episode_rewards']), 10), 
                    history['mean_rewards'], 'r-', label='Mean Reward (10 episodes)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True)
    
    # Parameter evolution
    params = history['best_params']
    q_weights = [p['q_weight'] for p in params]
    r_weights = [p['r_weight'] for p in params]
    
    axes[1].plot(q_weights, label='q_weight')
    axes[1].plot(r_weights, label='r_weight')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Parameter Value')
    axes[1].set_title('Parameter Evolution')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('nmpc_rl_tuning_results.png')
    print("Training plots saved to nmpc_rl_tuning_results.png")


if __name__ == "__main__":
    # Create meal scenario
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    scenario = CustomScenario(
        start_time=start_time,
        scenario=[(7, 45), (12, 70), (18, 80)]  # Meals at 7am, 12pm, 6pm
    )
    
    # Create environment
    env = NMPCTuningEnv(
        patient_name='adolescent#001',
        custom_scenario=scenario,
        episode_length=288,  # 1 day with 5-min sampling
        seed=42
    )
    
    # Create RL agent
    agent = SimpleRLAgent(learning_rate=0.01)
    
    # Train agent
    history, best_params = train_rl_agent(env, agent, num_episodes=50)
    
    # Plot results
    plot_training_history(history)
    
    # Save best parameters
    with open('best_nmpc_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("Best parameters saved to best_nmpc_params.json")
    
    # Test with best parameters
    print("\n" + "=" * 70)
    print("TESTING WITH BEST PARAMETERS")
    print("=" * 70)
    
    from simglucose.controller.nmpc_ctrller import NMPCController
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.sim_engine import SimObj, sim
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.actuator.pump import InsulinPump
    from datetime import timedelta
    
    # Create test environment
    patient = T1DPatient.withName('adolescent#001')
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    test_env = T1DSimEnv(patient, sensor, pump, scenario)
    
    # Create controller with best parameters
    best_controller = NMPCController(
        target_bg=140.0,
        prediction_horizon=int(best_params['prediction_horizon']),
        control_horizon=int(best_params['control_horizon']),
        sample_time=5.0,
        q_weight=best_params['q_weight'],
        r_weight=best_params['r_weight'],
        bg_min=70.0,
        bg_max=180.0
    )
    best_controller.opt_rate = best_params['opt_rate']
    
    # Run simulation
    sim_obj = SimObj(test_env, best_controller, timedelta(days=1), animate=False, path='./results')
    results = sim(sim_obj)
    
    # Calculate statistics
    bg_data = results['BG'].values
    time_in_range = np.sum((bg_data >= 70) & (bg_data <= 180)) / len(bg_data) * 100
    
    print(f"\nTest Results with Best Parameters:")
    print(f"  Mean BG: {np.mean(bg_data):.2f} mg/dL")
    print(f"  Time in Range: {time_in_range:.1f}%")
    print(f"  Min BG: {np.min(bg_data):.2f} mg/dL")
    print(f"  Max BG: {np.max(bg_data):.2f} mg/dL")

