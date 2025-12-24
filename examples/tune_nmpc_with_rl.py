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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from collections import deque
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress style warnings

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


def plot_training_history(history, save_path='nmpc_rl_tuning_results.png', dpi=300):
    """
    Create publication-quality plots of NMPC parameter tuning process.
    
    Parameters
    ----------
    history : dict
        Training history with episode_rewards, best_params, etc.
    save_path : str
        Path to save the figure
    dpi : int
        Resolution for saved figure (default: 300 for publication quality)
    """
    # Set publication-quality style (try different style names for compatibility)
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        try:
            plt.style.use('seaborn-paper')
        except:
            plt.style.use('seaborn-whitegrid')
            # Manually set some style parameters
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 2,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3
    })
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, 
                          left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    # Color palette for publication
    colors = {
        'reward': '#2E86AB',      # Blue
        'mean_reward': '#A23B72',  # Purple
        'q_weight': '#F18F01',     # Orange
        'r_weight': '#C73E1D',     # Red
        'pred_horizon': '#6A994E', # Green
        'ctrl_horizon': '#BC4749', # Dark red
        'opt_rate': '#7209B7'      # Purple
    }
    
    # Extract data
    episodes = np.arange(1, len(history['episode_rewards']) + 1)
    rewards = np.array(history['episode_rewards'])
    params = history['best_params']
    
    # ========== Subplot 1: Training Reward ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot individual episode rewards (transparent)
    ax1.plot(episodes, rewards, alpha=0.3, color=colors['reward'], 
             linewidth=0.8, label='Episode Reward', zorder=1)
    
    # Plot moving average (more prominent)
    window_size = min(10, len(rewards) // 5)
    if window_size > 1:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_episodes = episodes[window_size-1:]
        ax1.plot(moving_avg_episodes, moving_avg, color=colors['mean_reward'], 
                linewidth=2.5, label=f'Moving Average ({window_size} episodes)', zorder=2)
    
    # Highlight best episode
    best_idx = np.argmax(rewards)
    ax1.scatter(episodes[best_idx], rewards[best_idx], 
               color='gold', s=150, zorder=3, edgecolors='black', linewidth=1.5,
               label=f'Best Episode ({episodes[best_idx]})', marker='*')
    
    ax1.set_xlabel('Training Episode', fontweight='bold')
    ax1.set_ylabel('Cumulative Reward', fontweight='bold')
    ax1.set_title('(a) Reinforcement Learning Training Progress', fontweight='bold', pad=10)
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    
    # ========== Subplot 2: Weight Parameters ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    q_weights = [p['q_weight'] for p in params]
    r_weights = [p['r_weight'] for p in params]
    
    ax2.plot(episodes, q_weights, color=colors['q_weight'], 
            label=r'$q$ (glucose tracking)', linewidth=2, marker='o', markersize=4, markevery=max(1, len(episodes)//20))
    ax2.plot(episodes, r_weights, color=colors['r_weight'], 
            label=r'$r$ (insulin cost)', linewidth=2, marker='s', markersize=4, markevery=max(1, len(episodes)//20))
    
    ax2.set_xlabel('Training Episode', fontweight='bold')
    ax2.set_ylabel('Weight Parameter Value', fontweight='bold')
    ax2.set_title('(b) Cost Function Weights', fontweight='bold', pad=10)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    
    # ========== Subplot 3: Horizon Parameters ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    pred_horizons = [p['prediction_horizon'] for p in params]
    ctrl_horizons = [p['control_horizon'] for p in params]
    
    ax3.plot(episodes, pred_horizons, color=colors['pred_horizon'], 
            label='Prediction Horizon', linewidth=2, marker='o', markersize=4, markevery=max(1, len(episodes)//20))
    ax3.plot(episodes, ctrl_horizons, color=colors['ctrl_horizon'], 
            label='Control Horizon', linewidth=2, marker='s', markersize=4, markevery=max(1, len(episodes)//20))
    
    ax3.set_xlabel('Training Episode', fontweight='bold')
    ax3.set_ylabel('Horizon (minutes)', fontweight='bold')
    ax3.set_title('(c) Prediction and Control Horizons', fontweight='bold', pad=10)
    ax3.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    
    # ========== Subplot 4: Optimization Rate ==========
    ax4 = fig.add_subplot(gs[2, 0])
    
    opt_rates = [p['opt_rate'] for p in params]
    
    ax4.plot(episodes, opt_rates, color=colors['opt_rate'], 
            linewidth=2, marker='o', markersize=4, markevery=max(1, len(episodes)//20))
    
    ax4.set_xlabel('Training Episode', fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontweight='bold')
    ax4.set_title('(d) Optimization Learning Rate', fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)
    
    # ========== Subplot 5: Reward Distribution ==========
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Histogram of rewards
    n_bins = min(20, len(rewards) // 3)
    ax5.hist(rewards, bins=n_bins, color=colors['reward'], alpha=0.7, edgecolor='black', linewidth=1.2)
    ax5.axvline(np.mean(rewards), color=colors['mean_reward'], linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax5.axvline(np.max(rewards), color='gold', linestyle='--', 
               linewidth=2, label=f'Max: {np.max(rewards):.2f}')
    
    ax5.set_xlabel('Cumulative Reward', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('(e) Reward Distribution', fontweight='bold', pad=10)
    ax5.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add overall title
    fig.suptitle('NMPC Parameter Tuning via Reinforcement Learning', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Publication-quality plots saved to {save_path} (DPI: {dpi})")
    
    # Also save as PDF for vector graphics
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Vector graphics version saved to {pdf_path}")
    
    plt.close()


def save_training_history(history, filepath='nmpc_training_history.json'):
    """Save training history to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    history_save = {
        'episode_rewards': [float(r) for r in history['episode_rewards']],
        'episode_lengths': [int(l) for l in history['episode_lengths']],
        'best_params': history['best_params'],
        'mean_rewards': [float(r) for r in history.get('mean_rewards', [])]
    }
    with open(filepath, 'w') as f:
        json.dump(history_save, f, indent=2)
    print(f"Training history saved to {filepath}")


def load_training_history(filepath='nmpc_training_history.json'):
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        history = json.load(f)
    print(f"Training history loaded from {filepath}")
    return history


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
    
    # Save training history
    save_training_history(history)
    
    # Plot results (publication quality)
    plot_training_history(history, save_path='nmpc_rl_tuning_results.png', dpi=300)
    
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

