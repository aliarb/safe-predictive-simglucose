#!/usr/bin/env python3
"""
Standalone script to generate publication-quality plots from saved NMPC tuning history.

Usage:
    python plot_nmpc_tuning.py [history_file.json] [output_file.png]

If no arguments provided, uses default files:
    - Input: nmpc_training_history.json
    - Output: nmpc_rl_tuning_results.png
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


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


if __name__ == "__main__":
    # Parse command line arguments
    history_file = sys.argv[1] if len(sys.argv) > 1 else 'nmpc_training_history.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'nmpc_rl_tuning_results.png'
    
    print(f"Loading training history from {history_file}...")
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        print(f"Loaded {len(history['episode_rewards'])} episodes")
        
        # Generate plots
        plot_training_history(history, save_path=output_file, dpi=300)
        
    except FileNotFoundError:
        print(f"Error: File {history_file} not found.")
        print("Please run tune_nmpc_with_rl.py first to generate training history.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

