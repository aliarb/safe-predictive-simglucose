#!/usr/bin/env python3
"""
Quick script to run RL tuning and generate plots.
Uses fewer episodes for faster execution.
"""
import sys
sys.path.insert(0, '.')

from tune_nmpc_with_rl import (
    NMPCTuningEnv, SimpleRLAgent, train_rl_agent, 
    plot_training_history, save_training_history
)
from simglucose.simulation.scenario import CustomScenario
from datetime import datetime

if __name__ == "__main__":
    print("=" * 70)
    print("QUICK RL TUNING FOR PLOT GENERATION")
    print("=" * 70)
    
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
        episode_length=72,  # Very short episodes (6 hours) for faster training
        seed=42
    )
    
    # Create RL agent
    agent = SimpleRLAgent(learning_rate=0.01)
    
    # Train agent with fewer episodes for quick plot generation
    num_episodes = 20  # Reduced for faster execution
    print(f"\nTraining for {num_episodes} episodes (reduced for quick plot generation)...")
    history, best_params = train_rl_agent(env, agent, num_episodes=num_episodes)
    
    # Save training history
    save_training_history(history)
    
    # Plot results (publication quality)
    print("\nGenerating publication-quality plots...")
    plot_training_history(history, save_path='nmpc_rl_tuning_results.png', dpi=300)
    
    print("\n" + "=" * 70)
    print("PLOT GENERATION COMPLETE")
    print("=" * 70)
    print(f"Plots saved to: nmpc_rl_tuning_results.png")
    print(f"Training history saved to: nmpc_training_history.json")
    print("=" * 70)

