"""
Run training for the Knights vs Archers game with real-time visualization.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from agent_implementation import train_archer_agent, ArcherWrapper, evaluate_agent, ArcherPredictFunction, compare_with_baselines
from utils import create_environment
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train and visualize Knights vs Archers agent')
    parser.add_argument('--max-iterations', type=int, default=500, 
                        help='Maximum number of training iterations')
    parser.add_argument('--num-agents', type=int, default=1,
                        help='Number of archer agents')
    parser.add_argument('--max-zombies', type=int, default=10,
                        help='Maximum number of zombies')
    parser.add_argument('--checkpoint-dir', type=str, default='results',
                        help='Directory to save checkpoints')
    parser.add_argument('--plot-dir', type=str, default='training/training_plots',
                        help='Directory to save training plots')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--no-live-plot', action='store_true',
                        help='Disable live plotting (useful for headless servers)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create directories
    checkpoint_path = str(Path(args.checkpoint_dir).resolve())
    plot_dir = str(Path(args.plot_dir).resolve())
    Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
    Path(plot_dir).mkdir(exist_ok=True, parents=True)
    
    # Create the environment
    print("Creating environment...")
    env = create_environment(
        num_agents=args.num_agents,
        visual_observation=False,
        max_zombies=args.max_zombies,
        max_cycles=1000
    )
    
    # Apply custom wrapper
    env = ArcherWrapper(env)
    
    # Train the agent
    print(f"Training agent for up to {args.max_iterations} iterations...")
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    print(f"Training plots will be saved to: {plot_dir}")
    
    algo = train_archer_agent(
        env, 
        checkpoint_path, 
        max_iterations=args.max_iterations, 
        plot_dir=plot_dir
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    trained_agent = ArcherPredictFunction(env)
    mean_reward = evaluate_agent(env, num_episodes=args.eval_episodes)
    
    # Compare with baselines
    print("\nComparing with baselines...")
    baseline_results = compare_with_baselines(env, trained_agent, num_episodes=args.eval_episodes)
    
    # Print comparison results
    print("\nComparison Results:")
    for strategy, reward in baseline_results.items():
        print(f"{strategy}: {reward}")
    
    # Generate and display final results plot
    plt.figure(figsize=(10, 6))
    strategies = list(baseline_results.keys())
    rewards = [baseline_results[s] for s in strategies]
    
    plt.bar(strategies, rewards)
    plt.ylabel('Mean Reward')
    plt.title('Strategy Comparison')
    plt.savefig(f"{plot_dir}/strategy_comparison.png")
    
    # Show the plot if not in headless mode
    if not args.no_live_plot:
        plt.show()
    
    print(f"Training and evaluation complete. All results saved to {plot_dir}")

if __name__ == "__main__":
    main()
