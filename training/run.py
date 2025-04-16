"""
Run training for the Knights vs Archers game with real-time visualization.
"""

import sys
import os

# Rest of your imports and code...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from agent_implementation import CustomPredictFunction, CustomWrapper, train_archer_agent, evaluate_agent, compare_with_baselines
from utils import create_environment
import matplotlib.pyplot as plt
import json
import logging

logger = logging.getLogger(__name__)

# Import the NN parameters 
# (assuming they're defined in agent_implementation.py - adjust import as needed)
try:
    from agent_implementation import (
        BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, KL_COEFF, 
        CLIP_PARAM, VF_CLIP_PARAM, ENTROPY_COEFF, NUM_SGD_ITER
    )
except ImportError:
    raise ImportError("Neural network parameters not found")

def save_nn_params(save_dir):
    """
    Save neural network parameters as JSON
    
    Args:
        save_dir: Directory to save the parameters
    """
    params = {
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "GAMMA": GAMMA, 
        "LAMBDA": LAMBDA,
        "KL_COEFF": KL_COEFF,
        "CLIP_PARAM": CLIP_PARAM,
        "VF_CLIP_PARAM": VF_CLIP_PARAM,
        "ENTROPY_COEFF": ENTROPY_COEFF,
        "NUM_SGD_ITER": NUM_SGD_ITER
    }
    
    # Save as JSON
    with open(os.path.join(save_dir, "nn_params.json"), "w") as f:
        json.dump(params, f, indent=4)
    
    print(f"Neural network parameters saved to {save_dir}/nn_params.json")

def get_next_session_id(base_dir="training/training_plots"):
    """
    Find the next available session ID by checking existing directories.
    
    Args:
        base_dir: Base directory for training data
        
    Returns:
        Next available session ID and the full path to the session directory
    """
    # Ensure base directory exists
    Path(base_dir).mkdir(exist_ok=True, parents=True)
    
    # Get all existing session directories (named with integers)
    existing_sessions = [d for d in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    
    # Find next session ID
    if not existing_sessions:
        next_id = 1
    else:
        next_id = max(map(int, existing_sessions)) + 1
    
    # Create new session directory
    session_dir = os.path.join(base_dir, str(next_id))
    Path(session_dir).mkdir(exist_ok=True)
    
    print(f"Created new training session: {session_dir}")
    return next_id, session_dir

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
                        help='Base directory for training plots')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--no-live-plot', action='store_true',
                        help='Disable live plotting (useful for headless servers)')
    parser.add_argument('--screen', '-s', action='store_true',
                        help='Set render mode to human (show game)')
    parser.add_argument('--compare', nargs='?', const='Comparison_run', default=None, type=str,
                        help='If set, overlay training history from given training ID (default: Comparison_run)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    visual_observation = False
    render_mode = "human" if args.screen else None # "human" or None
    logger.info(f'Show game: {render_mode}')
    if render_mode == "human":
        logger.info(f'Press q to end game')
    logger.info(f'Use pixels: {visual_observation}')
    
    # Get next session ID and create session directory
    _, session_dir = get_next_session_id(args.plot_dir)
    
    # Save neural network parameters
    save_nn_params(session_dir)
    
    # Create checkpoint directory
    checkpoint_path = str(Path(args.checkpoint_dir).resolve())
    Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
    
    # Use session directory for plots
    plot_dir = session_dir
    
    # Create the environment
    print("Creating environment...")
    env = create_environment(
        num_agents=args.num_agents,
        visual_observation=visual_observation,
        max_zombies=args.max_zombies,
        max_cycles=1000,
        render_mode=render_mode
    )
    
    # Apply custom wrapper
    env = CustomWrapper(env)

    # Prepare comparison data if requested
    comparison_history = None
    comparison_id = args.compare
    if comparison_id:
        comp_hist_path = os.path.join(args.plot_dir, comparison_id, "training_history.json")
        if os.path.exists(comp_hist_path):
            with open(comp_hist_path, "r") as f:
                comparison_history = json.load(f)
            print(f"Loaded comparison training history from {comp_hist_path}")
        else:
            print(f"Warning: Comparison training history not found at {comp_hist_path}")

    # Pass comparison_history to the training monitor
    from training.training_monitor import setup_training_monitor
    monitor = setup_training_monitor(save_dir=plot_dir, live_plot=not args.no_live_plot, comparison_history=comparison_history)

    # Train the agent
    print(f"Training agent for up to {args.max_iterations} iterations...")
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    print(f"Training plots will be saved to: {plot_dir}")
    
    algo = train_archer_agent(
        env, 
        checkpoint_path, 
        max_iterations=args.max_iterations, 
        plot_dir=plot_dir,
        monitor=monitor
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    trained_agent = CustomPredictFunction(env)
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
    
    # Save session configuration
    with open(f"{plot_dir}/session_config.txt", "w") as f:
        f.write(f"Max iterations: {args.max_iterations}\n")
        f.write(f"Number of agents: {args.num_agents}\n")
        f.write(f"Max zombies: {args.max_zombies}\n")
        f.write(f"Evaluation episodes: {args.eval_episodes}\n")
        f.write(f"Final mean reward: {mean_reward}\n")
        f.write("\nBaseline comparison:\n")
        for strategy, reward in baseline_results.items():
            f.write(f"{strategy}: {reward}\n")
    
    # Show the plot if not in headless mode
    if not args.no_live_plot:
        plt.show()
    
    print(f"Training and evaluation complete. All results saved to {plot_dir}")

if __name__ == "__main__":
    main()