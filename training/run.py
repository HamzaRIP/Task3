"""
Run training for the Knights vs Archers game with real-time visualization.
"""

import sys
import os

# Get the absolute path to the module directory and parent directory
package_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(package_directory)
sys.path.append(parent_directory)

import argparse
from pathlib import Path
from training.submission_single import CustomPredictFunction, CustomWrapper, train_archer_agent, evaluate_agent, compare_with_baselines
from training.utils import create_environment
import matplotlib.pyplot as plt
import json
import logging
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator

logger = logging.getLogger(__name__)

# ... (rest of the code remains the same)

def train_with_tune(args):
    """
    Use Ray Tune to find optimal hyperparameters for the PPO algorithm.
    
    Args:
        args: Command line arguments
    """
    # Create the environment
    print("Creating environment for hyperparameter tuning...")
    env = create_environment(
        num_agents=args.num_agents,
        visual_observation=False,
        max_zombies=args.max_zombies,
        max_cycles=1000
    )
    
    # Apply custom wrapper
    env = CustomWrapper(env)
    
    # Define the parameter search space
    search_space = {
        "batch_size": tune.choice([512, 1024, 2048, 4096]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.uniform(0.95, 0.995),
        "lambda_": tune.uniform(0.9, 0.99),
        "kl_coeff": tune.uniform(0.1, 0.5),
        "clip_param": tune.uniform(0.1, 0.3),
        "vf_clip_param": tune.choice([5.0, 10.0, 20.0]),
        "entropy_coeff": tune.loguniform(0.0001, 0.01),
        "num_sgd_iter": tune.choice([5, 10, 15, 20]),
        "hidden_layers": tune.choice([
            [128, 128],
            [256, 256],
            [256, 256, 128],
            [512, 256, 128]
        ])
    }
    
    # Define the training function for Tune
    def train_function(config):
        # Override the global parameters with the ones from Tune
        from training.submission_single import (
            BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, KL_COEFF, 
            CLIP_PARAM, VF_CLIP_PARAM, ENTROPY_COEFF, NUM_SGD_ITER, 
            HIDDEN_LAYERS
        )
        
        # Override global variables with the ones from Tune
        globals()["BATCH_SIZE"] = config["batch_size"]
        globals()["LEARNING_RATE"] = config["learning_rate"]
        globals()["GAMMA"] = config["gamma"]
        globals()["LAMBDA"] = config["lambda_"]
        globals()["KL_COEFF"] = config["kl_coeff"]
        globals()["CLIP_PARAM"] = config["clip_param"]
        globals()["VF_CLIP_PARAM"] = config["vf_clip_param"]
        globals()["ENTROPY_COEFF"] = config["entropy_coeff"]
        globals()["NUM_SGD_ITER"] = config["num_sgd_iter"]
        globals()["HIDDEN_LAYERS"] = config["hidden_layers"]
        
        # Create a new session directory for this trial
        _, session_dir = get_next_session_id(args.plot_dir)
        
        # Save the hyperparameters for this trial
        save_nn_params(session_dir)
        
        # Set up training monitor
        from training.training_monitor import setup_training_monitor
        monitor = setup_training_monitor(save_dir=session_dir, live_plot=False)
        
        # Train the agent with the current hyperparameters
        algo = train_archer_agent(
            env, 
            args.checkpoint_dir, 
            max_iterations=args.max_iterations // 5,  # Reduce iterations for faster tuning
            plot_dir=session_dir,
            monitor=monitor
        )
        
        # Evaluate the trained agent
        mean_reward = evaluate_agent(env, num_episodes=5)
        
        # Report the results to Tune
        tune.report(mean_reward=mean_reward)
    
    # ... (rest of the code remains the same)

def train_with_grid_search(args):
    """
    Use a tactical grid search with educated parameter choices for PPO optimization.
    
    Args:
        args: Command line arguments
    """
    # Create the environment
    print("Creating environment for grid search hyperparameter tuning...")
    env = create_environment(
        num_agents=args.num_agents,
        visual_observation=False,
        max_zombies=args.max_zombies,
        max_cycles=1000
    )
    
    # Apply custom wrapper
    env = CustomWrapper(env)
    
    # Define a focused grid search space based on PPO literature and best practices
    # These are carefully selected values that have shown good performance in similar tasks
    grid_search_space = {
        # Batch size: Larger batch sizes provide more stable gradients but require more computation
        # 1024-2048 is a common sweet spot for PPO
        "batch_size": [1024, 2048],
        
        # Learning rate: Critical for convergence speed and stability
        # 3e-4 is often cited as a good default in PPO papers
        "learning_rate": [1e-4, 3e-4, 5e-4],
        
        # Gamma (discount factor): Higher values (closer to 1) prioritize long-term rewards
        # For tasks with clear short-term rewards, slightly lower values can work better
        "gamma": [0.99, 0.995],
        
        # Lambda (GAE parameter): Controls bias-variance tradeoff in advantage estimation
        # 0.95 is a common default that works well in many environments
        "lambda_": [0.9, 0.95],
        
        # KL coefficient: Prevents policy from changing too drastically
        "kl_coeff": [0.2, 0.3],
        
        # Clip parameter: Limits policy ratio to prevent too large policy updates
        # 0.2 is recommended in the original PPO paper
        "clip_param": [0.1, 0.2],
        
        # Value function clip parameter
        "vf_clip_param": [10.0],
        
        # Entropy coefficient: Encourages exploration
        # For complex environments, slightly higher values can help
        "entropy_coeff": [0.01, 0.005],
        
        # Number of SGD iterations: How many passes over each batch
        "num_sgd_iter": [10, 15],
        
        # Neural network architecture: Deeper networks can capture more complex patterns
        "hidden_layers": [
            [256, 256],
            [256, 256, 128]
        ]
    }
    
    # ... (rest of the code remains the same)

def train_with_refined_grid_search(args):
    """
    Use a refined grid search based on previous results to fine-tune PPO hyperparameters.
    
    Args:
        args: Command line arguments
    """
    # Create the environment
    print("Creating environment for refined grid search...")
    env = create_environment(
        num_agents=args.num_agents,
        visual_observation=False,
        max_zombies=args.max_zombies,
        max_cycles=1000
    )
    
    # Apply custom wrapper
    env = CustomWrapper(env)
    
    # Define a refined grid search space based on previous results
    # Focus on the parameter ranges that performed best in the initial grid search
    refined_grid_search_space = {
        # Batch size: 2048 consistently outperformed 1024
        "batch_size": [2048],
        
        # Learning rate: Mid-range learning rates performed best
        "learning_rate": [0.0002, 0.0003, 0.0004, 0.0005],
        
        # Gamma: Both high values performed well
        "gamma": [0.99, 0.995],
        
        # Lambda: Both values can work well
        "lambda_": [0.9, 0.95],
        
        # KL coefficient: 0.2 appeared in the best configuration
        "kl_coeff": [0.15, 0.2, 0.25],
        
        # Clip parameter: Lower values (0.1) performed better
        "clip_param": [0.05, 0.1, 0.15],
        
        # Value function clip parameter: Keep constant
        "vf_clip_param": [10.0],
        
        # Entropy coefficient: Lower values (0.005) performed better
        "entropy_coeff": [0.003, 0.005, 0.007],
        
        # Number of SGD iterations: Both 10 and 15 performed well
        "num_sgd_iter": [8, 10, 12],
        
        # Neural network architecture: Both performed well, with deeper slightly better
        "hidden_layers": [
            [256, 256, 128],
            [256, 256, 192],
            [384, 256, 128]
        ]
    }
    
    # ... (rest of the code remains the same)

def main():
    # Parse command line arguments
    args = parse_args()

    # Create checkpoint directory with absolute path
    checkpoint_path = str(Path(args.checkpoint_dir).resolve())
    Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
    
    # Store the absolute path in args for use in grid search and other functions
    args.checkpoint_dir = checkpoint_path

    # If tuning is enabled, run hyperparameter tuning
    if args.tune:
        best_config = train_with_tune(args)
        # Update the global parameters with the best ones found
        from training.submission_single import (
            BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, KL_COEFF, 
            CLIP_PARAM, VF_CLIP_PARAM, ENTROPY_COEFF, NUM_SGD_ITER, 
            HIDDEN_LAYERS
        )
        globals()["BATCH_SIZE"] = best_config["batch_size"]
        globals()["LEARNING_RATE"] = best_config["learning_rate"]
        globals()["GAMMA"] = best_config["gamma"]
        globals()["LAMBDA"] = best_config["lambda_"]
        globals()["KL_COEFF"] = best_config["kl_coeff"]
        globals()["CLIP_PARAM"] = best_config["clip_param"]
        globals()["VF_CLIP_PARAM"] = best_config["vf_clip_param"]
        globals()["ENTROPY_COEFF"] = best_config["entropy_coeff"]
        globals()["NUM_SGD_ITER"] = best_config["num_sgd_iter"]
        globals()["HIDDEN_LAYERS"] = best_config["hidden_layers"]
    # If grid search is enabled, run tactical grid search
    elif args.grid_search:
        best_config = train_with_grid_search(args)
        # Update the global parameters with the best ones found
        from training.submission_single import (
            BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, KL_COEFF, 
            CLIP_PARAM, VF_CLIP_PARAM, ENTROPY_COEFF, NUM_SGD_ITER, 
            HIDDEN_LAYERS
        )
        globals()["BATCH_SIZE"] = best_config["batch_size"]
        globals()["LEARNING_RATE"] = best_config["learning_rate"]
        globals()["GAMMA"] = best_config["gamma"]
        globals()["LAMBDA"] = best_config["lambda_"]
        globals()["KL_COEFF"] = best_config["kl_coeff"]
        globals()["CLIP_PARAM"] = best_config["clip_param"]
        globals()["VF_CLIP_PARAM"] = best_config["vf_clip_param"]
        globals()["ENTROPY_COEFF"] = best_config["entropy_coeff"]
        globals()["NUM_SGD_ITER"] = best_config["num_sgd_iter"]
        globals()["HIDDEN_LAYERS"] = best_config["hidden_layers"]
    # If refined grid search is enabled, run refined grid search
    elif args.refined_grid_search:
        best_config = train_with_refined_grid_search(args)
        # Update the global parameters with the best ones found
        from training.submission_single import (
            BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, KL_COEFF, 
            CLIP_PARAM, VF_CLIP_PARAM, ENTROPY_COEFF, NUM_SGD_ITER, 
            HIDDEN_LAYERS
        )
        globals()["BATCH_SIZE"] = best_config["batch_size"]
        globals()["LEARNING_RATE"] = best_config["learning_rate"]
        globals()["GAMMA"] = best_config["gamma"]
        globals()["LAMBDA"] = best_config["lambda_"]
        globals()["KL_COEFF"] = best_config["kl_coeff"]
        globals()["CLIP_PARAM"] = best_config["clip_param"]
        globals()["VF_CLIP_PARAM"] = best_config["vf_clip_param"]
        globals()["ENTROPY_COEFF"] = best_config["entropy_coeff"]
        globals()["NUM_SGD_ITER"] = best_config["num_sgd_iter"]
        globals()["HIDDEN_LAYERS"] = best_config["hidden_layers"]
    
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
    
    # ... (rest of the code remains the same)