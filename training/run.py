"""
Run training for the Knights vs Archers game with real-time visualization.
"""

import sys
import os

# Rest of your imports and code...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from submission_single import CustomPredictFunction, CustomWrapper, train_archer_agent, evaluate_agent, compare_with_baselines
from utils import create_environment
import matplotlib.pyplot as plt
import json
import logging
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator

logger = logging.getLogger(__name__)

# Import the NN parameters 
# (assuming they're defined in agent_implementation.py - adjust import as needed)
try:
    from submission_single import (
        BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, KL_COEFF, 
        CLIP_PARAM, VF_CLIP_PARAM, ENTROPY_COEFF, NUM_SGD_ITER, 
        HIDDEN_LAYERS
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
        "NUM_SGD_ITER": NUM_SGD_ITER,
        "HIDDEN_LAYERS": HIDDEN_LAYERS
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
        max_cycles=2500
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
        from submission_single import (
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
            max_iterations=args.max_iterations,  # Reduce iterations for faster tuning
            plot_dir=session_dir,
            monitor=monitor
        )
        
        # Evaluate the trained agent
        mean_reward = evaluate_agent(env, num_episodes=5)
        
        # Report the results to Tune
        tune.report(mean_reward=mean_reward)
    
    # Use BasicVariantGenerator which supports all parameter types
    search_alg = BasicVariantGenerator()
    
    # Set up the scheduler
    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=args.max_iterations,
        grace_period=args.max_iterations,
        reduction_factor=2
    )
    
    # Run the hyperparameter search
    print(f"Starting hyperparameter tuning with {args.tune_samples} samples...")
    tuner = tune.Tuner(
        train_function,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.tune_samples,
            scheduler=scheduler,
            search_alg=search_alg
        )
    )
    
    results = tuner.fit()
    
    # Get the best hyperparameters
    best_result = results.get_best_result(metric="mean_reward", mode="max")
    best_config = best_result.config
    best_reward = best_result.metrics["mean_reward"]
    
    print(f"\nBest hyperparameters found:")
    for param, value in best_config.items():
        print(f"{param}: {value}")
    print(f"Best mean reward: {best_reward}")
    
    # Save the best hyperparameters to a file
    best_params_dir = os.path.join(args.plot_dir, "best_params")
    Path(best_params_dir).mkdir(exist_ok=True)
    
    with open(os.path.join(best_params_dir, "best_params.json"), "w") as f:
        json.dump(best_config, f, indent=4)
    
    print(f"Best hyperparameters saved to {best_params_dir}/best_params.json")
    
    return best_config

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
        max_cycles=2500
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
    
    # Create all combinations of parameters
    import itertools
    keys = grid_search_space.keys()
    values = grid_search_space.values()
    
    # Calculate total combinations
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    
    # Limit the number of combinations if too many
    if total_combinations > args.max_grid_combinations:
        print(f"Warning: Grid search would generate {total_combinations} combinations.")
        print(f"Limiting to {args.max_grid_combinations} random combinations.")
        
        # Generate a subset of random combinations
        import random
        all_combinations = list(itertools.product(*values))
        random.seed(42)  # For reproducibility
        selected_combinations = random.sample(all_combinations, args.max_grid_combinations)
    else:
        selected_combinations = list(itertools.product(*values))
        print(f"Grid search will test {total_combinations} parameter combinations.")
    
    # Track the best parameters and reward
    best_params = None
    best_reward = float('-inf')
    
    # Create a directory for grid search results
    grid_search_dir = os.path.join(args.plot_dir, "grid_search_results")
    Path(grid_search_dir).mkdir(exist_ok=True, parents=True)
    
    # Save the grid search space for reference
    with open(os.path.join(grid_search_dir, "search_space.json"), "w") as f:
        json.dump(grid_search_space, f, indent=4)
    
    # Run the grid search
    print(f"Starting grid search with {len(selected_combinations)} parameter combinations...")
    
    results = []
    for i, combination in enumerate(selected_combinations):
        # Create parameter dictionary
        params = dict(zip(keys, combination))
        print(f"\nTesting combination {i+1}/{len(selected_combinations)}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Override the global parameters
        from submission_single import (
            BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, KL_COEFF, 
            CLIP_PARAM, VF_CLIP_PARAM, ENTROPY_COEFF, NUM_SGD_ITER, 
            HIDDEN_LAYERS
        )
        
        # Override global variables with the current combination
        globals()["BATCH_SIZE"] = params["batch_size"]
        globals()["LEARNING_RATE"] = params["learning_rate"]
        globals()["GAMMA"] = params["gamma"]
        globals()["LAMBDA"] = params["lambda_"]
        globals()["KL_COEFF"] = params["kl_coeff"]
        globals()["CLIP_PARAM"] = params["clip_param"]
        globals()["VF_CLIP_PARAM"] = params["vf_clip_param"]
        globals()["ENTROPY_COEFF"] = params["entropy_coeff"]
        globals()["NUM_SGD_ITER"] = params["num_sgd_iter"]
        globals()["HIDDEN_LAYERS"] = params["hidden_layers"]
        
        # Create a new session directory for this combination
        _, session_dir = get_next_session_id(args.plot_dir)
        
        # Save the hyperparameters for this trial
        save_nn_params(session_dir)
        
        # Set up training monitor
        from training.training_monitor import setup_training_monitor
        monitor = setup_training_monitor(save_dir=session_dir, live_plot=False)
        
        try:
            # Train the agent with the current hyperparameters
            algo = train_archer_agent(
                env, 
                args.checkpoint_dir, 
                max_iterations=1000,  # Reduce iterations for faster grid search
                plot_dir=session_dir,
                monitor=monitor
            )
            
            # Evaluate the trained agent
            mean_reward = evaluate_agent(env, num_episodes=5)
            
            # Record the results
            result = {
                "params": params,
                "mean_reward": mean_reward,
                "session_dir": session_dir
            }
            results.append(result)
            
            # Update best parameters if this combination is better
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_params = params.copy()
                print(f"New best reward: {best_reward}")
            
            # Save the current results
            with open(os.path.join(grid_search_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4, default=str)
                
        except Exception as e:
            print(f"Error during training with parameters: {params}")
            print(f"Error: {e}")
            # Continue with the next combination
    
    # Print and save the best parameters
    print(f"\nGrid search complete. Best parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best mean reward: {best_reward}")
    
    # Save the best hyperparameters to a file
    best_params_dir = os.path.join(args.plot_dir, "best_grid_params")
    Path(best_params_dir).mkdir(exist_ok=True)
    
    with open(os.path.join(best_params_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Best grid search parameters saved to {best_params_dir}/best_params.json")
    
    return best_params

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
        max_cycles=2500
    )
    
    # Apply custom wrapper
    env = CustomWrapper(env)
    
    # Define a refined grid search space based on previous results
    # Focus on the parameter ranges that performed best in the initial grid search
    refined_grid_search_space = {
        # Batch size: 2048 consistently outperformed 1024
        "batch_size": [2048],
        
        # Learning rate: Mid-range learning rates performed best
        "learning_rate": [0.0003],
        
        # Gamma: Both high values performed well
        "gamma": [0.99],
        
        # Lambda: Both values can work well
        "lambda_": [0.9],
        
        # KL coefficient: 0.2 appeared in the best configuration
        "kl_coeff": [0.2],
        
        # Clip parameter: Lower values (0.1) performed better
        "clip_param": [0.1],
        
        # Value function clip parameter: Keep constant
        "vf_clip_param": [10.0],
        
        # Entropy coefficient: Lower values (0.005) performed better
        "entropy_coeff": [0.01],
        
        # Number of SGD iterations: Both 10 and 15 performed well
        "num_sgd_iter": [10],
        
        # Neural network architecture: Both performed well, with deeper slightly better
        "hidden_layers": [
            # [256, 256, 128],
            # [256, 256, 256],
            # [512, 256, 128],
            # [512, 256, 128, 64],
            [512, 512, 512],
            # [512, 512, 512, 256],
            # [1024, 1024, 1024],
            # [2048, 2048, 2048],
            # [512, 512, 512, 512],
            # [512, 512, 512, 512, 512],
        ]
    }
    
    # Create all combinations of parameters
    import itertools
    keys = refined_grid_search_space.keys()
    values = refined_grid_search_space.values()
    
    # Calculate total combinations
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)
    
    # Limit the number of combinations if too many
    if total_combinations > args.max_grid_combinations:
        print(f"Warning: Refined grid search would generate {total_combinations} combinations.")
        print(f"Limiting to {args.max_grid_combinations} random combinations.")
        
        # Generate a subset of random combinations
        import random
        all_combinations = list(itertools.product(*values))
        random.seed(42)  # For reproducibility
        selected_combinations = random.sample(all_combinations, args.max_grid_combinations)
    else:
        selected_combinations = list(itertools.product(*values))
        print(f"Refined grid search will test {total_combinations} parameter combinations.")
    
    # Track the best parameters and reward
    best_params = None
    best_reward = float('-inf')
    
    # Create a directory for grid search results
    grid_search_dir = os.path.join(args.plot_dir, "refined_grid_search_results")
    Path(grid_search_dir).mkdir(exist_ok=True, parents=True)
    
    # Save the grid search space for reference
    with open(os.path.join(grid_search_dir, "search_space.json"), "w") as f:
        json.dump(refined_grid_search_space, f, indent=4)
    
    # Run the grid search
    print(f"Starting refined grid search with {len(selected_combinations)} parameter combinations...")
    
    results = []
    for i, combination in enumerate(selected_combinations):
        # Create parameter dictionary
        params = dict(zip(keys, combination))
        print(f"\nTesting combination {i+1}/{len(selected_combinations)}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # Override the global parameters
        from submission_single import (
            BATCH_SIZE, LEARNING_RATE, GAMMA, LAMBDA, KL_COEFF, 
            CLIP_PARAM, VF_CLIP_PARAM, ENTROPY_COEFF, NUM_SGD_ITER, 
            HIDDEN_LAYERS
        )
        
        # Override global variables with the current combination
        globals()["BATCH_SIZE"] = params["batch_size"]
        globals()["LEARNING_RATE"] = params["learning_rate"]
        globals()["GAMMA"] = params["gamma"]
        globals()["LAMBDA"] = params["lambda_"]
        globals()["KL_COEFF"] = params["kl_coeff"]
        globals()["CLIP_PARAM"] = params["clip_param"]
        globals()["VF_CLIP_PARAM"] = params["vf_clip_param"]
        globals()["ENTROPY_COEFF"] = params["entropy_coeff"]
        globals()["NUM_SGD_ITER"] = params["num_sgd_iter"]
        globals()["HIDDEN_LAYERS"] = params["hidden_layers"]
        
        # Create a new session directory for this combination
        _, session_dir = get_next_session_id(args.plot_dir)
        
        # Save the hyperparameters for this trial
        save_nn_params(session_dir)
        
        # Set up training monitor
        from training.training_monitor import setup_training_monitor
        monitor = setup_training_monitor(save_dir=session_dir, live_plot=False)
        
        try:
            # Train the agent with the current hyperparameters
            # Use more iterations for refined search to get more accurate results
            algo = train_archer_agent(
                env, 
                args.checkpoint_dir,
                max_iterations=500,  # More iterations for refined search
                plot_dir=session_dir,
                monitor=monitor
            )
            
            # Evaluate the trained agent with more episodes for more reliable results
            mean_reward = evaluate_agent(env, num_episodes=10)
            
            # Record the results
            result = {
                "params": params,
                "mean_reward": mean_reward,
                "session_dir": session_dir
            }
            results.append(result)
            
            # Update best parameters if this combination is better
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_params = params.copy()
                print(f"New best reward: {best_reward}")
            
            # Save the current results
            with open(os.path.join(grid_search_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4, default=str)
                
        except Exception as e:
            print(f"Error during training with parameters: {params}")
            print(f"Error: {e}")
            # Continue with the next combination
    
    # Print and save the best parameters
    print(f"\nRefined grid search complete. Best parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best mean reward: {best_reward}")
    
    # Save the best hyperparameters to a file
    best_params_dir = os.path.join(args.plot_dir, "best_refined_params")
    Path(best_params_dir).mkdir(exist_ok=True)
    
    with open(os.path.join(best_params_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Best refined parameters saved to {best_params_dir}/best_params.json")
    
    return best_params

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
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning with Ray Tune')
    parser.add_argument('--tune-samples', type=int, default=20,
                        help='Number of hyperparameter samples to try during tuning')
    parser.add_argument('--grid-search', action='store_true',
                        help='Run tactical grid search for hyperparameter optimization')
    parser.add_argument('--refined-grid-search', action='store_true',
                        help='Run refined grid search based on previous results')
    parser.add_argument('--max-grid-combinations', type=int, default=24,
                        help='Maximum number of grid search combinations to try')
    return parser.parse_args()

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
        from submission_single import (
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
        from submission_single import (
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
        from submission_single import (
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
        max_cycles=2500,
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
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    print(f"Training plots will be saved to: {plot_dir}")
    
    algo = train_archer_agent(
        env, 
        args.checkpoint_dir, 
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