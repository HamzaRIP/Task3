import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
# Set Ray's temporary directory to a location with more space
os.environ["TMPDIR"] = os.path.expanduser("~/ray_tmp")  # This will create a directory in your home folder

import gymnasium
from gymnasium import spaces
from pathlib import Path
import numpy as np
import torch
from typing import Callable
import matplotlib.pyplot as plt
import sys

# Get the absolute path to the module directory
package_directory = os.path.dirname(os.path.abspath(__file__))

from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env
import pettingzoo
import supersuit as ss


# Import from local utils.py file using absolute path
sys.path.append(package_directory)
from utils import create_environment
# Import our training monitor using absolute path
sys.path.append(os.path.join(package_directory, "training"))
from training_monitor import setup_training_monitor



from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env
import pettingzoo
import supersuit as ss


# Import from local utils.py file
from utils import create_environment
# Import our training monitor
from training.training_monitor import setup_training_monitor

# batch_size: 2048
# learning_rate: 0.0003
# gamma: 0.99
# lambda_: 0.9
# kl_coeff: 0.2
# clip_param: 0.1
# vf_clip_param: 10.0
# entropy_coeff: 0.005
# num_sgd_iter: 10
# hidden_layers: [256, 256, 128]


BATCH_SIZE = 2048        # No. steps collected for training in each batch, larger batches provide more stable gradients.
LEARNING_RATE = 0.0003   # Gradient update step size, controls how quickly the neural network weights are adjusted.
GAMMA = 0.99           # Discount factor for future rewards, values closer to 1 place more importance on long-term rewards.
LAMBDA = 0.9          # GAE (Generalized Advantage Estimation) parameter, controls bias-variance tradeoff in advantage estimation.
KL_COEFF = 0.2         # Coeff for KL divergence penalty, prevents policy updates from changing too drastically from previous policy.
CLIP_PARAM = 0.1       # PPO clipping parameter, limits policy ratio to prevent too large policy updates.
VF_CLIP_PARAM = 10.0    # Value function clipping parameter, limits how much the value function estimates can change per update.
ENTROPY_COEFF = 0.005   # Coeff for entropy bonus, encourages exploration by rewarding policies with higher action entropy.
NUM_SGD_ITER = 10      # No. SGD passes over the training data, determines how many times each batch is reused for optimization.

HIDDEN_LAYERS = [256, 256, 128]

class CustomWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = None
        self._initialized = False
        
        # Get environment parameters from the environment's configuration
        self.max_zombies = env.unwrapped.max_zombies if hasattr(env.unwrapped, 'max_zombies') else 4
        self.num_archers = env.unwrapped.num_archers if hasattr(env.unwrapped, 'num_archers') else 1

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        if not self._initialized:
            # Initialize observation space on first access
            original_space = spaces.flatten_space(self.env.observation_space(agent))
            # The model expects 68 input features
            self._observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(68,),  # Fixed size to match trained model
                dtype=np.float32
            )
            self._initialized = True
        return self._observation_space

    def _process_observation(self, obs):
        """
        Process the raw observation into enhanced features.
        
        Args:
            obs: Raw observation array of shape (N+1)x5
            
        Returns:
            Enhanced observation array with additional features
        """
        # Get the current agent's position and heading
        current_agent = obs[0]
        agent_pos = current_agent[1:3]  # x, y position
        agent_heading = current_agent[3:5]  # heading vector
        
        # Get zombie and archer rows
        zombie_rows = obs[-(self.max_zombies):]  # Last max_zombies rows
        archer_rows = obs[1:self.num_archers + 1]  # Rows after current agent
        
        # Filter out empty rows (where distance is 0)
        zombie_data = [(row[0], row[1:3], row[3:5]) for row in zombie_rows if row[0] > 0]
        archer_data = [(row[0], row[1:3], row[3:5]) for row in archer_rows if row[0] > 0]
        
        # 1. Calculate zombie threat level (weighted by distance and number)
        zombie_threat = 0.0
        if zombie_data:
            distances = [d for d, _, _ in zombie_data]
            zombie_threat = sum(1/d for d in distances if d < 0.5)  # Closer zombies contribute more to threat
        
        # 2. Calculate safe space score (distance from nearest entity)
        all_distances = [d for d, _, _ in zombie_data + archer_data]
        safe_space = min(all_distances) if all_distances else 1.0
        
        # 3. Calculate optimal shooting angle
        optimal_angle = 0.0
        if zombie_data:
            # Find nearest zombie
            nearest_zombie = min(zombie_data, key=lambda x: x[0])
            zombie_pos = nearest_zombie[1]
            # Calculate angle between agent heading and vector to zombie
            to_zombie = zombie_pos - agent_pos
            to_zombie = to_zombie / np.linalg.norm(to_zombie)
            optimal_angle = np.dot(agent_heading, to_zombie)
        
        # 4. Calculate zombie clustering (how spread out are the zombies)
        zombie_clustering = 0.0
        if len(zombie_data) > 1:
            zombie_positions = [pos for _, pos, _ in zombie_data]
            # Calculate average distance between zombies
            distances = []
            for i in range(len(zombie_positions)):
                for j in range(i+1, len(zombie_positions)):
                    dist = np.linalg.norm(zombie_positions[i] - zombie_positions[j])
                    distances.append(dist)
            zombie_clustering = np.mean(distances) if distances else 1.0
        
        # 5. Calculate escape route score (how many directions are blocked)
        escape_score = 0.0
        if zombie_data:
            # Check 8 directions around the agent
            directions = [
                (1,0), (1,1), (0,1), (-1,1),
                (-1,0), (-1,-1), (0,-1), (1,-1)
            ]
            for dx, dy in directions:
                direction = np.array([dx, dy])
                direction = direction / np.linalg.norm(direction)
                # Check if any zombie is in this direction
                blocked = any(
                    np.dot(zombie_pos - agent_pos, direction) > 0.7
                    for _, zombie_pos, _ in zombie_data
                )
                if not blocked:
                    escape_score += 1
            escape_score /= 8.0  # Normalize to [0,1]
        
        # 6. Calculate zombie movement prediction
        zombie_movement = 0.0
        if zombie_data:
            # Calculate average zombie velocity (heading)
            zombie_velocities = [vel for _, _, vel in zombie_data]
            avg_velocity = np.mean(zombie_velocities, axis=0)
            # Project onto agent's position to predict if zombies are moving towards agent
            to_agent = agent_pos - np.array([0.5, 0.5])  # Center of the map
            zombie_movement = np.dot(avg_velocity, to_agent)
        
        # 7. Calculate tactical position score
        tactical_score = 0.0
        if zombie_data:
            # Prefer positions that are not surrounded by zombies
            zombie_positions = [pos for _, pos, _ in zombie_data]
            center = np.mean(zombie_positions, axis=0)
            # Score is higher when agent is not in the center of zombie cluster
            tactical_score = 1.0 - np.linalg.norm(agent_pos - center)
        
        # 8. Calculate shooting opportunity score
        shooting_score = 0.0
        if zombie_data:
            # Find zombies that are in a good position to shoot
            for dist, pos, _ in zombie_data:
                if dist < 0.5:  # Only consider close zombies
                    to_zombie = pos - agent_pos
                    to_zombie = to_zombie / np.linalg.norm(to_zombie)
                    alignment = np.dot(agent_heading, to_zombie)
                    if alignment > 0.8:  # Good alignment for shooting
                        shooting_score += 1
            shooting_score = min(shooting_score / len(zombie_data), 1.0)
        
        # Combine all features
        enhanced_features = [
            zombie_threat,
            safe_space,
            optimal_angle,
            zombie_clustering,
            escape_score,
            zombie_movement,
            tactical_score,
            shooting_score
        ]
        
        # Combine original observation with enhanced features
        original_features = obs.flatten()
        combined_features = np.concatenate([original_features, enhanced_features])
        
        # Ensure the output has exactly 68 features
        if len(combined_features) < 68:
            # Pad with zeros if too short
            combined_features = np.pad(combined_features, (0, 68 - len(combined_features)))
        elif len(combined_features) > 68:
            # Truncate if too long
            combined_features = combined_features[:68]
            
        return combined_features

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        if obs is None:
            return None
        return self._process_observation(obs)

class CustomPredictFunction(Callable):
    """
    Prediction function for the trained archer agent.
    Loads a trained RLLib algorithm from a checkpoint and extracts the policies.
    """
    def __init__(self, env):
        # Get the absolute path to the module directory
        package_directory = os.path.dirname(os.path.abspath(__file__))
        # Load the trained model from checkpoint using absolute path
        checkpoint_path = os.path.join(package_directory, "results", "learner_group", "learner", "rl_module")
        self.modules = MultiRLModule.from_checkpoint(checkpoint_path)
    
    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action

def algo_config(id_env, env, policies, policies_to_train):
    """
    Configure the PPO algorithm for training the archer agent.
    """
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=4)
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    x: RLModuleSpec(
                        module_class=DefaultPPOTorchRLModule,
                        observation_space=env.observation_space(x), 
                        action_space=env.action_space(x),
                        model_config={
                            "fcnet_hiddens": HIDDEN_LAYERS,
                            "fcnet_activation": "relu",
                            "input_dim": env.observation_space(x).shape[0]
                        }
                    )
                    for x in policies
                },
            )
        )
        .training(
            train_batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            gamma=GAMMA,
            lambda_=LAMBDA,
            kl_coeff=KL_COEFF,
            clip_param=CLIP_PARAM,
            vf_clip_param=VF_CLIP_PARAM,
            entropy_coeff=ENTROPY_COEFF,
            num_epochs=NUM_SGD_ITER,
        )
        .debugging(log_level="ERROR")
    )
    return config

# And when creating the algorithm:

def train_archer_agent(env, checkpoint_path, max_iterations=1000, plot_dir="./training_plots", monitor=None):
    """
    Train the archer agent using PPO.
    
    Args:
        env: PettingZoo environment
        checkpoint_path: Path to save checkpoints
        max_iterations: Maximum number of training iterations
        plot_dir: Directory to save training plots
        monitor: Optional TrainingMonitor instance (for live comparison)
        
    Returns:
        Trained algorithm
    """
    # Set up training monitor
    if monitor is None:
        # Get the absolute path to the module directory
        package_directory = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(package_directory, "training"))
        from training_monitor import setup_training_monitor
        monitor = setup_training_monitor(save_dir=plot_dir, log_interval=1, live_plot=True)
    
    # Convert AEC environment to parallel for RLlib
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda config: rllib_env)
    
    # Fix seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define the configuration for the PPO algorithm
    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config(id_env, env, policies, policies_to_train)
    
    # Train the model
    algo = config.build_algo()
    best_reward = -float('inf')
    
    print("Starting training...")
    for i in range(max_iterations):
        result = algo.train()
        result.pop("config", None)
        
        # Extract training metrics
        metrics = {}
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            metrics["agent_returns"] = result["env_runners"]["agent_episode_returns_mean"]
            mean_reward = result["env_runners"]["agent_episode_returns_mean"]["archer_0"]
            print(f"Iteration {i}, Mean Reward: {mean_reward}")
            
            # Update training monitor
            monitor.update(i, metrics)
            
            # Save checkpoint if performance improves
            if mean_reward > best_reward:
                best_reward = mean_reward
                save_result = algo.save(checkpoint_path)
                path_to_checkpoint = save_result.checkpoint.path
                print(f"New best reward: {best_reward}, saved checkpoint to: {path_to_checkpoint}")
            
            # Early stopping if performance is good enough
            # if mean_reward > 10:
            #     print(f"Early stopping at iteration {i} with mean reward {mean_reward}")
            #     break
        else:
            # If metrics are structured differently, create a default metric for plotting
            default_metrics = {"agent_returns": {"archer_0": result.get("episode_reward_mean", 0)}}
            monitor.update(i, default_metrics)
        
        # Regular checkpoint saving
        if i % 10 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(f"Checkpoint saved to: {path_to_checkpoint}")
    
    # Finalize monitor
    monitor.finalize()
    
    print(f"Training completed. Best reward: {best_reward}")
    return algo

#######################
# TRAINING FUNCTIONS  #
#######################

def evaluate_agent(env, num_episodes=10):
    """
    Evaluate the trained agent.
    
    Args:
        env: PettingZoo environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Mean reward across episodes
    """
    # Create a prediction function using the trained model
    predict_fn = CustomPredictFunction(env)
    
    total_rewards = []
    for episode in range(num_episodes):
        env.reset()
        episode_rewards = {agent: 0 for agent in env.agents}
        
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            episode_rewards[agent] += reward
            
            if termination or truncation:
                action = None
            else:
                action = predict_fn(observation, agent)
            
            env.step(action)
            
            # Check if episode is done (all agents are done)
            all_done = True
            for a in env.agents:
                _, _, term, trunc, _ = env.last(a)
                if not (term or trunc):
                    all_done = False
                    break
                    
            if all_done:
                break
        
        # Calculate episode reward
        episode_reward = sum(episode_rewards.values())
        total_rewards.append(episode_reward)
        print(f"Episode {episode}: Total Reward = {episode_reward}")
    
    mean_reward = sum(total_rewards) / len(total_rewards)
    print(f"Evaluation complete. Mean reward over {num_episodes} episodes: {mean_reward}")
    return mean_reward

def compare_with_baselines(env, trained_agent, num_episodes=10):
    """
    Compare the trained agent with baseline strategies.
    
    Args:
        env: PettingZoo environment
        trained_agent: Trained agent prediction function
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary of mean rewards for each strategy
    """
    # Define baseline strategies
    baselines = {
        "trained_agent": trained_agent,
        "random": lambda obs, agent: env.action_space(agent).sample(),
        "always_shoot": lambda obs, agent: 5,  # Action 5 is shooting
        "always_move_right": lambda obs, agent: 3,  # Action 3 is moving right
    }
    
    results = {}
    
    for name, strategy in baselines.items():
        print(f"\nEvaluating {name} strategy...")
        total_rewards = []
        
        for episode in range(num_episodes):
            env.reset()
            episode_rewards = {agent: 0 for agent in env.agents}
            
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                episode_rewards[agent] += reward
                
                if termination or truncation:
                    action = None
                else:
                    action = strategy(observation, agent)
                
                env.step(action)
                
                # Check if episode is done (all agents are done)
                all_done = True
                for a in env.agents:
                    _, _, term, trunc, _ = env.last(a)
                    if not (term or trunc):
                        all_done = False
                        break
                        
                if all_done:
                    break
            
            # Calculate episode reward
            episode_reward = sum(episode_rewards.values())
            total_rewards.append(episode_reward)
            print(f"Episode {episode}: Total Reward = {episode_reward}")
        
        mean_reward = sum(total_rewards) / len(total_rewards)
        results[name] = mean_reward
        print(f"{name} strategy: Mean reward over {num_episodes} episodes: {mean_reward}")
    
    return results


if __name__ == "__main__":
    # Create the environment with a single archer
    num_agents = 1
    visual_observation = False
    max_zombies = 4
    
    print("Creating environment...")
    env = create_environment(
        num_agents=num_agents,
        visual_observation=visual_observation,
        max_zombies=max_zombies,
        max_cycles=1000
    )
    
    # Apply custom wrapper
    env = CustomWrapper(env)
    
    # Set up checkpoint path and plot directory
    checkpoint_path = str(Path("results").resolve())
    plot_dir = str(Path("training_plots").resolve())
    
    # Train the agent with visualization
    print("Training agent...")
    algo = train_archer_agent(env, checkpoint_path, max_iterations=500, plot_dir=plot_dir)
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    trained_agent = CustomPredictFunction(env)
    mean_reward = evaluate_agent(env, num_episodes=10)
    
    # Compare with baselines
    print("\nComparing with baselines...")
    baseline_results = compare_with_baselines(env, trained_agent, num_episodes=10)
    
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
    plt.show()
