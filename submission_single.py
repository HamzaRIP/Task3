import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
# Set Ray's temporary directory to a location with more space
os.environ["TMPDIR"] = os.path.expanduser("~/ray_tmp")  # This will create a directory in your home folder

import gymnasium
from gymnasium import spaces
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Callable, Tuple, List, Dict
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

# DQN Implementation
class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = [512, 512, 256]):
        super(DQNNetwork, self).__init__()
        
        # Build layers dynamically based on hidden_layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)  # Add dropout for regularization
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [64, 32, 16],
        learning_rate: float = 0.0001,  # Reduced learning rate
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.997,  # Slower decay
        buffer_size: int = 200000,  # Larger buffer
        batch_size: int = 128,  # Larger batch size
        target_update: int = 5,  # More frequent target updates
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.device = device
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_layers).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_layers).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, eps=1e-5)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        
        # Reward scaling
        self.reward_scale = 1.0
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
    
    def update_reward_stats(self, reward: float):
        """Update reward statistics for normalization."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std += (delta * delta2) / self.reward_count
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        if self.reward_count > 1:
            return (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Normalize rewards
        rewards = rewards * self.reward_scale
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using Double DQN
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """Save the model checkpoint."""
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory:  # Only create directory if path contains a directory
            os.makedirs(directory, exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']

def train_dqn(
    env,
    agent: DQNAgent,
    num_episodes: int,
    max_steps: int = 1000,
    save_path: str = "dqn_checkpoint.pth",
    save_interval: int = 100
):
    """
    Train the DQN agent in the environment.
    """
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Get the current agent
            agent_id = env.agents[0]  # We're using a single agent
            
            # Get observation for the current agent
            state = env.observe(agent_id)
            
            # Select and perform action
            action = agent.select_action(state)
            env.step(action)
            
            # Get next state and reward
            next_state = env.observe(agent_id)
            reward = env.rewards[agent_id]
            done = env.terminations[agent_id] or env.truncations[agent_id]
            
            # Update reward statistics
            agent.update_reward_stats(reward)
            
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            
            # Train the agent
            loss = agent.train_step()
            
            if done:
                break
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Store episode reward
        agent.episode_rewards.append(episode_reward)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(save_path)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}, "
                  f"Best Reward: {best_reward:.2f}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            agent.save(save_path)
    
    return agent

def plot_training_progress(agent, save_path: str):
    """Plot and save the training progress."""
    plt.figure(figsize=(10, 5))
    
    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(agent.episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot moving average
    window_size = 10
    moving_avg = np.convolve(agent.episode_rewards, 
                            np.ones(window_size)/window_size, 
                            mode='valid')
    plt.plot(moving_avg, label=f'{window_size}-episode moving average')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_dqn_agent(env, agent, num_episodes: int = 10):
    """Evaluate the trained DQN agent."""
    total_rewards = []
    
    for episode in range(num_episodes):
        env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get the current agent
            agent_id = env.agents[0]  # We're using a single agent
            
            # Get observation for the current agent
            state = env.observe(agent_id)
            
            # Select action
            action = agent.select_action(state, training=False)
            env.step(action)
            
            # Get next state and reward
            next_state = env.observe(agent_id)
            reward = env.rewards[agent_id]
            done = env.terminations[agent_id] or env.truncations[agent_id]
            
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")
    
    mean_reward = np.mean(total_rewards)
    print(f"Mean evaluation reward over {num_episodes} episodes: {mean_reward}")
    return mean_reward

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
    
    # Reset the environment to initialize it
    env.reset()
    
    # Get state and action dimensions
    state_dim = env.observation_space(env.agents[0]).shape[0]
    action_dim = env.action_space(env.agents[0]).n
    
    # Create DQN agent
    dqn_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=[512, 512, 256],
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.997,
        buffer_size=200000,
        batch_size=128,
        target_update=5
    )
    
    # Set up paths
    checkpoint_dir = "checkpoints"
    plot_dir = "plots"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    dqn_checkpoint_path = os.path.join(checkpoint_dir, "dqn_checkpoint.pth")
    dqn_plot_path = os.path.join(plot_dir, "dqn_training_plot.png")
    
    # Train the DQN agent
    print("Starting DQN training...")
    dqn_agent = train_dqn(
        env=env,
        agent=dqn_agent,
        num_episodes=1000,
        max_steps=1000,
        save_path=dqn_checkpoint_path,
        save_interval=100
    )
    
    # Plot DQN training progress
    plot_training_progress(dqn_agent, dqn_plot_path)
    
    # Evaluate the trained DQN agent
    print("\nEvaluating trained DQN agent...")
    dqn_mean_reward = evaluate_dqn_agent(env, dqn_agent, num_episodes=10)
    
    print(f"\nDQN Training completed. Final evaluation reward: {dqn_mean_reward}")
    print(f"DQN Training plot saved to: {dqn_plot_path}")
    print(f"DQN Model checkpoint saved to: {dqn_checkpoint_path}")
    
    # Continue with PPO training if desired
    # ... [Keep existing PPO training code] ...
