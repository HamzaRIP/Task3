import gymnasium
from gymnasium import spaces
from pathlib import Path
import numpy as np
import torch
from typing import Callable

from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import MultiRLModule

class ArcherWrapper(BaseWrapper):
    """
    Custom wrapper for the KAZ environment that flattens the observation space
    and adds feature engineering for better agent performance.
    """
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))
    
    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        if obs is None:
            return None
        
        # Flatten the observation for easier processing by neural networks
        flat_obs = obs.flatten()
        return flat_obs

class CustomPredictFunction(Callable):
    """
    Prediction function for the trained archer agent.
    Loads a trained RLLib algorithm from a checkpoint and extracts the policies.
    """
    def __init__(self, env):
        # Load the trained model from checkpoint
        checkpoint_path = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        try:
            self.modules = MultiRLModule.from_checkpoint(checkpoint_path)
            self.use_trained_model = True
        except (FileNotFoundError, ValueError):
            # Fallback to a simple heuristic strategy if model not found
            print("Warning: Trained model not found. Using fallback strategy.")
            self.use_trained_model = False
    
    def __call__(self, observation, agent, *args, **kwargs):
        if self.use_trained_model:
            # Use the trained model for prediction
            rl_module = self.modules[agent]
            fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
            fwd_outputs = rl_module.forward_inference(fwd_ins)
            action_dist_class = rl_module.get_inference_action_dist_cls()
            action_dist = action_dist_class.from_logits(
                fwd_outputs["action_dist_inputs"]
            )
            action = action_dist.sample()[0].numpy()
            return action
        else:
            # Fallback strategy: prioritize shooting when zombies are detected
            # This is a simple heuristic that looks for zombies in the observation
            # and shoots when they're detected
            
            # In the vectorized observation, zombies appear after the current agent,
            # archers, knights, swords, and arrows
            n_rows = observation.shape[0] // 5
            
            # Check if there are any zombies (non-zero rows in the zombie section)
            for i in range(1, n_rows):
                row = observation[i*5:(i+1)*5]
                if np.any(row != 0):
                    # If zombie is detected, shoot (action 5)
                    return 5
            
            # If no zombies detected, rotate to scan the environment (actions 0 or 1)
            return np.random.choice([0, 1])
