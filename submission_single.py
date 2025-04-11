import gymnasium
from gymnasium import spaces
from pathlib import Path
from typing import Optional

from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import MultiRLModule
import torch

class CustomWrapper(BaseWrapper):
    """
    Custom wrapper for the KAZ environment that flattens the observation space.
    
    Wrappers are useful to inject state pre-processing or features that do not need 
    to be learned by the agent. Pay attention to submit the same (or consistent) 
    wrapper you used during training.
    """
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))
    
    def observe(self, agent: AgentID) -> Optional[ObsType]:
        obs = super().observe(agent)
        if obs is None:
            return None
        flat_obs = obs.flatten()
        return flat_obs

class CustomPredictFunction:
    """
    Function to load the trained model and predict actions.
    
    This implementation loads a trained RLLib model from a checkpoint and uses it
    to predict actions for the archer agent.
    """
    def __init__(self, env):
        # Load the trained model from checkpoint
        package_directory = Path(__file__).resolve().parent
        best_checkpoint = (
            package_directory / "results" / "learner_group" / "learner" / "rl_module"
        ).resolve()
        
        # Try to load the trained model, fall back to a simple strategy if not found
        try:
            self.modules = MultiRLModule.from_checkpoint(best_checkpoint)
            self.use_trained_model = True
            print(f"Loaded trained model from {best_checkpoint}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load trained model: {e}")
            print("Using fallback strategy instead")
            self.use_trained_model = False
    
    def __call__(self, observation, agent, *args, **kwargs):
        if self.use_trained_model:
            # Use the trained model for prediction
            if agent not in self.modules:
                raise ValueError(f"No policy found for agent {agent}")
                
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
            # In the vectorized observation, zombies appear after the current agent,
            # archers, knights, swords, and arrows
            n_rows = observation.shape[0] // 5
            
            # Check if there are any zombies (non-zero rows in the zombie section)
            for i in range(1, n_rows):
                row = observation[i*5:(i+1)*5]
                if row[0] > 0:  # If distance to entity is > 0, entity exists
                    # If zombie is detected, shoot (action 5)
                    return 5
            
            # If no zombies detected, rotate to scan the environment (actions 0 or 1)
            return 0  # Rotate counter-clockwise
