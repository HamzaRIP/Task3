import gymnasium
from gymnasium import spaces
import numpy as np
import time
from pathlib import Path
import argparse

from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

from utils import create_environment
from submission_single import CustomWrapper, CustomPredictFunction

def visualize_game(num_agents=1, max_cycles=1000, max_zombies=10, fps=10):
    """
    Visualize the Knights-Archers-Zombies game with the trained agent.
    
    Args:
        num_agents: Number of archer agents (1 or 2)
        max_cycles: Maximum number of cycles for the game
        max_zombies: Maximum number of zombies in the game
        fps: Frames per second for visualization
    """
    # Create environment with render mode
    env = create_environment(
        num_agents=num_agents,
        visual_observation=False,
        max_zombies=max_zombies,
        max_cycles=max_cycles,
        render_mode="human"
    )
    
    # Apply custom wrapper
    env = CustomWrapper(env)
    
    # Create prediction function
    predict_fn = CustomPredictFunction(env)
    
    # Run the game
    print(f"Starting game visualization with {num_agents} archer(s)")
    print("Game controls: Close the window to end the visualization")
    
    total_rewards = 0
    env.reset()
    
    for agent in env.agent_iter():
        # Get observation, reward, termination, truncation, info
        observation, reward, termination, truncation, info = env.last()
        total_rewards += reward
        
        # Determine action
        if termination or truncation:
            action = None
        else:
            action = predict_fn(observation, agent)
            print(f"Agent: {agent}, Action: {action}, Reward: {reward}, Total Rewards: {total_rewards}")
        
        # Take action in environment
        env.step(action)
        
        # Control visualization speed
        time.sleep(1/fps)
        
        # Check if all agents are done
        if all(env.terminations.values()) or all(env.truncations.values()):
            break
    
    print(f"Game ended with total rewards: {total_rewards}")
    env.close()
    return total_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Knights-Archers-Zombies game')
    parser.add_argument('--num_agents', type=int, default=1, choices=[1, 2], 
                        help='Number of archer agents (1 or 2)')
    parser.add_argument('--max_cycles', type=int, default=1000, 
                        help='Maximum number of cycles for the game')
    parser.add_argument('--max_zombies', type=int, default=10, 
                        help='Maximum number of zombies in the game')
    parser.add_argument('--fps', type=int, default=10, 
                        help='Frames per second for visualization')
    
    args = parser.parse_args()
    
    visualize_game(
        num_agents=args.num_agents,
        max_cycles=args.max_cycles,
        max_zombies=args.max_zombies,
        fps=args.fps
    )
