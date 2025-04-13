import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from IPython.display import clear_output
from matplotlib.ticker import MaxNLocator
import json

class TrainingMonitor:
    """
    A class for monitoring and visualizing the training progress of RL agents.
    """
    def __init__(self, save_dir="./training_plots", log_interval=1, live_plot=True):
        """
        Initialize the training monitor.
        
        Args:
            save_dir: Directory to save plots and logs
            log_interval: How often to update plots (in iterations)
            live_plot: Whether to display live plots during training
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.iterations = []
        self.mean_rewards = {}
        self.best_rewards = {}
        self.log_interval = log_interval
        self.live_plot = live_plot
        self.start_time = time.time()
        
        # Create figure for plotting
        plt.ion() if live_plot else plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Dictionary to store all metrics for later analysis
        self.history = {
            "iterations": [],
            "mean_rewards": {},
            "best_rewards": {},
            "training_time": []
        }
        
    def update(self, iteration, metrics):
        """
        Update training metrics and plot progress.
        
        Args:
            iteration: Current training iteration
            metrics: Dictionary containing metrics like mean rewards per agent
        """
        self.iterations.append(iteration)
        elapsed_time = time.time() - self.start_time
        self.history["training_time"].append(elapsed_time)
        
        # Extract and store metrics
        if "agent_returns" in metrics:
            for agent_id, reward in metrics["agent_returns"].items():
                if agent_id not in self.mean_rewards:
                    self.mean_rewards[agent_id] = []
                    self.best_rewards[agent_id] = -float('inf')
                    self.history["mean_rewards"][agent_id] = []
                    self.history["best_rewards"][agent_id] = []
                
                self.mean_rewards[agent_id].append(reward)
                self.best_rewards[agent_id] = max(self.best_rewards[agent_id], reward)
                
                self.history["mean_rewards"][agent_id].append(reward)
                self.history["best_rewards"][agent_id].append(self.best_rewards[agent_id])
        
        # Update history
        self.history["iterations"].append(iteration)
        
        # Log progress
        if iteration % self.log_interval == 0:
            self._log_progress(iteration, metrics, elapsed_time)
            
            # Create and save visualization
            if self.live_plot:
                self._plot_progress()
            
            # Save metrics to disk
            self._save_metrics()
    
    def _log_progress(self, iteration, metrics, elapsed_time):
        """Log current training progress to console."""
        log_str = f"Iteration {iteration} | Time: {elapsed_time:.2f}s"
        
        if "agent_returns" in metrics:
            for agent_id, reward in metrics["agent_returns"].items():
                log_str += f" | {agent_id}: {reward:.2f} (best: {self.best_rewards[agent_id]:.2f})"
        
        print(log_str)
    
    def _plot_progress(self):
        """Create and display/save visualization of training progress."""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Plot mean rewards for each agent
            for agent_id, rewards in self.mean_rewards.items():
                self.ax.plot(self.iterations, rewards, label=f"{agent_id} Mean Reward")
                
                # Add moving average
                if len(rewards) > 5:
                    window_size = min(len(rewards), 10)
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    ma_x = self.iterations[window_size-1:]
                    self.ax.plot(ma_x, moving_avg, linestyle='--', 
                                alpha=0.7, label=f"{agent_id} Moving Avg (10)")
            
            # Add best rewards as horizontal lines
            for agent_id, best_reward in self.best_rewards.items():
                self.ax.axhline(y=best_reward, color='r', linestyle=':', alpha=0.5,
                              label=f"Best {agent_id}: {best_reward:.2f}")
            
            # Set labels and title
            self.ax.set_xlabel('Iterations')
            self.ax.set_ylabel('Reward')
            self.ax.set_title('Agent Training Progress')
            self.ax.legend(loc='lower right')
            self.ax.grid(True, alpha=0.3)
            
            # Make x-axis have integer ticks
            self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Save figure
            plt.savefig(self.save_dir / "training_progress.png")
            
            # Display the plot if in interactive environment
            if self.live_plot:
                try:
                    clear_output(wait=True)
                    plt.draw()
                    plt.pause(0.001)
                except Exception as e:
                    # If clear_output fails (not in notebook), just update the plot
                    plt.draw()
                    plt.pause(0.001)
        except Exception as e:
            print(f"Error plotting progress: {e}")
    
    def _save_metrics(self):
        """Save training metrics to disk."""
        with open(self.save_dir / "training_history.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_safe_history = {
                k: v if not isinstance(v, dict) else 
                   {agent: [float(x) for x in rewards] for agent, rewards in v.items()} 
                for k, v in self.history.items()
            }
            json_safe_history["iterations"] = [int(i) for i in json_safe_history["iterations"]]
            json_safe_history["training_time"] = [float(t) for t in json_safe_history["training_time"]]
            
            json.dump(json_safe_history, f, indent=2)
    
    def finalize(self):
        """Finalize training visualization and save final results."""
        # Create final plots with all data
        self._plot_progress()
        
        # Create and save summary plot
        self._create_summary_plot()
        
        if self.live_plot:
            plt.ioff()
        
        print(f"Training completed. Final metrics saved to {self.save_dir}")
    
    def _create_summary_plot(self):
        """Create and save a summary plot with training statistics."""
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Reward progression
        plt.subplot(2, 1, 1)
        for agent_id, rewards in self.mean_rewards.items():
            plt.plot(self.iterations, rewards, label=f"{agent_id} Reward")
        plt.xlabel('Iterations')
        plt.ylabel('Reward')
        plt.title('Reward Progression')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Learning curves (smoothed)
        plt.subplot(2, 1, 2)
        for agent_id, rewards in self.mean_rewards.items():
            if len(rewards) > 10:
                window_size = min(len(rewards), 20)
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ma_x = self.iterations[window_size-1:]
                plt.plot(ma_x, moving_avg, label=f"{agent_id} (Smoothed)")
        plt.xlabel('Iterations')
        plt.ylabel('Reward (Moving Average)')
        plt.title('Smoothed Learning Curves')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_summary.png")
        plt.close()


def setup_training_monitor(save_dir="./training_plots", log_interval=1, live_plot=True):
    """
    Create and return a training monitor instance.
    
    Args:
        save_dir: Directory to save plots and logs
        log_interval: How often to update plots (in iterations)
        live_plot: Whether to display live plots during training
        
    Returns:
        TrainingMonitor instance
    """
    return TrainingMonitor(save_dir=save_dir, log_interval=log_interval, live_plot=live_plot)


if __name__ == "__main__":
    # Example of how to use the training monitor with sample data
    monitor = setup_training_monitor()
    
    # Simulate training for demonstration
    for i in range(100):
        metrics = {
            "agent_returns": {
                "archer_0": np.random.normal(i*0.1, 1.0)  # Simulated increasing reward
            }
        }
        monitor.update(i, metrics)
        time.sleep(0.1)  # Simulate training time
    
    monitor.finalize()
