# Task 3: Reinforcement Learning Agent for Knights-Archers-Zombies

This document provides an overview of the implementation of a reinforcement learning agent for the Knights-Archers-Zombies (KAZ) environment using PettingZoo and Ray RLlib.

## Implementation Overview

The implementation consists of the following components:

1. **Environment Setup**: Using PettingZoo's KAZ environment with a single archer agent.
2. **Feature Representation**: A custom wrapper that flattens the vectorized state representation.
3. **RL Algorithm**: Proximal Policy Optimization (PPO) implemented using Ray RLlib.
4. **Training Process**: Training the agent to maximize rewards by shooting zombies.
5. **Evaluation**: Comparing the trained agent against simple baseline strategies.

## Model Architecture

The model uses a neural network with the following architecture:
- Input: Flattened vectorized state representation
- Hidden layers: Two fully connected layers with 128 neurons each and ReLU activation
- Output: Action probabilities for the 6 possible actions

## Training Strategy

The training strategy includes:
- PPO algorithm with appropriate hyperparameters
- Larger batch size (1024) for more stable training
- Higher learning rate (3e-4) for faster convergence
- Entropy coefficient (0.01) to encourage exploration
- Multiple SGD iterations per batch for better policy updates

## Baseline Comparison

The agent is compared against the following baseline strategies:
- Random action selection
- Always shooting
- Always moving right

## Usage Instructions

1. **Training the agent**:
   ```
   python agent_implementation.py
   ```

2. **Tournament submission**:
   The `submission_single_agent.py` file contains the `CustomPredictFunction` class that can be used for tournament evaluation.

## Implementation Details

The implementation follows the template provided in the repository and adapts it for the single-archer KAZ environment. The key components include:

1. **ArcherWrapper**: A custom wrapper that flattens the observation space for easier processing by neural networks.

2. **CustomPredictFunction**: A prediction function that loads a trained model from a checkpoint and uses it for inference. It includes a fallback strategy in case the trained model is not available.

3. **Training Configuration**: The PPO algorithm is configured with appropriate hyperparameters for training the archer agent.

4. **Evaluation**: The trained agent is evaluated against simple baseline strategies to assess its performance.

The implementation meets the requirements specified in the assignment instructions, including training an agent for a single-archer KAZ environment and comparing it against simple baselines.

---

## SSH Documentation

### Assuming .ssh/config is:
```
Host KUL
    User <r-studentno>
    HostName st.cs.kuleuven.be
    PasswordAuthentication no
    IdentitiesOnly yes
    IdentityFile ~/.ssh/KUL/id_rsa (or key directory)

Host *.student.cs.kuleuven.be
    User <r-studentno>
    PasswordAuthentication no
    IdentitiesOnly yes
    IdentityFile ~/.ssh/KUL/id_rsa (or key directory)
    ProxyJump KUL
```
Generate key at https://www.cs.kuleuven.be/restricted/ssh/
See `ssh.md` in original repo for more info: https://github.com/ML-KULeuven/ml-project-2024-2025/blob/main/ssh.md

### Accessing webpage of available servers
`ssh -L 10480:mysql.student.cs.kuleuven.be:443 KUL`
open `https://localhost:10480`

### Accessing server:
`cd <pcname>.student.cs.kuleuven.be`

### Once accessed:
`cd /cw/lvs/NoCsBack/vakken/H0T25A/ml-project`
`source /cw/lvs/NoCsBack/vakken/H0T25A/ml-project/venv/bin/activate`

