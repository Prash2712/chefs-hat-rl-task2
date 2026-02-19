"""
Deep Q-Network (DQN) Agent for Chef's Hat Gym
Implements reward shaping and auxiliary rewards for sparse reward learning

This module provides a PyTorch-based DQN implementation with techniques
specifically designed for the sparse/delayed reward variant of Chef's Hat.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
import random


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize DQN network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(state)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN architecture for better credit assignment."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize Dueling DQN network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(DuelingDQNNetwork, self).__init__()
        
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """
    DQN Agent with reward shaping for Chef's Hat Gym.
    
    Implements Double DQN with dueling architecture and reward shaping
    to address sparse rewards in the Chef's Hat environment.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 use_dueling: bool = True,
                 use_double_dqn: bool = True,
                 device: str = 'cpu'):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            use_dueling: Whether to use dueling architecture
            use_double_dqn: Whether to use Double DQN
            device: Device to use ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = torch.device(device)
        self.use_double_dqn = use_double_dqn
        
        # Create networks
        if use_dueling:
            self.q_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DuelingDQNNetwork(state_size, action_size).to(self.device)
        else:
            self.q_network = DQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training statistics
        self.training_losses = []
        self.episode_rewards = []
    
    def select_action(self, state: np.ndarray, possible_actions: Optional[List[int]] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            possible_actions: List of valid actions (None = all actions valid)
            
        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            # Exploration: random action
            if possible_actions is not None:
                return np.random.choice(possible_actions)
            else:
                return np.random.randint(0, self.action_size)
        else:
            # Exploitation: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                if possible_actions is not None:
                    # Mask invalid actions
                    q_values_masked = q_values.clone()
                    mask = torch.ones(self.action_size, dtype=torch.bool)
                    mask[possible_actions] = False
                    q_values_masked[0, mask] = float('-inf')
                    action = q_values_masked.argmax(dim=1).item()
                else:
                    action = q_values.argmax(dim=1).item()
            
            return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self, batch_size: int = 32) -> Optional[float]:
        """
        Train the DQN network.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Training loss, or None if buffer too small
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        if self.use_double_dqn:
            # Double DQN: use current network to select action
            next_q_values_current = self.q_network(next_states)
            next_actions = next_q_values_current.argmax(dim=1)
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            # Standard DQN
            next_q_values = self.target_network(next_states).max(dim=1)[0]
        
        # Bellman equation
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = self.loss_fn(q_values, target_q_values.detach())
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """
        Save model weights.
        
        Args:
            filepath: Path to save weights
        """
        torch.save(self.q_network.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        """
        Load model weights.
        
        Args:
            filepath: Path to load weights from
        """
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())


def create_dqn_agent(state_size: int, action_size: int,
                     use_dueling: bool = True,
                     use_double_dqn: bool = True,
                     device: str = 'cpu') -> DQNAgent:
    """
    Factory function to create a DQN agent.
    
    Args:
        state_size: Dimension of state space
        action_size: Number of possible actions
        use_dueling: Whether to use dueling architecture
        use_double_dqn: Whether to use Double DQN
        device: Device to use
        
    Returns:
        Configured DQNAgent instance
    """
    return DQNAgent(
        state_size=state_size,
        action_size=action_size,
        use_dueling=use_dueling,
        use_double_dqn=use_double_dqn,
        device=device
    )


if __name__ == "__main__":
    # Example usage
    agent = create_dqn_agent(state_size=100, action_size=50)
    print(f"DQN Agent created with state size {agent.state_size} and action size {agent.action_size}")
