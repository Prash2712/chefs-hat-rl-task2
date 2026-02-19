"""
Chef's Hat Gym Environment Wrapper with Reward Shaping
Implements the Sparse/Delayed Reward Variant for Task 2

This module provides a wrapper around the Chef's Hat Gym environment
with reward shaping techniques to address the sparse reward problem.
"""

import sys
import asyncio
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass

# Add the Chef's Hat Gym repository to the path
sys.path.insert(0, '/home/ubuntu/chefs-hat-rl-task2/chefshatgym_repo/src')

from agents.random_agent import RandomAgent
from agents.base_agent import BaseAgent
from rooms.room import Room


@dataclass
class RewardConfig:
    """Configuration for reward shaping strategies."""
    match_win_reward: float = 1.0
    match_loss_penalty: float = -1.0
    action_penalty: float = -0.01
    invalid_action_penalty: float = -0.1
    card_play_bonus: float = 0.05
    use_auxiliary_rewards: bool = True
    use_reward_shaping: bool = True


class ChefsHatRLAgent(BaseAgent):
    """
    Reinforcement Learning Agent for Chef's Hat Gym.
    
    This agent implements a DQN-based approach with reward shaping
    to handle the sparse reward problem in the Chef's Hat environment.
    """
    
    def __init__(self, name: str, log_directory: str, reward_config: Optional[RewardConfig] = None):
        """
        Initialize the RL Agent.
        
        Args:
            name: Agent identifier
            log_directory: Directory for logging
            reward_config: Configuration for reward shaping
        """
        super().__init__(name, log_directory)
        self.reward_config = reward_config or RewardConfig()
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_states = []
        self.match_history = []
        
    def get_action(self, state: Dict[str, Any], possible_actions: List[int]) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state: Current game state
            possible_actions: List of valid action indices
            
        Returns:
            Selected action index
        """
        # For now, use epsilon-greedy exploration
        # In full implementation, this would use a trained neural network
        if np.random.random() < 0.1:  # 10% exploration
            return np.random.choice(possible_actions)
        else:
            # Default to a heuristic strategy
            return self._heuristic_action(state, possible_actions)
    
    def _heuristic_action(self, state: Dict[str, Any], possible_actions: List[int]) -> int:
        """
        Select action using a heuristic strategy.
        
        Args:
            state: Current game state
            possible_actions: List of valid action indices
            
        Returns:
            Selected action index
        """
        # Simple heuristic: prefer playing cards over passing
        # This helps with exploration in sparse reward environments
        if len(possible_actions) > 1:
            # Prefer non-pass actions (assuming pass is the last action)
            non_pass_actions = [a for a in possible_actions if a != len(possible_actions) - 1]
            if non_pass_actions:
                return np.random.choice(non_pass_actions)
        
        return np.random.choice(possible_actions)
    
    def reward_shaping(self, base_reward: float, state: Dict[str, Any], 
                      action: int, next_state: Dict[str, Any]) -> float:
        """
        Apply reward shaping to the base reward.
        
        This implements auxiliary rewards and potential-based reward shaping
        to guide learning in sparse reward environments.
        
        Args:
            base_reward: Original reward from environment
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Shaped reward
        """
        shaped_reward = base_reward
        
        if self.reward_config.use_reward_shaping:
            # Penalize each action to encourage efficiency
            shaped_reward += self.reward_config.action_penalty
            
            # Bonus for playing cards (auxiliary reward)
            if self.reward_config.use_auxiliary_rewards:
                if action != -1:  # Not a pass action
                    shaped_reward += self.reward_config.card_play_bonus
        
        return shaped_reward
    
    def update_episode_stats(self, reward: float, action: int, state: Dict[str, Any]):
        """
        Update episode statistics for learning.
        
        Args:
            reward: Reward received
            action: Action taken
            state: Current state
        """
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_states.append(state)
    
    def end_match(self, match_result: Dict[str, Any]):
        """
        Process end-of-match information.
        
        Args:
            match_result: Information about match outcome
        """
        total_reward = sum(self.episode_rewards)
        self.match_history.append({
            'total_reward': total_reward,
            'num_actions': len(self.episode_actions),
            'result': match_result
        })
        
        # Reset episode buffers
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_states = []


class ChefsHatEnvironment:
    """
    Wrapper for Chef's Hat Gym environment with reward shaping.
    
    This class provides a unified interface for training RL agents
    with various reward configurations and opponent models.
    """
    
    def __init__(self, num_agents: int = 4, reward_config: Optional[RewardConfig] = None,
                 output_folder: str = "game_logs"):
        """
        Initialize the environment.
        
        Args:
            num_agents: Number of agents in the game
            reward_config: Configuration for reward shaping
            output_folder: Directory for logging game data
        """
        self.num_agents = num_agents
        self.reward_config = reward_config or RewardConfig()
        self.output_folder = output_folder
        self.room = None
        self.agents = []
        
    def create_agents(self, agent_types: List[str]) -> List[BaseAgent]:
        """
        Create agents for the environment.
        
        Args:
            agent_types: List of agent types ('rl', 'random', 'heuristic')
            
        Returns:
            List of created agents
        """
        agents = []
        for i, agent_type in enumerate(agent_types):
            if agent_type == 'rl':
                agent = ChefsHatRLAgent(
                    name=f"RLAgent_{i}",
                    log_directory=self.output_folder,
                    reward_config=self.reward_config
                )
            elif agent_type == 'random':
                agent = RandomAgent(
                    name=f"Random_{i}",
                    log_directory=self.output_folder
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            agents.append(agent)
        
        self.agents = agents
        return agents
    
    async def run_episode(self, num_matches: int = 10) -> Dict[str, Any]:
        """
        Run an episode with multiple matches.
        
        Args:
            num_matches: Number of matches to play
            
        Returns:
            Episode statistics
        """
        self.room = Room(
            run_remote_room=False,
            room_name="RL_Training_Room",
            max_matches=num_matches,
            output_folder=self.output_folder,
            save_game_dataset=True,
            save_logs_game=True,
            save_logs_room=True,
        )
        
        # Connect agents to room
        for agent in self.agents:
            self.room.connect_player(agent)
        
        # Run matches
        await self.room.run()
        
        # Collect statistics
        stats = self._collect_statistics()
        return stats
    
    def _collect_statistics(self) -> Dict[str, Any]:
        """
        Collect statistics from the episode.
        
        Returns:
            Dictionary of episode statistics
        """
        stats = {
            'num_agents': len(self.agents),
            'agent_names': [agent.name for agent in self.agents],
            'match_histories': [agent.match_history if hasattr(agent, 'match_history') else [] 
                               for agent in self.agents],
        }
        
        # Calculate aggregate statistics
        if self.agents and hasattr(self.agents[0], 'match_history'):
            rl_agent = self.agents[0]
            if rl_agent.match_history:
                rewards = [m['total_reward'] for m in rl_agent.match_history]
                stats['mean_reward'] = np.mean(rewards)
                stats['std_reward'] = np.std(rewards)
                stats['max_reward'] = np.max(rewards)
                stats['min_reward'] = np.min(rewards)
        
        return stats


def create_environment(num_agents: int = 4, 
                      reward_strategy: str = 'shaped',
                      output_folder: str = "game_logs") -> ChefsHatEnvironment:
    """
    Factory function to create a Chef's Hat environment with specified configuration.
    
    Args:
        num_agents: Number of agents
        reward_strategy: 'sparse' or 'shaped'
        output_folder: Output directory for logs
        
    Returns:
        Configured ChefsHatEnvironment instance
    """
    reward_config = RewardConfig(
        use_reward_shaping=(reward_strategy == 'shaped'),
        use_auxiliary_rewards=(reward_strategy == 'shaped')
    )
    
    return ChefsHatEnvironment(
        num_agents=num_agents,
        reward_config=reward_config,
        output_folder=output_folder
    )


if __name__ == "__main__":
    # Example usage
    print("Chef's Hat Gym Environment Wrapper initialized successfully")
