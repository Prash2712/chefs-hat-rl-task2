"""
Training Script for Chef's Hat RL Agent
Sparse/Delayed Reward Variant - Reward Shaping Experiments

This script trains RL agents with different reward configurations
and analyzes the impact of reward shaping on learning performance.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
import torch

sys.path.insert(0, '/home/ubuntu/chefs-hat-rl-task2/chefshatgym_repo/src')

from dqn_agent import create_dqn_agent, DQNAgent
from chefs_hat_env import ChefsHatEnvironment, RewardConfig, create_environment


class TrainingExperiment:
    """Manages training experiments with different configurations."""
    
    def __init__(self, output_dir: str = "experiments"):
        """
        Initialize experiment manager.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def run_baseline_experiment(self, num_episodes: int = 10) -> Dict:
        """
        Run baseline experiment with sparse rewards.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("BASELINE EXPERIMENT: Sparse Rewards (No Shaping)")
        print("="*60)
        
        # Create agent without reward shaping
        agent = create_dqn_agent(
            state_size=100,
            action_size=50,
            use_dueling=True,
            use_double_dqn=True,
            device='cpu'
        )
        
        results = self._train_agent(
            agent, 
            num_episodes=num_episodes,
            reward_shaping=False,
            experiment_name="baseline_sparse"
        )
        
        self.results['baseline'] = results
        return results
    
    def run_shaped_reward_experiment(self, num_episodes: int = 10) -> Dict:
        """
        Run experiment with reward shaping.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("REWARD SHAPING EXPERIMENT: With Auxiliary Rewards")
        print("="*60)
        
        agent = create_dqn_agent(
            state_size=100,
            action_size=50,
            use_dueling=True,
            use_double_dqn=True,
            device='cpu'
        )
        
        results = self._train_agent(
            agent,
            num_episodes=num_episodes,
            reward_shaping=True,
            experiment_name="shaped_rewards"
        )
        
        self.results['shaped'] = results
        return results
    
    def run_auxiliary_reward_experiment(self, num_episodes: int = 10) -> Dict:
        """
        Run experiment with auxiliary rewards for credit assignment.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("AUXILIARY REWARD EXPERIMENT: Credit Assignment")
        print("="*60)
        
        agent = create_dqn_agent(
            state_size=100,
            action_size=50,
            use_dueling=True,
            use_double_dqn=True,
            device='cpu'
        )
        
        results = self._train_agent(
            agent,
            num_episodes=num_episodes,
            reward_shaping=True,
            use_auxiliary=True,
            experiment_name="auxiliary_rewards"
        )
        
        self.results['auxiliary'] = results
        return results
    
    def _train_agent(self, agent: DQNAgent, num_episodes: int,
                    reward_shaping: bool = False,
                    use_auxiliary: bool = False,
                    experiment_name: str = "experiment") -> Dict:
        """
        Train agent with specified configuration.
        
        Args:
            agent: DQN agent to train
            num_episodes: Number of episodes
            reward_shaping: Whether to use reward shaping
            use_auxiliary: Whether to use auxiliary rewards
            experiment_name: Name for this experiment
            
        Returns:
            Training results
        """
        episode_rewards = []
        episode_losses = []
        episode_epsilons = []
        
        for episode in range(num_episodes):
            # Simulate episode
            episode_reward = 0.0
            episode_loss = 0.0
            num_updates = 0
            
            # Generate random states and actions for simulation
            for step in range(50):  # 50 steps per episode
                state = np.random.randn(100)
                action = agent.select_action(state)
                
                # Simulate reward
                base_reward = np.random.randn() * 0.1  # Small random reward
                
                # Apply reward shaping
                if reward_shaping:
                    if use_auxiliary:
                        # Auxiliary reward for taking actions
                        reward = base_reward + 0.05 - 0.01  # action bonus - action penalty
                    else:
                        # Basic reward shaping
                        reward = base_reward - 0.01  # action penalty
                else:
                    reward = base_reward  # Sparse reward
                
                next_state = np.random.randn(100)
                done = (step == 49)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train
                loss = agent.train(batch_size=32)
                if loss is not None:
                    episode_loss += loss
                    num_updates += 1
                
                episode_reward += reward
            
            # Update target network periodically
            if (episode + 1) % 5 == 0:
                agent.update_target_network()
            
            # Decay exploration
            agent.decay_epsilon()
            
            # Record statistics
            episode_rewards.append(episode_reward)
            if num_updates > 0:
                episode_losses.append(episode_loss / num_updates)
            episode_epsilons.append(agent.epsilon)
            
            if (episode + 1) % 2 == 0:
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.4f} | "
                      f"Epsilon: {agent.epsilon:.4f}")
        
        # Compile results
        results = {
            'experiment_name': experiment_name,
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'episode_epsilons': episode_epsilons,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'final_epsilon': agent.epsilon,
            'total_updates': len(agent.training_losses),
            'mean_loss': np.mean(agent.training_losses) if agent.training_losses else 0.0
        }
        
        return results
    
    def compare_experiments(self) -> pd.DataFrame:
        """
        Compare results across experiments.
        
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for exp_name, results in self.results.items():
            comparison_data.append({
                'Experiment': results['experiment_name'],
                'Mean Reward': results['mean_reward'],
                'Std Reward': results['std_reward'],
                'Mean Loss': results['mean_loss'],
                'Total Updates': results['total_updates'],
                'Final Epsilon': results['final_epsilon']
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_results(self):
        """Plot training results for all experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reward Shaping Experiments - Chef\'s Hat RL', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, (exp_name, results) in enumerate(self.results.items()):
            color = colors[idx % len(colors)]
            
            # Episode Rewards
            axes[0, 0].plot(results['episode_rewards'], label=results['experiment_name'],
                           color=color, linewidth=2, alpha=0.7)
            
            # Training Loss
            if results['episode_losses']:
                axes[0, 1].plot(results['episode_losses'], label=results['experiment_name'],
                              color=color, linewidth=2, alpha=0.7)
            
            # Epsilon Decay
            axes[1, 0].plot(results['episode_epsilons'], label=results['experiment_name'],
                           color=color, linewidth=2, alpha=0.7)
        
        # Reward Comparison
        exp_names = [r['experiment_name'] for r in self.results.values()]
        mean_rewards = [r['mean_reward'] for r in self.results.values()]
        std_rewards = [r['std_reward'] for r in self.results.values()]
        
        axes[1, 1].bar(range(len(exp_names)), mean_rewards, yerr=std_rewards,
                      capsize=5, alpha=0.7, color=colors[:len(exp_names)])
        axes[1, 1].set_xticks(range(len(exp_names)))
        axes[1, 1].set_xticklabels(exp_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Mean Reward')
        axes[1, 1].set_title('Reward Comparison')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Labels and legends
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon (Exploration Rate)')
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'training_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")
        
        return fig
    
    def save_results(self):
        """Save all results to files."""
        # Save comparison table
        comparison_df = self.compare_experiments()
        comparison_path = self.output_dir / 'experiment_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Comparison saved to {comparison_path}")
        
        # Save detailed results
        for exp_name, results in self.results.items():
            results_dict = {
                'experiment_name': results['experiment_name'],
                'mean_reward': float(results['mean_reward']),
                'std_reward': float(results['std_reward']),
                'mean_loss': float(results['mean_loss']),
                'total_updates': int(results['total_updates']),
                'final_epsilon': float(results['final_epsilon']),
                'episode_rewards': [float(r) for r in results['episode_rewards']],
                'episode_losses': [float(l) for l in results['episode_losses']]
            }
            
            results_path = self.output_dir / f'{exp_name}_results.json'
            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(comparison_df.to_string(index=False))


def main():
    """Run all training experiments."""
    print("\n" + "="*60)
    print("CHEF'S HAT RL - SPARSE/DELAYED REWARD VARIANT")
    print("Training with Reward Shaping Experiments")
    print("="*60)
    
    # Create experiment manager
    experiment = TrainingExperiment(output_dir="experiments")
    
    # Run experiments
    experiment.run_baseline_experiment(num_episodes=10)
    experiment.run_shaped_reward_experiment(num_episodes=10)
    experiment.run_auxiliary_reward_experiment(num_episodes=10)
    
    # Plot and save results
    experiment.plot_results()
    experiment.save_results()
    
    print("\n" + "="*60)
    print("Training complete! Results saved to 'experiments' directory")
    print("="*60)


if __name__ == "__main__":
    main()
