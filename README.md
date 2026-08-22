# Deep Q-Learning for Sparse-Reward Multi-Agent Play

Reinforcement-learning implementation for the **Chef's Hat Gym** environment, focused on learning under sparse and delayed rewards.

This project explores value-based deep RL using **Dueling DQN**, **Double DQN**, experience replay and configurable reward shaping in a competitive card-game setting.

## Technical focus

- Deep Q-Networks (DQN)
- Dueling value/advantage architecture
- Double DQN target estimation
- Experience replay
- Epsilon-greedy exploration
- Sparse and delayed rewards
- Reward shaping and auxiliary signals
- Multi-agent environment interaction
- Training/evaluation utilities

## Why this problem is interesting

Sparse terminal rewards make credit assignment difficult because the agent receives little information about which earlier actions contributed to the final outcome. Multi-agent interaction adds further non-stationarity because the effective environment changes with opponent behaviour.

The repository experiments with denser learning signals while retaining a value-based RL architecture.

## Repository structure

```text
.
├── dqn_agent.py
├── chefs_hat_env.py
├── train_agent.py
├── evaluate_agent.py
├── requirements.txt
├── Technical Documentation - Chef's Hat RL Agent.md
├── .gitignore
├── LICENSE
└── README.md
```

## Architecture

### Dueling DQN

The network decomposes the action-value function into a state-value stream and an advantage stream:

```text
Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
```

This can improve value estimation when many actions have similar effects in a state.

### Double DQN

Action selection and target evaluation are separated to reduce the overestimation bias associated with standard Q-learning targets.

### Experience replay

Past transitions are stored in a replay buffer and sampled during optimisation, reducing temporal correlation and improving data reuse.

## Reward configurations

The environment wrapper supports different reward strategies, including:

- sparse terminal rewards
- intermediate action-based shaping
- auxiliary reward signals

This enables controlled comparisons between learning from outcome-only feedback and learning from denser signals.

## Running the project

Install the dependencies and required Chef's Hat Gym environment, then run:

```bash
python train_agent.py
```

Evaluation utilities are available through:

```bash
python evaluate_agent.py
```

Generated experiment files are written by the training pipeline when it is executed.

## Results policy

The previous README included a numerical results table even though the corresponding experiment artifacts are not currently committed in this repository. Those figures have therefore been removed from the portfolio-facing documentation.

Any performance claims should be based on freshly generated or committed evaluation outputs rather than unsupported summary numbers.

## Current limitations

- Sparse reward remains a difficult learning signal
- Opponent behaviour introduces non-stationarity
- The action space varies with game state
- Hidden information makes the environment partially observable
- Long episodes make temporal credit assignment challenging

## Technical stack

`Python` · `PyTorch` · `DQN` · `Double DQN` · `Dueling Networks` · `NumPy` · `pandas` · `Matplotlib`

## Academic provenance

Originally developed for Coventry University reinforcement-learning coursework using the sparse/delayed reward variant. Academic metadata is retained here for provenance, while the repository is presented primarily around the engineering and RL methods demonstrated.

## References

- Mnih et al. (2015), *Human-level control through deep reinforcement learning*
- Wang et al. (2016), *Dueling Network Architectures for Deep Reinforcement Learning*
- van Hasselt et al. (2016), *Deep Reinforcement Learning with Double Q-learning*

## Author

**Prasanth Balisetty**  
Data Science & Machine Learning

[GitHub](https://github.com/Prash2712)
