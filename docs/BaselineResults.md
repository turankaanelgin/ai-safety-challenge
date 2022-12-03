Our work is based on the AI Safety Tanksworld environment which simulates a team-based battle among tanks which requires cooperation and competition among different agents. The environment serves as a test-bed for multi-agent reinforcement learning research, especially in the safety domain. One essential component of the environment is performance and safety trade-offs between competing objectives which we address in this paper with a formal baseline.

In AI Safety Tanksworld, agents have two main objectives which are inflicting damage on the opposing team and avoiding damage among allies. These two objectives compete with each other and satisfying them at the same time requires a complex strategy. In this sense, our baseline supports the research in risk-sensitive and multi-objective reinforcement learning in an environment which is more complex than other safety testbeds.

# Experimental Setup
For our baseline, we use centralized training - decentralized execution framework. That means five agents share a common policy but they draw their actions independent of each other. Based on the results in \cite{Yu2021TheSE}, we use state-of-the-art Proximal Policy Optimization (PPO) algorithm \cite{Schulman2017ProximalPO} and we find it to be effective in this benchmark as well as other multi-agent settings.
## Hyperparameters
We search through various hyperparameters to generate the Pareto frontier for the Tanksworld environment. In particular we consider two categories of training parameters:

- Utility function parameters
    - Damage taken weight ($\omega$): [0.0, 0.5, 1.0, 2.0]
- Learning parameters
    - Actor learning rate ($\eta_\theta$): [1e-3, 3e-4, 5e-5]
    - Critic learning rate ($\eta_\omega$): [1e-3, 5e-4, 1e-4]
    - Batch size ($B$): [32, 64, 128]

Reward parameters reflect system design goals to some degree. They impact the nature of the interaction between the agents and the environment and hence affect the nature of the learning problem (e.g., higher penalties generally result in more cautious behaviours). In general, as we vary the reward parameters, we expect to see agent performance that represents different tradeoffs between the accrual of positive and negative rewards.\\
On the other hand, the selection of learning parameters presents a natural tradeoff given the multi-objective nature of the problem. For example, larger batch sizes may be necessary to mitigate reward sparsity but increase the likelihood of aggregating positive and negative contributions within a single gradient calculation (in particular, given the high value selected for the discount factor).

We fix the rest of the hyperparameters with the following values based on \cite{Schulman2017ProximalPO}: 
| Number of epochs ($n\_epochs$)                   | 4     |
| Discount factor ($\gamma$)                       | 0.99  |
| GAE coefficient ($\lambda$)                      | 0.95  |
| Clip ratio ($\epsilon$)                          | 0.2   |
| KL divergence target ($d_{targ}$)                | 0.015 |
| Initial value of standard deviation ($\sigma_0$) | 0.6   |
