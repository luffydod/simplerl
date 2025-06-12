# SimpleRL

Simplerl is an educational reinforcement learning library implementing fundamental algorithms like Monte Carlo (MC), Temporal Difference (TD), SARSA, and Q-Learning. It provides clean, beginner-friendly Python implementations alongside classic environments such as GridWorld, CliffWalking, and FrozenLake. Designed for RL learners and researchers, Simplerl offers a practical starting point for understanding core reinforcement learning concepts through hands-on experimentation with essential algorithms and standardized testing scenarios.

## Sarsa

Q-function Bellman Expectation Equation:

$$Q\_\pi(s,a)=\mathcal{R}\_s^a+\gamma\sum\_{s^{\prime}\in\mathcal{S}}\mathcal{P}\_{ss^{\prime}}^a\sum\_{a^{\prime}\in\mathcal{A}}\pi(a^{\prime}|s^{\prime})Q\_\pi(s^{\prime},a^{\prime})$$

Using Samples to Represent Reward $R$ and Model $P$ Function. In online learning, the agent observes a trajectory $(s, a, r, s', a')$ at each timestep.

TD Learning Q-function:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma Q\left(s', a'\right) - Q(s, a) \right)$$

Sarsa Control at each time step: 

![sarsa_update](docs/images/sarsa_update.png)

- Policy Evaluation: Temporal Difference (TD)Learning: $Q \approx Q_{\pi}$
- Policy Improvement: ε-Greedy Strategy
