from .sarsa import SarsaBase
from tqdm import tqdm

class NStepsSarsa(SarsaBase):
    def __init__(self, env, logger=None, gamma=0.99, epsilon=0.01, alpha=0.01, total_time_steps=1000, n_steps=3):
        super().__init__(env, logger, gamma, epsilon, alpha)
        self.total_time_steps = total_time_steps
        self.n_steps = n_steps
        
    def learn(self):
        obs, _ = self.env.reset()
        # sample a_t
        action = self.epsilon_greedy_policy(obs)
        
        # Initialize buffer for trajectory
        states = []
        actions = []
        rewards = []
        
        time_step = 0
        episode_start_t = 0
        
        with tqdm(total=self.total_time_steps, desc="N-Steps Sarsa Learning") as pbar:
            while time_step < self.total_time_steps:
                # Store current state and action
                states.append(obs)
                actions.append(action)
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                rewards.append(reward)
                
                # Select next action
                next_action = self.epsilon_greedy_policy(next_obs)
                
                # If enough steps have been accumulated or episode ends, perform n-step update
                update_time = time_step - episode_start_t + 1
                if update_time >= self.n_steps or terminated or truncated:
                    # Determine how many steps to update
                    n = min(update_time, self.n_steps)
                    
                    # Calculate n-step return
                    G = 0
                    for i in range(n):
                        if i < len(rewards):  # Ensure not to exceed the index of rewards
                            G += (self.gamma ** i) * rewards[i]
                    
                    # Add bootstrapping value, unless it is a terminal state
                    if not (terminated or truncated) and n == self.n_steps:
                        state_index = self.env._get_q_state_index(next_obs)
                        G += (self.gamma ** n) * self.Q[state_index + (next_action,)]
                    
                    # Update Q value of the first state-action pair
                    first_state = states[0]
                    first_action = actions[0]
                    state_index = self.env._get_q_state_index(first_state)
                    
                    self.Q[state_index + (first_action,)] += \
                        self.alpha * (G - self.Q[state_index + (first_action,)])
                    
                    # Remove the first state-action pair that has been updated
                    states.pop(0)
                    actions.pop(0)
                    rewards.pop(0)
                
                # Update observation and action
                obs = next_obs
                action = next_action
                
                time_step += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                
                # If environment ends, reset environment and buffer
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    action = self.epsilon_greedy_policy(obs)
                    states = []
                    actions = []
                    rewards = []
                    episode_start_t = time_step

    def evaluate(self, num_episodes=10):
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done:
                action = self.epsilon_greedy_policy(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                steps += 1
                obs = next_obs.copy()
            
            self.logger.info(f"[episode-{episode+1}]-[reward={episode_reward:.2f}]-[ep_length={steps}]")