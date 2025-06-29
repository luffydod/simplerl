from abc import ABC, abstractmethod
import numpy as np
import logging
from tqdm import tqdm

class SarsaBase(ABC):
    def __init__(self, env, logger=None, gamma=0.99, epsilon=0.01, alpha=0.01):
        """
        Initialize the Sarsa algorithm
        Args:
            env: The environment
            gamma: The discount factor
            epsilon: The exploration rate
            alpha: The learning rate
        """
        self.env = env
        self.logger = logger if logger else logging.getLogger(__name__)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        
        assert hasattr(self.env, "_get_q_table"), "The environment must have a _get_q_table method"
        assert hasattr(self.env, "_get_q_state_index"), "The environment must have a _get_q_state_index method"
        
        self.Q = self.env._get_q_table()
    
    def epsilon_greedy_policy(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy policy
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_index = self.env._get_q_state_index(state)
            return np.argmax(self.Q[state_index])
        
    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass

class Sarsa(SarsaBase):
    def __init__(self, env, logger=None, gamma=0.99, epsilon=0.01, alpha=0.01, total_time_steps=1000):
        super().__init__(env, logger, gamma, epsilon, alpha)
        self.total_time_steps = total_time_steps
        

    def learn(self):
        obs, _ = self.env.reset()
        # sample a_t
        action = self.epsilon_greedy_policy(obs)
        
        with tqdm(total=self.total_time_steps, desc="Sarsa Learning") as pbar:
            for _ in range(self.total_time_steps):
                # take action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # sample a_t+1
                next_action = self.epsilon_greedy_policy(next_obs)
                
                # update Q by (obs, action, reward, next_obs, next_action)
                state_index = self.env._get_q_state_index(obs)
                next_state_index = self.env._get_q_state_index(next_obs)
                q_sa = self.Q[state_index + (action,)]
                q_sa_next = self.Q[next_state_index + (next_action,)]
                
                self.Q[state_index + (action,)] += \
                    self.alpha * (reward + self.gamma * q_sa_next - q_sa)
                    
                # update obs, action
                obs = next_obs
                action = next_action
                
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    action = self.epsilon_greedy_policy(obs)
                
                pbar.update(1)
                pbar.set_postfix(reward=reward)
            
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
            
