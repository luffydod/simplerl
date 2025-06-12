from abc import ABC, abstractmethod
from typing import Choice
import numpy as np

class SarsaBase(ABC):
    def __init__(self, env, gamma=0.99, epsilon=0.01, alpha=0.01, num_episodes=1000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_episodes = num_episodes
        
        assert hasattr(self.env, "_get_q_table"), "The environment must have a _get_q_table method"
        assert hasattr(self.env, "_get_q_index"), "The environment must have a _get_q_index method"
        assert hasattr(self.env, "_get_q_state_index"), "The environment must have a _get_q_state_index method"
        
        self.Q = self.env._get_q_table()
    
    def epsilon_greedy_policy(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy policy
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[self.env._get_q_state_index(state)])
        
    @abstractmethod
    def learn(self):
        pass

class Sarsa(SarsaBase):
    def __init__(self, env, gamma=0.99, epsilon=0.01, alpha=0.01, num_episodes=1000):
        super().__init__(env, gamma, epsilon, alpha, num_episodes)
        
    def learn(self):
        pass


