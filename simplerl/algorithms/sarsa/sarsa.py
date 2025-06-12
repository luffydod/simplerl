from abc import ABC, abstractmethod
from typing import Choice
import numpy as np

class SarsaBase(ABC):
    def __init__(self, env, gamma=0.99, epsilon=0.01, alpha=0.01, num_episodes=1000, max_steps=1000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.init_q_table(self.env.get_state_space(), self.env.get_action_space())
    
    def init_q_table(self, 
                     state_space: Choice[int, tuple], 
                     action_space: int):
        """
        Initialize the Q-table
        """
        if isinstance(state_space, int):
            self.Q = np.zeros((state_space, action_space))
        elif isinstance(state_space, tuple):
            self.Q = np.zeros((state_space[0], state_space[1], action_space))
    
    @abstractmethod
    def evaluate(self):
        """
        Evaluate the policy
        """
        pass
    
    @abstractmethod
    def improve(self):
        """
        Improve the policy
        """
        pass

class Sarsa(SarsaBase):
    def __init__(self, env, gamma=0.99, epsilon=0.01, alpha=0.01, num_episodes=1000, max_steps=1000):
        super().__init__(env, gamma, epsilon, alpha, num_episodes, max_steps)
        
    def evaluate(self):
        pass

    def improve(self):
        pass


