from abc import ABC, abstractmethod

class Sarsa(ABC):
    def __init__(self, env, gamma=0.99, epsilon=0.01, alpha=0.01, num_episodes=1000, max_steps=1000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
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
