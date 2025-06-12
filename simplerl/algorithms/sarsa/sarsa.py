from abc import ABC, abstractmethod
import numpy as np

class SarsaBase(ABC):
    def __init__(self, env, gamma=0.99, epsilon=0.01, alpha=0.01):
        """
        Initialize the Sarsa algorithm
        Args:
            env: The environment
            gamma: The discount factor
            epsilon: The exploration rate
            alpha: The learning rate
        """
        self.env = env
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

class Sarsa(SarsaBase):
    def __init__(self, env, gamma=0.99, epsilon=0.01, alpha=0.01, total_time_steps=1000):
        super().__init__(env, gamma, epsilon, alpha)
        self.total_time_steps = total_time_steps
        

    def learn(self):
        obs, _ = self.env.reset()
        # sample a_t
        action = self.epsilon_greedy_policy(obs)
        
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
            


