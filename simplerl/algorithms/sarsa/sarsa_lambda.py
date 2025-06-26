from abc import ABC, abstractmethod
import numpy as np
import logging
from tqdm import tqdm
from .sarsa import SarsaBase

class SarsaLambdaBase(SarsaBase):
    def __init__(self, env, logger=None, gamma=0.99, epsilon=0.01, alpha=0.01, lamb=0.9):
        """
        Initialize Sarsa(λ) algorithm
        Args:
            env: environment
            logger: logger
            gamma: discount factor
            epsilon: exploration rate
            alpha: learning rate
            lamb: lambda parameter, used to control the decay of eligibility traces
        """
        super().__init__(env, logger, gamma, epsilon, alpha)
        self.lamb = lamb
        self.E = np.zeros_like(self.Q)  # initialize eligibility traces


class ForwardSarsaLambda(SarsaLambdaBase):
    """
    Implementation of Forward Sarsa(λ) algorithm
    
    Use the conventional TD update method, through eligibility traces to propagate the reward signal.
    """
    def __init__(self, env, logger=None, gamma=0.99, epsilon=0.01, alpha=0.01, lamb=0.9, total_time_steps=1000):
        super().__init__(env, logger, gamma, epsilon, alpha, lamb)
        self.total_time_steps = total_time_steps

    def learn(self):
        obs, _ = self.env.reset()
        # sample action a_t
        action = self.epsilon_greedy_policy(obs)
        
        with tqdm(total=self.total_time_steps, desc="Forward Sarsa(λ) Learning") as pbar:
            for _ in range(self.total_time_steps):
                # execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # sample next action a_t+1
                next_action = self.epsilon_greedy_policy(next_obs)
                
                # get state index
                state_index = self.env._get_q_state_index(obs)
                next_state_index = self.env._get_q_state_index(next_obs)
                
                # calculate TD error
                q_sa = self.Q[state_index + (action,)]
                q_sa_next = self.Q[next_state_index + (next_action,)]
                td_error = reward + self.gamma * q_sa_next - q_sa
                
                # update eligibility traces
                self.E[state_index + (action,)] += 1.0
                
                # update Q value and eligibility traces for all state-action pairs
                self.Q += self.alpha * td_error * self.E
                self.E *= self.gamma * self.lamb
                
                # update observation and action
                obs = next_obs
                action = next_action
                
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    action = self.epsilon_greedy_policy(obs)
                    # reset eligibility traces
                    self.E = np.zeros_like(self.Q)
                
                pbar.update(1)
                pbar.set_postfix(reward=reward)
    
    def evaluate(self, num_episodes=10):
        return super().evaluate(num_episodes)


class BackwardSarsaLambda(SarsaLambdaBase):
    """
    Implementation of Backward Sarsa(λ) algorithm
    
    Use the backward perspective, update the previous states after an episode ends.
    This method is usually used for offline learning or batch updates.
    """
    def __init__(self, env, logger=None, gamma=0.99, epsilon=0.01, alpha=0.01, lamb=0.9, total_time_steps=1000):
        super().__init__(env, logger, gamma, epsilon, alpha, lamb)
        self.total_time_steps = total_time_steps

    def learn(self):
        total_steps = 0
        
        with tqdm(total=self.total_time_steps, desc="Backward Sarsa(λ) Learning") as pbar:
            while total_steps < self.total_time_steps:
                # reset environment and eligibility traces
                obs, _ = self.env.reset()
                self.E = np.zeros_like(self.Q)
                
                # save trajectory
                trajectory = []
                
                # sample initial action
                action = self.epsilon_greedy_policy(obs)
                
                done = False
                while not done and total_steps < self.total_time_steps:
                    # execute action
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    # sample next action
                    next_action = self.epsilon_greedy_policy(next_obs)
                    
                    # save current transition
                    trajectory.append((obs, action, reward, next_obs, next_action))
                    
                    # update state and action
                    obs = next_obs
                    action = next_action
                    done = terminated or truncated
                    
                    total_steps += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                
                # backward update Q value
                self._backward_update(trajectory)
    
    def _backward_update(self, trajectory):
        """
        Update Q value through backward perspective
        """
        # calculate the return of each time step
        returns = []
        G = 0
        
        # backward calculate the return
        for obs, action, reward, next_obs, next_action in reversed(trajectory):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # update Q value
        for i, (obs, action, _, _, _) in enumerate(trajectory):
            state_index = self.env._get_q_state_index(obs)
            G = returns[i]
            
            # update eligibility traces
            self.E[state_index + (action,)] += 1.0
            
            # calculate TD error and update Q value
            td_error = G - self.Q[state_index + (action,)]
            self.Q += self.alpha * td_error * self.E
            
            # decay eligibility traces
            self.E *= self.gamma * self.lamb
    
    def evaluate(self, num_episodes=10):
        return super().evaluate(num_episodes)