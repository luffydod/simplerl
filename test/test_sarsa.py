import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pytest
import logging 

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simplerl.algorithms.sarsa.sarsa import Sarsa
from simplerl.envs.maze_env import MazeEnv

@pytest.fixture(scope='module')
def get_maze_env():
    env = MazeEnv()
    yield env

@pytest.fixture(scope='module')
def get_sarsa(get_maze_env):
    sarsa = Sarsa(
        env=get_maze_env,
        gamma=0.99,  # 折扣因子
        epsilon=0.1,  # 探索率
        alpha=0.1,    # 学习率
        total_time_steps=10000  # 学习步数
    )
    yield sarsa
    
def test_sarsa_learn(get_sarsa):
    """测试SARSA算法学习"""
    get_sarsa.learn()
    
def test_sarsa_evaluate(get_maze_env, get_sarsa):
    """测试SARSA算法评估"""
    for episode in range(10):
        obs, _ = get_maze_env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # 使用贪婪策略（epsilon=0）
            action = np.argmax(get_sarsa.Q[get_maze_env._get_q_state_index(obs)])
            obs, reward, terminated, truncated, _ = get_maze_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
            
        logging.info(f"测试回合 {episode+1}: 奖励 = {episode_reward}, 步数 = {steps}")
        
def test_sarsa_q_value(get_maze_env, get_sarsa):
    """测试SARSA算法Q值表可视化"""
    if hasattr(get_maze_env, 'maze_size'):
        maze_height, maze_width = get_maze_env.maze_size
        
        # 为每个位置创建最佳动作的可视化
        best_actions = np.zeros((maze_height, maze_width))
        value_function = np.zeros((maze_height, maze_width))
        
        # 对每个状态计算最佳动作和值函数
        for i in range(maze_height):
            for j in range(maze_width):
                state = np.array([i, j])
                state_idx = get_maze_env._get_q_state_index(state)
                if state_idx < len(get_sarsa.Q):  # 确保状态索引有效
                    best_actions[i, j] = np.argmax(get_sarsa.Q[state_idx])
                    value_function[i, j] = np.max(get_sarsa.Q[state_idx])
        
        # 绘制值函数热图
        plt.figure(figsize=(10, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(value_function, cmap='hot')
        plt.colorbar(label='值函数')
        plt.title('SARSA值函数')
        
        plt.subplot(1, 2, 2)
        plt.imshow(best_actions, cmap='viridis')
        plt.colorbar(label='最佳动作')
        plt.title('SARSA策略')
        
        plt.tight_layout()
        plt.savefig('tmp/sarsa_visualization.png')
        logging.info("已保存可视化结果到 'tmp/sarsa_visualization.png'")
        plt.close()

if __name__ == "__main__":
    pytest.main()
