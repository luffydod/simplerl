import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pytest

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simplerl.algorithms.sarsa.sarsa import Sarsa
from simplerl.envs.maze_env import MazeEnv

def test_sarsa(capsys):
    """测试SARSA算法"""
    # 创建迷宫环境
    env = MazeEnv()
    
    # 创建SARSA算法实例
    sarsa = Sarsa(
        env=env,
        gamma=0.99,  # 折扣因子
        epsilon=0.1,  # 探索率
        alpha=0.1,    # 学习率
        total_time_steps=10000  # 学习步数
    )
    
    # 训练SARSA算法
    print("开始训练SARSA算法...")
    sarsa.learn()
    print("SARSA训练完成！")
    
    # 评估训练后的策略
    rewards = evaluate_policy(env, sarsa, n_episodes=10)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"评估结果 - 平均奖励: {mean_reward:.2f}, 标准差: {std_reward:.2f}")
    
    # 可视化Q值表并保存图片
    visualize_q_values(env, sarsa)
    
    # 捕获输出以便pytest显示
    captured = capsys.readouterr()
    print(captured.out)
    
    # return sarsa, rewards

def evaluate_policy(env, agent, n_episodes=10):
    """评估训练后的策略"""
    rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # 使用贪婪策略（epsilon=0）
            action = np.argmax(agent.Q[env._get_q_state_index(obs)])
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
            
        print(f"测试回合 {episode+1}: 奖励 = {episode_reward}, 步数 = {steps}")
        rewards.append(episode_reward)
    
    return rewards

def visualize_q_values(env, agent):
    """可视化Q值表"""
    if hasattr(env, 'maze_size'):
        maze_height, maze_width = env.maze_size
        
        # 为每个位置创建最佳动作的可视化
        best_actions = np.zeros((maze_height, maze_width))
        value_function = np.zeros((maze_height, maze_width))
        
        # 对每个状态计算最佳动作和值函数
        for i in range(maze_height):
            for j in range(maze_width):
                state = np.array([i, j])
                state_idx = env._get_q_state_index(state)
                if state_idx < len(agent.Q):  # 确保状态索引有效
                    best_actions[i, j] = np.argmax(agent.Q[state_idx])
                    value_function[i, j] = np.max(agent.Q[state_idx])
        
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
        print("已保存可视化结果到 'tmp/sarsa_visualization.png'")
        plt.close()

if __name__ == "__main__":
    pytest.main()
