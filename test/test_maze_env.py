import pytest
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simplerl.envs.maze_env import MazeEnv

class TestMazeEnv:
    
    def test_initialization(self):
        """测试环境初始化"""
        # 测试默认初始化
        env = MazeEnv()
        assert env.size == 10
        assert env.maze.shape == (10, 10)
        assert 0 <= env.start_pos[0] < 10 and 0 <= env.start_pos[1] < 10
        assert 0 <= env.goal_pos[0] < 10 and 0 <= env.goal_pos[1] < 10
        assert env.maze[env.start_pos] == 0
        assert env.maze[env.goal_pos] == 0
        
        # 测试自定义初始化
        custom_maze = np.zeros((10, 10))
        custom_maze[0, :] = 1  # 上墙
        custom_maze[-1, :] = 1  # 下墙
        custom_maze[:, 0] = 1  # 左墙
        custom_maze[:, -1] = 1  # 右墙
        start_pos = (1, 1)
        goal_pos = (8, 8)
        
        env = MazeEnv(maze=custom_maze, start_pos=start_pos, goal_pos=goal_pos)
        assert env.start_pos == start_pos
        assert env.goal_pos == goal_pos
        np.testing.assert_array_equal(env.maze, custom_maze)
    
    def test_get_q_table(self):
        """测试获取Q-table"""
        env = MazeEnv()
        q_table = env._get_q_table()
        assert q_table.shape == (10, 10, 4)
        assert q_table.dtype == np.float32
    
    def test_get_q_index(self):
        """测试获取Q-table index"""
        env = MazeEnv()
        obs = np.array([1, 2], dtype=np.int32)
        action = env.action_space.sample()
        q_index = env._get_q_index(obs, action)
        assert q_index == (1, 2, action)
        # 测试能否正常索引
        q_table = env._get_q_table()
        q_table[q_index] = 1.0
        assert q_table[q_index] == 1.0
        
    def test_reset(self):
        """测试环境重置"""
        env = MazeEnv()
        obs, info = env.reset()
        
        assert len(obs) == 2
        assert obs[0] == env.start_pos[0]
        assert obs[1] == env.start_pos[1]
        assert env.agent_pos == env.start_pos
        assert len(env.trajectory) == 1
        assert env.trajectory[0] == env.start_pos
    
    def test_step_valid_move(self):
        """测试有效移动"""
        # 创建一个简单的测试迷宫
        maze = np.zeros((10, 10))
        maze[0, :] = 1  # 上墙
        maze[-1, :] = 1  # 下墙
        maze[:, 0] = 1  # 左墙
        maze[:, -1] = 1  # 右墙
        
        env = MazeEnv(maze=maze, start_pos=(1, 1), goal_pos=(8, 8))
        env.reset()
        
        # 向右移动 (action=1)
        obs, reward, terminated, truncated, info = env.step(1)
        assert env.agent_pos == (1, 2)
        assert reward == 0.0
        assert not terminated
        assert not truncated
        
        # 向下移动 (action=2)
        obs, reward, terminated, truncated, info = env.step(2)
        assert env.agent_pos == (2, 2)
        assert reward == 0.0
        assert not terminated
        assert not truncated
        
        # 检查轨迹记录
        assert len(env.trajectory) == 3
        assert env.trajectory == [(1, 1), (1, 2), (2, 2)]
    
    def test_step_wall_collision(self):
        """测试撞墙情况"""
        maze = np.zeros((10, 10))
        maze[0, :] = 1  # 上墙
        maze[-1, :] = 1  # 下墙
        maze[:, 0] = 1  # 左墙
        maze[:, -1] = 1  # 右墙
        maze[2, 2] = 1  # 额外的墙
        
        env = MazeEnv(maze=maze, start_pos=(1, 2), goal_pos=(8, 8))
        env.reset()
        
        # 向下移动撞墙 (action=2)
        obs, reward, terminated, truncated, info = env.step(2)
        assert env.agent_pos == (1, 2)  # 位置不变
        assert reward == -1.0  # 撞墙惩罚
        assert not terminated
        assert not truncated
    
    def test_reach_goal(self):
        """测试到达目标"""
        env = MazeEnv(start_pos=(1, 1), goal_pos=(1, 2))
        env.reset()
        
        # 向右移动到达目标 (action=1)
        obs, reward, terminated, truncated, info = env.step(1)
        assert env.agent_pos == (1, 2)
        assert reward == 10.0  # 到达目标奖励
        assert terminated
        assert not truncated
    
    def test_max_steps(self):
        """测试最大步数限制"""
        env = MazeEnv(max_steps=3)
        env.reset()
        
        # 执行3步动作
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(1)
            if terminated or truncated:
                break
        
        assert truncated  # 应该因为达到最大步数而截断
    
    def test_random_maze(self):
        """测试随机迷宫生成"""
        env = MazeEnv(random_maze=True)
        # 验证迷宫中有通道和墙壁
        assert np.any(env.maze == 0)
        assert np.any(env.maze == 1)