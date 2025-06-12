import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root directory to the path to allow importing simplerl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplerl.envs import MazeEnv

# 测试夹具(fixture)，用于创建环境实例
@pytest.fixture
def env():
    return MazeEnv()

def test_initialization(env):
    """测试环境初始化"""
    # 测试迷宫大小
    assert env.size == 10
    assert env.maze.shape == (10, 10)
    
    # 测试起点和终点位置
    assert env.start_pos is not None
    assert env.goal_pos is not None
    assert env.start_pos != env.goal_pos
    
    # 测试智能体初始位置
    assert env.agent_pos == env.start_pos

def test_reset(env):
    """测试环境重置"""
    # 移动智能体
    action = 1  # 向右移动
    env.step(action)
    
    # 重置环境
    start_pos = env.reset()
    
    # 测试重置后的状态
    assert env.agent_pos == env.start_pos
    assert start_pos == env.start_pos
    assert not env.done
    assert len(env.trajectory) == 1
    assert env.trajectory[0] == env.start_pos

def test_step_movement(env):
    """测试智能体移动"""
    # 获取初始位置
    init_row, init_col = env.agent_pos
    
    # 测试向上移动
    if init_row > 0 and env.maze[init_row - 1, init_col] == 0:
        next_state, reward, done, _ = env.step(0)  # 向上
        assert next_state == (init_row - 1, init_col)
        env.reset()
    
    # 测试向右移动
    if init_col < env.size - 1 and env.maze[init_row, init_col + 1] == 0:
        next_state, reward, done, _ = env.step(1)  # 向右
        assert next_state == (init_row, init_col + 1)
        env.reset()
    
    # 测试向下移动
    if init_row < env.size - 1 and env.maze[init_row + 1, init_col] == 0:
        next_state, reward, done, _ = env.step(2)  # 向下
        assert next_state == (init_row + 1, init_col)
        env.reset()
    
    # 测试向左移动
    if init_col > 0 and env.maze[init_row, init_col - 1] == 0:
        next_state, reward, done, _ = env.step(3)  # 向左
        assert next_state == (init_row, init_col - 1)
        env.reset()

def test_wall_collision():
    """测试墙壁碰撞"""
    # 创建一个简单的迷宫，中间有一堵墙
    maze = np.zeros((10, 10), dtype=int)
    maze[5, 5] = 1  # 在中间放置一堵墙
    
    env = MazeEnv(maze=maze, start_pos=(5, 4), goal_pos=(5, 6))
    
    # 尝试向右移动（撞墙）
    next_state, reward, done, _ = env.step(1)
    
    # 智能体应该保持原位
    assert next_state == (5, 4)
    assert reward == -1.0
    assert not done

def test_reward_and_done():
    """测试奖励和结束条件"""
    # 创建一个简单的迷宫，起点和终点相邻
    maze = np.zeros((10, 10), dtype=int)
    
    env = MazeEnv(maze=maze, start_pos=(5, 4), goal_pos=(5, 5))
    
    # 向右移动到达目标
    next_state, reward, done, _ = env.step(1)
    
    # 验证奖励和结束状态
    assert next_state == (5, 5)
    assert reward == 10.0
    assert done

def test_random_maze():
    """测试随机迷宫生成"""
    env = MazeEnv(random_maze=True)
    
    # 测试迷宫大小
    assert env.maze.shape == (10, 10)
    
    # 确保起点和终点在可通行的瓦片上
    assert env.maze[env.start_pos] == 0
    assert env.maze[env.goal_pos] == 0

def test_trajectory(env):
    """测试轨迹记录"""
    # 记录初始轨迹
    init_trajectory = env.get_trajectory()
    assert len(init_trajectory) == 1
    assert init_trajectory[0] == env.start_pos
    
    # 走几步
    env.step(1)  # 向右
    env.step(2)  # 向下
    env.step(3)  # 向左
    
    # 检查轨迹
    trajectory = env.get_trajectory()
    assert len(trajectory) == 4
    assert trajectory[0] == env.start_pos

def test_render(env):
    """测试渲染功能"""
    # 走几步
    env.step(1)  # 向右
    env.step(2)  # 向下
    
    # 测试人类模式渲染（不显示图像，仅测试调用）
    result = env.render(mode='human', show_trajectory=True)
    assert result is None
    plt.close()  # 关闭绘图窗口
    
    # 测试rgb_array模式渲染
    img = env.render(mode='rgb_array', show_trajectory=True)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3  # 应该是一个3D数组(H, W, C)

def test_custom_maze():
    """测试自定义迷宫"""
    # 创建自定义迷宫
    custom_maze = np.zeros((10, 10), dtype=int)
    custom_maze[0, :] = 1  # 上边界
    custom_maze[9, :] = 1  # 下边界
    custom_maze[:, 0] = 1  # 左边界
    custom_maze[:, 9] = 1  # 右边界
    
    # 添加一些内部墙壁
    custom_maze[3, 3:7] = 1
    custom_maze[7, 3:7] = 1
    
    env = MazeEnv(maze=custom_maze, start_pos=(1, 1), goal_pos=(8, 8))
    
    # 验证迷宫是否正确加载
    np.testing.assert_array_equal(env.maze, custom_maze)
    assert env.start_pos == (1, 1)
    assert env.goal_pos == (8, 8)

def run_demo():
    """运行演示"""
    print("运行迷宫环境演示...")
    
    # 创建环境
    env = MazeEnv()
    
    # 重置环境
    state = env.reset()
    print(f"初始状态: {state}")
    
    # 随机行走100步或直到达到目标
    total_reward = 0
    for i in range(100):
        # 随机选择一个动作
        action = np.random.randint(0, 4)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 累积奖励
        total_reward += reward
        
        print(f"步骤 {i+1}: 动作={action}, 状态={next_state}, 奖励={reward}, 结束={done}")
        
        # 如果到达目标则停止
        if done:
            print(f"到达目标! 总步数: {i+1}, 总奖励: {total_reward}")
            break
    
    # 渲染最终状态
    print("渲染最终状态...")
    env.render()
    
    return env

if __name__ == "__main__":
    # 运行演示
    env = run_demo()
