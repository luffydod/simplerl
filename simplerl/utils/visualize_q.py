import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import gymnasium as gym
from simplerl.envs.maze_env import MazeEnv

def vis_best_policy_maze_env(q_table: np.ndarray, env: MazeEnv):
    """
    Visualize the best policy of a maze environment
    
    Parameters:
        q_table: The Q-table with shape (env.size, env.size, 4)
        env: The maze environment
    """
    # q_table shape: (env.size, env.size, 4)
    # 0: up, 1: right, 2: down, 3: left
    best_policy = np.argmax(q_table, axis=2)
    
    # visualize the best policy with arrows
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a grid for rendering
    grid = env.maze.copy()
    
    # Mark the end positions
    goal_row, goal_col = env.goal_pos
    
    # Create a color map
    cmap = colors.ListedColormap(['white', 'black', 'red'])
    
    # 0=channel(white), 1=wall(black), 2=goal(red)
    grid[goal_row, goal_col] = 2
    
    # Draw the grid
    ax.imshow(grid, cmap=cmap)
    
    # Add arrows to visualize the best policy
    for i in range(env.size):
        for j in range(env.size):
            # Skip walls and goal position
            if env.maze[i, j] == 1 or (i, j) == env.goal_pos:
                continue
            
            action = best_policy[i, j]
            
            # Define arrow directions
            if action == 0:  # up
                dx, dy = 0, -0.3
            elif action == 1:  # right
                dx, dy = 0.3, 0
            elif action == 2:  # down
                dx, dy = 0, 0.3
            else:  # left
                dx, dy = -0.3, 0
            
            # Add arrow
            ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    
    # Show grid lines
    ax.grid(True, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, env.size, 1))
    ax.set_yticks(np.arange(-0.5, env.size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Set title and labels
    ax.set_title('Best Policy Visualization')
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=15, label='通道'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=15, label='墙壁'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='终点'),
        plt.Line2D([0], [0], marker='>', color='blue', markersize=15, label='最佳动作')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    plt.tight_layout()
    plt.show()
