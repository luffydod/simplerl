import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
from typing import Tuple, List, Dict, Any, Optional

class MazeEnv:
    """
    Simple Maze Environment
    The agent moves in a 10x10 maze, 
    reaching the goal earns a reward of +10,
    hit the wall will be punished with a reward of -1,
    and the reward is 0 otherwise.
    """
    
    def __init__(self, maze: Optional[np.ndarray] = None, 
                 start_pos: Optional[Tuple[int, int]] = None,
                 goal_pos: Optional[Tuple[int, int]] = None,
                 random_maze: bool = False):
        """
        Initialize the 10x10 maze environment
        
        Parameters:
            maze: Optional, predefined maze layout, 0 represents channels, 1 represents walls
            start_pos: Optional, starting position of the agent (row, col)
            goal_pos: Optional, goal position (row, col)
            random_maze: Whether to generate a random maze
        """
        self.size = 10  # 10x10 maze
        
        # If no maze is provided, create a default maze or a random maze
        if maze is None:
            if random_maze:
                self.maze = self._generate_random_maze()
            else:
                # Default maze: outer walls, some random walls in the middle
                self.maze = np.zeros((self.size, self.size), dtype=int)
                # Set outer walls
                self.maze[0, :] = 1
                self.maze[-1, :] = 1
                self.maze[:, 0] = 1
                self.maze[:, -1] = 1
        else:
            assert maze.shape == (self.size, self.size), f"The maze size must be {self.size}x{self.size}"
            self.maze = maze
            
        # Set start and end positions
        if start_pos is None:
            # Default start position is the first non-wall position in the top-left corner
            for i in range(self.size):
                for j in range(self.size):
                    if self.maze[i, j] == 0:
                        self.start_pos = (i, j)
                        break
                if hasattr(self, 'start_pos'):
                    break
        else:
            assert 0 <= start_pos[0] < self.size and 0 <= start_pos[1] < self.size, "The start position must be within the maze"
            assert self.maze[start_pos] == 0, "The start position cannot be a wall"
            self.start_pos = start_pos
            
        if goal_pos is None:
            # Default end position is the first non-wall position in the bottom-right corner
            for i in range(self.size-1, -1, -1):
                for j in range(self.size-1, -1, -1):
                    if self.maze[i, j] == 0:
                        self.goal_pos = (i, j)
                        break
                if hasattr(self, 'goal_pos'):
                    break
        else:
            assert 0 <= goal_pos[0] < self.size and 0 <= goal_pos[1] < self.size, "The end position must be within the maze"
            assert self.maze[goal_pos] == 0, "The end position cannot be a wall"
            self.goal_pos = goal_pos
        
        # Ensure the start and end positions are different
        assert self.start_pos != self.goal_pos, "The start and end positions cannot be the same"
        
        # Action space: up(0), right(1), down(2), left(3)
        self.action_space = 4
        
        # State space: (row, col)
        self.agent_pos = self.start_pos
        self.done = False
        self.trajectory = [self.start_pos]  # record the trajectory
        
    def _generate_random_maze(self) -> np.ndarray:
        """Generate a random maze"""
        maze = np.ones((self.size, self.size), dtype=int)
        
        # Use depth-first search to generate the maze
        def carve_passages(x: int, y: int, maze: np.ndarray):
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and maze[nx, ny] == 1:
                    maze[nx, ny] = 0
                    maze[x + dx//2, y + dy//2] = 0
                    carve_passages(nx, ny, maze)
        
        # Start generating from a random point
        start_x, start_y = 1, 1
        maze[start_x, start_y] = 0
        carve_passages(start_x, start_y, maze)
        
        # Ensure the maze has a solution
        # Add some random channels
        for _ in range(self.size):
            x = random.randint(1, self.size-2)
            y = random.randint(1, self.size-2)
            maze[x, y] = 0
            
        return maze
        
    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment
        
        Returns:
            The initial position of the agent
        """
        self.agent_pos = self.start_pos
        self.done = False
        self.trajectory = [self.start_pos]
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Perform one step action
        
        Parameters:
            action: The action, 0=up, 1=right, 2=down, 3=left
            
        Returns:
            next_state: The next state (agent's new position)
            reward: The reward
            done: Whether the episode is done
            info: Additional information
        """
        assert 0 <= action < self.action_space, f"The action must be between 0 and {self.action_space-1}"
        
        # Calculate the new position
        row, col = self.agent_pos
        if action == 0:  # up
            new_pos = (max(0, row - 1), col)
        elif action == 1:  # right
            new_pos = (row, min(self.size - 1, col + 1))
        elif action == 2:  # down
            new_pos = (min(self.size - 1, row + 1), col)
        else:  # left
            new_pos = (row, max(0, col - 1))
        
        reward = 0.0
        # Check if the agent hits a wall
        if self.maze[new_pos] == 1:
            # The agent does not move when hitting a wall
            new_pos = self.agent_pos
            # hit the wall will be punished with a reward of -1
            reward -= 1.0
            
        # Update the position
        self.agent_pos = new_pos
        self.trajectory.append(new_pos)
        
        # Check if the agent has reached the goal
        if self.agent_pos == self.goal_pos:
            # reaching the goal earns a reward of +10
            reward += 10.0
            self.done = True
            
        return self.agent_pos, reward, self.done, {}
    
    def render(self, mode: str = 'human', show_trajectory: bool = True) -> Optional[np.ndarray]:
        """
        Render the current environment state
        
        Parameters:
            mode: The rendering mode, 'human' displays the image, 'rgb_array' returns the RGB array
            show_trajectory: Whether to show the trajectory
            
        Returns:
            If mode='rgb_array', return the RGB array, otherwise return None
        """
        # Create a grid for rendering
        grid = self.maze.copy()
        
        # Mark the start and end positions
        start_row, start_col = self.start_pos
        goal_row, goal_col = self.goal_pos
        
        # Create a color map
        cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
        
        # 0=channel(white), 1=wall (black), 2=start (green), 3=goal (red), 4=agent's current position (blue)
        grid[start_row, start_col] = 2
        grid[goal_row, goal_col] = 3
        
        # Mark the agent's current position
        agent_row, agent_col = self.agent_pos
        if (agent_row, agent_col) != self.goal_pos:  # If the agent is not at the goal
            grid[agent_row, agent_col] = 4
        
        # Create a figure
        plt.figure(figsize=(7, 7))
        plt.imshow(grid, cmap=cmap)
        
        # Show the grid lines
        plt.grid(True, color='black', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, self.size, 1), [])
        plt.yticks(np.arange(-0.5, self.size, 1), [])
        
        # Show the trajectory
        if show_trajectory and len(self.trajectory) > 1:
            traj_y, traj_x = zip(*self.trajectory)
            plt.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.5)
        
        plt.title('10x10 Maze Environment')
        
        if mode == 'human':
            plt.show()
            return None
        elif mode == 'rgb_array':
            fig = plt.gcf()
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close()
            return img
    
    def get_maze(self) -> np.ndarray:
        """Return the maze layout"""
        return self.maze.copy()
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Return the agent's trajectory"""
        return self.trajectory.copy() 