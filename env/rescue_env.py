import numpy as np

class RescueEnv:
    EMPTY = 0
    OBSTACLE = 1
    VICTIM = 2
    AGENT = 3

    ACTION_DELTAS = {0: (-1,  0), 1: ( 0,  1), 2: ( 1,  0), 3: ( 0, -1), 4: ( 0,  0)}

    def __init__(self, grid_size=(10, 10), num_agents=3, num_victims=5, num_obstacles=10, max_steps=357):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_victims = num_victims
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        
        self.agent_positions = []
        self.victim_positions = []
        
        self._place(self.grid, self.OBSTACLE, self.num_obstacles)
        self._place(self.grid, self.VICTIM, self.num_victims)
        self._place(self.grid, self.AGENT, self.num_agents)
        
        self.steps = 0
        self.rescued = 0
        return self._get_observation()

    def _place(self, grid, obj, num):
        free_idx = np.where(grid.flat == self.EMPTY)[0]

        if len(free_idx) < num:
            raise ValueError(f"Not enough empty cells to place {num} objects")

        chosen_idx = np.random.choice(free_idx, size=num, replace=False)

        for idx in chosen_idx:
            pos = (idx // self.grid_size[1], idx % self.grid_size[1])
            grid[pos] = obj

            if obj == self.AGENT:
                self.agent_positions.append(pos)
            elif obj == self.VICTIM:
                self.victim_positions.append(pos)

    def _is_valid_move(self, pos):
        return (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1] and 
                (self.grid[pos] == self.EMPTY or self.grid[pos] == self.VICTIM))

    def _get_valid_moves(self, pos):
        moves = []
        for action, (dx, dy) in self.ACTION_DELTAS.items():
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_move(new_pos) or action == 4:  
                moves.append(action)
        return moves
    
    def step(self, actions):
        rewards = [0] * self.num_agents
        done = False
        
        # Process each agent's action
        for agent_idx, action in enumerate(actions):
            current_pos = self.agent_positions[agent_idx]
            delta = self.ACTION_DELTAS[action]
            new_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
            
            # Check if move is valid
            if self._is_valid_move(new_pos):
                # Update agent position
                self.grid[current_pos] = self.EMPTY
                self.grid[new_pos] = self.AGENT
                self.agent_positions[agent_idx] = new_pos
                
                # Check if the new position contains a victim
                if new_pos in self.victim_positions:
                    victim_idx = self.victim_positions.index(new_pos)
                    rewards[agent_idx] += 10  # Reward for finding victim
                    self.victim_positions.pop(victim_idx)
                    self.rescued += 1
            else:
                rewards[agent_idx] -= 1  # Penalty for invalid move
        
        # Small penalty for each step to encourage efficiency
        for i in range(self.num_agents):
            rewards[i] -= 0.1
        
        self.steps += 1
        
        # Check if episode is done
        if not self.victim_positions or self.steps >= self.max_steps:
            done = True
        
        return self._get_observation(), rewards, done, {}

    def _get_observation(self):
        observations = []

        for agent_idx in range(self.num_agents):
            agent_pos = self.agent_positions[agent_idx]            
            local_obs = np.zeros((3, 3), dtype=int)
            local_victims = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    obs_i = agent_pos[0] + i
                    obs_j = agent_pos[1] + j
                    if 0 <= obs_i < self.grid_size[0] and 0 <= obs_j < self.grid_size[1]:
                        local_obs[i+1, j+1] = self.grid[obs_i, obs_j]
                        if self.grid[obs_i, obs_j] == self.VICTIM:
                            local_victims.append((obs_i, obs_j))

            # Create agent observation
            agent_obs = {
                'local_grid': local_obs,
                'position': agent_pos,
                'victim_positions': local_victims,
                'steps': self.steps
            }
            observations.append(agent_obs)
        
        return observations

    def render(self):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                cell = self.grid[i, j]
                if cell == self.EMPTY:
                    print('.', end=' ')
                elif cell == self.OBSTACLE:
                    print('#', end=' ')
                elif cell == self.VICTIM:
                    print('V', end=' ')
                elif cell == self.AGENT:
                    print('A', end=' ')
            print()
        print()


