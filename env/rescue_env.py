import numpy as np
from collections import defaultdict
class RescueEnv:
    EMPTY, OBSTACLE, VICTIM, AGENT = 0, 1, 2, 3

    ACTION_DELTAS = {0: (-1, 0), 1: ( 1, 0), 2: ( 0, -1), 3: ( 0, 1)}

    def __init__(self, grid_size=(10, 10), num_agents=3, num_victims=7, num_obstacles=11, max_steps=57, seed=None, lambda_cost = 0.0):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_victims = num_victims
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.static_obstacles = self._obstacle_mask()
        self.lambda_cost = lambda_cost
        self.reset()
    
    def _obstacle_mask(self):
        mask = np.zeros(self.grid_size, dtype=bool)
        idx = self.rng.choice(mask.size, size=self.num_obstacles, replace=False)
        mask.flat[idx] = True
        return mask  

    def _place_randomly(self, obj, num):
        free_coords = np.argwhere(self.grid == self.EMPTY)    

        if len(free_coords) < num:
            raise ValueError(f"Not enough empty cells to place {num} objects")

        chosen = self.rng.choice(len(free_coords), size=num, replace=False)
        for idx in chosen:
            r, c = free_coords[idx]
            self.grid[r, c] = obj

            if obj == self.AGENT:
                self.agent_positions.append((r, c))
            elif obj == self.VICTIM:
                self.victim_positions.add((r, c))

    def _is_valid_move(self, pos):
        return (0 <= pos[0] < self.grid_size[0] and 
                0 <= pos[1] < self.grid_size[1] and 
                self.grid[pos] in {self.EMPTY, self.VICTIM})
    
    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.grid[self.static_obstacles] = self.OBSTACLE
        
        self.agent_positions = []
        self.victim_positions = set()
        
        self._place_randomly(self.VICTIM, self.num_victims)
        self._place_randomly(self.AGENT, self.num_agents)

        self.visited_cells = [set() for _ in range(self.num_agents)]
        self.shared_map = set()
        self.global_visit_count = defaultdict(int)
        for agent_idx in range(self.num_agents):
            self.visited_cells[agent_idx].add(self.agent_positions[agent_idx])
            self.global_visit_count[self.agent_positions[agent_idx]] += 1

        self.steps = 0
        self.rescued = 0

        return self._get_observation()
    
    def _nearest_victim_distance(self, pos, agent):       
        min_dist = float('inf')
        for victim in agent.known_victims:
            distance = abs(pos[0] - victim[0]) + abs(pos[1] - victim[1])
            if distance < min_dist:
                min_dist = distance
        return min_dist

    def update_agent(self, agent, action):
        r, c = self.agent_positions[agent.id]
        dr, dc = self.ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        new_pos = (nr, nc)

        reward = -0.1  

        if self._is_valid_move(new_pos):
            self.grid[r, c] = self.EMPTY
            self.grid[nr, nc] = self.AGENT
            self.agent_positions[agent.id] = new_pos

            # Victim-related rewards
            if agent.known_victims:
                old_dist_to_victim = self._nearest_victim_distance((r, c), agent)
                new_dist_to_victim = self._nearest_victim_distance(new_pos, agent)
                if new_dist_to_victim < old_dist_to_victim:
                    reward += 0.17
                
            # Victim rescue reward
            if new_pos in self.victim_positions:
                self.victim_positions.remove(new_pos)
                self.rescued += 1
                reward += 10
                agent.rescued_victims.add(new_pos)
                agent.known_victims.remove(new_pos)
                agent.shared_victims.discard(new_pos)

            # Exploration rewards
            new_to_agent = new_pos not in self.visited_cells[agent.id]
            new_globally = self.global_visit_count[new_pos] == 0
            
            # Add position to visited cells only if it's new to this agent
            if new_to_agent:
                self.visited_cells[agent.id].add(new_pos)
            else:
                reward -= 0.23
            
            # Reward finding completely new cells (team efficiency)
            if new_globally:
                reward += 0.37  
            else:
                visits = self.global_visit_count[new_pos]
                if visits > 0:
                    reward -= 0.1 * min(visits, 3) 

            self.global_visit_count[new_pos] += 1
        else:
            reward -= 1.7

        return reward

    def step(self, actions, agents):
        payloads = [agent.send(self.steps) for agent in agents]
        inbox = set().union(*payloads)
        rewards = [-self.lambda_cost * len(p) for p in payloads]

        for message in inbox:
            for agent in agents:
               agent.receive(message)
               if message[0] == "C":
                   self.shared_map.add(message[1])

        for agent, action in zip(agents, actions):
            rwd = self.update_agent(agent, action)
            rewards[agent.id] += rwd

        self.steps += 1

        coverage = len(self.global_visit_count.keys())/(self.grid_size[0]*self.grid_size[1])
        redundancy = sum(1 for v in self.global_visit_count.values() if v>1)

        done = self.steps >= self.max_steps or self.rescued == self.num_victims
        info = {"steps": self.steps, "rescued": self.rescued, "comms": inbox, 
                "coverage": coverage, "redundancy": redundancy}

        return self._get_observation(), rewards, done, info

    def _get_observation(self):
        obs = []
        H, W = self.grid_size
        for idx, (r, c) in enumerate(self.agent_positions):
            omask = 0  
            vmask = 0  
            bit = 0
            local_victims = []

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    pos = (rr, cc)

                    # Check if position is out of bounds or is an obstacle
                    if not (0 <= rr < H and 0 <= cc < W) or self.grid[rr, cc] == self.OBSTACLE:
                        omask |= 1 << bit
                    else:
                        # Check for victims
                        if self.grid[rr, cc] == self.VICTIM:
                            local_victims.append(pos)
                    
                    # Create visited mask 
                    if pos in self.visited_cells[idx] or pos in self.shared_map:
                        vmask |= 1 << bit
                        
                    bit += 1
                    
            obs.append({"agent_id": idx, "position": (r, c), "obstacle_mask": omask, 
                        "victim_positions": local_victims, "visited_mask": vmask})
        return obs

    def render(self):
        sym = {self.EMPTY:'.', self.OBSTACLE:'#', self.VICTIM:'V', self.AGENT:'A'}
        for r in range(self.grid_size[0]):
            print(' '.join(sym[x] for x in self.grid[r]))
        print()


