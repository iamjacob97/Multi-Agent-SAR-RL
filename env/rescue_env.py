import numpy as np
from collections import defaultdict
class RescueEnv:
    EMPTY, OBSTACLE, VICTIM, AGENT = 0, 1, 2, 3

    ACTION_DELTAS = {0: (-1,  0), 1: ( 0,  1), 2: ( 1,  0), 3: ( 0, -1)}

    def __init__(self, grid_size=(10, 10), num_agents=3, num_victims=5, num_obstacles=10, max_steps=357, seed=None):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_victims = num_victims
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.static_obstacles = self._obstacle_mask()

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
        self.global_visit_count = defaultdict(int)
        for agent_idx in range(self.num_agents):
            self.visited_cells[agent_idx].add(self.agent_positions[agent_idx])
            self.global_visit_count[self.agent_positions[agent_idx]] += 1

        self.steps = 0
        self.rescued = 0

        return self._get_observation()
    
    def update_agent(self, idx, action):
        r, c = self.agent_positions[idx]
        dr, dc = self.ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc

        reward = -0.1

        if self._is_valid_move((nr, nc)):
            if (nr, nc) in self.victim_positions:
                self.victim_positions.remove((nr, nc))
                self.rescued += 1
                reward += 10
                self.rescued_now.append((nr, nc))

            self.grid[r, c] = self.EMPTY
            self.grid[nr, nc] = self.AGENT
            self.agent_positions[idx] = (nr, nc)

            if (nr, nc) in self.visited_cells[idx]:
                reward -= 0.2
            else:
                reward += 0.2
                self.visited_cells[idx].add((nr, nc))

            self.global_visit_count[(nr, nc)] += 1

        else:
            reward -= 1.0

        return reward

    def step(self, actions, agents=None):
        self.rescued_now = []
        rewards = []

        merged_payload = set()
        if agents is not None:
            payloads = [ag.send(self.steps) for ag in agents]
            for p in payloads:
                merged_payload.update(p)
            # bandwidth cost
            for i, pay in enumerate(payloads):
                rewards.append(-0.02 * len(pay))
                agents[i].bytes_sent += len(pay)
        else:
            rewards = [0.0] * self.num_agents

        for idx, action in enumerate(actions):
            rewards[idx] += self.update_agent(idx, action)

        self.steps += 1
        done = (self.rescued == self.num_victims) or (self.steps >= self.max_steps)

        obs_next = self._get_observation()
        info = {"rescued": self.rescued_now, "comms": merged_payload}

        if agents is not None:
            for ag in agents:
                ag.receive(merged_payload)

        return obs_next, rewards, done, info
    
    def _get_observation(self):
        obs = []
        for idx in range(self.num_agents):
            r, c = self.agent_positions[idx]
            patch = np.zeros((3, 3), dtype=int)
            victims_local = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.grid_size[0] and 0 <= cc < self.grid_size[1]:
                        patch[dr+1, dc+1] = self.grid[rr, cc]
                        if self.grid[rr, cc] == self.VICTIM:
                            victims_local.append((rr, cc))

            obs.append({
                "local_grid": patch,
                "position":   (r, c),
                "victim_positions": victims_local,
                "steps": self.steps
            })
        return obs

    def render(self):
        sym = {self.EMPTY:'.', self.OBSTACLE:'#', self.VICTIM:'V', self.AGENT:'A'}
        for r in range(self.grid_size[0]):
            print(' '.join(sym[x] for x in self.grid[r]))
        print()


