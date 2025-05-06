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
    
    def update_agent(self, agent, action):
        r, c = self.agent_positions[agent.id]
        dr, dc = self.ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc

        reward = -0.1

        if self._is_valid_move((nr, nc)):
            if (nr, nc) in self.victim_positions:
                self.victim_positions.remove((nr, nc))
                self.rescued += 1
                reward += 10
                agent.rescued.add((nr, nc))
                agent.known_victims.remove((nr, nc))

            self.grid[r, c] = self.EMPTY
            self.grid[nr, nc] = self.AGENT
            self.agent_positions[agent.id] = (nr, nc)

            if (nr, nc) in self.visited_cells[agent.id]:
                reward -= 0.2
                self.visited_flag[agent.id] = 1
            else:
                reward += 0.2
                self.visited_flag[agent.id] = 0
                self.visited_cells[agent.id].add((nr, nc))

            self.global_visit_count[(nr, nc)] += 1
        else:
            reward -= 1

        return reward

    def step(self, actions, agents):
        payloads = [agent.send(self.steps) for agent in agents]
        rewards = [-0.02 * len(payload) for payload in payloads]
        inbox = set().union(*payloads)

        self.visited_flag = [0] * self.num_agents

        for message in inbox:
            for agent in agents:
               agent.receive(message)
               if message[0] == "C":
                   self.visited_cells[agent.id].add(message[1])

        for agent, action in zip(agents, actions):
            rwd = self.update_agent(agent, action)
            rewards[agent.id] += rwd

        self.steps += 1
        
        done = self.steps >= self.max_steps or self.rescued == self.num_victims
        info = {"steps": self.steps, "rescued": self.rescued, "comms": inbox}


        return self._get_observation(), rewards, done, info

    def _get_observation(self):
        obs = []
        H, W = self.grid_size
        for idx, (r, c) in enumerate(self.agent_positions):
            mask = 0
            bit  = 0
            local_victims = []

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc

                    if not (0 <= rr < H and 0 <= cc < W) or self.grid[rr, cc] == self.OBSTACLE:
                        mask |= 1 << bit
                    else:
                        if self.grid[rr, cc] == self.VICTIM:
                            local_victims.append((rr, cc))
                    bit += 1
            visited_flag = self.visited_flag[idx]
            obs.append({"agent_id": idx,"position": (r, c),
                        "obstacle_mask": mask, "victim_positions": local_victims,
                        "steps": self.steps, "visited_flag": visited_flag})
        return obs

    def render(self):
        sym = {self.EMPTY:'.', self.OBSTACLE:'#', self.VICTIM:'V', self.AGENT:'A'}
        for r in range(self.grid_size[0]):
            print(' '.join(sym[x] for x in self.grid[r]))
        print()


