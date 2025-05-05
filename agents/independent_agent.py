import numpy as np
from collections import defaultdict

class IndependentAgent:
    def __init__(self, idx, action_space=4, alpha=0.5, alpha_min=0.05, gamma=0.95, epsilon=1.0, eps_min=0.05, eps_decay=0.997):
        self.id = idx
        self.action_space = action_space
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.q_table = defaultdict(lambda: np.zeros(self.action_space))

        self.known_victims = set()
        self.new_cells = set()

        self.last_state = None
        self.last_action = None
        self.bytes_sent = 0
        
    def _encode_state(self, obs):
        row, col = obs["position"]

        self.known_victims.update(obs["victim_positions"])

        if self.known_victims:
            vx, vy = min(self.known_victims,key=lambda v: max(abs(v[0] - row), abs(v[1] - col)))
            dx = np.clip(vx - row, -4, 4) + 4  # 0..8
            dy = np.clip(vy - col, -4, 4) + 4
        else:
            dx = dy = 9                        # sentinel “no target”

        local = obs["local_grid"]
        mask = 0
        bit = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if local[dr + 1, dc + 1] == 1:  # 1 == OBSTACLE
                    mask |= 1 << bit
                bit += 1
        return (row, col, dx, dy, mask)
        
    def act(self, obs):
        self.new_cells.add(obs["position"])

        state_key = self._encode_state(obs)
        self.last_state = state_key

        # ε-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = int(np.argmax(self.q_table[state_key]))
        self.last_action = action
        return action

    def update(self, next_obs, reward, done, info):
        for xy in info.get("rescued", []):
            self.known_victims.discard(tuple(xy))

        s  = self.last_state
        a  = self.last_action
        s2 = self._encode_state(next_obs)

        best_next = np.max(self.q_table[s2])
        td_target = reward + self.gamma * (0 if done else best_next)
        self.q_table[s][a] += self.alpha * (td_target - self.q_table[s][a])

        if done:
            # linear decay α
            self.alpha = max(self.alpha_min, self.alpha - (0.5 - self.alpha_min) / 1379)
            # exponential decay ε
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def send(self, t):
        return set()

    def receive(self, payload):
        pass