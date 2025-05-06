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
        self.rescued = set()

        self.last_state = None
        self.last_action = None
        self.bytes_sent = 0
        
    def _encode_state(self, obs):
        row, col = obs["position"]
        visited_flag = obs["visited_flag"]
        mask = obs["obstacle_mask"]

        if self.known_victims:
            vx, vy = min(
                self.known_victims,
                key=lambda v: max(abs(v[0] - row), abs(v[1] - col))
            )
            dx = np.clip(vx - row, -4, 4) + 4
            dy = np.clip(vy - col, -4, 4) + 4
        else:
            dx = dy = 9       

        return (row, col, visited_flag, dx, dy, mask)

    def act(self, obs):
        self.new_cells.add(obs["position"])
        self.known_victims.update(obs["victim_positions"])

        state_key = self._encode_state(obs)
        self.last_state = state_key

         # epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = int(np.argmax(self.q_table[state_key]))
        self.last_action = action

        return action

    def update(self, next_obs, reward, done, info):
        pass

    def send(self, t):
        payload = set()
        return payload

    def receive(self, message):
        if message[0] == "V":
            self.known_victims.add(message[1])
        elif message[0] == "R":
            self.known_victims.discard(message[1])