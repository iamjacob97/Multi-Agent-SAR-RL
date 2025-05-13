import numpy as np
from collections import defaultdict


class IndependentAgent:
    def __init__(self, idx, action_space=4, alpha=0.73, alpha_min=0.05, gamma=0.95, epsilon=0.97, eps_min=0.05, total_episodes = 357951):
        self.id = idx
        self.action_space = action_space
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        decay_episodes   = int(0.77 * total_episodes)
        self.alpha_step  = (self.alpha - self.alpha_min) / decay_episodes
        self.eps_decay   = (self.eps_min / self.epsilon) ** (1 / decay_episodes)

        self.q_table = defaultdict(lambda: np.zeros(self.action_space))
        self.q_table_size = 0

        self.known_victims = set()
        self.shared_victims = set()
        self.rescued_victims = set()

        self.new_cells = set()

        self.last_state = None
        self.last_action = None
        self.bytes_sent = 0
        
    def _encode_state(self, obs):
        row, col = obs["position"]
        vmask = obs["visited_mask"]
        omask = obs["obstacle_mask"]

        if self.known_victims:
            vx, vy = min(self.known_victims, key=lambda v: max(abs(v[0] - row), abs(v[1] - col)))
            dx = np.clip(vx - row, -4, 4) + 4
            dy = np.clip(vy - col, -4, 4) + 4
        else:
            dx = dy = 9       

        return (row, col, dx, dy, omask, vmask)

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
    
    def decay(self):
        self.alpha = max(self.alpha_min, self.alpha - self.alpha_step)
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def update(self, next_obs, reward, done, info):
        s  = self.last_state
        a  = self.last_action
        s2 = self._encode_state(next_obs)

        best_next = np.max(self.q_table[s2])
        td_target = reward + self.gamma * (0 if done else best_next)
        td_error = td_target - self.q_table[s][a]
        self.q_table[s][a] += self.alpha * td_error

        if done:
            self.decay()
            self.q_table_size = len(self.q_table)
        
        return td_error
    def get_payload(self):
        payload = set()

        for victim in self.known_victims:
            payload.add(("V", victim))
        for cell in self.new_cells:
            payload.add(("C", cell))
        self.new_cells.clear()
        for rescued in self.rescued_victims:
            payload.add(("R", rescued))
        self.rescued_victims.clear()

        return payload

    def send(self, t):
        payload = set()
        self.bytes_sent += len(payload) * 8
        return payload

    def receive(self, message):
        if message[0] == "V":
            self.known_victims.add(message[1])
            self.shared_victims.add(message[1])
        elif message[0] == "R":
            self.known_victims.discard(message[1])
            self.shared_victims.discard(message[1])

    def reset(self):
        self.known_victims.clear()
        self.shared_victims.clear()
        self.new_cells.clear()
        self.rescued_victims.clear()
        self.bytes_sent = 0