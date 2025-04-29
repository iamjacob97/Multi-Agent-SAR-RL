import numpy as np

class IndependentAgent:
    def __init__(self, state_size, action_size=5, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Initialize Q-table
        self.q_table = {}
        
    def _get_state_key(self, state):
        # Convert local grid to string representation
        local_grid_str = ''.join(map(str, state['local_grid'].flatten()))
        # Convert position to string
        pos_str = f"{state['position'][0]},{state['position'][1]}"
        # Convert victim positions to string
        victim_positions_str = '_'.join([f"{pos[0]},{pos[1]}" for pos in state['victim_positions']])
        
        return f"{local_grid_str}_{pos_str}_{victim_positions_str}"
    
    def get_action(self, state):
        state_key = self._get_state_key(state)

        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max * (not done))
        self.q_table[state_key][action] = new_value 