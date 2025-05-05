import numpy as np
from .independent_agent import IndependentAgent

class FullAgent(IndependentAgent):
    """
    Full Communication Agent:
    Continuously shares all discoveries with other agents
    """
    def __init__(self, idx, redundancy_penalty=0.1, **kwargs):
        super().__init__(idx, **kwargs)
        self.redundancy_penalty = redundancy_penalty  # Penalty for redundant exploration
        
        # Track information for communication
        self.victim_buffer = set()  # Victim positions to share
        self.received_victims = set()  # Victim positions received from other agents
        self.received_cells = set()  # Visited cells received from other agents
        self.other_agent_positions = {}  # Current positions of other agents {agent_id: position}
        
    def update(self, next_obs, reward, done, info):
        # Update victim buffer with newly discovered victims
        for victim_pos in next_obs["victim_positions"]:
            self.victim_buffer.add(tuple(victim_pos))
            
        # Add current position to visited cells
        self.known_visited.add(tuple(next_obs["position"]))
            
        # Standard Q-learning update from parent class
        super().update(next_obs, reward, done, info)
        
        # When victim is rescued, remove from buffers
        for xy in info.get("rescued", []):
            xy_tuple = tuple(xy)
            self.victim_buffer.discard(xy_tuple)
            self.received_victims.discard(xy_tuple)

    def _encode_state(self, obs):
        """
        Extended state encoding that considers received information 
        from other agents, including their positions
        """
        row, col = obs["position"]
        current_pos = (row, col)
        
        # Add current observations to known sets
        self.known_victims.update(obs["victim_positions"])
        
        # Also consider victims reported by other agents
        combined_victims = self.known_victims.union(self.received_victims)
        
        # Encode closest victim direction
        if combined_victims:
            # Find closest victim from both direct observation and received info
            vx, vy = min(combined_victims, 
                          key=lambda v: max(abs(v[0] - row), abs(v[1] - col)))
            dx = np.clip(vx - row, -4, 4) + 4
            dy = np.clip(vy - col, -4, 4) + 4
        else:
            dx = dy = 9
        
        # Encode local obstacles
        local = obs["local_grid"]
        mask = 0
        bit = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if local[dr + 1, dc + 1] == 1:  # 1 == OBSTACLE
                    mask |= 1 << bit
                bit += 1
        
        # Additional feature: is current position already visited by other agents?
        other_visited = 1 if current_pos in self.received_cells else 0
        
        # Return extended state representation
        return (row, col, dx, dy, mask, other_visited)
    
    def act(self, obs):
        """
        Modified act method that considers redundancy penalty
        """
        self.new_cells.add(tuple(obs["position"]))
        
        # Calculate redundancy penalty
        redundancy = 0
        current_pos = tuple(obs["position"])
        if current_pos in self.received_cells:
            redundancy = self.redundancy_penalty
        
        state_key = self._encode_state(obs)
        self.last_state = state_key
        
        # ε-greedy with redundancy consideration
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            # Apply redundancy penalty to Q-values for already visited states
            q_values = self.q_table[state_key].copy()
            if redundancy > 0:
                q_values -= redundancy
            action = int(np.argmax(q_values))
            
        self.last_action = action
        return action
        
    def send(self, t):
        """Send all discoveries continuously"""
        if not hasattr(self, 'last_state') or self.last_state is None:
            return {}
            
        # Get current position from state
        if len(self.last_state) >= 2:
            current_pos = (self.last_state[0], self.last_state[1])
        else:
            return {}
        
        # Prepare payload with all information
        payload = {
            "victims": list(self.victim_buffer),
            "visited": list(self.new_cells),
            "agent_id": self.id,
            "agent_pos": current_pos,
            "timestamp": t
        }
        
        # Estimate communication cost (8 bytes per coordinate pair)
        bytes_to_send = (len(payload["victims"]) * 8 + 
                         len(payload["visited"]) * 8 + 
                         8 +  # Agent position
                         4)   # Timestamp
        self.bytes_sent += bytes_to_send
        
        # Clear new cells after sending
        self.new_cells = set()
        
        return payload
    
    def receive(self, payload):
        """Process information received from other agents"""
        if not payload:
            return
            
        # Process each agent's payload
        if isinstance(payload, list):
            for p in payload:
                self._process_payload(p)
        else:
            self._process_payload(payload)
    
    def _process_payload(self, payload):
        """Process a single payload from another agent"""
        # Skip own messages
        if payload.get("agent_id") == self.id:
            return
            
        # Update with victims reported by other agents
        if "victims" in payload:
            for victim_pos in payload["victims"]:
                self.received_victims.add(tuple(victim_pos))
                
        # Update with cells visited by other agents
        if "visited" in payload:
            for cell in payload["visited"]:
                self.received_cells.add(tuple(cell))
                
        # Track other agent positions
        if "agent_pos" in payload and "agent_id" in payload:
            agent_id = payload["agent_id"]
            agent_pos = tuple(payload["agent_pos"])
            self.other_agent_positions[agent_id] = agent_pos 