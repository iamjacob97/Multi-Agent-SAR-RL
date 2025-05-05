import numpy as np
from agents.independent_agent import IndependentAgent

class LimitedAgent(IndependentAgent):
    """
    Limited Communication Agent: 
    Share all observations at fixed intervals
    """
    def __init__(self, idx, comm_interval=5, **kwargs):
        super().__init__(idx, **kwargs)
        self.comm_interval = comm_interval  # How often to share information
        self.last_comm = 0                  # Time step of last communication
        
        # Track additional information for communication
        self.comm_buffer = set()            # Buffer of victim positions to share
        self.received_victims = set()       # Victim positions received from other agents
        self.received_cells = set()         # Visited cells received from other agents
        
    def update(self, next_obs, reward, done, info):
        # Add newly discovered victims to communication buffer
        for victim_pos in next_obs["victim_positions"]:
            self.comm_buffer.add(tuple(victim_pos))
            
        # Standard Q-learning update from parent class
        super().update(next_obs, reward, done, info)
        
        # When victim is rescued, remove from buffer
        for xy in info.get("rescued", []):
            self.comm_buffer.discard(tuple(xy))
            self.received_victims.discard(tuple(xy))
    
    def _encode_state(self, obs):
        """
        Extended state encoding that considers received information 
        from other agents
        """
        row, col = obs["position"]
        
        # Add current observations to known sets
        self.known_victims.update(obs["victim_positions"])
        
        # Also consider victims reported by other agents
        combined_victims = self.known_victims.union(self.received_victims)
        
        if combined_victims:
            # Find closest victim from both direct observation and received info
            vx, vy = min(combined_victims, 
                          key=lambda v: max(abs(v[0] - row), abs(v[1] - col)))
            dx = np.clip(vx - row, -4, 4) + 4
            dy = np.clip(vy - col, -4, 4) + 4
        else:
            dx = dy = 9
            
        # Encode local obstacles as in parent class
        local = obs["local_grid"]
        mask = 0
        bit = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if local[dr + 1, dc + 1] == 1:  # 1 == OBSTACLE
                    mask |= 1 << bit
                bit += 1
                
        return (row, col, dx, dy, mask)
    
    def send(self, t):
        """Send information based on communication interval"""
        # Check if it's time to communicate
        if t - self.last_comm >= self.comm_interval:
            self.last_comm = t
            
            # Prepare payload: victim positions and visited cells
            payload = {
                "victims": list(self.comm_buffer),
                "visited": list(self.new_cells),
                "agent_id": self.id
            }
            
            # Estimate communication cost (simple model: 8 bytes per coordinate pair)
            bytes_to_send = len(payload["victims"]) * 8 + len(payload["visited"]) * 8
            self.bytes_sent += bytes_to_send
            
            # Reset new cells after sending
            self.new_cells = set()
            
            return payload
        
        # Not time to communicate yet
        return {}
    
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