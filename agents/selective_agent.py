import numpy as np
from .independent_agent import IndependentAgent

class SelectiveAgent(IndependentAgent):
    """
    Selective Communication Agent:
    Only shares critical information (e.g., closest victim)
    """
    def __init__(self, idx, victim_threshold=3, urgency_threshold=5, **kwargs):
        super().__init__(idx, **kwargs)
        self.victim_threshold = victim_threshold  # Distance threshold to report victim
        self.urgency_threshold = urgency_threshold  # Step threshold to report again

        # Track information for communication
        self.victim_buffer = {}  # {victim_pos: last_reported_timestep}
        self.received_victims = set()  # Victim positions received from other agents
        self.received_cells = set()  # Visited cells received from other agents
        self.last_reported = {}  # {victim_pos: last_reported_timestep}
        
    def update(self, next_obs, reward, done, info):
        # Update victim buffer with newly discovered victims
        current_step = next_obs["steps"]
        for victim_pos in next_obs["victim_positions"]:
            self.victim_buffer[tuple(victim_pos)] = current_step
            
        # Standard Q-learning update from parent class
        super().update(next_obs, reward, done, info)
        
        # When victim is rescued, remove from buffers
        for xy in info.get("rescued", []):
            xy_tuple = tuple(xy)
            if xy_tuple in self.victim_buffer:
                del self.victim_buffer[xy_tuple]
            if xy_tuple in self.last_reported:
                del self.last_reported[xy_tuple]
            self.received_victims.discard(xy_tuple)

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
    
    def _is_critical(self, victim_pos, agent_pos, current_step):
        """Determine if a victim position is critical enough to report"""
        # Calculate Manhattan distance to victim
        vx, vy = victim_pos
        ax, ay = agent_pos
        distance = abs(vx - ax) + abs(vy - ay)
        
        # Check if this victim was reported recently
        last_reported = self.last_reported.get(victim_pos, 0)
        time_since_report = current_step - last_reported
        
        # Report if close enough or it's been a while since last report
        return (distance <= self.victim_threshold or 
                time_since_report >= self.urgency_threshold)
    
    def send(self, t):
        """Send only critical information"""
        # Get current position
        if not hasattr(self, 'last_state') or self.last_state is None:
            return {}
            
        row, col = self.last_state[0], self.last_state[1]
        current_pos = (row, col)
        
        critical_victims = []
        
        # Determine which victims are critical enough to report
        for victim_pos, discovery_time in self.victim_buffer.items():
            if self._is_critical(victim_pos, current_pos, t):
                critical_victims.append(victim_pos)
                self.last_reported[victim_pos] = t
        
        # If no critical information, don't send anything
        if not critical_victims:
            return {}
        
        # Prepare payload with only critical information
        payload = {
            "victims": critical_victims,
            "agent_id": self.id,
            "agent_pos": current_pos,
        }
        
        # Estimate communication cost (8 bytes per coordinate pair)
        bytes_to_send = len(critical_victims) * 8 + 8  # Including agent position
        self.bytes_sent += bytes_to_send
        
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
                
        # Track other agent positions if provided
        if "agent_pos" in payload:
            # Could be used for coordination, currently just tracking
            other_agent_pos = tuple(payload["agent_pos"]) 