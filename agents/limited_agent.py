import numpy as np
from agents import IndependentAgent

class LimitedAgent(IndependentAgent):
    def __init__(self, idx, comm_interval=5, **kwargs):
        super().__init__(idx, **kwargs)
        self.comm_interval = comm_interval
        self.last_comm = -comm_interval
        
    def send(self, t):
        payload = set()
        # Check if it's time to communicate
        if t - self.last_comm >= self.comm_interval:
            self.last_comm = t
            # Prepare payload: victim positions and visited cells
            payload = self.get_payload()
            # Estimate communication cost 
            self.bytes_sent += len(payload) * 8
                    
        return payload
    
    def reset(self):
        super().reset()
        self.last_comm = -self.comm_interval
    
    