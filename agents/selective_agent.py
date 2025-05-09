import numpy as np
from agents import IndependentAgent

class SelectiveAgent(IndependentAgent):
    def __init__(self, idx, comm_interval=25, **kwargs):
        super().__init__(idx, **kwargs)
        self.comm_interval = comm_interval
        self.last_comm = -comm_interval
        
    def send(self, t):
        payload = set()

        if t - self.last_comm >= self.comm_interval or len(self.rescued_victims) > 0 or len(self.known_victims - self.shared_victims) > 0:
            self.last_comm = t
            self.shared_victims = self.known_victims
            payload = self.get_payload()

            self.bytes_sent += len(payload) * 8

        return payload
    
    def reset(self):
        super().reset()
        self.last_comm = -self.comm_interval
    
    