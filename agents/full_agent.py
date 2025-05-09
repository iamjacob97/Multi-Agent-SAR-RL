import numpy as np
from collections import defaultdict
from .independent_agent import IndependentAgent

class FullAgent(IndependentAgent):
    def send(self, t):
        payload = set()
        self.last_comm = t
        
        payload = self.get_payload()

        self.bytes_sent += len(payload) * 8
                    
        return payload
