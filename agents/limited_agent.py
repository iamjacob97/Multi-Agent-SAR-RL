from agents.independent_agent import IndependentAgent

class LimitedAgent(IndependentAgent):
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1, communication_interval=5):
        super().__init__(state_size, action_size, learning_rate, discount_factor, exploration_rate)
        self.communication_interval = communication_interval
        self.steps_since_last_communication = 0
        self.discovered_victims = set()  # Track victims this agent has discovered
    
    def communicate(self, other_agents, current_state):
        self.steps_since_last_communication += 1
        
        if self.steps_since_last_communication >= self.communication_interval:
            # Share all discovered victims
            message = {
                'victim_positions': list(self.discovered_victims),
                'agent_position': current_state['position']
            }
            
            # Send message to other agents
            for agent in other_agents:
                agent.receive_information(message)
            
            self.steps_since_last_communication = 0
    
    def receive_information(self, message):
        """
        Process received information from other agents
        
        Args:
            message: Dictionary containing information from other agents
        """
        if 'victim_positions' in message:
            # Add new victim positions to discovered victims
            self.discovered_victims.update(message['victim_positions'])
    
    def update_discovered_victims(self, local_victims):
        """
        Update discovered victims based on local observation
        
        Args:
            local_victims: List of victim positions in agent's local view
        """
        self.discovered_victims.update(local_victims) 