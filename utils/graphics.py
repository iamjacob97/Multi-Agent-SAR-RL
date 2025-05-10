import pygame
import numpy as np

class SARVisualizer:
    def __init__(self, grid_size, cell_size=60):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Search and Rescue Visualization")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 100, 255)
        self.YELLOW = (255, 255, 0)
        self.DARK_GRAY = (100, 100, 100)
        
        self.font = pygame.font.Font(None, 24)
        
        # Communication tracking
        self.communicating_agents = set()  # Set of agent IDs that are currently communicating
        self.comm_duration = 10  # How long agents stay yellow when communicating
        
    def draw_grid(self):
        for i in range(self.grid_size + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.GRAY,
                           (i * self.cell_size, 0),
                           (i * self.cell_size, self.window_size))
            # Horizontal lines
            pygame.draw.line(self.screen, self.GRAY,
                           (0, i * self.cell_size),
                           (self.window_size, i * self.cell_size))
    
    def draw_agent(self, pos, agent_id):
        x, y = pos
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        
        # Draw agent circle - yellow if communicating, blue otherwise
        color = self.YELLOW if agent_id in self.communicating_agents else self.BLUE
        pygame.draw.circle(self.screen, color,
                         (center_x, center_y), self.cell_size // 3)
        
        # Draw agent ID
        text = self.font.render(str(agent_id), True, self.WHITE)
        text_rect = text.get_rect(center=(center_x, center_y))
        self.screen.blit(text, text_rect)
    
    def draw_victim(self, pos, rescued=False):
        x, y = pos
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        
        # Draw victim as a cross
        color = self.GREEN if rescued else self.RED
        size = self.cell_size // 4
        pygame.draw.line(self.screen, color,
                        (center_x - size, center_y),
                        (center_x + size, center_y), 3)
        pygame.draw.line(self.screen, color,
                        (center_x, center_y - size),
                        (center_x, center_y + size), 3)
    
    def draw_obstacle(self, pos):
        x, y = pos
        rect = pygame.Rect(
            x * self.cell_size + 2,
            y * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(self.screen, self.DARK_GRAY, rect)
    
    def update(self, agents, victims, obstacles, communications=None):
        self.screen.fill(self.WHITE)
        self.draw_grid()
        
        # Draw obstacles first (so they appear behind everything)
        for obstacle in obstacles:
            self.draw_obstacle(obstacle['position'])
        
        # Draw victims
        for victim in victims:
            self.draw_victim(victim['position'], victim.get('rescued', False))
        
        # Update communicating agents
        if communications:
            self.communicating_agents = set()
            for agent1_id, agent2_id in communications:
                self.communicating_agents.add(agent1_id)
                self.communicating_agents.add(agent2_id)
        
        # Draw agents
        for agent in agents:
            self.draw_agent(agent['position'], agent['id'])
        
        pygame.display.flip()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
        return True
    
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == "__main__":
    # Create a 5x5 grid visualization
    visualizer = SARVisualizer(grid_size=5)
    
    # Example agents
    agents = [
        {'position': (1, 1), 'id': 1},
        {'position': (3, 3), 'id': 2},
        {'position': (2, 2), 'id': 3}
    ]
    
    # Example victims
    victims = [
        {'position': (0, 0), 'rescued': False},
        {'position': (4, 4), 'rescued': True}
    ]
    
    # Example obstacles
    obstacles = [
        {'position': (1, 3)},
        {'position': (3, 1)},
        {'position': (2, 2)}
    ]
    
    # Example communications
    communications = [(1, 3), (2, 3)]
    
    # Main loop
    running = True
    while running:
        running = visualizer.handle_events()
        visualizer.update(agents, victims, obstacles, communications)
        pygame.time.delay(100)  # Add a small delay to make the visualization visible
    
    visualizer.close() 