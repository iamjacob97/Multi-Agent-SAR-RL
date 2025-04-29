from env import RescueEnv
from agents import IndependentAgent
import numpy as np

def test_environment():
    # Create environment
    env = RescueEnv(grid_size=(10, 10), num_agents=3, num_victims=5, num_obstacles=10)
    
    # Create agents
    agents = [IndependentAgent(state_size=9, action_size=5) for _ in range(env.num_agents)]
    
    # Run a few episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        obs = env.reset()
        done = False
        total_rewards = [0] * env.num_agents
        
        while not done:
            # Get actions from agents
            actions = []
            for agent_idx, agent in enumerate(agents):
                action = agent.get_action(obs[agent_idx])
                actions.append(action)
            
            # Take step in environment
            next_obs, rewards, done, info = env.step(actions)
            
            # Update agents
            for agent_idx, agent in enumerate(agents):
                agent.update(obs[agent_idx], actions[agent_idx], rewards[agent_idx], next_obs[agent_idx], done)
                total_rewards[agent_idx] += rewards[agent_idx]
            
            # Render environment
            env.render()
            
            # Print information
            print(f"Actions: {actions}")
            print(f"Rewards: {rewards}")
            print(f"Total Rewards: {total_rewards}")
            print(f"Victims rescued: {env.rescued}")
            print(f"Steps: {env.steps}")
            print(f"Victims remaining: {len(env.victim_positions)}")
            
            # Update observations
            obs = next_obs
            
            # Break if done
            if done:
                break
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"Total Rewards: {total_rewards}")
        print(f"Victims Rescued: {env.rescued}")
        print(f"Steps Taken: {env.steps}")

if __name__ == "__main__":
    test_environment()




