from env import RescueEnv
from agents import IndependentAgent, LimitedAgent, SelectiveAgent, FullAgent
import numpy as np
import argparse

def run_episode(env, agents, verbose=False):
    obs = env.reset()
    done = False
    total_reward = [0.0] * len(agents)

    while not done:
        actions = [ag.act(o) for ag, o in zip(agents, obs)]
        next_obs, rewards, done, info = env.step(actions, agents)

        for i, ag in enumerate(agents):
            ag.update(next_obs[i], rewards[i], done, info)
            total_reward[i] += rewards[i]

        obs = next_obs
        if verbose:
            env.render()
    return total_reward, env.steps, info

def create_agents(agent_type, num_agents):
    agents = []
    
    for i in range(num_agents):
        if agent_type == 'independent':
            agents.append(IndependentAgent(i))
        elif agent_type == 'limited':
            agents.append(LimitedAgent(i, comm_interval=5))
        elif agent_type == 'selective':
            agents.append(SelectiveAgent(i, victim_threshold=3, urgency_threshold=5))
        elif agent_type == 'full':
            agents.append(FullAgent(i, redundancy_penalty=0.1))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
    return agents

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Search and Rescue simulation with different agent types')
    parser.add_argument('--agent-type', type=str, default='independent',
                        choices=['independent', 'limited', 'selective', 'full'],
                        help='Type of agent to use')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='Size of the grid environment')
    parser.add_argument('--num-agents', type=int, default=3,
                        help='Number of agents in the environment')
    parser.add_argument('--num-victims', type=int, default=5,
                        help='Number of victims to rescue')
    parser.add_argument('--verbose', action='store_true',
                        help='Display environment after each step')
    
    args = parser.parse_args()
    
    # Setup environment
    env = RescueEnv(
        grid_size=(args.grid_size, args.grid_size),
        num_agents=args.num_agents,
        num_victims=args.num_victims,
        seed=args.seed
    )
    
    # Create agents based on specified type
    agents = create_agents(args.agent_type, args.num_agents)
    
    # Print experiment setup
    print(f"Running {args.episodes} episodes with {args.agent_type} agents")
    print(f"Environment: {args.grid_size}x{args.grid_size} grid, {args.num_agents} agents, {args.num_victims} victims")
    
    # Run episodes
    episode_rewards = []
    episode_steps = []
    episode_rescued = []
    
    for ep in range(args.episodes):
        tr, steps, info = run_episode(env, agents, verbose=args.verbose)
        
        # Store metrics
        episode_rewards.append(sum(tr))
        episode_steps.append(steps)
        episode_rescued.append(env.rescued)
        
        # Print episode summary
        print(f"Episode {ep+1}: total reward={tr}, steps={steps}, rescued={env.rescued}/{args.num_victims}")
    
    # Print experiment summary
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = sum(episode_steps) / len(episode_steps)
    avg_rescued = sum(episode_rescued) / len(episode_rescued)
    
    print("\nExperiment Summary:")
    print(f"Agent Type: {args.agent_type}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Victims Rescued: {avg_rescued:.2f}/{args.num_victims}")
    
    # Print communication statistics
    total_bytes = sum(agent.bytes_sent for agent in agents)
    print(f"Total Communication: {total_bytes} bytes")
    print(f"Average Communication per Agent: {total_bytes/len(agents):.2f} bytes")




