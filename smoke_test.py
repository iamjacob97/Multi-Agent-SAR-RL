import numpy as np
import matplotlib.pyplot as plt
from env import RescueEnv
from agents import IndependentAgent, LimitedAgent, FullAgent, SelectiveAgent
import argparse
import time

# Configuration
GRID_SIZE = (5, 5)
NUM_AGENTS = 2
NUM_VICTIMS = 3
NUM_OBSTACLES = 5
MAX_STEPS = 17
SEED = 27
NUM_EPISODES = 1000

def check_environment():
    print("\nCHECKING ENVIRONMENT...")
    
    # Create environment
    env = RescueEnv(
        grid_size=GRID_SIZE, 
        num_agents=NUM_AGENTS, 
        num_victims=NUM_VICTIMS, 
        num_obstacles=NUM_OBSTACLES, 
        max_steps=MAX_STEPS, 
        seed=SEED
    )
    
    # Check initialization
    obs = env.reset()
    assert len(obs) == NUM_AGENTS, f"Expected {NUM_AGENTS} observations, got {len(obs)}"
    
    # Check observation structure
    for agent_obs in obs:
        assert "position" in agent_obs, "Observation missing 'position'"
        assert "obstacle_mask" in agent_obs, "Observation missing 'obstacle_mask'"
        assert "victim_positions" in agent_obs, "Observation missing 'victim_positions'"
        assert "visited_mask" in agent_obs, "Observation missing 'visited_mask'"
    
    # Check step function
    agents = [IndependentAgent(id) for id in range(NUM_AGENTS)]
    actions = [0] * NUM_AGENTS  # Just move all agents up
    next_obs, rewards, done, info = env.step(actions, agents)
    
    assert len(next_obs) == NUM_AGENTS, "Step returned wrong number of observations"
    assert len(rewards) == NUM_AGENTS, "Step returned wrong number of rewards"
    assert isinstance(done, bool), "Done should be a boolean"
    assert "steps" in info, "Info missing 'steps'"
    assert "rescued" in info, "Info missing 'rescued'"
    
    print("Environment checks passed!")
    
    # Render the environment
    print("\nInitial environment state:")
    env.render()
    
    return env

def check_agent_behavior(agent_type):
    print(f"\nCHECKING {agent_type.upper()} AGENT BEHAVIOR...")
    
    env = RescueEnv(
        grid_size=GRID_SIZE, 
        num_agents=NUM_AGENTS, 
        num_victims=NUM_VICTIMS, 
        num_obstacles=NUM_OBSTACLES, 
        max_steps=MAX_STEPS, 
        seed=SEED
    )
    
    # Create agents
    if agent_type == "independent":
        agents = [IndependentAgent(id, total_episodes=NUM_EPISODES) for id in range(NUM_AGENTS)]
    elif agent_type == "limited":
        agents = [LimitedAgent(id, comm_interval=5, total_episodes=NUM_EPISODES) for id in range(NUM_AGENTS)]
    elif agent_type == "selective":
        agents = [SelectiveAgent(id, comm_interval=25, total_episodes=NUM_EPISODES) for id in range(NUM_AGENTS)]
    elif agent_type == "full":
        agents = [FullAgent(id, total_episodes=NUM_EPISODES) for id in range(NUM_AGENTS)]
    
    # Run a single episode to check agent behavior
    obs = env.reset()
    for agent in agents:
        agent.reset()
    
    # Check action selection
    for i, (agent, o) in enumerate(zip(agents, obs)):
        action = agent.act(o)
        assert 0 <= action < 4, f"Agent {i} returned invalid action: {action}"
    
    # Check communication behavior
    for t in range(5):  # Run for a few steps
        # Get actions
        actions = [agent.act(o) for agent, o in zip(agents, obs)]
        
        # Step environment
        next_obs, rewards, done, info = env.step(actions, agents)
        
        # Check communication behavior
        print(f"Step {t+1}, Messages: {len(info['comms'])}")
        
        # Update agents
        for agent, o, r in zip(agents, next_obs, rewards):
            agent.update(o, r, done, info)
        
        obs = next_obs
        
        if done:
            break
    
    print(f"{agent_type.upper()} agent behavior check complete!")
    return agents

def run_test(agent_type):
    print(f"\nTESTING {agent_type.upper()} AGENTS...")
    start_time = time.time()
    
    # Create environment
    env = RescueEnv(
        grid_size=GRID_SIZE, 
        num_agents=NUM_AGENTS, 
        num_victims=NUM_VICTIMS, 
        num_obstacles=NUM_OBSTACLES, 
        max_steps=MAX_STEPS, 
        seed=SEED
    )
    
    # Create agents
    if agent_type == "independent":
        agents = [IndependentAgent(id, total_episodes=NUM_EPISODES) for id in range(NUM_AGENTS)]
    elif agent_type == "limited":
        agents = [LimitedAgent(id, comm_interval=5, total_episodes=NUM_EPISODES) for id in range(NUM_AGENTS)]
    elif agent_type == "selective":
        agents = [SelectiveAgent(id, comm_interval=25, total_episodes=NUM_EPISODES) for id in range(NUM_AGENTS)]
    elif agent_type == "full":
        agents = [FullAgent(id, total_episodes=NUM_EPISODES) for id in range(NUM_AGENTS)]
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Track metrics
    success_count = 0
    recent_td_errors = []
    episode_rewards = []
    episode_coverages = []
    episode_rescues = []
    communication_counts = []
    
    # For convergence detection
    convergence_window = 100
    convergence_threshold = 0.01
    converged = False
    convergence_episode = -1
    
    for episode in range(NUM_EPISODES):
        # Reset environment and agents
        obs = env.reset()
        for agent in agents:
            agent.reset()
        
        # Run episode
        done = False
        episode_td_errors = []
        episode_reward = 0
        
        while not done:
            # Get actions from all agents
            actions = [agent.act(o) for agent, o in zip(agents, obs)]
            
            # Step environment
            next_obs, rewards, done, info = env.step(actions, agents)
            
            # Update agents
            for agent, o, r in zip(agents, next_obs, rewards):
                td_error = agent.update(o, r, done, info)
                if td_error is not None:
                    episode_td_errors.append(abs(td_error))
            
            episode_reward += sum(rewards)
            obs = next_obs
        
        # Track metrics
        if info['rescued'] == NUM_VICTIMS:
            success_count += 1
        
        # Track TD errors for convergence detection
        avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
        recent_td_errors.append(avg_td_error)
        if len(recent_td_errors) > convergence_window:
            recent_td_errors.pop(0)
        
        # Track additional metrics
        episode_rewards.append(episode_reward)
        episode_coverages.append(info['coverage'])
        episode_rescues.append(info['rescued'])
        communication_counts.append(len(info['comms']))
        
        # Check for convergence
        if len(recent_td_errors) == convergence_window:
            avg_error = np.mean(recent_td_errors[-20:])  # Last 20 episodes
            if avg_error < convergence_threshold and not converged:
                converged = True
                convergence_episode = episode
                print(f"Q-learning appears to have converged at episode {episode+1}")
        
        # Display progress
        if (episode + 1) % 100 == 0 or episode == 0 or episode == NUM_EPISODES-1:
            success_rate = success_count / (episode + 1) * 100
            recent_td = np.mean(recent_td_errors[-min(100, len(recent_td_errors)):])
            recent_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
            print(f"Episode {episode+1}/{NUM_EPISODES}, TD Error: {recent_td:.5f}, "
                  f"Reward: {recent_reward:.2f}, Success Rate: {success_rate:.1f}%")
    
    elapsed_time = time.time() - start_time
    
    # Print final results
    print(f"\nRESULTS for {agent_type.upper()} agents after {elapsed_time:.1f} seconds:")
    print(f"Success Rate: {success_count/NUM_EPISODES*100:.1f}%")
    print(f"Final TD Error: {np.mean(recent_td_errors[-100:]):.5f}")
    print(f"Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Average Coverage: {np.mean(episode_coverages[-100:]):.2f}")
    print(f"Average Rescues: {np.mean(episode_rescues[-100:]):.2f}")
    print(f"Average Communications: {np.mean(communication_counts[-100:]):.2f}")
    
    for i, agent in enumerate(agents):
        print(f"Agent {i} Q-table size: {len(agent.q_table)} states, Bytes sent: {agent.bytes_sent}")
    
    if converged:
        print(f"Q-learning converged at episode {convergence_episode+1}")
    else:
        print("Q-learning did not converge within the episode limit")
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Moving averages
    window = 50
    rewards_ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    coverage_ma = np.convolve(episode_coverages, np.ones(window)/window, mode='valid')
    rescues_ma = np.convolve(episode_rescues, np.ones(window)/window, mode='valid')
    td_errors_ma = np.convolve(recent_td_errors, np.ones(window)/window, mode='valid')
    
    # Plot metrics
    axs[0, 0].plot(rewards_ma)
    axs[0, 0].set_title('Average Episode Reward')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    
    axs[0, 1].plot(coverage_ma)
    axs[0, 1].set_title('Average Coverage')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Coverage %')
    
    axs[1, 0].plot(rescues_ma)
    axs[1, 0].set_title('Average Victims Rescued')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_ylim(0, NUM_VICTIMS)
    
    axs[1, 1].plot(td_errors_ma)
    axs[1, 1].set_title('TD Error')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Error')
    
    plt.tight_layout()
    plt.savefig(f"{agent_type}_performance.png")
    print(f"Performance graph saved as {agent_type}_performance.png")
    
    return success_count/NUM_EPISODES*100, agents

def main():
    global NUM_EPISODES
    
    parser = argparse.ArgumentParser(description='Smoke test for SAR RL environment and agents')
    parser.add_argument('--agent', type=str, default='all', 
                        choices=['independent', 'limited', 'selective', 'full', 'all'],
                        help='Agent type to test (default: all)')
    parser.add_argument('--check-env', action='store_true', help='Run environment checks')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES, help='Number of episodes to run')
    args = parser.parse_args()
    
    NUM_EPISODES = args.episodes
    
    print("\n=== SAR RL Environment Smoke Test ===")
    
    # Environment checks
    if args.check_env:
        env = check_environment()
    
    results = {}
    
    if args.agent == 'all':
        for agent_type in ["independent", "limited", "selective", "full"]:
            check_agent_behavior(agent_type)
            success_rate, _ = run_test(agent_type)
            results[agent_type] = success_rate
    else:
        check_agent_behavior(args.agent)
        success_rate, _ = run_test(args.agent)
        results[args.agent] = success_rate
    
    # Compare results if multiple agents were tested
    if len(results) > 1:
        print("\n=== COMPARATIVE RESULTS ===")
        for agent_type, rate in results.items():
            print(f"{agent_type.upper()}: {rate:.1f}% success")

if __name__ == "__main__":
    main()