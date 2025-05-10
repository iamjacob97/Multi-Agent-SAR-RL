from env import RescueEnv
from agents import IndependentAgent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import time

# Configuration 
NUM_AGENTS = 3
NUM_VICTIMS = 7
NUM_OBSTACLES = 11
GRID_SIZE = (10, 10)
TOTAL_EPISODES = 357951
MAX_STEPS = 33
SEED = 42  
WINDOW_SIZE = 100  

def run_episode(env, agents, evaluation_mode=False):
    obs = env.reset()
    done = False
    total_reward = 0
    
    for agent in agents:
        agent.reset()
    
    while not done:
        if evaluation_mode:
            actions = []
            for agent, o in zip(agents, obs):
                original_epsilon = agent.epsilon
                agent.epsilon = 0
                action = agent.act(o)
                agent.epsilon = original_epsilon
                actions.append(action)
        else:
            actions = [agent.act(o) for agent, o in zip(agents, obs)]
            
        next_obs, rewards, done, info = env.step(actions, agents)
        
        for agent, o, r in zip(agents, next_obs, rewards):
            if not evaluation_mode:
                agent.update(o, r, done, info)
            total_reward += r

        obs = next_obs
    
    return {
        'total_reward': total_reward,
        'steps': info['steps'],
        'victims': info['rescued'],
        'coverage': info['coverage'],
        'success': info['rescued'] == env.num_victims,
        'q_table_size': agents[0].q_table_size  # Track Q-table size
    }

def main():
    print("Starting convergence test...")
    
    # Create environment and agent
    env = RescueEnv(
        grid_size=GRID_SIZE,
        num_agents=NUM_AGENTS,
        num_victims=NUM_VICTIMS,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        seed=SEED
    )
    
    agents = [IndependentAgent(i) for i in range(NUM_AGENTS)]
    
    # Setup logging
    training_log = []
    
    # Metrics for tracking rolling averages
    reward_window = deque(maxlen=WINDOW_SIZE)
    victims_window = deque(maxlen=WINDOW_SIZE)
    coverage_window = deque(maxlen=WINDOW_SIZE)
    q_table_window = deque(maxlen=WINDOW_SIZE)
    
    start_time = time.time()
    
    # Training loop
    for episode in range(1, TOTAL_EPISODES + 1):
        # Run training episode
        result = run_episode(env, agents)
        
        # Track metrics
        reward_window.append(result['total_reward'])
        victims_window.append(result['victims'])
        coverage_window.append(result['coverage'])
        q_table_window.append(result['q_table_size'])
        
        # Log every episode
        training_log.append({
            'episode': episode,
            'reward': result['total_reward'],
            'steps': result['steps'],
            'victims': result['victims'],
            'coverage': result['coverage'],
            'success': int(result['success']),
            'q_table_size': result['q_table_size']
        })
        
        # Print progress every 1000 episodes
        if episode % 1000 == 0:
            r_mean = np.mean(reward_window)
            v_mean = np.mean(victims_window)
            c_mean = np.mean(coverage_window)
            q_mean = np.mean(q_table_window)
            
            elapsed = time.time() - start_time
            eta = (elapsed / episode) * (TOTAL_EPISODES - episode)
            
            print(f"Episode {episode}/{TOTAL_EPISODES}  "
                  f"R mean={r_mean:.1f}  victims mean={v_mean:.1f}  "
                  f"coverage mean={c_mean:.2f}  Q-table mean={q_mean:.0f}  "
                  f"ETA {eta/60:.1f}m")
    
    # Save training log
    train_df = pd.DataFrame(training_log)
    train_df.to_csv("convergence_test_results.csv", index=False)
    
    # Generate convergence plots
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    window_size = 1000
    rewards_smooth = train_df['reward'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(train_df['episode'], rewards_smooth)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve: Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot victims rescued
    plt.subplot(2, 2, 2)
    victims_smooth = train_df['victims'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(train_df['episode'], victims_smooth)
    plt.xlabel('Episode')
    plt.ylabel('Victims Rescued')
    plt.title('Learning Curve: Victims Rescued')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot coverage
    plt.subplot(2, 2, 3)
    coverage_smooth = train_df['coverage'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(train_df['episode'], coverage_smooth)
    plt.xlabel('Episode')
    plt.ylabel('Coverage')
    plt.title('Learning Curve: Coverage')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Q-table size
    plt.subplot(2, 2, 4)
    q_table_smooth = train_df['q_table_size'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(train_df['episode'], q_table_smooth)
    plt.xlabel('Episode')
    plt.ylabel('Q-table Size')
    plt.title('Learning Curve: Q-table Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("convergence_test_plots.png", dpi=300)
    plt.close()
    
    print(f"\nConvergence test completed in {(time.time() - start_time)/60:.1f} minutes")
    print("Results saved to convergence_test_results.csv")
    print("Plots saved to convergence_test_plots.png")

if __name__ == "__main__":
    main() 