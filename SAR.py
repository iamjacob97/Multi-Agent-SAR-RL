from env import RescueEnv
from agents import IndependentAgent, LimitedAgent, SelectiveAgent, FullAgent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import argparse
import seaborn as sns
from multiprocessing import Process, Manager
from scipy import stats
from collections import deque

# Configuration
NUM_AGENTS = 3
NUM_VICTIMS = 7
NUM_OBSTACLES = 11
GRID_SIZE = (10, 10)
TOTAL_EPISODES = 357951
MAX_STEPS = 33 
SEEDS = [1, 3, 5, 7, 9, 11, 13, 15, 17]  
AGENT_TYPES = ['independent', 'limited', 'selective', 'full']
EVAL_EPISODES = 100  
EVAL_EVERY = 1000  
LOG_DIR = "results"


def create_agents(agent_type, num_agents):
    agents = []
    
    for i in range(num_agents):
        if agent_type == 'independent':
            agents.append(IndependentAgent(i))
        elif agent_type == 'limited':
            agents.append(LimitedAgent(i, comm_interval=5))
        elif agent_type == 'selective':
            agents.append(SelectiveAgent(i, comm_interval=25))
        elif agent_type == 'full':
            agents.append(FullAgent(i))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
    return agents


def run_episode(env, agents, evaluation_mode=False):
    obs = env.reset()
    done = False
    total_reward = 0
    
    for agent in agents:
        agent.reset()
    
    # Store initial bytes count to track new bytes sent during episode
    initial_bytes = sum(agent.bytes_sent for agent in agents)
    
    while not done:
        # In evaluation mode, force epsilon to 0 for pure exploitation
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
    
    # Calculate duplicate visits (redundancy metric)
    duplicates = sum(count-1 for count in env.global_visit_count.values() if count > 1)
    
    # Calculate bytes sent during this episode
    current_bytes = sum(agent.bytes_sent for agent in agents)
    bytes_sent = current_bytes - initial_bytes
    
    return {'total_reward': total_reward, 'steps': info['steps'], 'victims': info['rescued'], 
            'duplicates': duplicates, 'bytes': bytes_sent, 'coverage': info['coverage'],
            'success': info['rescued'] == env.num_victims}

def run_evaluation(env, agents, num_episodes):
    success_count = 0
    steps_list = []
    
    for _ in range(num_episodes):
        result = run_episode(env, agents, evaluation_mode=True)
        if result['success']:
            success_count += 1
        steps_list.append(result['steps'])
    
    success_rate = success_count / num_episodes
    steps_mean = np.mean(steps_list) if steps_list else 0
    
    return {'eval_success_rate': success_rate,'eval_steps_mean': steps_mean}

def train_and_evaluate(agent_type, seed, shared_results):
    print(f"[{agent_type} | seed {seed}] Starting training...")
    
    # Create environment and agents
    env = RescueEnv(grid_size=GRID_SIZE, num_agents=NUM_AGENTS, num_victims=NUM_VICTIMS,
                    num_obstacles=NUM_OBSTACLES, max_steps=MAX_STEPS, seed=seed)
    
    agents = create_agents(agent_type, NUM_AGENTS)
    
    # Setup logging
    os.makedirs(LOG_DIR, exist_ok=True)
    
    training_log = []
    eval_log = []
    
    # Metrics for tracking rolling averages using deques
    reward_window = deque(maxlen=100)
    victims_window = deque(maxlen=100)
    duplicates_window = deque(maxlen=100)
    bytes_window = deque(maxlen=100)
    
    start_time = time.time()
    
    # Training loop
    for episode in range(1, TOTAL_EPISODES + 1):
        # Run training episode
        result = run_episode(env, agents)
        
        # Track metrics
        reward_window.append(result['total_reward'])
        victims_window.append(result['victims'])
        duplicates_window.append(result['duplicates'])
        bytes_window.append(result['bytes'])
        
        # Log every episode
        training_log.append({'episode': episode, 'agent_type': agent_type, 'seed': seed,
                            'reward': result['total_reward'], 'steps': result['steps'],
                            'victims': result['victims'], 'duplicates': result['duplicates'],
                            'bytes': result['bytes'], 'coverage': result['coverage'],
                            'success': int(result['success'])})
        
        # Run evaluation every EVAL_EVERY episodes
        if episode % EVAL_EVERY == 0 or episode == TOTAL_EPISODES:
            eval_result = run_evaluation(env, agents, EVAL_EPISODES)
            
            eval_log.append({'episode': episode, 'agent_type': agent_type, 'seed': seed,
                            'eval_success_rate': eval_result['eval_success_rate'],
                            'eval_steps_mean': eval_result['eval_steps_mean']})
        
        # Print progress every 1000 episodes
        if episode % 1000 == 0:
            r_mean = np.mean(reward_window)
            v_mean = np.mean(victims_window)
            d_mean = np.mean(duplicates_window)
            b_mean = np.mean(bytes_window)
            
            elapsed = time.time() - start_time
            eta = (elapsed / episode) * (TOTAL_EPISODES - episode)
            
            print(f"[{agent_type} | seed {seed}] ep {episode}/{TOTAL_EPISODES}  "
                  f"R mean={r_mean:.1f}  victims mean={v_mean:.1f}  dupes mean={d_mean:.1f}  "
                  f"bytes mean={b_mean:.1f}  ETA {eta/60:.1f}m")
    
    # Save training and evaluation logs
    train_df = pd.DataFrame(training_log)
    train_df.to_csv(f"{LOG_DIR}/{agent_type}_seed{seed}_training.csv", index=False)
    
    eval_df = pd.DataFrame(eval_log)
    eval_df.to_csv(f"{LOG_DIR}/{agent_type}_seed{seed}_eval.csv", index=False)
    
    print(f"[{agent_type} | seed {seed}] Training completed in {(time.time() - start_time)/60:.1f} minutes")
    
    # Store results in shared dictionary for analysis
    shared_results[(agent_type, seed)] = {'train_df': train_df, 'eval_df': eval_df}

def generate_learning_curves(results):
    # Combine all training data
    all_training = []
    for (agent_type, seed), data in results.items():
        df = data['train_df'].copy()
        df['agent_seed'] = f"{agent_type}_{seed}"
        all_training.append(df)
    
    combined_df = pd.concat(all_training)
    
    # Create smoothed curves using rolling mean
    plt.figure(figsize=(15, 10))
    
    # First subplot for rewards
    plt.subplot(2, 1, 1)
    for agent_type in AGENT_TYPES:
        agent_data = combined_df[combined_df['agent_type'] == agent_type]
        
        # Group by episode and calculate mean and CI across seeds
        episode_groups = agent_data.groupby('episode')
        means = episode_groups['reward'].mean()
        ci = 1.96 * episode_groups['reward'].std() / np.sqrt(len(SEEDS))
        
        # Apply rolling window for smoothing
        window_size = 500
        means_smooth = means.rolling(window=window_size, min_periods=1).mean()
        ci_smooth = ci.rolling(window=window_size, min_periods=1).mean()
        
        # Plot mean and confidence interval
        episodes = means_smooth.index
        plt.plot(episodes, means_smooth, label=agent_type.capitalize())
        plt.fill_between(episodes, means_smooth - ci_smooth, means_smooth + ci_smooth, alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curves: Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Second subplot for victims rescued
    plt.subplot(2, 1, 2)
    for agent_type in AGENT_TYPES:
        agent_data = combined_df[combined_df['agent_type'] == agent_type]
        
        # Group by episode and calculate mean and CI across seeds
        episode_groups = agent_data.groupby('episode')
        means = episode_groups['victims'].mean()
        ci = 1.96 * episode_groups['victims'].std() / np.sqrt(len(SEEDS))
        
        # Apply rolling window for smoothing
        window_size = 500
        means_smooth = means.rolling(window=window_size, min_periods=1).mean()
        ci_smooth = ci.rolling(window=window_size, min_periods=1).mean()
        
        # Plot mean and confidence interval
        episodes = means_smooth.index
        plt.plot(episodes, means_smooth, label=agent_type.capitalize())
        plt.fill_between(episodes, means_smooth - ci_smooth, means_smooth + ci_smooth, alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel('Victims Rescued')
    plt.title('Learning Curves: Victims Rescued')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{LOG_DIR}/learn_curves.png", dpi=300)
    plt.close()

def generate_efficiency_bars(results):
    # Extract final performance metrics from training data
    summary_data = []
    
    for (agent_type, seed), data in results.items():
        # Get last 5000 episodes for a stable estimate
        df = data['train_df'].tail(5000)
        
        summary_data.append({'agent_type': agent_type, 'seed': seed, 'steps_mean': df['steps'].mean(),
                            'duplicates_mean': df['duplicates'].mean(),'bytes_mean': df['bytes'].mean(),})
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create bar plots
    plt.figure(figsize=(15, 6))
    
    # Setup plot style
    sns.set_style("whitegrid")
    colors = sns.color_palette("muted", len(AGENT_TYPES))
    
    metrics = ['steps_mean', 'duplicates_mean', 'bytes_mean']
    titles = ['Average Steps per Episode', 'Average Duplicate Visits', 'Average Bytes Sent']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i+1)
        
        # Calculate mean and confidence intervals per agent type
        agent_stats = []
        for j, agent_type in enumerate(AGENT_TYPES):
            agent_data = summary_df[summary_df['agent_type'] == agent_type]
            mean = agent_data[metric].mean()
            ci = 1.96 * agent_data[metric].std() / np.sqrt(len(agent_data))
            agent_stats.append((mean, ci))
        
        # Plot bars with error bars
        x = np.arange(len(AGENT_TYPES))
        means = [stat[0] for stat in agent_stats]
        errors = [stat[1] for stat in agent_stats]
        
        plt.bar(x, means, yerr=errors, color=colors, alpha=0.7, capsize=10,
                width=0.6, edgecolor='black', linewidth=1.5)
        
        plt.xticks(x, [a.capitalize() for a in AGENT_TYPES])
        plt.title(title)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
        plt.tight_layout()
    plt.savefig(f"{LOG_DIR}/bars_efficiency.png", dpi=300)
    plt.close()
    

def generate_pareto_plot(results):
    # Extract final performance metrics
    pareto_data = []
    
    for (agent_type, seed), data in results.items():
        # Get last 5000 episodes
        df = data['train_df'].tail(5000)
        
        pareto_data.append({'agent_type': agent_type, 'seed': seed, 'reward': df['reward'].mean(),
                            'bytes': df['bytes'].mean()})
    
    pareto_df = pd.DataFrame(pareto_data)
    
    # Create Pareto plot
    plt.figure(figsize=(10, 8))
    
    # Setup markers and colors
    markers = {'independent': 'o', 'limited': 's', 'selective': '^', 'full': 'D'}
    colors = {'independent': 'blue', 'limited': 'green', 'selective': 'orange', 'full': 'red'}
    
    # Plot each agent type
    for agent_type in AGENT_TYPES:
        agent_data = pareto_df[pareto_df['agent_type'] == agent_type]
        plt.scatter(agent_data['bytes'], agent_data['reward'], 
                   marker=markers[agent_type], color=colors[agent_type], s=100,
                   label=agent_type.capitalize(), edgecolors='black', alpha=0.7)
    
    # Add labels and annotations for each point
    for idx, row in pareto_df.iterrows():
        plt.annotate(f"Seed {row['seed']}", 
                    (row['bytes'], row['reward']),
                    xytext=(7, 7), textcoords='offset points',
                    fontsize=8)
    
    # Calculate average per strategy and add larger markers
    for agent_type in AGENT_TYPES:
        agent_data = pareto_df[pareto_df['agent_type'] == agent_type]
        avg_bytes = agent_data['bytes'].mean()
        avg_reward = agent_data['reward'].mean()
        plt.scatter(avg_bytes, avg_reward, 
                   marker=markers[agent_type], color=colors[agent_type], s=200,
                   edgecolors='black', linewidth=2)
        plt.annotate(f"{agent_type.capitalize()} Avg", 
                    (avg_bytes, avg_reward),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')
        
    plt.xlabel('Average Bytes Sent per Episode')
    plt.ylabel('Average Reward per Episode')
    plt.title('Pareto Frontier: Reward vs. Communication Cost')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{LOG_DIR}/pareto_reward_bytes.png", dpi=300)
    plt.close()
    

def calculate_statistics(results):
    # Extract final performance metrics
    final_metrics = []
    
    for (agent_type, seed), data in results.items():
        # Get last 5000 episodes
        df = data['train_df'].tail(5000)
        
        final_metrics.append({'agent_type': agent_type, 'seed': seed, 'reward': df['reward'].mean(),
                            'steps': df['steps'].mean(), 'victims': df['victims'].mean(),
                            'duplicates': df['duplicates'].mean(), 'bytes': df['bytes'].mean(),
                            'coverage': df['coverage'].mean(), 'success_rate': df['success'].mean()})
    
    metrics_df = pd.DataFrame(final_metrics)
    
    # Create summary table with one row per strategy×seed
    summary_df = metrics_df.copy()
    summary_df.to_csv(f"{LOG_DIR}/summary_table.csv", index=False)
    
    # Open a text file for statistics
    with open(f"{LOG_DIR}/stats.txt", "w") as f:
        f.write("STATISTICAL ANALYSIS\n")
        f.write("===================\n\n")
        
        # Compare each pair of agent types
        metrics_to_compare = ['reward', 'steps', 'duplicates', 'bytes', 'coverage', 'success_rate']
        
        for i, agent1 in enumerate(AGENT_TYPES):
            for j, agent2 in enumerate(AGENT_TYPES[i+1:], i+1):
                f.write(f"Comparison: {agent1.capitalize()} vs {agent2.capitalize()}\n")
                f.write("-" * 50 + "\n")
                
                for metric in metrics_to_compare:
                    # Get data for both agent types
                    data1 = metrics_df[metrics_df['agent_type'] == agent1][metric].values
                    data2 = metrics_df[metrics_df['agent_type'] == agent2][metric].values
                    
                    # Perform paired t-test
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    
                    # Calculate Cohen's d effect size
                    mean1, mean2 = np.mean(data1), np.mean(data2)
                    pooled_std = np.sqrt((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2)
                    cohen_d = (mean2 - mean1) / pooled_std if pooled_std != 0 else 0
                    
                    # Interpret effect size
                    if abs(cohen_d) < 0.2:
                        effect = "negligible"
                    elif abs(cohen_d) < 0.5:
                        effect = "small"
                    elif abs(cohen_d) < 0.8:
                        effect = "medium"
                    else:
                        effect = "large"
                    
                    # Determine which is better (depends on the metric)
                    if metric in ['reward', 'victims', 'coverage', 'success_rate']:
                        better = agent1 if mean1 > mean2 else agent2
                    else:  # For 'steps', 'duplicates', 'bytes', lower is better
                        better = agent1 if mean1 < mean2 else agent2
                    
                    # Write results
                    f.write(f"Metric: {metric}\n")
                    f.write(f"  {agent1}: {mean1:.2f}, {agent2}: {mean2:.2f}\n")
                    f.write(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}\n")
                    f.write(f"  Cohen's d: {cohen_d:.4f} ({effect} effect)\n")
                    f.write(f"  Better strategy: {better}\n\n")
                
                f.write("\n")
        
    return summary_df


def main():
    global LOG_DIR, TOTAL_EPISODES
    
    parser = argparse.ArgumentParser(description='Search and Rescue Multi-Agent Training')
    parser.add_argument('--agents', nargs='+', choices=AGENT_TYPES + ['all'], default=['all'],
                        help='Agent types to train')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Random seeds to use')
    parser.add_argument('--episodes', type=int, default=TOTAL_EPISODES,
                        help='Number of training episodes')
    parser.add_argument('--logdir', type=str, default=LOG_DIR,
                        help='Directory to save logs and plots')
    
    args = parser.parse_args()
    
    # Set global variables
    LOG_DIR = args.logdir
    TOTAL_EPISODES = args.episodes
    
    # Determine which agents to train
    agent_types = AGENT_TYPES if 'all' in args.agents else args.agents
    seeds = args.seeds
    
    print(f"Starting SAR experiment with agent types: {agent_types}")
    print(f"Seeds: {seeds}")
    print(f"Episodes: {TOTAL_EPISODES}")
    print(f"Log directory: {LOG_DIR}")
    
    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Use multiprocessing to run experiments in parallel
    manager = Manager()
    shared_results = manager.dict()
    processes = []
    
    # Create a process for each agent_type × seed combination
    for agent_type in agent_types:
        for seed in seeds:
            p = Process(target=train_and_evaluate, args=(agent_type, seed, shared_results))
            processes.append(p)
            p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All training processes completed. Generating analysis...")
    
    # Generate analysis plots and statistics
    generate_learning_curves(shared_results)
    generate_efficiency_bars(shared_results)
    generate_pareto_plot(shared_results)
    summary_df = calculate_statistics(shared_results)
    
    print(f"Analysis complete. Results saved to {LOG_DIR}/")


if __name__ == "__main__":
    main()
