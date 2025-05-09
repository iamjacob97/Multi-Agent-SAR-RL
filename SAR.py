from env import RescueEnv
from agents import IndependentAgent, LimitedAgent, SelectiveAgent, FullAgent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import argparse
from datetime import datetime
import copy
from collections import deque

# Configuration
NUM_AGENTS = 3
NUM_VICTIMS = 7
NUM_OBSTACLES = 11
GRID_SIZE = (10, 10)
TOTAL_EPISODES = 17935
MAX_STEPS = 79
SEEDS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]  
EVAL_SEED = 10  # Fixed seed for evaluation
EVAL_EPISODES = 10101  # Number of evaluation episodes per agent type
AGENT_TYPES = ['independent', 'limited', 'selective', 'full']
LOG_DIR = "results"

def create_agents(agent_type, num_agents, total_episodes=TOTAL_EPISODES):
    agents = []
    
    for i in range(num_agents):
        if agent_type == 'independent':
            agents.append(IndependentAgent(i, total_episodes=total_episodes))
        elif agent_type == 'limited':
            agents.append(LimitedAgent(i, comm_interval=5, total_episodes=total_episodes))
        elif agent_type == 'selective':
            agents.append(SelectiveAgent(i, comm_interval=25, total_episodes=total_episodes))
        elif agent_type == 'full':
            agents.append(FullAgent(i, total_episodes=total_episodes))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
    return agents

def run_episode(env, agents, verbose=False, evaluation_mode=False):
    obs = env.reset()
    done = False
    total_reward = [0.0] * len(agents)
    ep_steps = 0
    comm_count = 0
    bytes_sent = 0
    
    for agent in agents:
        agent.reset()
    
    while not done:
        # In evaluation mode, force epsilon to 0 for pure exploitation
        if evaluation_mode:
            actions = []
            for ag, o in zip(agents, obs):
                # Store original epsilon
                original_epsilon = ag.epsilon
                ag.epsilon = 0
                action = ag.act(o)
                # Restore original epsilon after action selection
                ag.epsilon = original_epsilon
                actions.append(action)
        else:
            actions = [ag.act(o) for ag, o in zip(agents, obs)]
            
        next_obs, rewards, done, info = env.step(actions, agents)
        
        # Track communication
        comm_count += len(info['comms'])
        bytes_sent += sum(ag.bytes_sent for ag in agents)
        
        for i, (ag, o, rw) in enumerate(zip(agents, next_obs, rewards)):
            # Only update Q-values during training, not evaluation
            if not evaluation_mode:
                ag.update(o, rw, done, info)
            total_reward[i] += rw

        obs = next_obs
        ep_steps += 1
        
        if verbose and ep_steps % 10 == 0:
            env.render()
            
    # Calculate coverage and duplicate visits
    total_cells = env.grid_size[0] * env.grid_size[1]
    visited_cells_union = set().union(*env.visited_cells)
    coverage = len(visited_cells_union) / total_cells
    duplicate_visits = sum(v-1 for v in env.global_visit_count.values())
    
    return {
        'total_reward': sum(total_reward),
        'steps': ep_steps,
        'rescued': info['rescued'],
        'coverage': coverage,
        'comm_count': comm_count,
        'bytes_sent': bytes_sent,
        'duplicate_visits': duplicate_visits,
        'success': info['rescued'] == env.num_victims
    }

def run_experiment(agent_type, seed, log_dir, total_episodes=TOTAL_EPISODES):
    print(f"Running experiment with agent_type={agent_type}, seed={seed}")
    
    # Create environment and agents
    env = RescueEnv(
        grid_size=GRID_SIZE,
        num_agents=NUM_AGENTS,
        num_victims=NUM_VICTIMS,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        seed=seed
    )
    
    agents = create_agents(agent_type, NUM_AGENTS, total_episodes)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{agent_type}_seed{seed}_{timestamp}_training.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Initialize metrics tracking
    metrics = []
    eval_interval = 500  # Evaluate every 500 episodes to reduce data size
    
    # Progress tracking
    start_time = time.time()
    
    # Run episodes
    for episode in range(1, total_episodes + 1):
        # Run episode
        result = run_episode(env, agents)
        
        # Keep the last 500 episodes' results for rolling average
        if episode == 1:
            # Initialize deques to store metrics for rolling average
            window_size = 500
            rolling_results = {
                'total_reward': deque(maxlen=window_size),
                'steps': deque(maxlen=window_size),
                'rescued': deque(maxlen=window_size),
                'coverage': deque(maxlen=window_size),
                'comm_count': deque(maxlen=window_size),
                'bytes_sent': deque(maxlen=window_size),
                'duplicate_visits': deque(maxlen=window_size),
                'success_rate': deque(maxlen=window_size)
            }
        
        # Add current episode results to rolling window
        rolling_results['total_reward'].append(result['total_reward'])
        rolling_results['steps'].append(result['steps'])
        rolling_results['rescued'].append(result['rescued'])
        rolling_results['coverage'].append(result['coverage'])
        rolling_results['comm_count'].append(result['comm_count'])
        rolling_results['bytes_sent'].append(result['bytes_sent'])
        rolling_results['duplicate_visits'].append(result['duplicate_visits'])
        rolling_results['success_rate'].append(1 if result['success'] else 0)
        
        # Track metrics at regular intervals or at first/last episode
        if episode % eval_interval == 0 or episode == 1 or episode == total_episodes:
            print(f"Episode {episode}/{total_episodes}, Time elapsed: {(time.time() - start_time):.2f}s")
            
            # Calculate rolling averages
            rolling_avg = {key: sum(values) / len(values) for key, values in rolling_results.items()}
            
            metrics.append({
                'episode': episode,
                'seed': seed,
                'agent_type': agent_type,
                'total_reward': rolling_avg['total_reward'],
                'steps': rolling_avg['steps'],
                'rescued': rolling_avg['rescued'],
                'coverage': rolling_avg['coverage'],
                'comm_count': rolling_avg['comm_count'],
                'bytes_sent': rolling_avg['bytes_sent'],
                'duplicate_visits': rolling_avg['duplicate_visits'],
                'success_rate': rolling_avg['success_rate'],
                'epsilon': agents[0].epsilon,
                'alpha': agents[0].alpha
            })
            
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(log_path, index=False)
    print(f"Training results saved to {log_path}")
    
    # Return the trained agents for evaluation
    return agents

def run_evaluation(agent_type, trained_agents, log_dir):
    """Run evaluation episodes with fixed seed and epsilon=0"""
    print(f"Evaluating agent_type={agent_type} with epsilon=0")
    
    # Create a new environment with the evaluation seed
    env = RescueEnv(
        grid_size=GRID_SIZE,
        num_agents=NUM_AGENTS,
        num_victims=NUM_VICTIMS,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        seed=EVAL_SEED
    )
    
    # Make deep copies of agents to avoid modifying the originals
    agents = copy.deepcopy(trained_agents)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{agent_type}_eval_{timestamp}.csv")
    
    # Initialize metrics tracking
    metrics = []
    
    # Run evaluation episodes
    for episode in range(1, EVAL_EPISODES + 1):
        # Run episode with evaluation_mode=True to force epsilon=0
        result = run_episode(env, agents, evaluation_mode=True)
        
        metrics.append({
            'episode': episode,
            'agent_type': agent_type,
            'total_reward': result['total_reward'],
            'steps': result['steps'],
            'rescued': result['rescued'],
            'coverage': result['coverage'],
            'comm_count': result['comm_count'],
            'bytes_sent': result['bytes_sent'],
            'duplicate_visits': result['duplicate_visits'],
            'success_rate': 1 if result['success'] else 0,
            'epsilon': 0,  # Fixed at 0 for evaluation
        })
        
        if episode % 10 == 0:
            print(f"Evaluation episode {episode}/{EVAL_EPISODES}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(log_path, index=False)
    print(f"Evaluation results saved to {log_path}")
    
    return metrics_df

def plot_results(data, plots_dir, is_evaluation=False):
    """Generate plots comparing the communication strategies"""
    os.makedirs(plots_dir, exist_ok=True)
    
    title_prefix = "Evaluation " if is_evaluation else ""
    
    # Group data by agent type and episode, calculating means and standard deviations
    grouped = data.groupby(['agent_type', 'episode']).agg({
        'total_reward': ['mean', 'std'],
        'steps': ['mean', 'std'],
        'rescued': ['mean', 'std'],
        'coverage': ['mean', 'std'],
        'comm_count': ['mean', 'std'],
        'bytes_sent': ['mean', 'std'],
        'duplicate_visits': ['mean', 'std'],
        'success_rate': ['mean', 'std']
    }).reset_index()
    
    # Plot metrics over episodes
    metrics = {
        'total_reward': 'Total Reward',
        'steps': 'Episode Length',
        'rescued': 'Victims Rescued',
        'coverage': 'Environment Coverage',
        'comm_count': 'Communication Count',
        'bytes_sent': 'Bytes Sent',
        'duplicate_visits': 'Duplicate Visits',
        'success_rate': 'Success Rate'
    }
    
    for metric, title in metrics.items():
        plt.figure(figsize=(10, 6))
        
        for agent_type in AGENT_TYPES:
            # Get data for this agent type
            agent_data = grouped[grouped['agent_type'] == agent_type]
            
            # Skip if no data for this agent type
            if len(agent_data) == 0:
                continue
                
            # Plot mean with standard deviation
            plt.errorbar(
                agent_data['episode'], 
                agent_data[(metric, 'mean')],
                yerr=agent_data[(metric, 'std')],
                marker='o',
                label=agent_type
            )
        
        plt.title(f'{title_prefix}{title} vs Episodes')
        plt.xlabel('Episodes')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        file_prefix = "eval_" if is_evaluation else ""
        plt.savefig(os.path.join(plots_dir, f'{file_prefix}{metric}_vs_episodes.png'))
        plt.close()
    
    # Create bar plots for final performance
    final_results = data.groupby('agent_type').mean().reset_index()
    
    for metric, title in metrics.items():
        plt.figure(figsize=(10, 6))
        
        plt.bar(final_results['agent_type'], final_results[metric])
        plt.title(f'{title_prefix}Final {title} by Communication Strategy')
        plt.xlabel('Communication Strategy')
        plt.ylabel(title)
        plt.grid(axis='y')
        plt.tight_layout()
        file_prefix = "eval_" if is_evaluation else ""
        plt.savefig(os.path.join(plots_dir, f'{file_prefix}{metric}_final.png'))
        plt.close()
    
    # Create a combined metrics plot
    plt.figure(figsize=(12, 8))
    
    # Set final_results index to agent_type for easier access
    final_results.set_index('agent_type', inplace=True)
    
    # Plot relationship between communication amount and performance
    plt.scatter(final_results['bytes_sent'], final_results['rescued'] / final_results['steps'], 
                s=100, label='Rescue efficiency')
    plt.scatter(final_results['bytes_sent'], final_results['coverage'], 
                s=100, label='Coverage')
    plt.scatter(final_results['bytes_sent'], final_results['duplicate_visits'] / final_results['steps'], 
                s=100, label='Redundancy rate')
    
    # Add labels to points
    for idx in final_results.index:
        row = final_results.loc[idx]
        plt.annotate(idx, (row['bytes_sent'], row['rescued'] / row['steps']), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title(f'{title_prefix}Communication Amount vs Performance Metrics')
    plt.xlabel('Bytes Sent')
    plt.ylabel('Performance Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_prefix = "eval_" if is_evaluation else ""
    plt.savefig(os.path.join(plots_dir, f'{file_prefix}communication_vs_performance.png'))
    plt.close()
    
    # Create performance summary
    summary = final_results[['total_reward', 'steps', 'rescued', 'coverage', 
                       'comm_count', 'bytes_sent', 'duplicate_visits', 'success_rate']]
    summary['rescue_efficiency'] = summary['rescued'] / summary['steps']
    
    # Save summary to CSV
    file_prefix = "eval_" if is_evaluation else ""
    summary.to_csv(os.path.join(plots_dir, f'{file_prefix}performance_summary.csv'))
    
    print(f"{title_prefix}Plots and summary saved to {plots_dir}")
    print(f"\n{title_prefix}Performance Summary:")
    print(summary)
    
    return summary

def analyze_evaluation_results(eval_dir):
    """Analyze the evaluation results and generate a report"""
    # Get all evaluation CSV files
    all_files = [f for f in os.listdir(eval_dir) if f.endswith('.csv') and 'eval' in f]
    
    if not all_files:
        print("No evaluation results found to analyze")
        return
    
    # Load and combine data
    data_frames = []
    for file in all_files:
        data = pd.read_csv(os.path.join(eval_dir, file))
        data_frames.append(data)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    # Generate plots
    plots_dir = os.path.join(eval_dir, 'eval_plots')
    summary = plot_results(combined_data, plots_dir, is_evaluation=True)
    
    # Create a text report based on evaluation results
    report_path = os.path.join(eval_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("MULTI-AGENT COMMUNICATION STRATEGIES EVALUATION ANALYSIS\n")
        f.write("===================================================\n\n")
        
        f.write("PERFORMANCE SUMMARY (EPSILON=0)\n")
        f.write("----------------------------\n")
        f.write(summary.to_string())
        f.write("\n\n")
        
        # Add insights about each strategy
        f.write("COMMUNICATION STRATEGY INSIGHTS\n")
        f.write("-----------------------------\n")
        
        # Independent
        f.write("Independent Strategy (No Communication):\n")
        if 'independent' in summary.index:
            ind = summary.loc['independent']
            f.write(f"- Rescued {ind['rescued']:.2f} victims in {ind['steps']:.2f} steps\n")
            f.write(f"- Coverage: {ind['coverage']*100:.2f}%\n")
            f.write(f"- Duplicate visits: {ind['duplicate_visits']:.2f}\n")
            f.write(f"- No communication overhead ({ind['bytes_sent']:.2f} bytes)\n\n")
        
        # Limited
        f.write("Limited Strategy (Time-Based Communication):\n")
        if 'limited' in summary.index:
            lim = summary.loc['limited']
            ind = summary.loc['independent'] if 'independent' in summary.index else None
            f.write(f"- Rescued {lim['rescued']:.2f} victims in {lim['steps']:.2f} steps\n")
            f.write(f"- Coverage: {lim['coverage']*100:.2f}%\n")
            f.write(f"- Duplicate visits: {lim['duplicate_visits']:.2f}\n")
            f.write(f"- Communication overhead: {lim['bytes_sent']:.2f} bytes\n")
            if ind is not None:
                f.write(f"- Improvement over Independent: {(lim['rescue_efficiency']/ind['rescue_efficiency']-1)*100:.2f}% rescue efficiency\n\n")
            else:
                f.write("\n")
        
        # Selective
        f.write("Selective Strategy (Need-Based Communication):\n")
        if 'selective' in summary.index:
            sel = summary.loc['selective']
            ind = summary.loc['independent'] if 'independent' in summary.index else None
            f.write(f"- Rescued {sel['rescued']:.2f} victims in {sel['steps']:.2f} steps\n")
            f.write(f"- Coverage: {sel['coverage']*100:.2f}%\n")
            f.write(f"- Duplicate visits: {sel['duplicate_visits']:.2f}\n")
            f.write(f"- Communication overhead: {sel['bytes_sent']:.2f} bytes\n")
            if ind is not None:
                f.write(f"- Improvement over Independent: {(sel['rescue_efficiency']/ind['rescue_efficiency']-1)*100:.2f}% rescue efficiency\n\n")
            else:
                f.write("\n")
        
        # Full
        f.write("Full Strategy (Continuous Communication):\n")
        if 'full' in summary.index:
            ful = summary.loc['full']
            ind = summary.loc['independent'] if 'independent' in summary.index else None
            f.write(f"- Rescued {ful['rescued']:.2f} victims in {ful['steps']:.2f} steps\n")
            f.write(f"- Coverage: {ful['coverage']*100:.2f}%\n")
            f.write(f"- Duplicate visits: {ful['duplicate_visits']:.2f}\n")
            f.write(f"- Communication overhead: {ful['bytes_sent']:.2f} bytes\n")
            if ind is not None:
                f.write(f"- Improvement over Independent: {(ful['rescue_efficiency']/ind['rescue_efficiency']-1)*100:.2f}% rescue efficiency\n\n")
            else:
                f.write("\n")
        
        # Determine best strategy for different metrics
        f.write("OPTIMAL COMMUNICATION STRATEGY (BASED ON EVALUATION)\n")
        f.write("---------------------------------------------\n")
        
        best_reward = summary['total_reward'].idxmax()
        best_efficiency = summary['rescue_efficiency'].idxmax()
        best_coverage = summary['coverage'].idxmax()
        least_redundancy = summary['duplicate_visits'].idxmin()
        best_bytes_efficiency = (summary['total_reward'] / summary['bytes_sent'].replace(0, 1)).idxmax()
        
        f.write(f"Best for overall reward: {best_reward}\n")
        f.write(f"Best for rescue efficiency: {best_efficiency}\n")
        f.write(f"Best for environment coverage: {best_coverage}\n")
        f.write(f"Best for minimizing redundancy: {least_redundancy}\n")
        f.write(f"Best reward-to-communication ratio: {best_bytes_efficiency}\n\n")
        
        f.write("CONCLUSION (BASED ON EVALUATION WITH EPSILON=0)\n")
        f.write("-------------------------------------------\n")
        f.write("The optimal balance of communication depends on specific priorities:\n")
        f.write("- If minimizing communication overhead is critical: Independent strategy\n")
        f.write("- If maximizing rescue speed is the priority: Full strategy\n")
        f.write("- If balancing efficiency and communication cost is desired: Selective or Limited strategy\n")
        f.write("\nFor real-world applications, these insights can guide the development of adaptive communication\n")
        f.write("strategies that adjust based on environment complexity and mission requirements.\n")
    
    print(f"Evaluation report generated at {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Search and Rescue Experiment Runner")
    global EVAL_EPISODES
    
    parser.add_argument('--episodes', type=int, default=TOTAL_EPISODES, help='Number of episodes to run')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS, help='Seeds to use for experiments')
    parser.add_argument('--types', type=str, nargs='+', default=AGENT_TYPES, help='Agent types to evaluate')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze existing results')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation on trained agents')
    parser.add_argument('--eval_episodes', type=int, default=EVAL_EPISODES, help='Number of evaluation episodes')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='Directory for results')
    args = parser.parse_args()
    
    # Update global variables based on arguments
    EVAL_EPISODES = args.eval_episodes
    
    # Create results directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    if not args.analyze_only and not args.eval_only:
        start_time = time.time()
        
        # Dictionary to store trained agents for each agent type
        all_trained_agents = {}
        
        # Run training experiments for each agent type and seed
        for agent_type in args.types:
            # Use first seed for the agent that will be used in evaluation
            trained_agents = run_experiment(agent_type, args.seeds[0], args.log_dir, args.episodes)
            all_trained_agents[agent_type] = trained_agents
            
            # Run the remaining seeds for better training statistics
            for seed in args.seeds[1:]:
                run_experiment(agent_type, seed, args.log_dir, args.episodes)
        
        total_time = time.time() - start_time
        print(f"All training experiments completed in {total_time/60:.2f} minutes")
        
        # Run evaluation phase for each agent type
        for agent_type, trained_agents in all_trained_agents.items():
            run_evaluation(agent_type, trained_agents, args.log_dir)
        
        # Analyze evaluation results
        analyze_evaluation_results(args.log_dir)
    
    elif args.eval_only:
        # This section would need access to already trained agents
        # In a real implementation, you would need to load them from files
        print("Evaluation-only mode requires trained agents. Please run training first.")
    
    else:
        # Analyze existing evaluation results
        analyze_evaluation_results(args.log_dir)

if __name__ == "__main__":
    main()
