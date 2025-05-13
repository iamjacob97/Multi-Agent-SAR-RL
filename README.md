# Multi-Agent Search and Rescue Reinforcement Learning

This project explores how the frequency and type of communication affect search efficiency, redundancy, and adaptability in multi-agent search-and-rescue operations. The environment is a grid-based disaster zone, where multiple autonomous agents must navigate obstacles, and locate, and rescue targets. The agents are AI-driven search-and-rescue units using different strategies:

1. Independent Agents – No information sharing. 
2. Limited Communication – Agents share data at fixed intervals. 
3. Selective Communication – Only share data for new discoveries and rescues. 
4. Full Communication – Agents continuously share all discoveries. 

The key research question is: What is the optimal balance of communication to maximize efficiency while minimizing redundant searches? To address this, experiments will compare the four communication strategies using reinforcement learning (Q-learning). Performance will be evaluated based on rescue time, search coverage, and adaptability to dynamic conditions. Results will be quantified and visually demonstrated through graphs, providing insights into AI-driven multi-agent collaboration.

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/Multi-Agent-SAR-RL.git
cd Multi-Agent-SAR-RL


2. Install the required dependencies:

pip install -r requirements.txt


## Dependencies

- numpy>=1.21.0 - For numerical operations and array handling
- pandas>=2.0.0 - For data manipulation and analysis
- matplotlib>=3.5.0 - For plotting results and metrics
- seaborn>=0.12.0 - For enhanced statistical visualizations
- scipy>=1.10.0 - For scientific computing and statistics
- pygame>=2.5.0 - For visualization and simulation


## Project Structure

- `env/` - Environment implementation
  - `rescue_env.py` - Main environment class
- `agents/` - Agent implementations
  - `independent_agent.py` - Independent agent implementation
  - `limited_agent.py` - Limited communication agent
  - `selective_agent.py` - Selective communication agent
  - `full_agent.py` - Full communication agent
- `utils/` - Utility functions
  - `graphics.py` - Visualization utilities
- `SAR.py` - Main training script
- `smoke_test.py` - Testing script
- `convergence_test.py` - Convergence analysis script

## Results

Results are saved in the `results/` directory, including:
- Learning curves
- Efficiency metrics
- Statistical analysis
- Visualization plots


