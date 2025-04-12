Multi-Agent Search and Rescue Reinforcement Learning

This project explores how the frequency and type of communication affect search efficiency, redundancy, and adaptability in multi-agent search-and-rescue operations. The environment is a grid-based disaster zone, where multiple autonomous agents must navigate obstacles, and locate, and rescue targets. The agents are AI-driven search-and-rescue units using different strategies:

1. Independent Agents – No information sharing. 
2. Limited Communication – Agents share data at fixed intervals. 
3. Selective Communication – Only critical information is shared (e.g., closest victim). 
4. Full Communication – Agents continuously share all discoveries. 

The key research question is: What is the optimal balance of communication to maximize efficiency while minimizing redundant searches? To address this, experiments will compare the four communication strategies using reinforcement learning (Q-learning or PPO). Performance will be evaluated based on rescue time, search coverage, and adaptability to dynamic conditions. Results will be quantified and visually demonstrated through graphs, heatmaps, and animations, providing insights into AI-driven multi-agent collaboration.