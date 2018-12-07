# CS238_Final_Project
CS238 Final Project -- Reinforcement Learning and Charitable Donations

# Steps: 
# 1. File 1 (final_data_generation.py): Generate data or incorporate real data
#     -- real data may have extra steps
#     -- Output: CSV
# 2. File 2 (final_pre_process.py): Pre-process data 
#     -- Generate three files -- one for each MDP
#     -- Input: raw data generated (csv)
#     -- Output: 3 CSV's
# 3. File 3 (final_Q_learn.py): Find optimal policy for each file
#     -- Input: Processed data files
#     -- Output: Three optimal policy text files
# 4. File 4 (final_policy_performance.py): Test each optimal policy with x random policies to compare efficiency
#     -- Input: Processed CSV's; Optimal policy files
#     -- Prints scores
