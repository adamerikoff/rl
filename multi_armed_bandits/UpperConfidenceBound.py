import numpy as np
from env.MAB import MAB

# Parameters
k_arms = 5  # Number of arms
episodes = 1000  # Number of rounds
confidence_param = 2  # Exploration parameter (c)

# Initialize environment and variables
mab = MAB(k_arms)
action_values = np.zeros(k_arms)  # Estimated rewards
action_counts = np.zeros(k_arms)  # Number of pulls for each arm

# Run UCB Algorithm
for t in range(1, episodes + 1):
    # Calculate UCB scores for each arm
    ucb_scores = action_values + confidence_param * np.sqrt(np.log(t) / (action_counts + 1e-5))  # Add small value to avoid division by zero
    
    # Select the arm with the highest UCB score
    arm = np.argmax(ucb_scores)
    
    # Pull the arm and observe the reward
    reward = mab.pull_arm(arm)
    
    # Update action values and counts
    action_counts[arm] += 1
    action_values[arm] += (reward - action_values[arm]) / action_counts[arm]
    
    # Log progress
    print(f"Round {t}: Pulled arm {arm}, received reward {reward}.")
    print(f"Action values: {action_values}\n")

# Final Results
print("Final action counts:", action_counts)
print("Final action values:", action_values)
mab.print_arms()
