from env.MAB import MAB
import random
import numpy as np

# Parameters
k_arms = 5  # Number of arms
episodes = 1000  # Number of episodes
initial_value = 0.0  # Initial action value estimate for each arm

# Initialize the MAB environment
mab = MAB(k_arms)

# Initialize variables
action_values = np.zeros(k_arms)  # Estimated values for each arm
action_counts = np.zeros(k_arms)  # Counts of how many times each arm was pulled
returns = np.zeros(k_arms)  # Sum of rewards for each arm (to calculate the average reward)

# Run the Monte Carlo algorithm
for episode in range(episodes):
    # Choose an action using exploration
    arm = random.randint(0, k_arms - 1)  # Random action selection
    
    # Pull the arm and get the reward
    reward = mab.pull_arm(arm)
    
    # Update the sum of rewards for the chosen arm
    returns[arm] += reward
    action_counts[arm] += 1
    
    # Update the action value (average reward for the arm)
    action_values[arm] = returns[arm] / action_counts[arm]  # Monte Carlo update
    
    # Log the episode result
    print(f"Episode {episode + 1}: Pulled arm {arm}, received reward {reward}.")
    print(f"Action values: {action_values}\n")

# Final results
print("Final action counts:")
for i, value in enumerate(action_counts):
    print(f"ARM_COUNTS[{i}]: {value:.2f}")

print("Final estimated action values:")
for i, value in enumerate(action_values):
    print(f"ARM_VALUES[{i}]: {value:.2f}")

print("ARM success probabilities:")
mab.print_arms()
