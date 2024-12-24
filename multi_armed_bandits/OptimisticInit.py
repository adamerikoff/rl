from env.MAB import MAB
import random
import numpy as np

# Parameters
k_arms = 5  # Number of arms
episodes = 1000  # Number of iterations
optimistic_value = 5.0  # Optimistic initial value for action estimates

# Initialize the MAB environment
mab = MAB(k_arms)

# Initialize variables
action_values = np.full(k_arms, optimistic_value)  # Optimistic initial values for each arm
action_counts = np.zeros(k_arms)  # Counts of how many times each arm was pulled

# Run the optimistic initialization algorithm
for episode in range(episodes):
    # Choose an action (always exploit, as optimistic initialization drives exploration naturally)
    arm = np.argmax(action_values)
    
    # Pull the arm and get the reward
    reward = mab.pull_arm(arm)
    
    # Update the action-value estimate using incremental formula
    action_counts[arm] += 1
    action_values[arm] += (reward - action_values[arm]) / action_counts[arm]
    
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
