from env.MAB import MAB
import random
import numpy as np

# Parameters
k_arms = 5  # Number of arms
episodes = 1000  # Number of iterations
initial_epsilon = 1.0  # Initial exploration rate
decay_rate = 0.99  # Epsilon decay rate
min_epsilon = 0.01  # Minimum epsilon value

# Initialize the MAB environment
mab = MAB(k_arms)

# Initialize variables
action_values = np.zeros(k_arms)  # Estimated values for each arm
action_counts = np.zeros(k_arms)  # Counts of how many times each arm was pulled
epsilon = initial_epsilon

# Run the greedy exploration with decaying epsilon
for episode in range(episodes):
    # Choose an action
    if random.random() < epsilon:  # Explore
        arm = random.randint(0, k_arms - 1)
    else:  # Exploit
        arm = np.argmax(action_values)
    
    # Pull the arm and get the reward
    reward = mab.pull_arm(arm)
    
    # Update the action-value estimate using incremental formula
    action_counts[arm] += 1
    action_values[arm] += (reward - action_values[arm]) / action_counts[arm]
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * decay_rate)
    
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
