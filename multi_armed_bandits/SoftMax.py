from env.MAB import MAB
import numpy as np
import random

# Parameters
k_arms = 5  # Number of arms
episodes = 1000  # Number of iterations
initial_temp = 10.0  # Initial temperature
min_temp = 0.1  # Minimum temperature
decay_rate = 0.99  # Temperature decay rate

# Initialize the MAB environment
mab = MAB(k_arms)

# Initialize variables
action_values = np.zeros(k_arms)  # Estimated values for each arm
action_counts = np.zeros(k_arms)  # Counts of how many times each arm was pulled

# Softmax function for computing probabilities
def compute_softmax_probabilities(q_values, temp):
    scaled_q = q_values / temp  # Scale Q-values by temperature
    norm_q = scaled_q - np.max(scaled_q)  # Normalize for numeric stability
    exp_q = np.exp(norm_q)  # Exponentiate the normalized Q-values
    return exp_q / np.sum(exp_q)  # Return probabilities

# Run the softmax algorithm
temperature = initial_temp
for episode in range(episodes):
    # Compute action probabilities using softmax
    probabilities = compute_softmax_probabilities(action_values, temperature)
    
    # Select an action based on the probabilities
    arm = np.random.choice(np.arange(k_arms), p=probabilities)
    
    # Pull the arm and get the reward
    reward = mab.pull_arm(arm)
    
    # Update the action-value estimate using incremental formula
    action_counts[arm] += 1
    action_values[arm] += (reward - action_values[arm]) / action_counts[arm]
    
    # Decay the temperature
    temperature = max(min_temp, temperature * decay_rate)
    
    # Log the episode result
    print(f"Episode {episode + 1}: Pulled arm {arm}, received reward {reward}.")
    print(f"Action values: {action_values}")
    print("Action probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"probability ARM[{i}]: {prob:.6f}")

# Final results
print("Final action counts:")
for i, value in enumerate(action_counts):
    print(f"ARM_COUNTS[{i}]: {value:.2f}")
print("Final estimated action values:")
for i, value in enumerate(action_values):
    print(f"ARM_VALUES[{i}]: {value:.2f}")
print("ARM success probabilities:")
mab.print_arms()
