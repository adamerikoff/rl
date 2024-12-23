import matplotlib.pyplot as plt
from env.FrozenLake import FrozenLake 
import random
import numpy as np

grid = [
    ["S", "F", "F", "F", "F", "F", "F", "F"],
    ["F", "H", "F", "F", "F", "F", "F", "F"],
    ["F", "F", "F", "F", "F", "F", "F", "F"],
    ["F", "F", "F", "H", "F", "F", "F", "F"],
    ["F", "F", "F", "F", "F", "F", "F", "F"],
    ["F", "F", "F", "F", "F", "H", "F", "F"],
    ["F", "F", "F", "F", "F", "F", "F", "F"],
    ["F", "F", "F", "F", "F", "F", "F", "G"]
]
start = (0, 0)
goal = (7, 7)
holes = {
    (1, 1), (3, 3), (5, 5), 
    (2, 6), (4, 2), (6, 4), 
    (0, 7), (7, 0) 
}

# Create the environment
env = FrozenLake(grid, start, goal, holes)

# Parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 10000
max_timesteps = 50

# Action mapping
actions = ["up", "down", "left", "right"]
num_actions = len(actions)

# Initialize Q-table
state_space = (len(grid), len(grid[0]))
Q = np.zeros(state_space + (num_actions,))

# Tracking metrics
cumulative_rewards = []  # To track rewards per episode
errors = []  # To track Q-value updates

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    timestep = 1

    while not done and timestep <= max_timesteps:
        # Choose action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(num_actions)
        else:
            action_idx = np.argmax(Q[state])

        # Take action
        next_state, reward, done = env.step(actions[action_idx])
        total_reward += reward

        # Penalize for exceeding time limit
        if timestep == max_timesteps and not done:
            reward = -1  # Apply a penalty for not reaching the goal within the limit

        # Track Q-value update (error)
        old_value = Q[state][action_idx]
        new_value = old_value + alpha * (
            reward + gamma * np.max(Q[next_state]) - old_value
        )
        errors.append(abs(new_value - old_value))

        # Update Q-value
        Q[state][action_idx] = new_value

        # Move to the next state
        state = next_state
        timestep += 1

    cumulative_rewards.append(total_reward)  # Track cumulative reward for this episode

print("Training complete. Final Q-Table:")
print(Q)

# Plot cumulative rewards and Q-value update errors
plt.figure(figsize=(25, 6))

# Cumulative rewards plot
plt.subplot(1, 2, 1)
plt.plot(cumulative_rewards)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards Over Episodes")

# Q-value update errors plot
plt.subplot(1, 2, 2)
plt.plot(errors)
plt.xlabel("Updates")
plt.ylabel("Q-value Update Error")
plt.title("Q-value Update Error Over Time")

plt.tight_layout()
plt.show()

# Test the trained Q-table
print("\n\nUsing the trained Q-table for a single episode:")
state = env.reset()
done = False
timestep = 1

while not done and timestep <= max_timesteps:
    # Choose action based on the learned Q-table (greedy)
    action_idx = np.argmax(Q[state]) 
    action = actions[action_idx]

    # Take action
    next_state, reward, done = env.step(action)

    # Print information
    print(f"TimeStep: {timestep}, Action: {action}, State: {state}, Possible Actions: {actions},\n Q-Values: {Q[state]}, Reward: {reward}")

    # Move to the next state
    state = next_state
    timestep += 1

    # Render the environment (optional)
    env.render()
