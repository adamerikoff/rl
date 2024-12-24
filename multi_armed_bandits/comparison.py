from env.MAB import MAB
import random
import numpy as np
import matplotlib.pyplot as plt

# Parameters
k_arms = 100
episodes = 1000
optimistic_value = 5.0
initial_epsilon = 1.0
decay_rate = 0.99
min_epsilon = 0.01

# Initialize the MAB environment
mab = MAB(k_arms)
true_best_arm = max(mab.arms, key=mab.arms.get)  # Identify the true best arm

# Tracking metrics
def track_metrics(algorithm_name, arm_pulls, cumulative_rewards, regrets):
    print(f"\n{algorithm_name} Results:")
    print(f"Total Cumulative Reward: {cumulative_rewards[-1]:.2f}")
    print(f"Total Regret: {regrets[-1]:.2f}")
    print(f"Best Arm Selected Frequency: {arm_pulls[true_best_arm] / episodes:.2f}")
    plt.plot(cumulative_rewards, label=f"{algorithm_name} - Cumulative Reward")
    plt.plot(regrets, label=f"{algorithm_name} - Regret")

# Optimistic Initialization
action_values_optimistic = np.full(k_arms, optimistic_value)
action_counts_optimistic = np.zeros(k_arms)
cumulative_rewards_optimistic = []
regrets_optimistic = []
total_reward_optimistic = 0

for episode in range(episodes):
    arm = np.argmax(action_values_optimistic)
    reward = mab.pull_arm(arm)
    total_reward_optimistic += reward
    action_counts_optimistic[arm] += 1
    action_values_optimistic[arm] += (reward - action_values_optimistic[arm]) / action_counts_optimistic[arm]
    optimal_reward = mab.arms[true_best_arm]
    regret = (episode + 1) * optimal_reward - total_reward_optimistic
    cumulative_rewards_optimistic.append(total_reward_optimistic)
    regrets_optimistic.append(regret)

track_metrics("Optimistic Initialization", action_counts_optimistic, cumulative_rewards_optimistic, regrets_optimistic)

# ε-Greedy with Decaying ε
action_values_epsilon = np.zeros(k_arms)
action_counts_epsilon = np.zeros(k_arms)
epsilon = initial_epsilon
cumulative_rewards_epsilon = []
regrets_epsilon = []
total_reward_epsilon = 0

for episode in range(episodes):
    if random.random() < epsilon:
        arm = random.randint(0, k_arms - 1)
    else:
        arm = np.argmax(action_values_epsilon)
    reward = mab.pull_arm(arm)
    total_reward_epsilon += reward
    action_counts_epsilon[arm] += 1
    action_values_epsilon[arm] += (reward - action_values_epsilon[arm]) / action_counts_epsilon[arm]
    epsilon = max(min_epsilon, epsilon * decay_rate)
    optimal_reward = mab.arms[true_best_arm]
    regret = (episode + 1) * optimal_reward - total_reward_epsilon
    cumulative_rewards_epsilon.append(total_reward_epsilon)
    regrets_epsilon.append(regret)

track_metrics("ε-Greedy with Decaying ε", action_counts_epsilon, cumulative_rewards_epsilon, regrets_epsilon)

# Plot comparison
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward / Regret")
plt.title("Convergence Comparison: Optimistic Initialization vs ε-Greedy")
plt.legend()
plt.show()
