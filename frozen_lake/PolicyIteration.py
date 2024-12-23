import numpy as np
from env.FrozenLake import FrozenLake

actions = ["up", "down", "left", "right"]
moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def evaluate_policy(env, policy, gamma=0.99, theta=1e-6):
    V = np.zeros(len(env.grid) * len(env.grid[0]))

    while True:
        delta = 0
        for i in range(len(env.grid)):
            for j in range(len(env.grid[0])):
                state = (i, j)
                state_idx = i * len(env.grid[0]) + j

                # Skip terminal states
                if state == env.goal or state in env.holes:
                    continue

                v = V[state_idx]
                action_idx = policy[state_idx]
                action = actions[action_idx]  # Convert index to action name
                move = moves[action]  # Now we use the correct action name

                # Compute next state directly based on the action
                next_state = (state[0] + move[0], state[1] + move[1])
                if not (0 <= next_state[0] < len(env.grid) and 0 <= next_state[1] < len(env.grid[0])):
                    next_state = state  # Stay in place if out of bounds
                
                # Reward for moving
                reward = -1 if next_state in env.holes else 0
                if next_state == env.goal:
                    reward = 1

                next_idx = next_state[0] * len(env.grid[0]) + next_state[1]
                V[state_idx] = reward + gamma * V[next_idx]
                delta = max(delta, abs(v - V[state_idx]))

        if delta < theta:
            break

    return V


def improve_policy(env, V, gamma=0.99):
    policy = np.zeros(len(env.grid) * len(env.grid[0]), dtype=int)

    for i in range(len(env.grid)):
        for j in range(len(env.grid[0])):
            state = (i, j)
            state_idx = i * len(env.grid[0]) + j

            # Skip terminal states
            if state == env.goal or state in env.holes:
                continue

            best_action = None
            best_value = float('-inf')
            for action_idx, action in enumerate(actions):
                move = moves[action]
                next_state = (state[0] + move[0], state[1] + move[1])
                if not (0 <= next_state[0] < len(env.grid) and 0 <= next_state[1] < len(env.grid[0])):
                    next_state = state  # Stay in place if out of bounds
                
                reward = -1 if next_state in env.holes else 0
                if next_state == env.goal:
                    reward = 1

                next_idx = next_state[0] * len(env.grid[0]) + next_state[1]
                value = reward + gamma * V[next_idx]
                if value > best_value:
                    best_value = value
                    best_action = action_idx

            policy[state_idx] = best_action

    return policy

def print_policy(policy, grid):
    action_symbols = ["↑", "↓", "←", "→"]  # Symbols for up, down, left, right
    policy_grid = policy.reshape(len(grid), -1)
    for i in range(len(grid)):
        row = ""
        for j in range(len(grid[0])):
            state_idx = i * len(grid[0]) + j
            row += action_symbols[policy_grid[i, j]] + " "
        print(row)
    print()

# Define the grid, start, goal, and holes
grid = [["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"]]

start = (0, 0)
goal = (3, 3)
holes = [(1, 1), (1, 3), (2, 3), (3, 0)]

env = FrozenLake(grid, start, goal, holes)

# Initialize a random policy
state_space_size = len(grid) * len(grid[0])
policy = np.random.randint(0, len(actions), state_space_size)

# Policy Iteration
max_iterations = 100  # Set a maximum number of iterations for safety
iteration = 0
while iteration < max_iterations:
    print(f"Policy Iteration {iteration}")
    print("Current Policy:")
    print_policy(policy, grid)

    # Policy Evaluation
    V = evaluate_policy(env, policy, gamma=0.99)
    print(f"State-Value Function after Evaluation:\n{V.reshape(len(grid), -1)}\n")

    # Policy Improvement
    new_policy = improve_policy(env, V, gamma=0.99)
    print("Improved Policy:")
    print_policy(new_policy, grid)

    # Check if the policy has stabilized
    if np.array_equal(policy, new_policy):
        print("Policy has stabilized.\n")
        break
    else:
        policy = new_policy
        iteration += 1

if iteration == max_iterations:
    print("Reached maximum iterations without stabilization.\n")

print("Optimal Policy Found:")
print_policy(policy, grid)
