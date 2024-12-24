import random

class MAB:
    def __init__(self, k_arms):
        self.k = k_arms
        self.arms = {i: random.random() for i in range(k_arms)}

    def pull_arm(self, arm):
        reward_prob = self.arms[arm]
        return 1 if random.random() < reward_prob else 0
    
    def print_arms(self):
        for key, value in self.arms.items():
            print(f"ARM[{key}]: {value:.2f}")