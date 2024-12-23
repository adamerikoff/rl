import random

class FrozenLake:
    def __init__(self, grid, start, goal, holes, is_slippery=True):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.holes = holes
        self.is_slippery = is_slippery
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        if action not in moves:
            raise ValueError("Invalid action!")

        move = moves[action]
        next_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        if self.is_slippery and random.random() < 0.2:
            next_pos = (-next_pos[0], -next_pos[1])

        if not (0 <= next_pos[0] < len(self.grid) and 0 <= next_pos[1] < len(self.grid[0])):
            next_pos = self.agent_pos

        self.agent_pos = next_pos

        if next_pos == self.goal:
            return next_pos, 1, True
        elif next_pos in self.holes:
            return next_pos, -1, True
        else:
            return next_pos, 0, False

    def render(self):
        for i, row in enumerate(self.grid):
            row_str = ""
            for j, tile in enumerate(row):
                if (i, j) == self.agent_pos:
                    row_str += "A "  # Agent
                else:
                    row_str += f"{tile} "
            print(row_str)
        print()