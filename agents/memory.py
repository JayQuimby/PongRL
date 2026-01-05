import math
import random
from game.static import *
from typing import List

class AgentMemory:
    def __init__(self) -> None:
        self.max_mem = MEMORY_SIZE
        self.memories: List = []
        self.reward_ind: int = 2

    def __call__(self, k):
        return random.sample(self.memories, min(max(1, k), len(self)))

    def __len__(self):
        return len(self.memories)

    def remember(self, action):
        if abs(action[2]) > MIN_REWARD_VAL:
            if len(self.memories) < self.max_mem:
                self.memories.append(action)
            else:
                self._purge_memory()
                self.memories.append(action)

    def sort_memories(self):
        self.memories.sort(key=lambda x: x[self.reward_ind], reverse=True)

    def weighted_random_sample(self, percent=0.5):
        mems = []
        lim = int(self.max_mem * percent)

        for memory in self.memories:
            prob = 1 / (1 + math.exp(-memory[self.reward_ind]))
            if random.random() < prob:
                mems.append(memory)
            
            if len(mems) >= lim:
                break
        return mems

    def _purge_memory(self):
        self.sort_memories()
        new_memories = self.weighted_random_sample()
        self.memories = new_memories

    def clear(self):
        self.memories = []