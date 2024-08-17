import math
import random
from global_vars import *

class AgentMemory:
    def __init__(self) -> None:
        self.max_mem = MEMORY_SIZE
        self.memories = []
        self.reward_ind = 2

    def __call__(self, x):
        x = min(max(x, 0.1), 1)
        return random.sample(self.memories, k=(int(len(self.memories)*x)))#self.random_sample(x)

    def __len__(self):
        return len(self.memories)

    def remember(self, action):
        if abs(action[2]) > MIN_REWARD_VAL:
            if len(self.memories) < self.max_mem:
                self.memories.append(action)
            else:
                self.purge_memory()
                self.memories.append(action)

    def sort_memories(self):
        self.memories.sort(key=lambda x: x[self.reward_ind], reverse=True)

    def random_sample(self, percent=0.5):
        mems = []
        lim = int(self.max_mem * percent)

        for memory in self.memories:
            prob = 1 / (1 + math.exp(-memory[self.reward_ind]))
            if random.random() < prob:
                mems.append(memory)
            
            if len(mems) >= lim:
                break
        return mems

    def purge_memory(self):
        self.sort_memories()
        new_memories = self.random_sample()
        self.memories = new_memories

    def clear(self):
        self.memories = []