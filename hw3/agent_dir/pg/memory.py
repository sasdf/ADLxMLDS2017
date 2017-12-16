import random
import numpy as np
import bisect
import math


memSZ = 2 ** 14
class PrioritizedMemory(object):
    def __init__(self):
        self.tree = [0] * (memSZ * 2)
        self.memory = [None] * memSZ
        self.errors = []
        self.ptr = 0
        self.size = 0
        self.minProb = 0

    def append(self, data, tdError):
        self.memory[self.ptr] = data
        self.update(self.ptr, tdError)
        self.ptr = (self.ptr + 1) % memSZ
        if self.size < memSZ:
            self.size += 1


    def update(self, index, tdError):
        tdError = min(tdError, 10)
        index += memSZ
        if self.tree[index] > 0:
            self.errors.pop(bisect.bisect_left(self.errors, self.tree[index]))
        if tdError > 0:
            bisect.insort_left(self.errors, tdError)

        diff = tdError - self.tree[index]
        while index > 0:
            self.tree[index] += diff
            index //= 2

    def sample(self):
        index = 1
        r = random.random() * self.tree[index]
        while index < memSZ:
            index *= 2
            if r > self.tree[index]:
                r -= self.tree[index]
                index += 1
        minError = 0
        if len(self.errors) > 0:
            minError = self.errors[0]
        importance = (minError + 1e-2) / (self.tree[index] + 1e-2)
        importance = min(max(importance, 1e-2), 10)
        #  importance = 1
        index -= memSZ
        return self.memory[index] + [importance, index]

    def __len__(self):
        return self.size

class SimpleMemory(object):
    def __init__(self):
        self.memory = []
        self.errors = self.memory
        self.tree = [0] * memSZ

    def append(self, data, tdError):
        self.memory.append(data)
        if len(self.memory) > memSZ:
            self.memory.pop(0)

    def update(self, index, tdError):
        pass

    def sample(self):
        index = math.floor(random.random() * len(self.memory))
        importance = 1
        return self.memory[index] + [importance, index]

    def __len__(self):
        return len(self.memory)

Memory = SimpleMemory
#  Memory = PrioritizedMemory
