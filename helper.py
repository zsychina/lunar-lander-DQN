from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    memory = ReplayMemory(100)
    memory.push(1, 2, 3, 4)
    memory.push(5, 6, 7, 8)
    memory.push(1, 2, 3, 4)
    memory.push(5, 6, 7, 8)
    memory.push(1, 2, 3, 4)
    memory.push(5, 6, 7, 8)
    memory.push(1, 2, 3, 4)
    memory.push(5, 6, 7, 8)
    transitions = memory.sample(5)
    print(transitions)
    print(*transitions)
    print(zip(*transitions))   
    print(*zip(*transitions))
    batch = Transition(*zip(*transitions))
    print(batch) 
    