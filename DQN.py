import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
if __name__ == '__main__':
    import numpy as np
    dqn = DQN(4, 2)
    state = [0.1, 2.1, -0.7, 0.4]
    state = torch.Tensor(state)
    action = np.argmax(dqn(state))
    print(action)
