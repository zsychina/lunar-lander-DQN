import torch
from DQN import DQN
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v2", render_mode='human')

action_n = env.action_space.n
state_n = len(env.observation_space.sample())

policy_net = DQN(state_n, action_n).to(device)
target_net = DQN(state_n, action_n).to(device)

policy_net.load_state_dict(torch.load('policy.pt'))
target_net.load_state_dict(torch.load('target.pt'))

def select_action(state):
    return policy_net(state).max(1)[1].view(1, 1)

state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
while True:
    action = select_action(state)
    state_next, reward, terminated, truncated, info = env.step(action.item())
    state_next = torch.tensor(state_next, dtype=torch.float32, device=device).unsqueeze(0)
    if terminated or truncated:
        observation, info = env.reset()
    state = state_next
        
env.close()


