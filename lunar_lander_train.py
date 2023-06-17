import gym
import torch
import torch.optim as optim
import torch.nn as nn
import math
import random

from helper import ReplayMemory, Transition
from DQN import DQN

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
episode_num = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v2")

action_n = env.action_space.n
state_n = len(env.observation_space.sample())

# print(state_n, action_n)
policy_net = DQN(state_n, action_n).to(device)
target_net = DQN(state_n, action_n).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else: # random
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < 100:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
 
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)   
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)    

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_value = (next_state_values * GAMMA) + reward_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_value.unsqueeze(1)) # 真实值与预测值

    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

episode_rewards = []
for episode_idx in range(episode_num):
    state, _ = env.reset()
    episode_reward = 0
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    terminated = False
    while not terminated:
        action = select_action(state)
        state_next, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward        
        reward = torch.tensor([reward], device=device)

        if terminated:
            state_next = None
        else:
            state_next = torch.tensor(state_next, dtype=torch.float32, device=device).unsqueeze(0)
            
        memory.push(state, action, state_next, reward)
        
        state = state_next
        
        optimize_model()
        
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        
        for key in target_net_state_dict:
            target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU) * target_net_state_dict[key]
        target_net.load_state_dict(target_net_state_dict)
        
        if terminated or truncated:
            break
    
    episode_rewards.append(episode_reward)
    print("Episode {}, reward {}".format(episode_idx, episode_reward))

env.close()

# plot
import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.show()
plt.savefig('lander_rewards.png')



# save
torch.save(policy_net.state_dict(), 'policy.pt')
torch.save(target_net.state_dict(), 'target.pt')





# if __name__ == '__main__':
#     BATCH_SIZE = 3
#     state, _ = env.reset()
#     state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#     action = select_action(state)
#     _, reward, _, _, _ = env.step(action.item()) 
#     # print(reward)
#     reward = torch.tensor([reward], device=device) 
#     # print(reward)
#     # print(state, state.shape)
    
#     memory.push(state, action, state, reward)
#     memory.push(state, action, state, reward)
#     memory.push(state, action, state, reward)
#     memory.push(state, action, state, reward)
       
#     transitions = memory.sample(BATCH_SIZE)
#     # print(transitions)
#     batch = Transition(*zip(*transitions))
#     # print(batch)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
#     # print(non_final_mask)
    
#     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#     # print(non_final_next_states)
    
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)

#     # print(state_batch)
#     # print(action_batch)
#     # print(reward_batch)
    
#     # 相当于state_values = Q[state, :]
#     # state_values = policy_net(state_batch)
#     # print(state_values)
    
#     # 相当于state_action_values = Q[state, action]
#     state_action_values = policy_net(state_batch).gather(1, action_batch)
#     # print(state_action_values)
    
#     # 相当于next_state_values = max(Q[next_state, :])，预测的未来最大值
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     with torch.no_grad():
#         # print(target_net(non_final_next_states))
#         next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
#     # print(next_state_values)
#     expected_state_action_value = (next_state_values * GAMMA) + reward_batch
#     print(expected_state_action_value)
#     print(expected_state_action_value.unsqueeze(1))
#     print(state_action_values)
    
    

