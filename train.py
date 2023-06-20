import gym
import torch
import numpy as np
from ddpg import DDPG





# create the environment
env_name='MountainCarContinuous-v0'
env = gym.make(env_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define different parameters for training the agent
max_episode=100
max_time_steps=5000
ep_r = 0
total_step = 0
score_hist=[]
# for rensering the environmnet
render=True
render_interval=10
# for reproducibility
env.seed(0)
torch.manual_seed(0)
np.random.seed(0)
#Environment action ans states
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) 

# Exploration Noise
exploration_noise=0.1
exploration_noise=0.1 * max_action

# Create a DDPG instance
agent = DDPG(state_dim, action_dim)

# Train the agent for max_episodes
for i in range(max_episode):
    total_reward = 0
    step =0
    state = env.reset()
    for  t in range(max_time_steps):
        action = agent.select_action(state)
        # Add Gaussian noise to actions for exploration
        action = (action + np.random.normal(0, 1, size=action_dim)).clip(-max_action, max_action)
        #action += ou_noise.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if render and i >= render_interval : env.render()
        agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
        state = next_state
        if done:
            break
        step += 1
        
    score_hist.append(total_reward)
    total_step += step+1
    print("Episode: \t{}  Total Reward: \t{:0.2f}".format( i, total_reward))
    agent.update()
    if i % 10 == 0:
        agent.save()
env.close()

