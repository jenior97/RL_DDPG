import gym
import numpy as np
import torch


# create the environment
env_name='MountainCarContinuous-v0'
env = gym.make(env_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_iteration=100
  
for i in range(test_iteration):
    state = env.reset()
    for t in count():
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(np.float32(action))
        ep_r += reward
        print(reward)
        env.render()
        if done: 
            print("reward{}".format(reward))
            print("Episode \t{}, the episode reward is \t{:0.2f}".format(i, ep_r))
            ep_r = 0
            env.render()
            break
        state = next_state