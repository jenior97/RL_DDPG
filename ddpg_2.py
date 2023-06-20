import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import copy



class Replay_buffer():

    def __init__(self, max_size = 10**6):
        
        # replay buffer max_size = 10**6
        # storage 안에 저장되는 memory의 형태 : tuple with (state, next_state, action, reward, done)
        # drop_idx를 통해 oldest부터 교체됨 & max_size의 나머지 형태로 계산함으로써 0 ~ (max_size-1)의 숫자가 계속 순환됨
        self.storage = []
        self.max_size = max_size
        self.drop_idx = 0

    def push(self, data):

        # memory overflow일 때, oldest부터 교체
        # data로 들어오는 형태는 위 tuple 형태
        if len(self.storage) == self.max_size:
            self.storage[int(self.drop_idx)] = data
            self.drop_idx = (self.drop_idx + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):

        # random sample을 통해 storage 안에 있는 memory를 batch_size만큼 꺼냄
        # batch_size만큼의 sample을 추출해서 array 형식으로 append 함
        # 이렇게 하면, output의 형태는 np.array with (batch_size , state/action/reward/done의 size)로 나타남
        idx = np.random.randint(0, len(self.storage), size = batch_size)
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = [], [], [], [], []

        for i in idx:
            state, next_state, action, reward, done = self.storage[i]
            batch_state.append(np.array(state, copy=False))
            batch_next_state.append(np.array(next_state, copy=False))
            batch_action.append(np.array(action, copy=False))
            batch_reward.append(np.array(reward, copy=False))
            batch_done.append(np.array(done, copy=False))

        return np.array(batch_state), np.array(batch_next_state), np.array(batch_action), np.array(batch_reward).reshape(-1, 1), np.array(batch_done).reshape(-1, 1)
    

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################

# optimizer : Adam
# actor_lr : 10**-4, critic_lr : 10**-3
# Q : L2 weight decay of 10**-2
# discount factor : 0.99
# soft target update tau : 10**-3
# low-dimensional network의 경우, minibatch with 64 size & pixel input의 경우, minibatch with 16 size 

# Initialization
# Actor와 Critic 모두 동일한 initialization weight을 가짐
# final layer을 제외한 모든 layer의 경우 -> uniform distribution with [-1/sqrt(f) , 1/sqrt(f)] where f = fan-in of the layer
# final layer의 경우, low-dimensional network의 경우 [-3 * 10**-3 , 3 * 10**3] & pixel input의 경우 [-3 * 10 ** -4 , 3 * 10 ** 4]에서의 uniform distribution


# Actor의 경우, 
# low-dimensional network의 경우, 2 hidden layers with 400 & 300 units
# pixel input의 경우, 3 convolutional layers with 32 filters at each layer without pooling + 2 fully-connected layer with 200 units
# final ouput layer of the actor : tanh

# Batch Normalization
# low-dimensional network의 경우, 
# state_input & Actor network의 모든 layer, Q network prior to the action input ==> Batch Normalization을 진행함

###################################################################################################################################################

class Actor(nn.Module):

    # output : continuous action value (1 single optimized action for the state)
    # final layer tanh를 붙여줌으로써 action을 bound시켜준다고 표현하고 있음 (여기에 max_action을 곱해줘야 되는지는 모르겠는데 tanh <= |1|이므로 마지막에 max_action을 곱해줌)

    def __init__(self, n_states, action_dim, actor_hidden1, actor_hidden2, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # select_action을 할 때, batch_size = 1로 들어가는데 이 때 BatchNorm1d가 동작하지 않아서 해결해야 함
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_states),
            nn.Linear(n_states, actor_hidden1), 
            nn.BatchNorm1d(actor_hidden1),
            nn.ReLU(), 
            nn.Linear(actor_hidden1, actor_hidden2), 
            nn.BatchNorm1d(actor_hidden2),            
            nn.ReLU(), 
            nn.Linear(actor_hidden2, 1), 
            nn.Tanh(), 
            )
        
    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):

    # input : state & action
    # output : Q-value
    def __init__(self, n_states, action_dim, critic_hidden1, critic_hidden2):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states + action_dim, critic_hidden1), 
            nn.ReLU(), 
            nn.Linear(critic_hidden1, critic_hidden2), 
            nn.ReLU(), 
            nn.Linear(critic_hidden2, action_dim), 
        )
        
    def forward(self, state, action):
        # state = nn.BatchNorm1d(len(state))
        return self.net(torch.cat((state, action), 1))
    

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################



class OU_Noise(object):

    # size: the size of the noise vector to be generated
    # mu: the mean of the noise, set to 0 by default
    # theta: the rate of mean reversion, controlling how quickly the noise returns to the mean
    # sigma: the volatility of the noise, controlling the magnitude of fluctuations
    # theta = 0.15, sigma = 0.2라고 잡았다고 하고 있음

    def __init__(self, size, seed, mu = 0., theta = 0.15, sigma = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample.
        This method uses the current state of the noise and generates the next sample
        """
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state



###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################

batch_size = 64
update_iteration = 200
tau = 0.001 # tau for soft updating
gamma = 0.99 # discount factor
directory = './'
actor_hidden1 = 400 # 1st hidden layer for actor
actor_hidden2 = 300 # 2nd hidden layer for actor
critic_hidden1 = 400 # 1st hidden layer for critic
critic_hidden2 = 300 # 2nd hidden layer for critic
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):

        self.replay_buffer = Replay_buffer()
        
        self.actor = Actor(state_dim, action_dim, actor_hidden1, actor_hidden2, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim,  actor_hidden1, actor_hidden2, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 10**-3)

        self.critic = Critic(state_dim, action_dim, critic_hidden1, critic_hidden2).to(device)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden1, critic_hidden2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 10**-2)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):

        # input : current state & output : action to take in that state -> Actor network에 쓰임
        # Normalization을 시켰기 때문에 select_action 할 때도 마찬가지로 input을 normalize해서 줘야하는가? -> 논문에서는 그렇다고 말하고 있는 것 같긴 함
        # select_action을 할 때, BatchNorm1d를 적용 안시키기 위해 .eval() 붙임
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def update(self):

        c_loss = []
        a_loss = []

        for iter in range(update_iteration):

            # replay buffer에서 batch만큼 sampling함
            state, next_state, action, reward, done = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1-done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            # Target Q value 계산
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Current Q estimate 가져옴
            current_Q = self.critic(state, action)

            # Critic loss -> current Q와 Target Q간의 MSE
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor loss -> 
            # Compute actor loss as the negative mean Q value using the critic network and the actor network
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target network update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
           
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
            c_loss.append(critic_loss.cpu().data.numpy())
            a_loss.append(actor_loss.cpu().data.numpy())
        
        return c_loss, a_loss


    def save(self):
        # save 
        torch.save(self.actor.state_dict(), directory + env_name + '_actor.pth')
        torch.save(self.critic.state_dict(), directory + env_name + '_critic.pth')
    def load(self):
        # load
        self.actor.load_state_dict(torch.load(directory + env_name + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + env_name + 'critic.pth'))








###################################################################################################################################################
###################################################################################################################################################

import gym

# Environment
env_name = 'Hopper-v3'
env = gym.make(env_name)

# Define different parameters for training the agent
max_episode = 5000 # episode 
max_time_steps = 5000 # 
reward_hist = [] # reward history
critic_loss_hist = []
actor_loss_hist = []

# Rendering
render = True
render_interval = 10

env.seed(0)
torch.manual_seed(0)
np.random.seed(0)

#Environment action ans states
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Create a DDPG instance
agent = DDPG(state_dim, action_dim, max_action)

# Train the agent for max_episodes
for i in range(max_episode):
    
    step = 0
    episode_reward = 0
    state = env.reset()

    for  t in range(max_time_steps):
        # 들어오는 state를 actor model에 넣고 action을 고름
        action = agent.select_action(state)
        
        # exploration을 위해 noise 추가
        noise = OU_Noise(size = len(action), seed = 0)
        exploration_noise = noise.sample()
        action += exploration_noise
        
        # step
        next_state, reward, done, info = env.step(action)
        
        # 각 step마다의 reward 저장
        reward_hist.append(reward)
        
        # rendering
        if render and i >= render_interval : env.render()
        
        # replay buffer 추가
        agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
        
        # state update
        state = next_state

        if done:
            break
        
        step += 1
        
    episode_reward += 0
    c_loss, a_loss = agent.update()

    critic_loss_hist.append(c_loss)
    actor_loss_hist.append(a_loss)
    print("Episode: \t{}  Reward: \t{:0.5f}  Critic Loss: \t{:0.5f}  Actor Loss: \t{:0.5f}".format( i, reward_hist[-1], c_loss[-1], a_loss[-1]))    
    
    if i % 10 == 0:
        agent.save()

# save loss & reward

# reward 갯수 = 이론상 update_iteration * max_time_steps(다 다름) * max_episode
np.save(env_name + '_reward.npy', reward_hist, allow_pickle = True)
# critic_loss 갯수 = list with size of max_episode [update_iteration 수만큼, ... , update_iteration 수만큼 ]
np.save(env_name + '_critic_loss.npy', critic_loss_hist, allow_pickle = True)
# actor_loss 갯수 = list with size of max_episode [update_iteration 수만큼, ... , update_iteration 수만큼 ]
np.save(env_name + '_actor_loss.npy', actor_loss_hist, allow_pickle = True)
env.close()




