import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import csv





env_name = 'Hopper-v2'
env = gym.make(env_name)
toTensor = torch.Tensor
FloatTensor = torch.FloatTensor

action_space = env.action_space.shape[0]
max_action = env.action_space.high[0]
state_space = env.observation_space.shape[0]

episode = 10000
buffer_size= 10**6
batch_size = 64
gamma = 0.99
critic_lr = 10**-3
actor_lr = 10**-4
tau = 10**-3



class RelayBuffer:

	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.memory = []

	def push(self, data):		
		# memory에 overflow나면 oldest부터 교체
		
		self.memory.append(data)
		if len(self.memory) > self.buffer_size:
			del self.memory[0]

	def sample(self, batch_size):
		# random하게 batch_size만큼 sampling 함
		
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


################################################################################################################################
################################################################################################################################
################################################################################################################################

def fanin_init(size, fanin=None):
	# layer initialization을 하기 위해서 ! -> uniform distribution으로
	
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


class ActorNet(nn.Module):
	def __init__(self, input = state_space, output = action_space, dis = 0.003):
		super(ActorNet, self).__init__()
		self.fc1 = nn.Linear(input, 400)
		self.fc2 = nn.Linear(400, 300)
		self.fc3 = nn.Linear(300, output)
		self.tanh = nn.Tanh()
		self.init_weight(dis)
	
	def init_weight(self, dis):
		# actor network layer의 initialization

		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		self.fc3.weight.data.uniform_(-dis, dis)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# action을 bound해주기 위해서 마지막 layer에 tanh를 사용함
		x = self.tanh(self.fc3(x))
		return x


class CriticNet(nn.Module):
	def __init__(self, input = state_space, output = 1, dis = 0.0003):
		super(CriticNet, self).__init__()
		self.fc1 = nn.Linear(input, 400)
		self.fc2 = nn.Linear(400+action_space, 300)
		self.fc3 = nn.Linear(300, output)
		self.init_weight(dis)
	
	def init_weight(self, dis):
		# critic network layer의 initialization

		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		self.fc3.weight.data.uniform_(-dis, dis)
	
	def forward(self, xs):
		x, a = xs
		x = F.relu(self.fc1(x))
		# 두번째 layer에서 state + action의 형태로 input이 들어감
		x = F.relu(self.fc2(torch.cat([x, a], 1)))
		x = self.fc3(x)
		return x




################################################################################################################################
################################################################################################################################
################################################################################################################################



class ou_noise:
    # Exploration noise를 추가하기 위해서
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'ou_noise(mu={}, sigma={})'.format(self.mu, self.sigma)



################################################################################################################################
################################################################################################################################
################################################################################################################################


# target network 따로 설정
actor_target_net = ActorNet().cuda()
actor_policy_net = ActorNet().cuda()

critic_target_net = CriticNet().cuda()
critic_policy_net = CriticNet().cuda()



# 저장된 parameter를 불러오기
if os.path.isfile(env_name + 'actor_target.pth'):
	actor_target_net.load_state_dict(torch.load(env_name + 'actor_target.pth'))
if os.path.isfile(env_name + 'actor_policy.pth'):
	actor_policy_net.load_state_dict(torch.load(env_name + 'actor_policy.pth'))
if os.path.isfile(env_name + 'critic_target.pth'):
	critic_target_net.load_state_dict(torch.load(env_name + 'critic_target.pth'))
if os.path.isfile(env_name + 'critic_policy.pth'):
	critic_policy_net.load_state_dict(torch.load(env_name + 'critic_policy.pth'))

# 저장된 replay buffer memory 불러오기
relay_buffer = RelayBuffer(buffer_size)
if os.path.isfile(env_name + 'buffer_memory.pth'):
	relay_buffer = torch.load(env_name + 'buffer_memory.pth')



# critic network에서는 L2 weight decay of 10**-2를 사용했다고 나와 있음
optimizer_critic = optim.Adam(critic_policy_net.parameters(), lr = critic_lr, weight_decay = 0.01)
optimizer_actor = optim.Adam(actor_policy_net.parameters(), lr = actor_lr)



# exploration noise 정의
noise = ou_noise(mu = np.zeros(action_space))

def select_action(observation, n_step):
	# action selection

	state = Variable(toTensor([observation])).cuda()
	# action += noise 형태로 action selection 진행
	action = actor_policy_net(state).data.cpu() + toTensor(noise())
	return action


def critic_loss_func(predicted, target):
	# target network에서의 Q_value와 policy network에서의 Q_value의 MSE

	return torch.sum((target - predicted)**2) / batch_size




def soft_update(target, policy, tau):
	# soft update for target network

	for target_param, policy_param in zip(target.parameters(), policy.parameters()):
		target_param.data = tau * policy_param + (1-tau) * target_param


def train():

	# sampling
	if len(relay_buffer) < batch_size:
		return
	else:
		sample_batch = relay_buffer.sample(batch_size)
	
	s, a, r, n_s, D = zip(*sample_batch)
	state_batch = Variable(torch.cat(s, 0)).cuda()
	action_batch = Variable(torch.cat(a, 0)).cuda()
	reward_batch = Variable(torch.cat(r, 0)).cuda()
	next_state_batch = Variable(torch.cat(n_s, 0)).cuda()

	# compute y
	optimizer_critic.zero_grad()
	ya = actor_target_net(next_state_batch)
	ys = critic_target_net([next_state_batch, ya])
	y = reward_batch + gamma * ys

	# D(done) == True이면 y는 0값을 줌
	for i in range(len(D)):
		if D[i]:
			y[i] = 0

	# compute predicted value of critic policy net
	predicted = critic_policy_net([state_batch, action_batch])

	# compute loss
	critic_loss = critic_loss_func(predicted, y)
	critic_loss.backward()
	optimizer_critic.step()

	# train actor network
	optimizer_actor.zero_grad()
	act = actor_policy_net(state_batch)
	predicted = -critic_policy_net([state_batch, act])
	# Compute actor loss as the negative mean Q value using the critic network and the actor network
	actor_loss = predicted.mean()
	actor_loss.backward()
	optimizer_actor.step()

	# update target network
	soft_update(critic_target_net, critic_policy_net, tau)
	soft_update(actor_target_net, actor_policy_net, tau)

	c_loss = np.array(critic_loss.cpu().data.numpy())
	a_loss = np.array(actor_loss.cpu().data.numpy())

	return c_loss, a_loss







################################################################################################################################
################################################################################################################################
################################################################################################################################


step = 0
R = 0
n_step = 0
Return = []
loss_history = []
return_writer = csv.writer(open(env_name + "_Return.csv", 'w'))

for episode in range(episode):
	observation = env.reset()
	done = False
	step = 0
	R = 0
	while not done:
		action = select_action(observation, n_step)
		action = torch.clamp(action, min = -1, max = 1)
		next_observation, reward, done, _ = env.step(action)

		transition = [
			FloatTensor([observation]),
			action,
			FloatTensor([reward]),
			FloatTensor([next_observation]),
			done
		]
		relay_buffer.push(transition)

		train_loss = train()
		loss_history.append(train_loss)

		R += reward
		step += 1
		n_step += 1
		observation = next_observation

	return_writer.writerow([R])
	Return.append(R)

	print('episode: %3d,\tStep: %5d,\tReturn: %f,\tActor_loss: %5f,\tCritic_loss: %5f ' %(episode, step, R, R, R))

	torch.save(actor_target_net.state_dict(), env_name + 'actor_target.pth')
	torch.save(actor_policy_net.state_dict(), env_name + 'actor_policy.pth')
	torch.save(critic_target_net.state_dict(), env_name + 'critic_target.pth')
	torch.save(critic_policy_net.state_dict(), env_name + 'critic_policy.pth')
	np.save(env_name + 'loss_history.npy', loss_history)
	torch.save(relay_buffer, env_name + 'buffer_memory.pth')