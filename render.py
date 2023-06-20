

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import random
import matplotlib.pyplot as plt
import csv

env_name = 'Hopper-v2'
env = gym.make(env_name)


action_space = env.action_space.shape[0]
max_action = env.action_space.high[0]
state_space = env.observation_space.shape[0]
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


actor_policy_net = ActorNet()
actor_policy_net.eval()
actor_policy_net.load_state_dict(torch.load('/home/kim/catkin_ws/src/ddpg/' + env_name + 'actor_policy.pth'))


def select_action(obs):
	# action selection

	s = Variable(torch.Tensor(([obs])))
	# action += noise 형태로 action selection 진행
	a = actor_policy_net(s)
	return a


episode = 100

for episode in range(episode):
	observation = env.reset()
	done = False
	step = 0
	R = 0

	while not done:
		env.render()
		action = select_action(observation)
		# print(action)

		next_observation, reward, done, _ = env.step(action.detach().numpy())
		# print(reward)
		observation = next_observation
		R += reward
		step += 1
	print(episode, R, step, done)