# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, optim
import numpy as np


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, lr_a, lr_c, gama, tau, memory_size, batch_size):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = self.to_var(a_bound)
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gama = gama
        self.tau = tau
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory = [0 for i in range(memory_size)]

        self.a_eval = ActorNet(self.a_dim, self.s_dim, self.a_bound).cuda()
        self.a_target = ActorNet(self.a_dim, self.s_dim, self.a_bound).cuda()

        self.c_eval = CriticNet(self.a_dim, self.s_dim).cuda()
        self.c_target = CriticNet(self.a_dim, self.s_dim).cuda()

        for ap, cp in zip(self.a_target.parameters(), self.c_target.parameters()):
            ap.requires_grad = False
            cp.requires_grad = False

    def choose_action(self, s):
        s = Variable(torch.from_numpy(s).float()).cuda().unsqueeze(0)
        action = self.a_eval.forward(s).cpu().data[0].numpy().tolist()
        return action

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        memory_idx = self.memory_counter % self.memory_size
        self.memory[memory_idx] = [s, a, r, s_]
        self.memory_counter += 1

    def learn(self):

        for ae, at, ce, ct in zip(self.a_eval.parameters(), self.a_target.parameters(),
                                  self.c_eval.parameters(), self.c_target.parameters()):
            a, b, c, d = ae.data, at.data, ce.data, ct.data
            at.data.copy_(self.tau * a + (1 - self.tau) * b)
            ct.data.copy_(self.tau * c + (1 - self.tau) * d)

        batch_idx = np.random.choice(min(self.memory_counter, self.memory_size), self.batch_size)
        memory_batch = [self.memory[i] for i in batch_idx]
        s, a, r, s_ = list(zip(*memory_batch))
        s, a, r, s_ = self.to_var(s), self.to_var(a), self.to_var(r), self.to_var(s_)

        action_eval = self.a_eval.forward(s)
        action_target = self.a_target.forward(s_)

        q_eval = self.c_eval.forward(s, a)
        q = self.c_eval.forward(s, action_eval)
        q_target = self.c_target.forward(s_, action_target)

        c_loss = torch.mean((r.unsqueeze(1) + self.gama * q_target - q_eval) ** 2)
        c_optimizer = optim.Adam(self.c_eval.parameters(), lr=self.lr_c)

        c_optimizer.zero_grad()
        c_loss.backward()
        c_optimizer.step()

        a_loss = -torch.mean(q)
        a_optimizer = optim.Adam(self.a_eval.parameters(), lr=self.lr_a)

        a_optimizer.zero_grad()
        a_loss.backward()
        a_optimizer.step()

    def to_var(self, x_list):
        return Variable(torch.from_numpy(np.stack(x_list)).float()).cuda()


class ActorNet(nn.Module):
    def __init__(self, a_dim, s_dim, a_bound):
        super(ActorNet, self).__init__()
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound

        self.fc = nn.Sequential(nn.Linear(self.s_dim, 30),
                                nn.ReLU(),
                                nn.Linear(30, self.a_dim),
                                nn.Tanh())

    def forward(self, s):
        out = self.fc(s)
        out = out * self.a_bound.unsqueeze(0)
        return out


class CriticNet(nn.Module):
    def __init__(self, a_dim, s_dim):
        super(CriticNet, self).__init__()
        self.a_dim = a_dim
        self.s_dim = s_dim

        self.fc1 = nn.Linear(self.a_dim, 30, bias=False)
        self.fc2 = nn.Linear(self.s_dim, 30)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, s, a):
        s_out = self.fc2(s)
        a_out = self.fc1(a)
        out = F.relu(s_out + a_out)

        return self.fc3(out)
