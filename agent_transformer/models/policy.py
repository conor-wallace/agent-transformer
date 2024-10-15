import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import OneHotCategorical
from torch.distributions.categorical import Categorical


class PolicyNet(nn.Module):

    def __init__(self, obs_dim, hidden_dim, act_dim, embedding_dim=None):
        super(PolicyNet, self).__init__()
        if embedding_dim is not None:
            self.fc1 = nn.Linear(obs_dim + embedding_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, act_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy(x), dim=-1)
        values = self.value(x)
        return policy, values


class RecurrentPolicyNet(nn.Module):

    def __init__(self, input_dim, act_dim, hidden_dim, oracle=False):
        super(RecurrentPolicyNet, self).__init__()
        self.oracle = oracle
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, act_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):

        batch_size = hidden[0].shape[1]
        x = x.reshape((-1, batch_size, self.lstm.input_size))

        hs = []
        for xi in x:
            h, hidden = self.lstm(xi.unsqueeze(0), hidden)
            hs.append(h)

        h = torch.flatten(torch.cat(hs), 0, 1)
        h = F.relu(self.fc1(h))
        policy = F.softmax(self.policy(h), dim=-1)
        values = self.value(h)
        return policy, values, hidden

    def act(
        self,
        obs,
        action,
        hidden,
        opp_obs=None,
        opp_action=None,
        embedding=None
    ):
        if self.oracle:
            x = torch.cat((obs, action, opp_obs, opp_action), dim=-1)
        elif embedding is not None:
            x = torch.cat((obs, embedding), dim=-1)
        else:
            x = torch.cat((obs, action), dim=-1)
        policy, value, hidden = self(x, hidden)
        policy = Categorical(policy)
        action = policy.sample()
        return action, value, hidden
