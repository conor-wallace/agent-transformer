import copy
import numpy as np
import itertools
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, OneHotCategorical


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


def after_first_landmark(obs, id):
    agent_obs = obs[4:6]

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_second_landmark(obs, id):
    agent_obs = obs[6:8]

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_third_landmark(obs, id):
    agent_obs = obs[8:10]

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_farthest_landmark(obs, id):
    landmark1_rel_pos = obs[4:6]
    landmark2_rel_pos = obs[6:8]
    landmark3_rel_pos = obs[8:10]

    landmark1_dist = (landmark1_rel_pos**2).sum()
    landmark2_dist = (landmark2_rel_pos**2).sum()
    landmark3_dist = (landmark3_rel_pos**2).sum()
    dists = np.array([landmark1_dist, landmark2_dist, landmark3_dist])
    index = np.argmax(dists)
    if index == 0:
        agent_obs = landmark1_rel_pos
    elif index == 1:
        agent_obs = landmark2_rel_pos
    else:
        agent_obs = landmark3_rel_pos

    # print("Agent obs: ", agent_obs)

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_closest_landmark(obs, id):
    landmark1_rel_pos = obs[4:6]
    landmark2_rel_pos = obs[6:8]
    landmark3_rel_pos = obs[8:10]

    landmark1_dist = (landmark1_rel_pos**2).sum()
    landmark2_dist = (landmark2_rel_pos**2).sum()
    landmark3_dist = (landmark3_rel_pos**2).sum()
    dists = np.array([landmark1_dist, landmark2_dist, landmark3_dist])
    index = np.argmin(dists)
    if index == 0:
        agent_obs = landmark1_rel_pos
    elif index == 1:
        agent_obs = landmark2_rel_pos
    else:
        agent_obs = landmark3_rel_pos

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class IA2CPolicyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IA2CPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, net_input, id):
        # if id == 4:
        #     print(f"IA2C agent {id} input: {net_input}")
        out = F.relu(self.fc1(torch.Tensor(net_input)))
        out = F.relu(self.fc2(out))
        pol_out = F.softmax(self.policy(out), dim=-1)
        
        m = OneHotCategorical(pol_out)
        action = m.sample()
        return action.detach().numpy()


class MADDPGNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(MADDPGNet, self).__init__()
        self.in_fn = nn.BatchNorm1d(input_dim)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x, id):
        h = F.relu(self.fc1(self.in_fn(torch.Tensor(x).unsqueeze(0))))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        action = onehot_from_logits(out)
        return action[0].detach().numpy()


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MLPLayer, self).__init__()
        self._layer_N = 1

        active_func = nn.Tanh()
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain('tanh')

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        # print("fc1 output sum: ", x.sum())
        # print("FC2: ", self.fc2[0])
        # print("Fc2 linear weight sum: ", self.fc2[0][0].weight.sum())
        # print("Fc2 linear bias sum: ", self.fc2[0][0].bias.sum())
        # print("Fc2 norm weight sum: ", self.fc2[0][2].weight.sum())
        # print("Fc2 norm bias sum: ", self.fc2[0][2].bias.sum())
        for i in range(self._layer_N):
            # print("i = ", i)
            x = self.fc2[i](x)
            # print("fc2 output sum: ", x.sum())
        return x


class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dim, cat_self=True, attn_internal=False, embedding_size=None):
        super(MLPBase, self).__init__()

        self._layer_N = 1
        self.hidden_size = hidden_dim

        self.feature_norm = nn.LayerNorm(input_dim)

        self.mlp = MLPLayer(input_dim, self.hidden_size)

    def forward(self, x):
        # print("Input sum: ", x.sum())
        x = self.feature_norm(x)

        # print("Normalized input sum: ", x.sum())
        x = self.mlp(x)

        return x


class PPONet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(PPONet, self).__init__()

        self.base = MLPBase(input_dim, hidden)
        self.act = nn.Linear(hidden, output_dim)
    
    def forward(self, x, id):
        # print("Agent id: ", id)
        # print("Feature norm weight sum: ", self.feature_norm.weight.sum())
        # print("Fc1 linear weight sum: ", self.fc1[0].weight.sum())
        # print("Fc1 norm weight sum: ", self.fc1[2].weight.sum())
        # print("Fc2 linear weight sum: ", self.fc2[0].weight.sum())
        # print("Fc2 norm weight sum: ", self.fc2[2].weight.sum())
        # print("Act weight sum: ", self.act.weight.sum())

        x = torch.Tensor(x).unsqueeze(0)
        # print("Input sum: ", x.sum())
        x = self.base(x)
        # print("base output sum: ", x.sum())
        logits = self.act(x)
        # print("X: ", logits)
        probs = Categorical(logits=logits).probs
        # print("Logits: ", probs)
        action = onehot_from_logits(probs)
        return action[0].detach().numpy()


# ia2c_agents = [IA2CPolicyNet(16, 64, 5) for _ in range(3)]
# for i in range(3):
#     save_dict = torch.load('pretrained_parameters/params_ia2c_agent_' + str(i) + '.pt')
#     ia2c_agents[i].load_state_dict(save_dict['agent_params']['actor_critic'])
# maddpg_agents = [MADDPGNet(16, 5, 64) for _ in range(3)]
# for i in range(3):
#     save_dict = torch.load('pretrained_parameters/params_maddpg_agent_' + str(i) + '.pt')
#     maddpg_agents[i].load_state_dict(save_dict['policy_params'])
#     maddpg_agents[i].eval()

ippo_agents = [PPONet(18, 5, 64) for _ in range(4)]
for i in range(4):
    save_dict = torch.load('pretrained_opponents/pretrained_parameters/spread/params_ippo_agent_' + str(i) + '.pt')
    modified_save_dict = {
        'base.feature_norm.weight': save_dict['base.feature_norm.weight'],
        'base.feature_norm.bias': save_dict['base.feature_norm.bias'],
        'base.mlp.fc1.0.bias': save_dict['base.mlp.fc1.0.bias'],
        'base.mlp.fc1.0.weight': save_dict['base.mlp.fc1.0.weight'],
        'base.mlp.fc1.2.weight': save_dict['base.mlp.fc1.2.weight'],
        'base.mlp.fc1.2.bias': save_dict['base.mlp.fc1.2.bias'],
        'base.mlp.fc_h.0.bias': save_dict['base.mlp.fc_h.0.bias'],
        'base.mlp.fc_h.0.weight': save_dict['base.mlp.fc_h.0.weight'],
        'base.mlp.fc_h.2.weight': save_dict['base.mlp.fc_h.2.weight'],
        'base.mlp.fc_h.2.bias': save_dict['base.mlp.fc_h.2.bias'],
        'base.mlp.fc2.0.0.weight': save_dict['base.mlp.fc2.0.0.weight'],
        'base.mlp.fc2.0.0.bias': save_dict['base.mlp.fc2.0.0.bias'],
        'base.mlp.fc2.0.2.weight': save_dict['base.mlp.fc2.0.2.weight'],
        'base.mlp.fc2.0.2.bias': save_dict['base.mlp.fc2.0.2.bias'],
        'act.weight': save_dict['act.action_out.linear.weight'],
        'act.bias': save_dict['act.action_out.linear.bias']
    }
    ippo_agents[i].load_state_dict(modified_save_dict)
    ippo_agents[i].eval()
mappo_agents = [PPONet(18, 5, 64) for _ in range(6)]
for i in range(6):
    save_dict = torch.load('pretrained_opponents/pretrained_parameters/spread/params_mappo_agent_' + str(i) + '.pt')
    modified_save_dict = {
        'base.feature_norm.weight': save_dict['base.feature_norm.weight'],
        'base.feature_norm.bias': save_dict['base.feature_norm.bias'],
        'base.mlp.fc1.0.bias': save_dict['base.mlp.fc1.0.bias'],
        'base.mlp.fc1.0.weight': save_dict['base.mlp.fc1.0.weight'],
        'base.mlp.fc1.2.weight': save_dict['base.mlp.fc1.2.weight'],
        'base.mlp.fc1.2.bias': save_dict['base.mlp.fc1.2.bias'],
        'base.mlp.fc_h.0.bias': save_dict['base.mlp.fc_h.0.bias'],
        'base.mlp.fc_h.0.weight': save_dict['base.mlp.fc_h.0.weight'],
        'base.mlp.fc_h.2.weight': save_dict['base.mlp.fc_h.2.weight'],
        'base.mlp.fc_h.2.bias': save_dict['base.mlp.fc_h.2.bias'],
        'base.mlp.fc2.0.0.weight': save_dict['base.mlp.fc2.0.0.weight'],
        'base.mlp.fc2.0.0.bias': save_dict['base.mlp.fc2.0.0.bias'],
        'base.mlp.fc2.0.2.weight': save_dict['base.mlp.fc2.0.2.weight'],
        'base.mlp.fc2.0.2.bias': save_dict['base.mlp.fc2.0.2.bias'],
        'act.weight': save_dict['act.action_out.linear.weight'],
        'act.bias': save_dict['act.action_out.linear.bias']
    }
    mappo_agents[i].load_state_dict(modified_save_dict)
    mappo_agents[i].eval()


opp = [after_first_landmark, after_second_landmark, after_third_landmark, after_farthest_landmark, after_closest_landmark]
opp += ippo_agents
opp += mappo_agents

# ippo_agents = [5, 6, 7, 8]
# mappo_agents = [9, 10, 11, 12]

opp_indices = [i for i in range(len(opp))]
# index = np.array(list(itertools.combinations_with_replacement(opp_indices, 2)))

# 0: ippo1
# 1: ippo2
# 2: ippo3
# 3: ippo4
# 4: mappo1
# 5: mappo2
# 6: mappo3
# 7: mappo4
# 8: mappo5
# 9: mappo6
# 10: after_first_landmark
# 11: after_second_landmark
# 12: after_third_landmark
# 13: after_farthest_landmark
# 14: after_closest_landmark

index = np.array([
    [0, 6],  # ippo1, mappo3
    [1, 10], # ippo2, after_first_landmark
    [2, 8],  # ippo3, mappo5
    [3, 12], # ippo4, after_third_landmark
    [4, 6],  # mappo1, mappo3
    [5, 0],  # mappo2, ippo1
    [9, 1],  # mappo6, ippo2
    [7, 2],  # mappo4, ippo3
    [11, 3], # after_second_landmark, ippo4
    [5, 10], # mappo2, after_first_landmark
    [7, 12], # mappo4, after_third_landmark
    [9, 6],  # mappo6, mappo3
    [11, 8], # after_second_landmark, mappo5
])

# index = np.array([[1,  8,  0],
#                    [2,  5,  8],
#                    [3,  0,  1],
#                    [8,  8, 9],
#                    [6,  0,  1],
#                    [7,  3,  4],
#                    [7,  1,  2],
#                    [3,  7,  3],
#                    [9,  3,  5],
#                    [0,  4,  6]])
pretrained_agents = []

for i in range(len(index)):
    pretrained_agents.append([opp[index[i][0]], opp[index[i][1]]])


@torch.no_grad()
def get_opponent_actions(obs, task_id):
    actions = []
    for i, o in enumerate(obs):
        actions.append(pretrained_agents[task_id][i](o, index[task_id, i]))
    return actions
