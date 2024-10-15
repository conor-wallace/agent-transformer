import torch
import torch.nn as nn

from .distributions import Categorical


class ActionHead(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, act_dim, input_dim):
        super(ActionHead, self).__init__()

        self.multi_discrete = True
        self.action_out = Categorical(input_dim, act_dim)

    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """

        action_logit = self.action_out(x)
        action = action_logit.mode() if deterministic else action_logit.sample()
        action_log_prob = action_logit.log_probs(action)

        return action, action_log_prob

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """

        action_probs = []
        for action_out in self.action_outs:
            action_logit = action_out(x)
            action_prob = action_logit.probs
            action_probs.append(action_prob)
        action_probs = torch.cat(action_probs, -1)

        return action_probs

    def evaluate_actions(self, x, action):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action = torch.transpose(action, 0, 1)
        action_log_probs = []
        dist_entropy = []
        for action_out, act in zip(self.action_outs, action):
            action_logit = action_out(x)
            action_log_probs.append(action_logit.log_probs(act))
            dist_entropy.append(action_logit.entropy().mean())

        action_log_probs = torch.cat(action_log_probs, -1) # ! could be wrong
        dist_entropy = sum(dist_entropy) / len(dist_entropy)

        return action_log_probs, dist_entropy