import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ActionHead
from .utils import get_clones, init


class PPO(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        recurrent_dim,
        hidden_dim,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.recurrent_dim = recurrent_dim
        self.hidden_dim = hidden_dim

        self.policy = RNNActor(
            self.state_dim,
            self.act_dim,
            self.hidden_dim
        )
        self.critic = RNNCritic(
            self.state_dim, self.hidden_dim
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def act(self, obs, rnn_states, masks, deterministic=False):
        self.policy.eval()

        obs = obs.to(self.device)
        rnn_states = rnn_states.to(self.device)
        masks = masks.to(self.device)

        move_action, comm_action, rnn_states = self.policy(obs, rnn_states, masks, deterministic)

        return move_action.squeeze(-1), comm_action.squeeze(-1), rnn_states

    def get_action(self, obs, rnn_states, masks, deterministic=False):
        obs = obs.to(self.device)
        rnn_states = rnn_states.to(self.device)
        masks = masks.to(self.device)

        action, prob, rnn_states = self.policy(obs, rnn_states, masks)

        return action, prob, rnn_states

    def get_values(self, obs, rnn_states, masks):
        return self.critic(obs, rnn_states, masks)

    def evaluate_actions(self, obs, rnn_states, masks, actions):
        return self.policy.evaluate_actions(obs, rnn_states, masks, actions)

    def load_params(self, checkpoint_path):
        actor_dict = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(actor_dict)


class RNNActor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim,
        device=torch.device("cpu"),
    ):
        super(RNNActor, self).__init__()

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.base = MLPBase(obs_dim, hidden_dim)
        self.rnn = RNNLayer(hidden_dim, hidden_dim)
        self.act = ActionHead(act_dim, hidden_dim)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """

        obs = obs.to(dtype=torch.float32, device=self.device)
        rnn_states = rnn_states.to(dtype=torch.float32, device=self.device)
        masks = masks.to(dtype=torch.float32, device=self.device)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actions, probs = self.act(actor_features, available_actions, deterministic)

        return actions, probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, masks, actions):
        obs = obs.to(dtype=torch.float32, device=self.device)
        rnn_states = rnn_states.to(dtype=torch.float32, device=self.device)
        masks = masks.to(dtype=torch.float32, device=self.device)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        log_probs, entropy = self.act.evaluate_actions(actor_features, actions)

        return log_probs, entropy


class RNNCritic(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        obs_dim,
        hidden_dim,
        device=torch.device("cpu"),
    ):
        super(RNNCritic, self).__init__()

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.base = MLPBase(obs_dim, hidden_dim)
        self.rnn = RNNLayer(hidden_dim, hidden_dim)
        self.v_out = nn.Linear(hidden_dim, 1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """

        obs = obs.to(dtype=torch.float32, device=self.device)
        rnn_states = rnn_states.to(dtype=torch.float32, device=self.device)
        masks = masks.to(dtype=torch.float32, device=self.device)

        critic_features = self.base(obs)
        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values


class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPBase, self).__init__()

        self.feature_norm = nn.LayerNorm(input_dim)
        self.mlp = MLPLayer(input_dim, hidden_dim)

    def forward(self, x):
        x = self.feature_norm(x)
        x = self.mlp(x)

        return x


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPLayer, self).__init__()

        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain("relu")

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim)), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fc2 = get_clones(self.fc_h, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2[0](x)
        return x


class RNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RNNLayer, self).__init__()

        self.rnn = nn.GRU(input_dim, output_dim, num_layers=1)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            # print("X has the same shape as H")
            # print("X shape = ", x.unsqueeze(0).shape)
            # print("H shape = ", (hxs * masks.repeat(1, 1).unsqueeze(-1)).transpose(0, 1).contiguous().shape)
            x, hxs = self.rnn(
                x.unsqueeze(0),
                (hxs * masks.repeat(1, 1).unsqueeze(-1)).transpose(0, 1).contiguous(),
            )
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            print("X does not have the same shape as H")
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (
                    hxs * masks[start_idx].view(1, -1, 1).repeat(1, 1, 1)
                ).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs
