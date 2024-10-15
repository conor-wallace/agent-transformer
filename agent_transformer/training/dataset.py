import numpy as np
import random
import torch
from typing import Any, Dict, List


def cast(x):
    return x.reshape(-1, *x.shape[2:])


def flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def create_seq_dataloader(
    trajectories: List[Dict[str, Any]],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    scale: float,
    state_dim: int,
    act_dim: int,
    oppnt_state_dim: int,
    oppnt_act_dims: List[int],
    num_opponents: int,
    max_len: int = 20,
    max_ep_len: int = 25,
    batch_size: int = 256,
    num_workers: int = 2,
    model_type: str = 'dt'
):
    dataset = SeqDataset(
        trajectories=trajectories,
        state_mean=state_mean,
        state_std=state_std,
        state_dim=state_dim,
        scale=scale,
        act_dim=act_dim,
        oppnt_state_dim=oppnt_state_dim,
        oppnt_act_dims=oppnt_act_dims,
        num_opponents=num_opponents,
        max_len=max_len,
        max_ep_len=max_ep_len
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)


def create_online_seq_dataloader(
    trajectories: List[Dict[str, Any]],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    scale: float,
    state_dim: int,
    act_dim: int,
    oppnt_state_dims: List[int],
    oppnt_act_dims: List[int],
    num_opponents: int,
    max_len: int = 20,
    max_ep_len: int = 25,
    batch_size: int = 256,
    num_workers: int = 2,
    model_type: str = 'dt'
):
    dataset = OnlineSeqDataset(
        trajectories=trajectories,
        state_mean=state_mean,
        state_std=state_std,
        state_dim=state_dim,
        scale=scale,
        act_dim=act_dim,
        oppnt_state_dims=oppnt_state_dims,
        oppnt_act_dims=oppnt_act_dims,
        num_opponents=num_opponents,
        max_len=max_len,
        max_ep_len=max_ep_len
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)


def create_vae_dataloader(
    trajectories: List[Dict[str, Any]],
    state_dim: int,
    move_act_dim: int,
    comm_act_dim: int,
    hidden_dim: int,
    max_len: int = 5,
    max_ep_len: int = 25,
    batch_size: int = 256,
    num_workers: int = 2,
):
    dataset = VAEDataset(
        trajectories=trajectories,
        state_dim=state_dim,
        move_act_dim=move_act_dim,
        comm_act_dim=comm_act_dim,
        hidden_dim=hidden_dim,
        max_len=max_len,
        max_ep_len=max_ep_len
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)


def create_liam_dataloader(
    trajectories: List[Dict[str, Any]],
    state_dim: int,
    move_act_dim: int,
    comm_act_dim: int,
    hidden_dim: int,
    max_len: int = 5,
    max_ep_len: int = 25,
    batch_size: int = 256,
    num_workers: int = 2,
):
    dataset = LIAMDataset(
        trajectories=trajectories,
        state_dim=state_dim,
        move_act_dim=move_act_dim,
        comm_act_dim=comm_act_dim,
        hidden_dim=hidden_dim,
        max_len=max_len,
        max_ep_len=max_ep_len
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)


def create_cql_dataloader(
    trajectories: List[Dict[str, Any]],
    state_dim: int,
    move_act_dim: int,
    comm_act_dim: int,
    hidden_dim: int,
    max_ep_len: int = 25,
    batch_size: int = 256,
    num_workers: int = 2,
):
    dataset = CQLDataset(
        trajectories=trajectories,
        state_dim=state_dim,
        move_act_dim=move_act_dim,
        comm_act_dim=comm_act_dim,
        hidden_dim=hidden_dim,
        max_ep_len=max_ep_len
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)


def create_cbom_dataloader(
    trajectories: List[Dict[str, Any]],
    state_dim: int,
    move_act_dim: int,
    comm_act_dim: int,
    hidden_dim: int,
    max_ep_len: int = 25,
    batch_size: int = 256,
    num_workers: int = 2,
):
    dataset = CBOMDataset(
        trajectories=trajectories,
        state_dim=state_dim,
        move_act_dim=move_act_dim,
        comm_act_dim=comm_act_dim,
        hidden_dim=hidden_dim,
        max_ep_len=max_ep_len
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)


class SeqDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        scale: float,
        state_dim: int,
        act_dim: int,
        oppnt_state_dim: int,
        oppnt_act_dims: List[int],
        num_opponents: int,
        max_len: int = 20,
        max_ep_len: int = 25
    ):
        self.trajectories = trajectories
        self.inds = []
        for t, traj in enumerate(self.trajectories):
            for si in range(traj['rewards'].shape[0]):
                self.inds.append((t, si))

        self.num_opponents = num_opponents
        self.scale = scale
        self.state_mean = state_mean
        self.state_std = state_std
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.oppnt_state_dim = oppnt_state_dim
        self.oppnt_act_dims = oppnt_act_dims
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.task2idx = {None: 0, 'finetune': 1, 'opponent_classification': 2}

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        ti, si = self.inds[idx]
        traj = self.trajectories[ti]

        # get sequences from dataset
        # s = traj['observations'][si:si + self.max_len].reshape(-1, self.state_dim)
        # a = traj['actions'][si:si + self.max_len].reshape(-1, self.act_dim)
        # o_s = traj['opponent_observations'][si:si + self.max_len].reshape(-1, self.oppnt_state_dim)
        # o_a = traj['opponent_actions'][si:si + self.max_len].reshape(-1, self.num_opponents, self.oppnt_act_dims[0])
        # o = traj['opponent'][si:si + self.max_len].reshape(-1)
        # r = traj['rewards'][si:si + self.max_len].reshape(-1, 1)
        # o_r = traj['opponent_rewards'][si:si + self.max_len].reshape(-1, 1)
        # timesteps = np.arange(si, si + s.shape[0]).reshape(-1)
        # timesteps[timesteps >= self.max_ep_len] = self.max_ep_len-1  # padding cutoff
        # rtg = discount_cumsum(traj['rewards'][si:], gamma=1.)[:s.shape[0]].reshape(-1, 1)

        s = traj['observations'][max(si - self.max_len, 0):max(si, 1)].reshape(-1, self.state_dim)
        a = traj['actions'][max(si - self.max_len, 0):max(si, 1)].reshape(-1, self.act_dim)
        o_s = traj['opponent_observations'][max(si - self.max_len, 0):max(si, 1)].reshape(-1, self.oppnt_state_dim)
        o_a = traj['opponent_actions'][max(si - self.max_len, 0):max(si, 1)].reshape(-1, self.num_opponents, self.oppnt_act_dims[0])
        o = traj['opponent'][max(si - self.max_len, 0):max(si, 1)].reshape(-1)
        r = traj['rewards'][max(si - self.max_len, 0):max(si, 1)].reshape(-1, 1)
        o_r = traj['opponent_rewards'][max(si - self.max_len, 0):max(si, 1)].reshape(-1, 1)
        timesteps = np.arange(max(si - s.shape[0], 0), max(si, 1)).reshape(-1)
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len-1  # padding cutoff
        rtg = discount_cumsum(traj['rewards'][max(si - self.max_len, 0):], gamma=1.)[:s.shape[0]].reshape(-1, 1)

        if rtg.shape[0] < s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        # padding and state + reward normalization
        tlen = s.shape[0]
        s = np.concatenate([np.zeros((max(self.max_len - tlen, 0), self.state_dim)), s], axis=0)
        s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((max(self.max_len - tlen, 0), self.act_dim)), a], axis=0)
        o_s = np.concatenate([np.zeros((max(self.max_len - tlen, 0), self.oppnt_state_dim)), o_s], axis=0)
        o_a = np.concatenate([np.ones((max(self.max_len - tlen, 0), self.num_opponents, self.oppnt_act_dims[0])), o_a], axis=0)
        o = np.concatenate([np.zeros((max(self.max_len - tlen, 0))), o], axis=0)
        r = np.concatenate([np.zeros((max(self.max_len - tlen, 0), 1)), r], axis=0)
        o_r = np.concatenate([np.zeros((max(self.max_len - tlen, 0), 1)), o_r], axis=0)
        rtg = np.concatenate([np.zeros((max(self.max_len - tlen, 0), 1)), rtg], axis=0) / self.scale
        timesteps = np.concatenate([np.zeros((max(self.max_len - tlen, 0))), timesteps], axis=0)
        mask = np.concatenate([np.zeros((max(self.max_len - tlen, 0))), np.ones((tlen))], axis=0)

        task_name = traj.get('task', None)
        task_idx = self.task2idx[task_name]
        task = np.ones((self.max_len)) * task_idx

        # print(f"trajectory({ti}), index({si})")
        # print(f"tlen = {tlen}")
        # print(f"(max_len - tlen) = {self.max_len - tlen}")
        # print("s shape = ", s.shape)
        # print("a shape = ", a.shape)
        # print("o_s shape = ", o_s.shape)
        # print("o_a shape = ", o_a.shape)
        # print("rtg shape = ", rtg.shape)
        # print("timesteps shape = ", timesteps.shape)
        # print("mask shape = ", mask.shape)

        # if si == 0:
        #     print(f"Opponent observations at trajectory({ti}), index({si}): ", o_s)

        return s, a, o_s, o_a, r, timesteps, mask


class OnlineSeqDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        scale: float,
        state_dim: int,
        act_dim: int,
        oppnt_state_dims: List[int],
        oppnt_act_dims: List[int],
        num_opponents: int,
        max_len: int = 20,
        max_ep_len: int = 25
    ):
        episode_length, num_envs = trajectories["observations"].shape[0:2]
        batch_size = episode_length * num_envs
        self.envinds = np.arange(num_envs)
        np.random.shuffle(self.envinds)
        self.flatinds = np.arange(batch_size).reshape(episode_length, num_envs)
        self.inds = self.flatinds[:, self.envinds].ravel()

        # print("Trajectory obs shape: ", trajectories["observations"].shape)
        # print("Trajectory prev_actions shape: ", trajectories["prev_actions"].shape)
        # print("Trajectory actions shape: ", trajectories["actions"].shape)
        # print("Trajectory returns shape: ", trajectories["returns"].shape)
        # print("Trajectory hiddens 1 shape: ", trajectories["hiddens"][0].shape)
        # print("Trajectory hiddens 2 shape: ", trajectories["hiddens"][1].shape)

        self.obs = cast(trajectories["observations"])
        self.prev_actions = cast(trajectories["prev_actions"])
        self.actions = cast(trajectories["actions"])
        self.opponent_observations = cast(trajectories["opponent_observations"])
        self.opponent_actions = cast(trajectories["opponent_actions"])
        self.rewards = cast(trajectories["rewards"])
        self.returns = cast(trajectories["returns"])
        self.advantages = cast(trajectories["advantages"])
        self.hiddens = (
            trajectories["hiddens"][0],
            trajectories["hiddens"][1]
        )

        # print("Batch obs shape: ", self.obs.shape)
        # print("Batch prev_actions shape: ", self.prev_actions.shape)
        # print("Batch actions shape: ", self.actions.shape)
        # print("Batch returns shape: ", self.returns.shape)
        # print("Batch hiddens 1 shape: ", self.hiddens[0].shape)
        # print("Batch hiddens 2 shape: ", self.hiddens[1].shape)

        self.num_opponents = num_opponents
        self.scale = scale
        self.state_mean = state_mean
        self.state_std = state_std
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.oppnt_state_dim = sum(oppnt_state_dims)
        self.oppnt_act_dims = oppnt_act_dims
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.task2idx = {None: 0, 'finetune': 1, 'opponent_classification': 2}

    def generate(self):
        obs_batch = self.obs[self.inds]
        prev_actions_batch = self.prev_actions[self.inds]
        actions_batch = self.actions[self.inds]
        returns_batch = self.returns[self.inds]
        advantages_batch = self.advantages[self.inds]
        hiddens_batch = (
            self.hiddens[0][:, self.envinds],
            self.hiddens[1][:, self.envinds]
        )

        yield obs_batch, prev_actions_batch, actions_batch, returns_batch, advantages_batch, hiddens_batch

    def __iter__(self):
        return self.generate()

    # def __len__(self):
    #     return len(self.inds)

    # def __getitem__(self, idx):

    #     s = traj['observations'][si:si + self.max_len].reshape(-1, self.state_dim)
    #     p_a = traj['actions'][si:si + self.max_len].reshape(-1, self.act_dim)
    #     a = traj['actions'][si:si + self.max_len].reshape(-1, self.act_dim)
    #     o_s = traj['opponent_observations'][si:si + self.max_len].reshape(-1, self.oppnt_state_dim)
    #     o_a = traj['opponent_actions'][si:si + self.max_len].reshape(-1, self.num_opponents, self.oppnt_act_dims[0])
    #     returns = traj['returns'][si:si + self.max_len].reshape(-1, 1)
    #     rtg = discount_cumsum(traj['rewards'][si:], gamma=1.)[:s.shape[0]].reshape(-1, 1)
    #     timesteps = np.arange(si, si + s.shape[0]).reshape(-1)
    #     timesteps[timesteps >= self.max_ep_len] = self.max_ep_len-1  # padding cutoff

    #     tlen = s.shape[0]
    #     mask = np.concatenate([np.zeros((max(self.max_len - tlen, 0))), np.ones((tlen))], axis=0)

    #     return s, p_a, rtg, timesteps, mask, a, o_s, o_a, returns


class VAEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        state_dim: int,
        move_act_dim: int,
        comm_act_dim: int,
        hidden_dim: int,
        max_len: int = 5,
        max_ep_len: int = 25
    ):
        self.trajectories = trajectories
        self.inds = []
        for t, traj in enumerate(self.trajectories):
            self.trajectories[t]['prev_move_actions'] = np.concatenate((np.zeros((1, move_act_dim)), self.trajectories[t]['move_actions'][:-1]))
            self.trajectories[t]['prev_comm_actions'] = np.concatenate((np.zeros((1, comm_act_dim)), self.trajectories[t]['comm_actions'][:-1]))
            for si in range(0, traj['rewards'].shape[0] - max_len, max_len):
                self.inds.append((t, si))

        self.state_dim = state_dim
        self.move_act_dim = move_act_dim
        self.comm_act_dim = comm_act_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.max_ep_len = max_ep_len

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        ti, si = self.inds[idx]
        traj = self.trajectories[ti]

        # get sequences from dataset
        s_a = traj['observations'][si:si + self.max_len].reshape(-1, self.state_dim)
        m_a = traj['move_actions'][si:si + self.max_len].reshape(-1, self.move_act_dim)
        c_a = traj['comm_actions'][si:si + self.max_len].reshape(-1, self.comm_act_dim)
        p_m_a = traj['prev_move_actions'][si:si + self.max_len].reshape(-1, self.move_act_dim)
        p_c_a = traj['prev_comm_actions'][si:si + self.max_len].reshape(-1, self.comm_act_dim)
        n_s_a = traj['observations'][si + 1:si + self.max_len + 1].reshape(-1, self.state_dim)
        n_m_a = traj['move_actions'][si + 1:si + self.max_len + 1].reshape(-1, self.move_act_dim)
        n_c_a = traj['comm_actions'][si + 1:si + self.max_len + 1].reshape(-1, self.comm_act_dim)
        s_o = traj['opponent_observations'][si:si + self.max_len].reshape(-1, self.state_dim)
        m_o = traj['opponent_move_actions'][si:si + self.max_len].reshape(-1, self.move_act_dim)
        c_o = traj['opponent_comm_actions'][si:si + self.max_len].reshape(-1, self.comm_act_dim)
        h1 = torch.zeros((1, self.hidden_dim))
        h2 = torch.zeros((1, self.hidden_dim))
        r = traj['rewards'][si:si + self.max_len].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][si:si + self.max_len].reshape(-1, 1)
        else:
            d = traj['dones'][si:si + self.max_len].reshape(-1, 1)

        return s_a, m_a, c_a, p_m_a, p_c_a, n_s_a, n_m_a, n_c_a, s_o, m_o, c_o, h1, h2, r, d


class LIAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        state_dim: int,
        move_act_dim: int,
        comm_act_dim: int,
        hidden_dim: int,
        max_len: int = 5,
        max_ep_len: int = 25
    ):
        self.trajectories = trajectories
        self.inds = []
        for t, traj in enumerate(self.trajectories):
            self.trajectories[t]['prev_move_actions'] = np.concatenate((np.zeros((1, move_act_dim)), self.trajectories[t]['move_actions'][:-1]))
            self.trajectories[t]['prev_comm_actions'] = np.concatenate((np.zeros((1, comm_act_dim)), self.trajectories[t]['comm_actions'][:-1]))
            for si in range(0, traj['rewards'].shape[0] - max_len, max_len):
                self.inds.append((t, si))

        self.state_dim = state_dim
        self.move_act_dim = move_act_dim
        self.comm_act_dim = comm_act_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.max_ep_len = max_ep_len

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        ti, si = self.inds[idx]
        traj = self.trajectories[ti]

        # get sequences from dataset
        s_a = traj['observations'][si:si + self.max_len].reshape(-1, self.state_dim)
        m_a = traj['move_actions'][si:si + self.max_len].reshape(-1, self.move_act_dim)
        c_a = traj['comm_actions'][si:si + self.max_len].reshape(-1, self.comm_act_dim)
        p_m_a = traj['prev_move_actions'][si:si + self.max_len].reshape(-1, self.move_act_dim)
        p_c_a = traj['prev_comm_actions'][si:si + self.max_len].reshape(-1, self.comm_act_dim)
        n_s_a = traj['observations'][si + 1:si + self.max_len + 1].reshape(-1, self.state_dim)
        n_m_a = traj['move_actions'][si + 1:si + self.max_len + 1].reshape(-1, self.move_act_dim)
        n_c_a = traj['comm_actions'][si + 1:si + self.max_len + 1].reshape(-1, self.comm_act_dim)
        s_o = traj['opponent_observations'][si:si + self.max_len].reshape(-1, self.state_dim)
        m_o = traj['opponent_move_actions'][si:si + self.max_len].reshape(-1, self.move_act_dim)
        c_o = traj['opponent_comm_actions'][si:si + self.max_len].reshape(-1, self.comm_act_dim)
        h1 = torch.zeros((1, self.hidden_dim))
        h2 = torch.zeros((1, self.hidden_dim))
        r = traj['rewards'][si:si + self.max_len].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][si:si + self.max_len].reshape(-1, 1)
        else:
            d = traj['dones'][si:si + self.max_len].reshape(-1, 1)

        return s_a, m_a, c_a, p_m_a, p_c_a, n_s_a, n_m_a, n_c_a, s_o, m_o, c_o, h1, h2, r, d


class CQLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        state_dim: int,
        move_act_dim: int,
        comm_act_dim: int,
        hidden_dim: int,
        max_ep_len: int = 25
    ):
        self.trajectories = trajectories
        self.inds = []
        for t, traj in enumerate(self.trajectories):
            self.trajectories[t]['prev_move_actions'] = np.concatenate((np.zeros((1, move_act_dim)), self.trajectories[t]['move_actions'][:-1]))
            self.trajectories[t]['prev_comm_actions'] = np.concatenate((np.zeros((1, comm_act_dim)), self.trajectories[t]['comm_actions'][:-1]))
            for si in range(traj['rewards'].shape[0] - 1):
                self.inds.append((t, si))

        self.state_dim = state_dim
        self.move_act_dim = move_act_dim
        self.comm_act_dim = comm_act_dim
        self.hidden_dim = hidden_dim
        self.max_ep_len = max_ep_len

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        ti, si = self.inds[idx]
        traj = self.trajectories[ti]

        # get sequences from dataset
        s_a = traj['observations'][si].reshape(self.state_dim)
        m_a = traj['move_actions'][si].reshape(self.move_act_dim)
        c_a = traj['comm_actions'][si].reshape(self.comm_act_dim)
        if si == 0:
            p_m_a = np.zeros((self.move_act_dim))
            p_c_a = np.zeros((self.comm_act_dim))
        else:
            p_m_a = traj['move_actions'][si - 1].reshape(self.move_act_dim)
            p_c_a = traj['comm_actions'][si - 1].reshape(self.comm_act_dim)
        n_s_a = traj['observations'][si + 1].reshape(self.state_dim)
        n_m_a = traj['move_actions'][si + 1].reshape(self.move_act_dim)
        n_c_a = traj['comm_actions'][si + 1].reshape(self.comm_act_dim)
        s_o = traj['opponent_observations'][si].reshape(self.state_dim)
        m_o = traj['opponent_move_actions'][si].reshape(self.move_act_dim)
        c_o = traj['opponent_comm_actions'][si].reshape(self.comm_act_dim)
        h1 = torch.zeros((self.hidden_dim))
        h2 = torch.zeros((self.hidden_dim))
        r = traj['rewards'][si].reshape(1)
        if 'terminals' in traj:
            d = traj['terminals'][si].reshape(-1)
        else:
            d = traj['dones'][si].reshape(-1)

        return s_a, m_a, c_a, n_s_a, n_m_a, n_c_a, s_o, m_o, c_o, r, d


class CBOMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        state_dim: int,
        move_act_dim: int,
        comm_act_dim: int,
        hidden_dim: int,
        max_ep_len: int = 25
    ):
        self.trajectories = trajectories
        # import sys
        # print("Dataset trajectories opponent ids: ", [t['opponent'] for t in trajectories])
        # sys.exit()
        self.inds = []
        for t, traj in enumerate(self.trajectories):
            for si in range(traj['rewards'].shape[0] - 1):
                self.inds.append((t, si))

        self.state_dim = state_dim
        self.move_act_dim = move_act_dim
        self.comm_act_dim = comm_act_dim
        self.hidden_dim = hidden_dim
        self.max_ep_len = max_ep_len

        self.task2idx = {None: 0, 'finetune': 1, 'opponent_classification': 2}

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        ti, si = self.inds[idx]
        traj = self.trajectories[ti]

        # get sequences from dataset
        s_a = traj['observations'][si].reshape(self.state_dim)
        m_a = traj['move_actions'][si].reshape(self.move_act_dim)
        c_a = traj['comm_actions'][si].reshape(self.comm_act_dim)
        if si == 0:
            p_m_a = np.zeros((self.move_act_dim), dtype=np.float32)
            p_c_a = np.zeros((self.comm_act_dim), dtype=np.float32)
        else:
            p_m_a = traj['move_actions'][si - 1].reshape(self.move_act_dim)
            p_c_a = traj['comm_actions'][si - 1].reshape(self.comm_act_dim)
        n_s_a = traj['observations'][si + 1].reshape(self.state_dim)
        n_m_a = traj['move_actions'][si + 1].reshape(self.move_act_dim)
        n_c_a = traj['comm_actions'][si + 1].reshape(self.comm_act_dim)
        # s_o = traj['opponent_observations'][si].reshape(self.state_dim)
        # m_o = traj['opponent_move_actions'][si].reshape(self.move_act_dim)
        # c_o = traj['opponent_comm_actions'][si].reshape(self.comm_act_dim)
        o = traj['opponent'][si].reshape(-1)
        r = traj['rewards'][si].reshape(1)
        if 'terminals' in traj:
            d = traj['terminals'][si].reshape(-1)
        else:
            d = traj['dones'][si].reshape(-1)

        task_name = traj.get('task', None)
        task = self.task2idx[task_name]

        return s_a, m_a, c_a, p_m_a, p_c_a, n_s_a, n_m_a, n_c_a, o, r, d, task
