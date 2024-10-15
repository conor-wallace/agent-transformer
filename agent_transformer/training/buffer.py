import numpy as np
from typing import Any, Dict, List


class ReplayBuffer:
    def __init__(self, capacity: int = int(1e6), num_opponents: int = 3, trajectories: List[Dict[str, Any]] = []):
        self.capacity = capacity * num_opponents
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]
        self.online_trajectories = []

        self.start_idx = 0
        self.online_start_idx = 0

    def __len__(self):
        return len(self.trajectories) + len(self.online_trajectories)

    def add_new_trajectories(self, trajectories=None, online_trajectories=None):
        if trajectories:
            self.trajectories.extend(trajectories)
            self.trajectories = self.trajectories[-self.capacity :]

            # if len(self.trajectories) < self.capacity:
            #     self.trajectories.extend(trajectories)
            #     self.trajectories = self.trajectories[-self.capacity :]
            # else:
            #     self.trajectories[
            #         self.start_idx : self.start_idx + len(trajectories)
            #     ] = trajectories
            #     self.start_idx = (self.start_idx + len(trajectories)) % self.capacity

            assert len(self.trajectories) <= self.capacity
        else:
            self.online_trajectories.extend(online_trajectories)
            self.online_trajectories = self.online_trajectories[-self.capacity :]

            # if len(self.online_trajectories) < self.capacity:
            #     self.online_trajectories.extend(online_trajectories)
            #     self.online_trajectories = self.online_trajectories[-self.capacity :]
            # else:
            #     self.online_trajectories[
            #         self.online_start_idx : self.online_start_idx + len(online_trajectories)
            #     ] = online_trajectories
            #     self.online_start_idx = (self.online_start_idx + len(online_trajectories)) % self.capacity

            assert len(self.online_trajectories) <= self.capacity