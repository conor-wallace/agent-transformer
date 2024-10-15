import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opponent_transformer.training.trainer import Trainer


class CQLTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tau = 1e-3
        self.gamma = 0.99

        self.optimizer1 = torch.optim.Adam(params=self.model.network.parameters(), lr=1e-3)

    def train_step(self, batch, finetune: bool = False):
        states, move_actions, comm_actions, next_states, next_move_actions, next_comm_actions, oppnt_states, oppnt_move_actions, oppnt_comm_actions, hidden1, hidden2, rewards, dones = batch

        states = states.to(dtype=torch.float32, device=self.device)
        move_actions = move_actions.to(dtype=torch.long, device=self.device)
        comm_actions = comm_actions.to(dtype=torch.long, device=self.device)
        next_states = next_states.to(dtype=torch.float32, device=self.device)
        next_move_actions = next_move_actions.to(dtype=torch.long, device=self.device)
        next_comm_actions = next_comm_actions.to(dtype=torch.long, device=self.device)
        oppnt_states = oppnt_states.to(dtype=torch.float32, device=self.device)
        oppnt_move_actions = oppnt_move_actions.to(dtype=torch.long, device=self.device)
        oppnt_comm_actions = oppnt_comm_actions.to(dtype=torch.long, device=self.device)
        hidden1 = hidden1.unsqueeze(0).to(dtype=torch.float32, device=self.device)
        hidden2 = hidden2.unsqueeze(0).to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.long, device=self.device)

        actions = torch.cat((move_actions, comm_actions), dim=-1)
        next_actions = torch.cat((next_move_actions, next_comm_actions), dim=-1)
        oppnt_actions = torch.cat((oppnt_move_actions, oppnt_comm_actions), dim=-1)
        move_actions = move_actions.argmax(dim=-1).unsqueeze(-1)
        comm_actions = comm_actions.argmax(dim=-1).unsqueeze(-1)
        next_move_actions = next_move_actions.argmax(dim=-1).unsqueeze(-1)
        next_comm_actions = next_comm_actions.argmax(dim=-1).unsqueeze(-1)

        with torch.no_grad():
            Q_targets_next_move, Q_targets_next_comm  = self.model.target_net(next_states)
            Q_targets_next_move = Q_targets_next_move.detach().max(1)[0].unsqueeze(1)
            Q_targets_next_comm = Q_targets_next_comm.detach().max(1)[0].unsqueeze(1)
            Q_targets_move = rewards + (self.gamma * Q_targets_next_move * (1 - dones))
            Q_targets_comm = rewards + (self.gamma * Q_targets_next_comm * (1 - dones))

        Q_a_s_move, Q_a_s_comm = self.model.network(states)
        Q_expected_move = Q_a_s_move.gather(1, move_actions)
        Q_expected_comm = Q_a_s_comm.gather(1, comm_actions)

        cql1_move_loss = self.cql_loss(Q_a_s_move, move_actions)
        cql1_comm_loss = self.cql_loss(Q_a_s_comm, comm_actions)

        bellman_move_error = F.mse_loss(Q_expected_move, Q_targets_move)
        bellman_comm_error = F.mse_loss(Q_expected_comm, Q_targets_comm)

        q1_move_loss = cql1_move_loss + 0.5 * bellman_move_error
        q1_comm_loss = cql1_comm_loss + 0.5 * bellman_comm_error

        self.optimizer1.zero_grad()
        (q1_move_loss + q1_comm_loss).backward()
        nn.utils.clip_grad_norm_(self.model.network.parameters(), 1.)
        self.optimizer1.step()

        self.soft_update(self.model.network, self.model.target_net)

        with torch.no_grad():
            self.diagnostics['training/loss'].append((q1_move_loss + q1_comm_loss).item())

        return (q1_move_loss + q1_comm_loss).detach().cpu().item()

    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)

        return (logsumexp - q_a).mean()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
