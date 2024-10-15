import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opponent_transformer.training.trainer import Trainer


class CQLLIAMTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tau = 1e-3
        self.gamma = 0.99

        self.optimizer1 = torch.optim.Adam(params=self.model.network.parameters(), lr=3e-4)
        self.optimizer2 = torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), lr=7e-4)

    def train_step(self, batch, finetune: bool = False):
        states, move_actions, comm_actions, prev_move_actions, prev_comm_actions, next_states, next_move_actions, next_comm_actions, oppnt_states, oppnt_move_actions, oppnt_comm_actions, hidden1, hidden2, rewards, dones = batch

        states = states.to(dtype=torch.float32, device=self.device).permute((1, 0, 2))
        move_actions = move_actions.to(dtype=torch.long, device=self.device).permute((1, 0, 2))
        comm_actions = comm_actions.to(dtype=torch.long, device=self.device).permute((1, 0, 2))
        prev_move_actions = prev_move_actions.to(dtype=torch.long, device=self.device).permute((1, 0, 2))
        prev_comm_actions = prev_comm_actions.to(dtype=torch.long, device=self.device).permute((1, 0, 2))
        next_states = next_states.to(dtype=torch.float32, device=self.device).permute((1, 0, 2))
        next_move_actions = next_move_actions.to(dtype=torch.long, device=self.device).permute((1, 0, 2))
        next_comm_actions = next_comm_actions.to(dtype=torch.long, device=self.device).permute((1, 0, 2))
        oppnt_states = oppnt_states.to(dtype=torch.float32, device=self.device).permute((1, 0, 2))
        oppnt_move_actions = oppnt_move_actions.to(dtype=torch.long, device=self.device).permute((1, 0, 2))
        oppnt_comm_actions = oppnt_comm_actions.to(dtype=torch.long, device=self.device).permute((1, 0, 2))
        hidden1 = hidden1.to(dtype=torch.float32, device=self.device).permute((1, 0, 2))
        hidden2 = hidden2.to(dtype=torch.float32, device=self.device).permute((1, 0, 2))
        rewards = rewards.to(dtype=torch.float32, device=self.device).permute((1, 0, 2))
        dones = dones.to(dtype=torch.long, device=self.device).permute((1, 0, 2))

        hidden = (hidden1, hidden2)
        actions = torch.cat((move_actions, comm_actions), dim=-1)
        prev_actions = torch.cat((prev_move_actions, prev_comm_actions), dim=-1)
        next_actions = torch.cat((next_move_actions, next_comm_actions), dim=-1)
        oppnt_actions = torch.cat((oppnt_move_actions, oppnt_comm_actions), dim=-1)
        move_actions = move_actions.argmax(dim=-1).unsqueeze(-1)
        comm_actions = comm_actions.argmax(dim=-1).unsqueeze(-1)
        next_move_actions = next_move_actions.argmax(dim=-1).unsqueeze(-1)
        next_comm_actions = next_comm_actions.argmax(dim=-1).unsqueeze(-1)

        with torch.no_grad():
            targets_embeddings, _ = self.model.compute_embedding(next_states, actions, hidden)
            # print("Target embeddings shape = ", targets_embeddings.shape)
            # print("Next states shape = ", next_states.shape)
            targets_aug_states = torch.cat((next_states, targets_embeddings.detach()), dim=-1)
            Q_targets_next_move, Q_targets_next_comm  = self.model.target_net(targets_aug_states)
            # print("Q shape = ", Q_targets_next_move.shape)
            Q_targets_next_move = Q_targets_next_move.detach().max(-1)[0].unsqueeze(-1)
            # print("Q shape = ", Q_targets_next_move.shape)
            Q_targets_next_comm = Q_targets_next_comm.detach().max(-1)[0].unsqueeze(-1)
            Q_targets_move = rewards + (self.gamma * Q_targets_next_move * (1 - dones))
            Q_targets_comm = rewards + (self.gamma * Q_targets_next_comm * (1 - dones))

        # print("Q shape = ", Q_targets_next_move.shape)

        embeddings, _ = self.model.compute_embedding(states, prev_actions, hidden)
        aug_states = torch.cat((states, embeddings.detach()), dim=-1)
        Q_a_s_move, Q_a_s_comm = self.model.network(aug_states)
        Q_expected_move = Q_a_s_move.gather(dim=-1, index=move_actions)
        Q_expected_comm = Q_a_s_comm.gather(dim=-1, index=comm_actions)

        # print("Q expected shape = ", Q_expected_move.shape)
        # print("Q targets shape = ", Q_targets_move.shape)

        cql1_move_loss = self.cql_loss(Q_a_s_move, move_actions)
        cql1_comm_loss = self.cql_loss(Q_a_s_comm, comm_actions)

        bellman_move_error = F.mse_loss(Q_expected_move, Q_targets_move)
        bellman_comm_error = F.mse_loss(Q_expected_comm, Q_targets_comm)

        q1_move_loss = cql1_move_loss + 0.5 * bellman_move_error
        q1_comm_loss = cql1_comm_loss + 0.5 * bellman_comm_error

        recon_loss1, recon_loss2, move_acc, comm_acc = self.liam_loss(embeddings, oppnt_states, oppnt_actions)

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        (q1_move_loss + q1_comm_loss).backward()
        (recon_loss1 + recon_loss2).backward()
        nn.utils.clip_grad_norm_(self.model.network.parameters(), 1.)
        nn.utils.clip_grad_norm_(self.model.encoder.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 0.5)
        self.optimizer1.step()
        self.optimizer2.step()

        # self.soft_update(self.model.network, self.model.target_net)

        with torch.no_grad():
            self.diagnostics['training/loss'].append((q1_move_loss + q1_comm_loss + recon_loss1 + recon_loss2).item())
            self.diagnostics['training/modelled_obs_loss'].append(recon_loss1.item())
            self.diagnostics['training/modelled_act_loss'].append(recon_loss2.item())
            self.diagnostics['training/opponent_move_accuracy'].append(move_acc)
            self.diagnostics['training/opponent_comm_accuracy'].append(comm_acc)

        return (q1_move_loss + q1_comm_loss + recon_loss1 + recon_loss2).detach().cpu().item()

    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=-1, keepdim=True)
        q_a = q_values.gather(-1, current_action)

        return (logsumexp - q_a).mean()

    def liam_loss(self, embeddings, oppnt_states, oppnt_actions):
        rec_loss1, rec_loss2, move_acc, comm_acc = self.model.eval_decoding(embeddings, oppnt_states, oppnt_actions)

        return rec_loss1.mean(), rec_loss2.mean(), move_acc, comm_acc

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
