import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from opponent_transformer.training.trainer import Trainer


class CBOMTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tau = 1e-3
        self.gamma = 0.99

        self.optimizer1 = torch.optim.Adam(params=self.model.network.parameters(), lr=1e-3)
        self.optimizer2 = torch.optim.Adam(params=self.model.opponent_model.parameters(), lr=1e-3)

    def train_iteration(self, num_steps: int, iter_num: int, dataloader, finetune: bool = False):

        train_losses = []
        logs = dict()
        self.diagnostics['training/loss'] = []
        self.diagnostics['training/cql_loss'] = []
        self.diagnostics['training/oppnt_id_loss'] = []
        self.diagnostics['training/opponent_accuracy'] = []

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {iter_num}"):
                train_loss = self.train_step(batch, finetune=finetune)
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        logs['time/total'] = time.time() - self.start_time
        logs['training/lr'] = self.optimizer.param_groups[0]["lr"]
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = np.mean(self.diagnostics[k])

        print('=' * 80)
        print(f'Iteration {iter_num}')
        for k, v in logs.items():
            print(f'{k}: {v}')

        return logs

    def train_step(self, batch, finetune: bool = False):
        states, move_actions, comm_actions, prev_move_actions, prev_comm_actions, next_states, next_move_actions, next_comm_actions, oppnt_ids, rewards, dones, task = batch

        states = states.to(dtype=torch.float32, device=self.device)
        move_actions = move_actions.to(dtype=torch.long, device=self.device)
        comm_actions = comm_actions.to(dtype=torch.long, device=self.device)
        prev_move_actions = prev_move_actions.to(dtype=torch.long, device=self.device)
        prev_comm_actions = prev_comm_actions.to(dtype=torch.long, device=self.device)
        next_states = next_states.to(dtype=torch.float32, device=self.device)
        next_move_actions = next_move_actions.to(dtype=torch.long, device=self.device)
        next_comm_actions = next_comm_actions.to(dtype=torch.long, device=self.device)
        oppnt_targets = oppnt_ids.to(dtype=torch.long, device=self.device).reshape(-1)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.long, device=self.device)

        actions = torch.cat((move_actions, comm_actions), dim=-1)
        prev_actions = torch.cat((prev_move_actions, prev_comm_actions), dim=-1)
        next_actions = torch.cat((next_move_actions, next_comm_actions), dim=-1)
        move_actions = move_actions.argmax(dim=-1).unsqueeze(-1)
        comm_actions = comm_actions.argmax(dim=-1).unsqueeze(-1)
        next_move_actions = next_move_actions.argmax(dim=-1).unsqueeze(-1)
        next_comm_actions = next_comm_actions.argmax(dim=-1).unsqueeze(-1)

        oppnt_preds = self.model.predict_opponent(states, prev_actions)

        if finetune:
            q1_move_loss = torch.tensor(0)
            q1_comm_loss = torch.tensor(0)
        else:
            with torch.no_grad():
                target_oppnt_preds = self.model.predict_opponent(next_states, actions)
                # target_oppnt_preds = F.one_hot(target_oppnt_preds.argmax(-1), self.model.num_opponents).detach()
                target_oppnt_preds = F.one_hot(oppnt_targets, self.model.num_opponents)
                target_input_tensor = torch.cat((next_states, target_oppnt_preds), dim=-1)

                Q_targets_next_move, Q_targets_next_comm  = self.model.target_net(target_input_tensor)
                Q_targets_next_move = Q_targets_next_move.detach().max(1)[0].unsqueeze(1)
                Q_targets_next_comm = Q_targets_next_comm.detach().max(1)[0].unsqueeze(1)
                Q_targets_move = rewards + (self.gamma * Q_targets_next_move * (1 - dones))
                Q_targets_comm = rewards + (self.gamma * Q_targets_next_comm * (1 - dones))

            # oppnt_pred_ids = F.one_hot(oppnt_preds.argmax(-1), self.model.num_opponents).detach()
            oppnt_pred_ids = F.one_hot(oppnt_targets, self.model.num_opponents)
            target_input_tensor = torch.cat((states, oppnt_pred_ids), dim=-1)

            Q_a_s_move, Q_a_s_comm = self.model.network(target_input_tensor)
            Q_expected_move = Q_a_s_move.gather(1, move_actions)
            Q_expected_comm = Q_a_s_comm.gather(1, comm_actions)

            Q_a_s_move = Q_a_s_move[(task == 0)]
            move_actions = move_actions[(task == 0)]
            Q_a_s_comm = Q_a_s_comm[(task == 0)]
            comm_actions = comm_actions[(task == 0)]
            Q_expected_move = Q_expected_move[(task == 0)]
            Q_targets_move = Q_targets_move[(task == 0)]
            Q_expected_comm = Q_expected_comm[(task == 0)]
            Q_targets_comm = Q_targets_comm[(task == 0)]

            # if Q_a_s_move.shape[0] != 64:
            #     print("Q_a_s_move shape = ", Q_a_s_move.shape)
            #     print("move_actions shape = ", move_actions.shape)
            #     print("Q_a_s_comm shape = ", Q_a_s_comm.shape)
            #     print("comm_actions shape = ", comm_actions.shape)
            #     print("Q_expected_move shape = ", Q_expected_move.shape)
            #     print("Q_targets_move shape = ", Q_targets_move.shape)
            #     print("Q_expected_comm shape = ", Q_expected_comm.shape)
            #     print("Q_targets_comm shape = ", Q_targets_comm.shape)
            #     print("Task: ", task)

            cql1_move_loss = self.cql_loss(Q_a_s_move, move_actions)
            cql1_comm_loss = self.cql_loss(Q_a_s_comm, comm_actions)

            bellman_move_error = F.mse_loss(Q_expected_move, Q_targets_move)
            bellman_comm_error = F.mse_loss(Q_expected_comm, Q_targets_comm)

            q1_move_loss = cql1_move_loss + 0.5 * bellman_move_error
            q1_comm_loss = cql1_comm_loss + 0.5 * bellman_comm_error

        # Opponent ID Loss
        oppnt_id_loss = F.cross_entropy(oppnt_preds, oppnt_targets)

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        if not finetune:
            (q1_move_loss + q1_comm_loss).backward()
        oppnt_id_loss.backward()
        nn.utils.clip_grad_norm_(self.model.network.parameters(), 1.)
        nn.utils.clip_grad_norm_(self.model.opponent_model.parameters(), 1.)
        self.optimizer1.step()
        self.optimizer2.step()

        self.soft_update(self.model.network, self.model.target_net)

        with torch.no_grad():
            self.diagnostics['training/cql_loss'].append((q1_move_loss + q1_comm_loss).item())
            self.diagnostics['training/oppnt_id_loss'].append(oppnt_id_loss.item())
            self.diagnostics['training/opponent_accuracy'].append(torch.sum(oppnt_preds.argmax(dim=-1) == oppnt_targets).cpu() / oppnt_preds.shape[0])

        return (q1_move_loss + q1_comm_loss + oppnt_id_loss).item()

    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)

        return (logsumexp - q_a).mean()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
