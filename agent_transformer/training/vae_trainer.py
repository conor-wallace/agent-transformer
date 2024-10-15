import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from opponent_transformer.training.trainer import Trainer
from opponent_transformer.training.seq_trainer import cre_loss_fn


class VAETrainer(Trainer):

    def train_iteration(self, num_steps: int, iter_num: int, dataloader, finetune: bool = False):

        train_losses = []
        logs = dict()
        self.diagnostics['training/loss'] = []
        self.diagnostics['training/action_loss'] = []
        self.diagnostics['training/agent_action_accuracy'] = []
        # self.diagnostics['training/opponent_policy_loss'] = []
        self.diagnostics['training/opponent_states_loss'] = []
        # self.diagnostics['training/opponent_returns_loss'] = []
        self.diagnostics['training/opponent_actions_loss'] = []
        # self.diagnostics['training/opponent_policy_accuracy'] = []
        self.diagnostics['training/opponent_actions_accuracy'] = []

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {iter_num}"):
                train_loss = self.train_step(batch, finetune=finetune)
                train_losses.append(train_loss)
                if self.policy_scheduler is not None:
                    self.policy_scheduler.step()
                if self.opponent_scheduler is not None:
                    self.opponent_scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        logs['time/total'] = time.time() - self.start_time
        logs['training/lr'] = self.policy_optimizer.param_groups[0]["lr"]
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
        agent_states, agent_actions, agent_probs, oppnt_states, oppnt_actions, rewards, oppnt_rewards, rtg, advantages, dones, oppnt_policy, timesteps, task, attention_mask = batch

        agent_states = agent_states.to(dtype=torch.float32, device=self.device)
        agent_actions = agent_actions.to(dtype=torch.float32, device=self.device)
        agent_probs = agent_probs.to(dtype=torch.float32, device=self.device)
        oppnt_states = oppnt_states.to(dtype=torch.float32, device=self.device)
        oppnt_actions = oppnt_actions.to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        oppnt_rewards = oppnt_rewards.to(dtype=torch.float32, device=self.device)
        rtg = rtg.to(dtype=torch.float32, device=self.device)
        advantages = advantages.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.long, device=self.device)
        oppnt_policy = oppnt_policy.to(dtype=torch.long, device=self.device)
        timesteps = timesteps.to(dtype=torch.long, device=self.device)
        task = task.to(dtype=torch.long, device=self.device)
        attention_mask = attention_mask.to(dtype=torch.long, device=self.device)

        action_target = torch.clone(agent_actions)
        oppnt_states_target = torch.clone(oppnt_states)
        oppnt_policy_target = torch.clone(oppnt_policy)
        oppnt_actions_target = torch.clone(oppnt_actions)
        oppnt_returns_target = torch.clone(oppnt_rewards)

        oppnt_outputs = self.model.oppnt_model(
            states=agent_states, actions=agent_actions
        )

        state_preds, action_preds, return_preds = self.model.forward(
            states=agent_states, actions=agent_actions, returns_to_go=rtg[:,:-1], timesteps=timesteps, attention_mask=attention_mask, oppnt_outputs=oppnt_outputs['embeddings'].detach()
        )

        # Compute Policy Loss
        if finetune:
            # move_action_loss, move_action_nll, move_action_entropy = nll_loss_fn(
            #     move_action_preds,  # a_hat_dist
            #     move_action_target,
            #     attention_mask,
            #     self.model.temperature().detach(),  # no gradient taken here
            # )

            # comm_action_loss, comm_action_nll, comm_action_entropy = nll_loss_fn(
            #     comm_action_preds,  # a_hat_dist
            #     comm_action_target,
            #     attention_mask,
            #     self.model.temperature().detach(),  # no gradient taken here
            # )
            critic_loss = a2c_critic_loss_fn(advantages, attention_mask)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .25)
            self.critic_optimizer.step()

            action_loss = a2c_actor_loss_fn(agent_probs, agent_actions, advantages, attention_mask)
        else:
            act_dim = action_preds.shape[2]

            action_loss = cre_loss_fn(
                action_preds.reshape(-1, act_dim),  # a_hat_dist
                action_target.reshape(-1, act_dim).argmax(dim=-1),
                attention_mask.reshape(-1),
                task.reshape(-1)
            )

        policy_loss = action_loss

        # Compute Opponent Model Loss
        oppnt_preds = self.model.oppnt_model.predict_opponent(oppnt_outputs['embeddings'])
        opponent_loss = torch.tensor(0.0)
        # if 'policy' in oppnt_preds:
        #     oppnt_policy_preds = oppnt_preds['policy']
        #     oppnt_policy_dim = oppnt_policy_preds.shape[2]
        #     oppnt_policy_preds = oppnt_policy_preds.reshape(-1, oppnt_policy_dim)[attention_mask.reshape(-1) > 0]
        #     oppnt_policy_target = oppnt_policy_target.reshape(-1)[attention_mask.reshape(-1) > 0]

        #     # Opponent ID Loss
        #     oppnt_policy_loss = F.cross_entropy(oppnt_policy_preds, oppnt_policy_target)
        #     loss = loss + oppnt_policy_loss
        if 'states' in oppnt_preds:
            oppnt_states_preds = oppnt_preds['states']
            oppnt_states_dim = oppnt_states_preds.shape[2]
            oppnt_states_preds = oppnt_states_preds.reshape(-1, oppnt_states_dim)[attention_mask.reshape(-1) > 0]
            oppnt_states_target = oppnt_states_target.reshape(-1, oppnt_states_dim)[attention_mask.reshape(-1) > 0]

            # Opponent ID Loss
            oppnt_states_loss = F.mse_loss(oppnt_states_preds, oppnt_states_target)
            opponent_loss = opponent_loss + oppnt_states_loss
        if 'actions' in oppnt_preds:
            oppnt_actions_preds = oppnt_preds['actions']
            oppnt_actions_dim = oppnt_actions_preds.shape[2]
            oppnt_actions_preds = oppnt_actions_preds.reshape(-1, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]
            oppnt_actions_target = oppnt_actions_target.reshape(-1, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]

            # Opponent ID Loss
            # oppnt_actions_loss = -torch.log(torch.sum(oppnt_actions_target * oppnt_actions_preds, dim=-1)).mean()
            oppnt_actions_loss = F.cross_entropy(oppnt_actions_preds, oppnt_actions_target)
            opponent_loss = opponent_loss + oppnt_actions_loss
        # if 'returns' in oppnt_preds:
        #     oppnt_returns_preds = oppnt_preds['returns']
        #     oppnt_returns_dim = oppnt_returns_preds.shape[2]
        #     oppnt_returns_preds = oppnt_returns_preds.reshape(-1, oppnt_returns_dim)[attention_mask.reshape(-1) > 0]
        #     oppnt_returns_target = oppnt_returns_target.reshape(-1, oppnt_returns_dim)[attention_mask.reshape(-1) > 0]

        #     # Opponent ID Loss
        #     oppnt_returns_loss = F.mse_loss(oppnt_returns_preds, oppnt_returns_target)
        #     loss = loss + oppnt_returns_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.policy_optimizer.step()

        if self.opponent_optimizer is not None:
            self.opponent_optimizer.zero_grad()
            opponent_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.oppnt_model.parameters(), .25)
            self.opponent_optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/loss'].append(policy_loss.item())
            self.diagnostics['training/action_loss'].append(action_loss.item())

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            self.diagnostics['training/agent_action_accuracy'].append(torch.sum(action_preds.argmax(dim=-1) == action_target.argmax(dim=-1)).detach().cpu().item() / action_preds.shape[0])
            if oppnt_preds is not None:
                # self.diagnostics['training/opponent_policy_loss'].append(oppnt_policy_loss.item())
                self.diagnostics['training/opponent_states_loss'].append(oppnt_states_loss.item())
                # self.diagnostics['training/opponent_returns_loss'].append(oppnt_returns_loss.item())
                self.diagnostics['training/opponent_actions_loss'].append(oppnt_actions_loss.item())
                # self.diagnostics['training/opponent_policy_accuracy'].append(torch.sum(oppnt_policy_preds.argmax(dim=-1) == oppnt_policy_target).cpu() / oppnt_policy_preds.shape[0])
                self.diagnostics['training/opponent_actions_accuracy'].append(torch.sum(oppnt_actions_preds.argmax(dim=-1) == oppnt_actions_target.argmax(dim=-1)).cpu() / oppnt_actions_preds.shape[0])

            # self.diagnostics['training/move_action_entropy'] = move_action_entropy.item()
            # self.diagnostics['training/comm_action_entropy'] = comm_action_entropy.item()
            # self.diagnostics['training/entropy'] = loss.item()
            # print("Agent move action pred shape = ", agent_move_action_preds.shape)
            # print("Agent move action targets shape = ", agent_move_action_target.shape)
            # print("Opponent move action pred shape = ", oppnt_move_action_preds.shape)
            # print("Opponent move action targets shape = ", oppnt_move_action_target.shape)
            # print("Agent move action accuracy shape = ", torch.sum(agent_move_action_preds.argmax(dim=-1) == agent_move_action_target).shape)
            # self.diagnostics['training/agent_move_action_accuracy'] = torch.sum(agent_move_action_preds.argmax(dim=-1) == agent_move_action_target).detach().cpu().item() / agent_move_action_preds.shape[0]
            # self.diagnostics['training/agent_comm_action_accuracy'] = torch.sum(agent_comm_action_preds.argmax(dim=-1) == agent_comm_action_target).detach().cpu().item() / agent_comm_action_preds.shape[0]
            # self.diagnostics['training/opponent_accuracy'].append(torch.sum(oppnt_preds.argmax(dim=-1) == oppnt_target).cpu() / oppnt_preds.shape[0])
            # if len(online_oppnt_preds) > 0:
            #     self.diagnostics['training/online_opponent_accuracy'].append(torch.sum(online_oppnt_preds.argmax(dim=-1) == online_oppnt_target).cpu() / online_oppnt_preds.shape[0])
            # else:
            #     self.diagnostics['training/online_opponent_accuracy'].append(0.0)
            # if oppnt_move_action_preds is not None:
            #     self.diagnostics['training/oppnt_move_action_accuracy'] = torch.sum(oppnt_move_action_preds.argmax(dim=-1) == oppnt_move_action_target).detach().cpu().item() / oppnt_move_action_preds.shape[0]
            # if oppnt_comm_action_preds is not None:
            #     self.diagnostics['training/oppnt_comm_action_accuracy'] = torch.sum(oppnt_comm_action_preds.argmax(dim=-1) == oppnt_comm_action_target).detach().cpu().item() / oppnt_comm_action_preds.shape[0]
            # if oppnt_states_preds is not None:
            #     self.diagnostics['training/oppnt_states_error'] = torch.mean((oppnt_states_preds-oppnt_states_target)**2).detach().cpu().item()

        return policy_loss.detach().cpu().item()