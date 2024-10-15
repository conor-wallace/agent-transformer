import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import time


class Trainer:

    def __init__(self, model, optimizer, scheduler, device: str = "cuda"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.diagnostics = dict()
        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num, dataloader):
        train_losses = []
        logs = dict()
        self.diagnostics['training/loss'] = []
        self.diagnostics['training/opponent_states_loss'] = []
        self.diagnostics['training/opponent_actions_loss'] = []
        self.diagnostics['training/opponent_actions_accuracy'] = []

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {iter_num}"):
                train_loss = self.train_step(batch)
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
        
        logs['time/training'] = time.time() - train_start
        logs['training/lr'] = self.optimizer.param_groups[0]["lr"]
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = np.mean(self.diagnostics[k])

        print('=' * 80)
        print(f'Iteration {iter_num}')
        for k, v in logs.items():
            print(f'{k}: {v}')
        
        self.model.save_params('trained_parameters/transformer_opponent_model.pt')

        return logs

    def train_step(self, batch):
        agent_states, agent_actions, oppnt_states, oppnt_actions, rtg, timesteps, attention_mask = batch

        agent_states = agent_states.to(dtype=torch.float32, device=self.device)
        agent_actions = agent_actions.to(dtype=torch.float32, device=self.device)
        oppnt_states = oppnt_states.to(dtype=torch.float32, device=self.device)
        oppnt_actions = oppnt_actions.to(dtype=torch.float32, device=self.device)
        rtg = rtg.to(dtype=torch.float32, device=self.device)
        timesteps = timesteps.to(dtype=torch.long, device=self.device)
        attention_mask = attention_mask.to(dtype=torch.long, device=self.device)

        oppnt_states_target = torch.clone(oppnt_states)
        oppnt_actions_target = torch.clone(oppnt_actions)

        embeddings = self.model(
            states=agent_states, actions=agent_actions, returns_to_go=rtg, timesteps=timesteps, attention_mask=attention_mask,
        )
        oppnt_states_preds, oppnt_actions_preds = self.model.predict_opponent(embeddings)

        # Opponent States Loss
        oppnt_states_dim = oppnt_states_preds.shape[-1]
        oppnt_states_preds = oppnt_states_preds.reshape(-1, oppnt_states_dim)[attention_mask.reshape(-1) > 0]
        oppnt_states_target = oppnt_states_target.reshape(-1, oppnt_states_dim)[attention_mask.reshape(-1) > 0]

        oppnt_states_loss = F.mse_loss(oppnt_states_preds, oppnt_states_target)

        # Opponent Actions Loss
        num_opponents = oppnt_actions_preds.shape[2]
        oppnt_actions_dim = oppnt_actions_preds.shape[3]
        oppnt_actions_preds = oppnt_actions_preds.reshape(-1, num_opponents, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]
        oppnt_actions_target = oppnt_actions_target.reshape(-1, num_opponents, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]

        oppnt_actions_loss = F.cross_entropy(oppnt_actions_preds, oppnt_actions_target)

        # oppnt_actions_loss = torch.tensor(0).to(dtype=torch.float32, device=self.device)
        # for i, oppnt_actions_pred in enumerate(oppnt_actions_preds):
        #     oppnt_actions_pred = oppnt_actions_pred.reshape(-1, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]
        #     oppnt_actions_loss += F.cross_entropy(oppnt_actions_pred, oppnt_actions_target[:, i])

        # Opponent Modeling Loss
        opponent_loss = oppnt_states_loss + oppnt_actions_loss

        self.optimizer.zero_grad()
        opponent_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/loss'].append(opponent_loss.item())
            self.diagnostics['training/opponent_states_loss'].append(oppnt_states_loss.item())
            self.diagnostics['training/opponent_actions_loss'].append(oppnt_actions_loss.item())
            # self.diagnostics['training/opponent_actions_accuracy'].append(torch.sum(oppnt_actions_preds.argmax(dim=-1) == oppnt_actions_target.argmax(dim=-1)).cpu() / oppnt_actions_preds.shape[0])

            oppnt_actions_acc = torch.sum(oppnt_actions_preds.argmax(dim=-1) == oppnt_actions_target.argmax(dim=-1)).cpu() / (oppnt_actions_preds.shape[0] * oppnt_actions_preds.shape[1])

            # oppnt_actions_acc = torch.tensor(0).to(dtype=torch.float32)
            # for i, oppnt_actions_pred in enumerate(oppnt_actions_preds):
            #     oppnt_actions_pred = oppnt_actions_pred.reshape(-1, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]
            #     oppnt_actions_acc += torch.sum(oppnt_actions_pred.argmax(dim=-1) == oppnt_actions_target[:, i].argmax(dim=-1)).cpu() / oppnt_actions_pred.shape[0]

            # oppnt_actions_acc = oppnt_actions_acc / len(oppnt_actions_preds)

            self.diagnostics['training/opponent_actions_accuracy'].append(oppnt_actions_acc)

        return opponent_loss.detach().cpu().item()


class VAETrainer:

    def __init__(self, model, optimizer, scheduler, device: str = "cuda"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.diagnostics = dict()
        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num, dataloader):
        train_losses = []
        logs = dict()
        self.diagnostics['training/loss'] = []
        self.diagnostics['training/opponent_states_loss'] = []
        self.diagnostics['training/opponent_actions_loss'] = []
        self.diagnostics['training/opponent_actions_accuracy'] = []

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {iter_num}"):
                train_loss = self.train_step(batch)
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
        
        logs['time/training'] = time.time() - train_start
        logs['training/lr'] = self.optimizer.param_groups[0]["lr"]
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = np.mean(self.diagnostics[k])

        print('=' * 80)
        print(f'Iteration {iter_num}')
        for k, v in logs.items():
            print(f'{k}: {v}')

        self.model.save_params('trained_parameters/vae_opponent_model.pt')

        return logs

    def train_step(self, batch):
        agent_states, agent_actions, oppnt_states, oppnt_actions, rtg, timesteps, attention_mask = batch

        agent_states = agent_states.to(dtype=torch.float32, device=self.device)
        agent_actions = agent_actions.to(dtype=torch.float32, device=self.device)
        oppnt_states = oppnt_states.to(dtype=torch.float32, device=self.device)
        oppnt_actions = oppnt_actions.to(dtype=torch.float32, device=self.device)
        rtg = rtg.to(dtype=torch.float32, device=self.device)
        timesteps = timesteps.to(dtype=torch.long, device=self.device)
        attention_mask = attention_mask.to(dtype=torch.long, device=self.device)

        oppnt_states_target = torch.clone(oppnt_states)
        oppnt_actions_target = torch.clone(oppnt_actions)

        oppnt_outputs = self.model(
            states=agent_states, actions=agent_actions, returns_to_go=rtg, timesteps=timesteps, attention_mask=attention_mask,
        )
        oppnt_preds = self.model.predict_opponent(oppnt_outputs['embeddings'])

        oppnt_states_preds = oppnt_preds['states']
        oppnt_states_dim = oppnt_states_preds.shape[2]
        oppnt_states_preds = oppnt_states_preds.reshape(-1, oppnt_states_dim)[attention_mask.reshape(-1) > 0]
        oppnt_states_target = oppnt_states_target.reshape(-1, oppnt_states_dim)[attention_mask.reshape(-1) > 0]
        # oppnt_states_preds = oppnt_states_preds
        # oppnt_states_target = oppnt_states_target

        # Opponent States Loss
        oppnt_states_loss = F.mse_loss(oppnt_states_preds, oppnt_states_target)
        # oppnt_states_loss = 0.5 * ((oppnt_states_preds - oppnt_states_target) ** 2).sum(-1)
        # oppnt_states_loss = oppnt_states_loss.mean()

        oppnt_actions_preds = oppnt_preds['actions']
        num_opponents = len(oppnt_actions_preds)
        oppnt_actions_dim = oppnt_actions_preds[0].shape[2]
        oppnt_actions_target = oppnt_actions_target.reshape(-1, num_opponents, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]
        # oppnt_actions_target = oppnt_actions_target

        # Opponent Actions Loss
        oppnt_actions_loss = torch.tensor(0).to(dtype=torch.float32, device=self.device)
        for i, oppnt_actions_pred in enumerate(oppnt_actions_preds):
            oppnt_actions_pred = oppnt_actions_pred.reshape(-1, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]
            # oppnt_actions_pred = oppnt_actions_pred
            oppnt_actions_loss += F.cross_entropy(oppnt_actions_pred, oppnt_actions_target[:, i])
            # oppnt_actions_loss = -torch.log(torch.sum(oppnt_actions_target[:, i] * oppnt_actions_pred, dim=-1))
        # oppnt_actions_loss = oppnt_actions_loss.mean()

        # Opponent Modeling Loss
        opponent_loss = oppnt_states_loss + oppnt_actions_loss

        self.optimizer.zero_grad()
        opponent_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/loss'].append(opponent_loss.item())
            self.diagnostics['training/opponent_states_loss'].append(oppnt_states_loss.item())
            self.diagnostics['training/opponent_actions_loss'].append(oppnt_actions_loss.item())
            # self.diagnostics['training/opponent_actions_accuracy'].append(torch.sum(oppnt_actions_preds.argmax(dim=-1) == oppnt_actions_target.argmax(dim=-1)).cpu() / oppnt_actions_preds.shape[0])

            oppnt_actions_acc = torch.tensor(0).to(dtype=torch.float32)
            for i, oppnt_actions_pred in enumerate(oppnt_actions_preds):
                oppnt_actions_pred = oppnt_actions_pred.reshape(-1, oppnt_actions_dim)[attention_mask.reshape(-1) > 0]
                # oppnt_actions_acc += torch.sum(oppnt_actions_pred.argmax(dim=-1) == oppnt_actions_target[:, i].argmax(dim=-1)).cpu() / oppnt_actions_pred.shape[0]
                # oppnt_actions_pred = oppnt_actions_pred[:, -1]
                oppnt_actions_acc += torch.sum(oppnt_actions_pred.argmax(dim=-1) == oppnt_actions_target[:, i].argmax(dim=-1)).cpu() / oppnt_actions_pred.shape[0]

            oppnt_actions_acc = oppnt_actions_acc / len(oppnt_actions_preds)
            self.diagnostics['training/opponent_actions_accuracy'].append(oppnt_actions_acc)

        return opponent_loss.detach().cpu().item()
