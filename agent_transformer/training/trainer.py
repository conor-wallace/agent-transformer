import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from opponent_transformer.models.pretrained_opponents import get_opponent_actions
from opponent_transformer.training.running_mean import RunningMeanStd


class Trainer:

    def __init__(
        self,
        policy,
        policy_optimizer,
        obs_dim,
        act_dim,
        opp_obs_dims,
        opp_act_dims,
        hidden_dim,
        episode_length,
        num_envs,
        num_opponents,
        num_opponent_policies,
        gamma,
        gae_lambda,
        oracle=False,
        opponent_model=None,
        opponent_optimizer=None,
        policy_scheduler=None,
        opponent_scheduler=None,
        target_return=None,
        return_scale=None,
        eval_fns=None,
        device: str = "cuda",
        clip_coef: float = 0.01,
        ent_coef: float = 0.001
    ):
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.policy_scheduler = policy_scheduler

        self.oracle = oracle
        self.opponent_model = opponent_model
        self.opponent_optimizer = opponent_optimizer
        self.opponent_scheduler = opponent_scheduler

        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.diagnostics = dict()

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.opp_obs_dims = opp_obs_dims
        self.opp_act_dims = opp_act_dims
        self.hidden_dim = hidden_dim
        self.episode_length = episode_length
        self.num_opponents = num_opponents
        self.num_opponent_policies = num_opponent_policies
        self.target_return = target_return
        self.return_scale = return_scale
        self.running_mean = RunningMeanStd(shape=1)

        self.hiddens = (
            torch.zeros((1, self.num_envs, self.hidden_dim), device=self.device, dtype=torch.float32),
            torch.zeros((1, self.num_envs, self.hidden_dim), device=self.device, dtype=torch.float32)
        )

        self.start_time = time.time()

    @torch.no_grad()
    def rollout_episode(self, envs):
        num_envs = envs.num_envs
        episode_obs = np.zeros((self.episode_length, num_envs, self.obs_dim))
        episode_actions = np.zeros((self.episode_length + 1, num_envs, self.act_dim))
        episode_hiddens = [
            self.hiddens[0].cpu().detach().numpy(),
            self.hiddens[1].cpu().detach().numpy()
        ]
        episode_rewards = np.zeros((self.episode_length, num_envs, 1))
        episode_values = np.zeros((self.episode_length, num_envs, 1))
        episode_dones = np.zeros((self.episode_length, num_envs, 1))

        episode_opp_obs = [np.zeros((self.episode_length, num_envs, opp_obs_dim)) for opp_obs_dim in self.opp_obs_dims]
        episode_opp_actions = [np.zeros((self.episode_length, num_envs, opp_act_dim)) for opp_act_dim in self.opp_act_dims]

        all_obs = envs.reset()
        obs = torch.tensor([o[3] for o in all_obs], device=self.device, dtype=torch.float32)
        # obs = torch.tensor(all_obs[self.num_opponents], device=self.device, dtype=torch.float32)
        opp_obs = [o[:3] for o in all_obs]
        # opp_obs = np.array([all_obs[i] for i in range(self.num_opponents)]).transpose(1, 0, 2)
        dones = torch.zeros((num_envs, 1))
        actions = torch.zeros((num_envs, self.act_dim), device=self.device, dtype=torch.float32)

        # sequence data
        obs_seq = torch.tensor([o[3] for o in all_obs], device=self.device, dtype=torch.float32).reshape(num_envs, 1, self.obs_dim)
        actions_seq = torch.zeros((num_envs, 1, self.act_dim), device=self.device, dtype=torch.float32)
        rtg_seq = torch.tensor([self.target_return] * num_envs, device=self.device, dtype=torch.float32).reshape(num_envs, 1, 1)
        timesteps = torch.zeros((num_envs, 1), device=self.device, dtype=torch.long)

        # Pretrained opponent ids
        tasks = np.random.choice(range(self.num_opponent_policies), size=num_envs)

        for step in range(self.episode_length):
            if self.opponent_model is not None:
                embedding = self.opponent_model.predict(obs_seq, actions_seq, rtg_seq, timesteps)
                actions, value, self.hiddens = self.policy.act(obs, actions, self.hiddens, embedding=embedding)
            else:
                actions, value, self.hiddens = self.policy.act(obs, actions, self.hiddens)

            opp_actions = [get_opponent_actions(opp_obs[id], tasks[id]) for id in range(num_envs)]

            env_actions = [[opp_actions[id][0], opp_actions[id][1], opp_actions[id][2],
                            actions[id].cpu().detach().numpy()] for id in range(num_envs)]
            all_next_obs, rewards, dones, _ = envs.step(env_actions)

            next_obs = torch.tensor([o[3] for o in all_next_obs], device=self.device, dtype=torch.float32)
            next_opp_obs = [o[:3] for o in all_next_obs]
            rewards = torch.tensor([r[3] for r in rewards]).unsqueeze(1)
            dones = torch.tensor([d[0] for d in dones]).unsqueeze(1).long().numpy()

            episode_obs[step] = obs.cpu().numpy()
            episode_actions[step + 1] = actions.cpu().detach().numpy()
            episode_rewards[step] = rewards
            episode_values[step] = value.cpu().detach().numpy()
            episode_dones[step] = dones

            for i in range(self.num_opponents):
                episode_opp_obs[i][step] = torch.Tensor([opp_obs[id][i] for id in range(num_envs)])
                episode_opp_actions[i][step] = torch.Tensor([opp_actions[id][i] for id in range(num_envs)])

            obs = next_obs
            opp_obs = next_opp_obs
            pred_return = rtg_seq[:, -1] - torch.tensor(rewards / self.return_scale, device=self.device, dtype=torch.float32)

            obs_seq = torch.cat([obs_seq, obs.reshape(num_envs, 1, self.obs_dim)], dim=1)
            actions_seq = torch.cat([actions_seq, actions.reshape(num_envs, 1, self.act_dim)], dim=1)
            rtg_seq = torch.cat([rtg_seq, pred_return.reshape(num_envs, 1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((num_envs, 1), device=self.device, dtype=torch.long) * step], dim=1)

        if self.opponent_model is not None:
            embedding = self.opponent_model.predict(obs_seq, actions_seq, rtg_seq, timesteps)
            _, last_value, _ = self.policy.act(obs, actions, self.hiddens, embedding=embedding)
        else:
            _, last_value, _ = self.policy.act(obs, actions, self.hiddens)

        last_value = last_value.cpu().detach().numpy()

        episode_returns, episode_advantages = self.compute_returns(last_value, dones, episode_rewards, episode_values, episode_dones)

        trajectories = {
            "observations": episode_obs,
            "prev_actions": episode_actions[:-1],
            "actions": episode_actions[1:],
            "hiddens": episode_hiddens,
            "opponent_observations": np.array([episode_opp_obs[o] for o in range(self.num_opponents)]),
            "opponent_actions": np.array([episode_opp_actions[o][:, i] for o in range(self.num_opponents)]),
            "rewards": episode_rewards,
            "returns": episode_returns,
            "advantages": episode_advantages
        }
        # for i in range(num_envs):
        #     trajectory_i = {
        #         "observations": episode_obs[:, i],
        #         "prev_actions": episode_actions[:-1, i],
        #         "actions": episode_actions[1:, i],
        #         "opponent_observations": np.array([episode_opp_obs[o][:, i] for o in range(self.num_opponents)]).transpose(1, 0, 2).reshape(self.episode_length, -1),
        #         "opponent_actions": np.array([episode_opp_actions[o][:, i] for o in range(self.num_opponents)]).transpose(1, 0, 2),
        #         "rewards": episode_rewards[:, i],
        #         "returns": episode_returns[:-1, i],
        #     }

        #     trajectories.append(trajectory_i)

        return trajectories

    def compute_returns(self, last_value, last_done, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for step in reversed(range(self.episode_length)):
            if step == self.episode_length - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones[step + 1]
                nextvalues = values[step + 1]
            delta = rewards[step] + self.gamma * nextvalues * nextnonterminal - values[step]
            advantages[step] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        # returns = np.zeros_like(rewards)
        # values = (torch.from_numpy(values) * torch.sqrt(self.running_mean.var) + self.running_mean.mean).cpu().numpy()
        # gae = 0
        # for step in reversed(range(self.episode_length)):
        #     delta = rewards[step] + self.gamma * values[step + 1] * (1. - dones[step]) - \
        #             values[step]
        #     gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
        #     returns[step] = gae + values[step]
        # self.running_mean.update(torch.from_numpy(returns[:-1]))
        # return ((torch.from_numpy(returns) - self.running_mean.mean) / torch.sqrt(self.running_mean.var)).cpu().numpy()

        return returns, advantages

    def train_iteration(self, num_steps: int, iter_num: int, dataloader):

        logs = dict()
        self.diagnostics['training/action_loss'] = []
        self.diagnostics['training/value_loss'] = []
        self.diagnostics['training/entropy_loss'] = []
        self.diagnostics['training/loss'] = []
        self.diagnostics['training/opponent_obs_loss'] = []
        self.diagnostics['training/opponent_act_loss'] = []
        self.diagnostics['training/opponent_actions_accuracy'] = []
        train_start = time.time()

        for _ in range(num_steps):
            for i, batch in enumerate(dataloader):
                action_loss, value_loss, entropy_loss, loss, opp_obs_loss, opp_act_loss, accuracy = self.train_step(batch)
                self.diagnostics['training/action_loss'].append(action_loss)
                self.diagnostics['training/value_loss'].append(value_loss)
                self.diagnostics['training/entropy_loss'].append(entropy_loss)
                self.diagnostics['training/loss'].append(loss)
                self.diagnostics['training/opponent_obs_loss'].append(opp_obs_loss)
                self.diagnostics['training/opponent_act_loss'].append(opp_act_loss)
                self.diagnostics['training/opponent_actions_accuracy'].append(accuracy)
                if self.policy_scheduler is not None:
                    self.policy_scheduler.step()
                if self.opponent_scheduler is not None:
                    self.opponent_scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/lr'] = self.policy_optimizer.param_groups[0]["lr"]

        for k in self.diagnostics:
            logs[k] = np.mean(self.diagnostics[k])

        return logs

    def train_step(self, batch):
        obs_batch, prev_actions_batch, actions_batch, returns_batch, advantages_batch, hiddens_batch = batch

        # print('obs_batch: ', obs_batch.shape)
        # print('prev_actions_batch: ', prev_actions_batch.shape)
        # print('rtg_batch: ', rtg_batch.shape)
        # print('timesteps_batch: ', timesteps_batch.shape)
        # print('mask_batch: ', mask_batch.shape)
        # print('actions_batch: ', actions_batch.shape)
        # print('opp_obs_batch: ', opp_obs_batch.shape)
        # print('opp_actions_batch: ', opp_actions_batch.shape)
        # print('returns_batch: ', returns_batch.shape)

        obs_batch = obs_batch.to(device=self.device, dtype=torch.float32).squeeze(0)
        prev_actions_batch = prev_actions_batch.to(device=self.device, dtype=torch.float32).squeeze(0)
        actions_batch = actions_batch.to(device=self.device, dtype=torch.float32).squeeze(0)
        returns_batch = returns_batch.to(device=self.device, dtype=torch.float32).squeeze(0)
        advantages_batch = advantages_batch.to(device=self.device, dtype=torch.float32).squeeze(0)
        hiddens_batch[0] = hiddens_batch[0].to(device=self.device, dtype=torch.float32).squeeze(0)
        hiddens_batch[1] = hiddens_batch[1].to(device=self.device, dtype=torch.float32).squeeze(0)

        batch_size = obs_batch.shape[0]

        if self.opponent_model is not None:
            outputs = self.opponent_model(obs_batch, prev_actions_batch, rtg_batch, timesteps_batch, mask_batch)
            embedding_batch = outputs['embeddings']
            x = torch.cat((obs_batch, embedding_batch.detach()), dim=-1)
            policy, values, _ = self.policy(x, hiddens_batch)
        else:
            x = torch.cat((obs_batch, prev_actions_batch), dim=-1)
            # print("Obs batch shape: ", obs_batch.shape)
            # print("Prev actions batch shape: ", prev_actions_batch.shape)
            # print("Input tensor shape: ", x.shape)
            # print("Hiddens 1 shape: ", hiddens_batch[0].shape)
            # print("Hiddens 2 shape: ", hiddens_batch[1].shape)
            policy, values, _ = self.policy(x, hiddens_batch)

        # x = torch.cat((obs_batch, prev_actions_batch), dim=-1)
        # policy, values, _ = self.policy(x, hidden_batch)
        # policy = policy.permute((1, 0, 2))
        # values = values.permute((1, 0, 2))
        log_probs = torch.sum(torch.log(policy + 1e-20) * actions_batch, dim=-1)
        entropy = -torch.sum(policy * torch.log(policy + 1e-20), dim=-1).mean()

        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        action_loss = -(advantages_batch * log_probs.unsqueeze(-1)).mean()
        value_loss = (returns_batch - values).pow(2).mean()
        loss = value_loss + action_loss - entropy * self.ent_coef

        self.policy_optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.25)

        self.policy_optimizer.step()

        if self.opponent_model is not None:
            opp_preds = self.opponent_model.predict_opponent(embedding_batch)
            opp_obs_preds = opp_preds['states']
            opp_actions_preds = opp_preds['actions']
            opp_obs_dim = opp_obs_preds.shape[-1]
            opp_act_dim = opp_actions_preds[0].shape[-1]

            opp_obs_preds = opp_obs_preds.reshape(-1, opp_obs_dim)[mask_batch.reshape(-1) > 0]
            opp_obs_batch = opp_obs_batch.reshape(-1, opp_obs_dim)[mask_batch.reshape(-1) > 0]

            opp_obs_loss = F.mse_loss(opp_obs_preds, opp_obs_batch)

            opp_actions_loss = torch.tensor(0).to(dtype=torch.float32, device=self.device)
            opp_actions_acc = torch.tensor(0).to(dtype=torch.float32)
            opp_actions_batch = opp_actions_batch.reshape(-1, self.num_opponents, opp_act_dim)[mask_batch.reshape(-1) > 0]
            for i, opp_actions_pred in enumerate(opp_actions_preds):
                opp_actions_pred = opp_actions_pred.reshape(-1, opp_act_dim)[mask_batch.reshape(-1) > 0]
                opp_actions_loss += F.cross_entropy(opp_actions_pred, opp_actions_batch[:, i])
                opp_actions_acc += torch.sum(opp_actions_pred.argmax(dim=-1) == opp_actions_batch[:, i].argmax(dim=-1)).cpu() / opp_actions_pred.shape[0]

            opp_actions_acc = opp_actions_acc / len(opp_actions_preds)

            opp_loss = opp_obs_loss + opp_actions_loss

            self.opponent_optimizer.zero_grad()
            opp_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.opponent_model.parameters(), .25)
            self.opponent_optimizer.step()
        else:
            opp_obs_loss = torch.tensor(0).to(dtype=torch.float32, device=self.device)
            opp_actions_loss = torch.tensor(0).to(dtype=torch.float32, device=self.device)
            opp_actions_acc = torch.tensor(0).to(dtype=torch.float32)

        return action_loss.detach().cpu().item(), value_loss.detach().cpu().item(), entropy.detach().cpu().item(), loss.detach().cpu().item(), opp_obs_loss.detach().cpu().item(), opp_actions_loss.detach().cpu().item(), opp_actions_acc