import os
import random
import time
from dataclasses import dataclass

import multiagent.scenarios as scenarios
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from multiagent.environment import MultiAgentEnv
from torch.distributions import OneHotCategorical
from torch.utils.tensorboard import SummaryWriter

from opponent_transformer.envs import DummyVecEnv
from opponent_transformer.models.pretrained_opponents import get_opponent_actions
from opponent_transformer.models.opponent_models import BERTRAOOpponentTransformer
from opponent_transformer.training.running_mean import RunningMeanStd


# Grid search:
# lr: 3e-4, 1e-4
# seq length: 20, 10, 5
# batch size: 5, 10, 25
# embedding dim: 16, 32, 64


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 4
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "bhd445"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "simple_tag"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the policy model optimizer"""
    opponent_learning_rate: float = 3e-4
    """the learning rate of the opponent model optimizer"""
    num_opponent_policies: int = 10
    """the number of opponent policies to sample from"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_eval_envs: int = 100
    """the number of parallel game environments for evaluation"""
    num_steps: int = 5
    """the number of steps to run in each environment per policy rollout"""
    episode_length: int = 50
    """the maximum length of an episode"""
    seq_length: int = 5
    """the length of the transformer sequence"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    target_return: float = 23.0
    """the target return for the opponent model"""
    scale: float = 10.0
    """returns-to-go scale factor"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    embedding_dim: int = 32
    """the embedding dimension for the opponent model"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id):
    scenario = scenarios.load(f"{env_id}.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        discrete_action=True
    )
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def discount_cumsum(x, gamma=1.0):
    discount_cumsum = torch.zeros_like(x)
    discount_cumsum[0] = x[0]
    for t in range(1, x.shape[0]):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t-1]
    return discount_cumsum


class Agent(nn.Module):
    def __init__(self, envs, embedding_dim):
        super().__init__()
        
        self.num_opponents = 3
        self.act_dim = envs.action_space[3].n
        self.obs_dim = envs.observation_space[3].shape[0]
        self.opp_act_dim = sum([envs.action_space[i].n for i in range(3)])
        self.opp_obs_dim = sum([envs.observation_space[i].shape[0] for i in range(3)])
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(self.obs_dim + embedding_dim, 128)
        self.fc1 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, self.act_dim)
        self.critic = nn.Linear(128, 1)

    def act(self, x, lstm_state):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, lstm_state = self.lstm(x, lstm_state)
        h = F.relu(self.fc1(h))
        pol_out = F.softmax(self.actor(h), dim=-1)
        val_out = self.critic(h)
        m = OneHotCategorical(pol_out)
        action = m.sample()
        return action[0], val_out, lstm_state

    def get_value(self, x, lstm_state):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, lstm_state = self.lstm(x, lstm_state)
        h = F.relu(self.fc1(h))
        value = self.critic(h)
        return value

    def evaluate(self, x, lstm_state, action):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, lstm_state = self.lstm(x, lstm_state)
        h = F.relu(self.fc1(h))
        logits = F.softmax(self.actor(h), dim=-1)
        values = self.critic(h)
        log_probs = torch.sum(torch.log(logits + 1e-20) * action, dim=-1)
        entropy = -torch.sum(logits * torch.log(logits + 1e-20), dim=-1).mean()
        return log_probs, entropy, values


def evaluate(agent, opponent_model, args):
    eval_envs = [make_env(args.env_id) for _ in range(args.num_eval_envs)]
    eval_envs = DummyVecEnv(eval_envs)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    tasks = np.random.choice(range(args.num_opponent_policies), size=args.num_eval_envs)
    hidden = (torch.zeros((1, args.num_eval_envs, agent.lstm.hidden_size)).to(device),
              torch.zeros((1, args.num_eval_envs, agent.lstm.hidden_size)).to(device))
    obs = eval_envs.reset()
    agent_obs = torch.FloatTensor([o[3] for o in obs]).to(device)
    modelled_agent_obs = [o[:3] for o in obs]
    actions = torch.zeros((args.num_eval_envs, agent.act_dim)).to(device)
    dones = torch.zeros((args.num_eval_envs, 1)).to(device)
    average_reward = torch.zeros(args.num_eval_envs, 1)

    obs_seq = torch.zeros((args.num_eval_envs, args.seq_length, agent.obs_dim), device=device, dtype=torch.float32)
    obs_seq[:, -1] = agent_obs
    last_actions_seq = torch.ones((args.num_eval_envs, args.seq_length, agent.act_dim), device=device, dtype=torch.float32) * -10.0
    reward_seq = torch.zeros((args.num_eval_envs, args.seq_length, 1), device=device, dtype=torch.float32)
    timestep = torch.zeros((args.num_eval_envs, args.seq_length), device=device, dtype=torch.long)
    mask = torch.zeros((args.num_eval_envs, args.seq_length), device=device, dtype=torch.float32)
    mask[:, -1] = torch.ones((args.num_eval_envs,), device=device, dtype=torch.float32)

    observation_losses = []
    action_accuracies = []
    for t in range(args.episode_length):
        modelled_agent_actions = [get_opponent_actions(modelled_agent_obs[id], tasks[id]) for id in range(args.num_eval_envs)]
        modelled_agent_obs_torch = torch.Tensor(modelled_agent_obs).to(device).reshape(args.num_eval_envs, agent.opp_obs_dim)
        modelled_agent_act_torch = torch.Tensor(modelled_agent_actions).to(device)
        with torch.no_grad():
            embeddings = opponent_model(obs_seq, last_actions_seq, reward_seq, timestep, mask)
            opp_obs_preds, opp_actions_preds = opponent_model.predict_opponent(embeddings)
            x = torch.cat((agent_obs, embeddings[:, -1]), dim=-1)
            actions, _, hidden = agent.act(
                x,
                hidden
            )

        env_actions = [
            [
                modelled_agent_actions[id][0],
                modelled_agent_actions[id][1],
                modelled_agent_actions[id][2],
                actions[id].cpu().detach().numpy()
            ] for id in range(args.num_eval_envs)
        ]
        next_obs, rewards, _, _ = eval_envs.step(env_actions)

        next_agent_obs = torch.FloatTensor([o[3] for o in next_obs])
        next_modelled_agent_obs = [o[:3] for o in next_obs]
        rewards = torch.FloatTensor([r[3] for r in rewards]).unsqueeze(1)
        rewards_aug = rewards.clone()
        rewards_aug[rewards_aug<-2] += 10
        average_reward += rewards_aug
        pred_reward = reward_seq[:, -1] + rewards
        agent_obs = next_agent_obs.to(device)
        modelled_agent_obs = next_modelled_agent_obs

        obs_seq = torch.cat((obs_seq, agent_obs.unsqueeze(1)), dim=1)[:, -args.seq_length:]
        last_actions_seq = torch.cat((last_actions_seq, actions.unsqueeze(1)), dim=1)[:, -args.seq_length:]
        reward_seq = torch.cat((reward_seq, rewards.reshape(args.num_eval_envs, 1, 1)), dim=1)[:, -args.seq_length:]
        timestep = torch.cat((timestep, torch.ones((args.num_eval_envs, 1), device=device, dtype=torch.long) * (t + 1)), dim=1)[:, -args.seq_length:]
        mask = torch.cat((mask, torch.ones((args.num_eval_envs, 1), device=device, dtype=torch.float32)), dim=1)[:, -args.seq_length:]

        # Compute reconstruction accuracy
        opp_obs_preds = opp_obs_preds[:, -1]
        opp_obs_loss = F.mse_loss(opp_obs_preds, modelled_agent_obs_torch)
        observation_losses.append(opp_obs_loss.item())

        opp_actions_preds = opp_actions_preds[:, -1]
        opp_actions_acc = torch.sum(opp_actions_preds.argmax(dim=-1) == modelled_agent_act_torch.argmax(dim=-1)).cpu() / (opp_actions_preds.shape[0] * opp_actions_preds.shape[1])
        action_accuracies.append(opp_actions_acc.item())

    eval_envs.close()

    observation_losses = torch.tensor(observation_losses)
    action_accuracies = torch.tensor(action_accuracies)

    return average_reward.mean().item(), torch.mean(observation_losses).item(), torch.mean(action_accuracies).item()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.episode_length // args.num_minibatches)
    args.num_iterations = args.total_timesteps // (args.episode_length * args.num_envs)
    print(f"Num iterations = {args.num_iterations}")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = [make_env(args.env_id) for _ in range(args.num_envs)]
    for i in range(args.num_envs):
        envs[i].seed(i)

    envs = DummyVecEnv(envs)

    agent = Agent(envs, 128).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    opponent_model = BERTRAOOpponentTransformer(
        state_dim=agent.obs_dim,
        act_dim=agent.act_dim,
        oppnt_state_dims=[envs.observation_space[i].shape[0] for i in range(3)],
        oppnt_act_dims=[envs.action_space[i].n for i in range(3)],
        hidden_dim=128,
        embedding_dim=args.embedding_dim,
        num_opponents=3,
        max_ep_len=args.episode_length,
        max_length=args.seq_length,
        n_layer=4,
        n_head=4,
        n_inner=2*128,
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        activation_function='relu',
        model_opponent_policy=False,
        model_opponent_states=True,
        model_opponent_actions=True,
        model_opponent_returns=False,
        use_embedding_layer=False
    ).to(device)
    # opponent_model.load_params('trained_parameters/transformer_opponent_model.pt')
    opponent_optimizer = optim.Adam(opponent_model.parameters(), lr=args.opponent_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # Policy Storage
    obs = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, agent.obs_dim)).to(device)
    last_actions = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, agent.act_dim)).to(device)
    actions = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, agent.act_dim)).to(device)
    opp_obs = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, agent.opp_obs_dim)).to(device)
    opp_actions = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, agent.opp_act_dim)).to(device)
    rewards = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs)).to(device)
    dones = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs)).to(device)
    values = torch.zeros((args.num_minibatches, args.batch_size + 1, args.num_envs)).to(device)
    returns = torch.zeros((args.num_minibatches, args.batch_size + 1, args.num_envs)).to(device)
    lstm_states = (
        torch.zeros(args.num_minibatches, agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(args.num_minibatches, agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device)
    )

    # Opponent Storage
    obs_seqs = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, args.seq_length, agent.obs_dim)).to(device)
    last_actions_seqs = torch.ones((args.num_minibatches, args.batch_size, args.num_envs, args.seq_length, agent.act_dim)).to(device) * -10.0
    opp_obs_seqs = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, args.seq_length, agent.opp_obs_dim)).to(device)
    opp_actions_seqs = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, args.seq_length, agent.opp_act_dim)).to(device)
    reward_seqs = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, args.seq_length, 1)).to(device)
    timesteps = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, args.seq_length)).to(device, dtype=torch.long)
    masks = torch.zeros((args.num_minibatches, args.batch_size, args.num_envs, args.seq_length)).to(device)

    # Running mean for rewards
    running_mean = RunningMeanStd(shape=1, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    counter = 0
    episodes_passed = -1
    start_time = time.time()
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    for iteration in range(1, args.num_iterations + 1):
        for idx in range(args.num_minibatches):
            if counter == 0:
                # reset environment and initialize data
                # policy data
                all_obs = envs.reset()
                next_obs = torch.tensor([o[3] for o in all_obs]).to(device, dtype=torch.float32)
                next_opp_obs = [o[:3] for o in all_obs]
                last_action = torch.zeros(args.num_envs, agent.act_dim).to(device, dtype=torch.float32)
                next_done = torch.zeros(args.num_envs).to(device)
                next_lstm_state = (
                    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
                    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
                )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

                # opponent data
                obs_seq = torch.zeros((args.num_envs, args.seq_length, agent.obs_dim), device=device, dtype=torch.float32)
                obs_seq[:, -1] = next_obs
                last_actions_seq = torch.ones((args.num_envs, args.seq_length, agent.act_dim), device=device, dtype=torch.float32) * -10.0
                reward_seq = torch.zeros((args.num_envs, args.seq_length, 1), device=device, dtype=torch.float32)
                # rtg_seq[:, -1] = torch.tensor([args.target_return] * args.num_envs, device=device, dtype=torch.float32).reshape(args.num_envs, 1)
                timestep = torch.zeros((args.num_envs, args.seq_length), device=device, dtype=torch.long)
                mask = torch.zeros((args.num_envs, args.seq_length), device=device, dtype=torch.float32)
                mask[:, -1] = torch.ones((args.num_envs,), device=device, dtype=torch.float32)
                opp_obs_seq = torch.zeros((args.num_envs, args.seq_length, agent.opp_obs_dim), device=device, dtype=torch.float32)
                opp_actions_seq = torch.zeros((args.num_envs, args.seq_length, agent.opp_act_dim), device=device, dtype=torch.float32)

                # shuffle the set of opponent policies
                tasks = np.random.choice(range(args.num_opponent_policies), size=args.num_envs)

            # reset lstm state to the current state
            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
            lstm_states[0][idx] = next_lstm_state[0].clone()
            lstm_states[1][idx] = next_lstm_state[1].clone()

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.batch_size):
                # if torch.isnan(obs_seq).any():
                    # print("Observations: ", obs_seq)
                # if counter > 0:
                #     print("Last actions: ", last_action)
                #     print("Last opponent actions: ", opp_action)

                # get opponent actions
                opp_action = [get_opponent_actions(next_opp_obs[id], tasks[id]) for id in range(args.num_envs)]
                opp_obs_flat = torch.Tensor(next_opp_obs).reshape(args.num_envs, agent.opp_obs_dim).to(device, dtype=torch.float32)
                opp_action_flat = torch.Tensor(opp_action).reshape(args.num_envs, agent.opp_act_dim).to(device, dtype=torch.float32)

                opp_obs_seq = torch.cat((opp_obs_seq, opp_obs_flat.unsqueeze(1)), dim=1)[:, -args.seq_length:]
                opp_actions_seq = torch.cat((opp_actions_seq, opp_action_flat.unsqueeze(1)), dim=1)[:, -args.seq_length:]

                # ALGO LOGIC: action logic
                inference_start_time = time.time()
                with torch.no_grad():
                    embeddings = opponent_model(obs_seq, last_actions_seq, reward_seq, timestep, mask)
                    x = torch.cat((next_obs, embeddings[:, -1]), dim=-1)
                    action, value, next_lstm_state = agent.act(x, next_lstm_state)

                inference_time = time.time() - inference_start_time
                # print("Inference time: ", inference_time)

                global_step += args.num_envs
                counter += 1
                obs[idx, step] = next_obs
                opp_obs[idx, step] = opp_obs_flat
                opp_actions[idx, step] = opp_action_flat
                last_actions[idx, step] = last_action
                dones[idx, step] = next_done
                actions[idx, step] = action
                values[idx, step] = value.flatten()
                last_action = action

                obs_seqs[idx, step] = obs_seq
                last_actions_seqs[idx, step] = last_actions_seq
                opp_obs_seqs[idx, step] = opp_obs_seq
                opp_actions_seqs[idx, step] = opp_actions_seq
                reward_seqs[idx, step] = reward_seq
                timesteps[idx, step] = timestep
                masks[idx, step] = mask

                all_actions = [
                    [
                        opp_action[id][0],
                        opp_action[id][1],
                        opp_action[id][2],
                        last_action[id].cpu().detach().numpy()
                    ] for id in range(args.num_envs)
                ]

                # TRY NOT TO MODIFY: execute the game and log data.
                
                all_next_obs, all_rewards, all_dones, _ = envs.step(all_actions)
                next_obs = torch.tensor([o[3] for o in all_next_obs]).to(device, dtype=torch.float32)
                next_opp_obs = [o[:3] for o in all_next_obs]
                next_rewards = torch.tensor([r[3] for r in all_rewards]).unsqueeze(-1).to(device, dtype=torch.float32)
                rewards[idx, step] = next_rewards.view(-1)
                pred_reward = reward_seq[:, -1] + next_rewards

                obs_seq = torch.cat((obs_seq, next_obs.unsqueeze(1)), dim=1)[:, -args.seq_length:]
                last_actions_seq = torch.cat((last_actions_seq, last_action.unsqueeze(1)), dim=1)[:, -args.seq_length:]
                reward_seq = torch.cat((reward_seq, next_rewards.reshape(args.num_envs, 1, 1)), dim=1)[:, -args.seq_length:]
                timestep = torch.cat((timestep, torch.ones((args.num_envs, 1), device=device, dtype=torch.long) * counter), dim=1)[:, -args.seq_length:]
                mask = torch.cat((mask, torch.ones((args.num_envs, 1), device=device, dtype=torch.float32)), dim=1)[:, -args.seq_length:]

                if counter == args.episode_length:
                    next_done = torch.ones(args.num_envs).to(device)
                    episodes_passed += 1
                    counter = 0
                else:
                    next_done = torch.zeros(args.num_envs).to(device)

            # bootstrap value if not done
            with torch.no_grad():
                if next_done[0]:
                    next_value = torch.zeros(args.num_envs).to(device)
                else:
                    opp_action = [get_opponent_actions(next_opp_obs[id], tasks[id]) for id in range(args.num_envs)]
                    opp_obs_flat = torch.Tensor(next_opp_obs).reshape(args.num_envs, agent.opp_obs_dim).to(device, dtype=torch.float32)
                    opp_action_flat = torch.Tensor(opp_action).reshape(args.num_envs, agent.opp_act_dim).to(device, dtype=torch.float32)

                    embeddings = opponent_model(obs_seq, last_actions_seq, reward_seq, timestep, mask)
                    x = torch.cat((next_obs, embeddings[:, -1]), dim=-1)
                    next_value = agent.get_value(
                        x,
                        next_lstm_state
                    ).reshape(1, -1)

                values[idx, -1] = next_value.detach()

                lastgaelam = 0
                values[idx] = values[idx] * torch.sqrt(running_mean.var) + running_mean.mean
                for t in reversed(range(args.batch_size)):
                    delta = rewards[idx, t] + args.gamma * values[idx, t + 1] * (1.0 - dones[idx, t]) - values[idx, t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * (1.0 - dones[idx, t]) * lastgaelam
                    returns[idx, t] = lastgaelam + values[idx, t]
                
                running_mean.update(returns[idx, :-1].unsqueeze(-1))
                returns[idx] = (returns[idx] - running_mean.mean) / torch.sqrt(running_mean.var)

        # Compute returns-to-go
        rewards_flat = rewards.reshape(-1, args.num_envs)
        returns_to_go = discount_cumsum(rewards_flat)
        # pad for sequence length
        returns_to_go = torch.cat((torch.zeros((args.seq_length - 1, args.num_envs)), returns_to_go))
        rtg_seqs = torch.cat([returns_to_go[i:i+args.seq_length].unsqueeze(0) for i in range(args.episode_length)], dim=0)
        rtg_seqs = rtg_seqs.reshape(args.num_minibatches, args.batch_size, args.num_envs, args.seq_length, 1) / args.scale

        train_average_reward = rewards.reshape(-1, args.num_envs).sum(0).mean().item()

        # flatten the batch
        b_obs = obs.clone()
        b_last_actions = last_actions.clone()
        b_actions = actions.clone()
        b_opp_obs = opp_obs.clone()
        b_opp_actions = opp_actions.clone()
        b_returns = returns.clone()
        b_values = values.clone()
        b_hidden = (
            lstm_states[0].clone(),
            lstm_states[1].clone()
        )

        b_obs_seqs = obs_seqs.clone()
        b_last_actions_seqs = last_actions_seqs.clone()
        b_reward_seqs = reward_seqs.clone()
        b_rtg_seqs = rtg_seqs.clone()
        b_timesteps = timesteps.clone()
        b_masks = masks.clone()
        b_opp_obs_seqs = opp_obs_seqs.clone()
        b_opp_actions_seqs = opp_actions_seqs.clone()

        # Optimizing the policy and value network
        for epoch in range(args.update_epochs):
            opp_actions_accuracies = []
            for idx in range(args.num_minibatches):
                # policy batch
                mb_obs = b_obs[idx]
                mb_last_actions = b_last_actions[idx]
                mb_actions = b_actions[idx]
                mb_opp_obs = b_opp_obs[idx]
                mb_opp_actions = b_opp_actions[idx]
                mb_returns = b_returns[idx, :-1]
                mb_hidden = (
                    b_hidden[0][idx],
                    b_hidden[1][idx]
                )

                # transformer batch
                mb_obs_seq = b_obs_seqs[idx].reshape(-1, args.seq_length, agent.obs_dim)
                mb_last_actions_seq = b_last_actions_seqs[idx].reshape(-1, args.seq_length, agent.act_dim)
                mb_reward_seq = b_reward_seqs[idx].reshape(-1, args.seq_length, 1)
                mb_rtg_seq = b_rtg_seqs[idx].reshape(-1, args.seq_length, 1)
                mb_timestep = b_timesteps[idx].reshape(-1, args.seq_length)
                mb_mask = b_masks[idx].reshape(-1, args.seq_length)
                mb_opp_obs_seq = b_opp_obs_seqs[idx].reshape(-1, args.seq_length, agent.opp_obs_dim)
                mb_opp_actions_seq = b_opp_actions_seqs[idx].reshape(-1, args.seq_length, agent.opp_act_dim)

                embeddings = opponent_model(mb_obs_seq, mb_last_actions_seq, mb_reward_seq, mb_timestep, mb_mask)
                opp_obs_preds, opp_actions_preds = opponent_model.predict_opponent(embeddings)

                # get the last embeddings according to the current timestep
                mb_embeddings = embeddings.reshape(args.batch_size, args.num_envs, args.seq_length, -1)[:, :, -1].detach()

                opp_obs_preds = opp_obs_preds.reshape(-1, agent.opp_obs_dim)[mb_mask.reshape(-1) > 0]
                mb_opp_obs_seq = mb_opp_obs_seq.reshape(-1, agent.opp_obs_dim)[mb_mask.reshape(-1) > 0]
                opp_obs_loss = F.mse_loss(opp_obs_preds, mb_opp_obs_seq)

                opp_actions_preds = opp_actions_preds.reshape(-1, agent.num_opponents, agent.opp_act_dim // agent.num_opponents)[mb_mask.reshape(-1) > 0]
                mb_opp_actions_seq = mb_opp_actions_seq.reshape(-1, agent.num_opponents, agent.opp_act_dim // agent.num_opponents)[mb_mask.reshape(-1) > 0]
                opp_actions_loss = F.cross_entropy(opp_actions_preds, mb_opp_actions_seq)

                opp_actions_acc = torch.sum(opp_actions_preds.argmax(dim=-1) == mb_opp_actions_seq.argmax(dim=-1)).cpu() / (opp_actions_preds.shape[0] * opp_actions_preds.shape[1])
                opp_actions_accuracies.append(opp_actions_acc.item())

                opponent_loss = opp_obs_loss + opp_actions_loss

                opponent_optimizer.zero_grad()
                opponent_loss.backward()
                nn.utils.clip_grad_norm_(opponent_model.parameters(), args.max_grad_norm)
                opponent_optimizer.step()

                x = torch.cat((mb_obs, mb_embeddings), dim=-1)
                new_log_probs, entropy, new_values = agent.evaluate(
                    x,
                    mb_hidden,
                    mb_actions
                )

                new_values = new_values.squeeze(-1)
                mb_advantages = mb_returns - new_values.detach()

                action_loss = -(mb_advantages.detach() * new_log_probs).mean()

                value_loss = (mb_returns - new_values).pow(2).mean()

                entropy_loss = entropy.mean()

                # print("batch = ", idx)
                # print("Policy Loss: ", action_loss.item())
                # print("Value Loss: ", value_loss.item())
                # print("Entropy Loss: ", entropy_loss.item())

                loss = action_loss - args.ent_coef * entropy_loss + value_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if (episodes_passed % 100 == 0) and (counter == 0):
            average_reward, eval_obs_loss, eval_act_acc = evaluate(agent, opponent_model, args)
            print(f"Episode={episodes_passed}, episodic_return={average_reward}, eval_obs_loss={eval_obs_loss}, eval_act_acc={eval_act_acc}")
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], episodes_passed)
            writer.add_scalar("losses/value_loss", value_loss.item(), episodes_passed)
            writer.add_scalar("losses/policy_loss", action_loss.item(), episodes_passed)
            writer.add_scalar("losses/entropy", entropy_loss.item(), episodes_passed)
            writer.add_scalar("losses/obs_recon_loss", opp_obs_loss.item(), episodes_passed)
            writer.add_scalar("losses/act_recon_loss", opp_actions_loss.item(), episodes_passed)
            writer.add_scalar("losses/explained_variance", explained_var, episodes_passed)
            writer.add_scalar("charts/train_act_recon_acc", np.mean(opp_actions_accuracies), episodes_passed)
            writer.add_scalar("charts/eval_act_recon_acc", eval_act_acc, episodes_passed)
            writer.add_scalar("charts/eval_obs_recon_loss", eval_obs_loss, episodes_passed)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), episodes_passed)
            writer.add_scalar("charts/episodic_return", average_reward, episodes_passed)
            writer.add_scalar("charts/train_episodic_return", train_average_reward, episodes_passed)

    envs.close()
    writer.close()
