import numpy as np
import torch
import torch.nn.functional as F


def to_onehot(x: torch.Tensor, act_dim: int):
    x_onehot = torch.zeros(x.shape)
    x = F.softmax(x, dim=-1)
    max_idx = x.argmax(dim=-1)
    x_onehot = F.one_hot(max_idx.long(), act_dim)

    return x_onehot


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    final_value: torch.Tensor,
    final_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    episode_length: int = 25,
):
    final_value = final_value
    final_done = final_done

    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(episode_length)):
        if t == episode_length - 1:
            nextnonterminal = 1.0 - final_done
            nextvalues = final_value
        else:
            nextnonterminal = 1.0 - dones[:, t + 1]
            nextvalues = values[:, t + 1]
        delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:, t]
        advantages[:, t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values

    return returns, advantages


def evaluate_episode(
        env,
        agent_state_dim,
        agent_move_act_dim,
        agent_comm_act_dim,
        oppnt_state_dim,
        oppnt_move_act_dim,
        oppnt_comm_act_dim,
        agent_model,
        oppnt_model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        agent_state_mean=0.,
        agent_state_std=1.,
        oppnt_state_mean=0.,
        oppnt_state_std=1.,
        agent_idx=0,
        oppnt_idx=1,
):

    agent_model.eval()
    agent_model.to(device=device)
    oppnt_model.eval()
    oppnt_model.to(device=device)

    agent_state_mean = torch.from_numpy(agent_state_mean).to(device=device)
    agent_state_std = torch.from_numpy(agent_state_std).to(device=device)
    oppnt_state_mean = torch.from_numpy(oppnt_state_mean).to(device=device)
    oppnt_state_std = torch.from_numpy(oppnt_state_std).to(device=device)

    state = env.reset()
    agent_state, oppnt_state = state[0], state[1]

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    agent_states = torch.from_numpy(agent_state).reshape(1, agent_state_dim).to(device=device, dtype=torch.float32)
    agent_move_actions = torch.zeros((0, agent_move_act_dim), device=device, dtype=torch.float32)
    agent_comm_actions = torch.zeros((0, agent_comm_act_dim), device=device, dtype=torch.float32)
    oppnt_states = torch.zeros((0, oppnt_state_dim), device=device, dtype=torch.float32)
    oppnt_move_actions = torch.zeros((0, oppnt_move_act_dim), device=device, dtype=torch.float32)
    oppnt_comm_actions = torch.zeros((0, oppnt_comm_act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    values = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []


    oppnt_state = torch.from_numpy(oppnt_state).reshape(1, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_hidden = torch.zeros((1, oppnt_model.recurrent_dim, oppnt_model.hidden_dim)).to(self.device)
    oppnt_mask = torch.ones((1, 1)).to(self.device)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        agent_move_actions = torch.cat([agent_move_actions, torch.zeros((1, agent_move_act_dim), device=device)], dim=0)
        agent_comm_actions = torch.cat([agent_comm_actions, torch.zeros((1, agent_comm_act_dim), device=device)], dim=0)
        oppnt_move_actions = torch.cat([oppnt_move_actions, torch.zeros((1, oppnt_move_act_dim), device=device)], dim=0)
        oppnt_comm_actions = torch.cat([oppnt_comm_actions, torch.zeros((1, oppnt_comm_act_dim), device=device)], dim=0)
        oppnt_states = torch.cat([oppnt_states, torch.zeros((1, oppnt_state_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        values = torch.cat([values, torch.zeros(1, device=device)])

        oppnt_action, oppnt_probs, oppnt_hidden = oppnt_model.get_action(oppnt_state, oppnt_hidden, oppnt_mask)
        oppnt_move_action_onehot = to_onehot(oppnt_probs[0])
        oppnt_comm_action_onehot = to_onehot(oppnt_probs[1])
        oppnt_actions = np.concatenate((oppnt_move_action_onehot, oppnt_comm_action_onehot))

        agent_move_action, agent_comm_action, oppnt_state, oppnt_move_action, oppnt_comm_action = agent_model.get_action(
            (agent_states.to(dtype=torch.float32) - agent_state_mean) / agent_state_std,
            agent_move_actions.to(dtype=torch.float32),
            agent_comm_actions.to(dtype=torch.float32),
            (oppnt_states.to(dtype=torch.float32) - oppnt_state_mean) / oppnt_state_std,
            oppnt_move_actions.to(dtype=torch.float32),
            oppnt_comm_actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        oppnt_states[-1] = oppnt_state
        oppnt_state = oppnt_state.detach().cpu().numpy()

        agent_move_action_onehot = to_onehot(agent_move_action)
        agent_comm_action_onehot = to_onehot(agent_comm_action)
        oppnt_move_action_onehot = to_onehot(oppnt_move_action)
        oppnt_comm_action_onehot = to_onehot(oppnt_comm_action)

        agent_move_actions[-1] = agent_move_action_onehot
        agent_move_action_onehot = agent_move_action_onehot.detach().cpu().numpy()
        agent_comm_actions[-1] = agent_comm_action_onehot
        agent_comm_action_onehot = agent_comm_action_onehot.detach().cpu().numpy()
        oppnt_move_actions[-1] = oppnt_move_action_onehot
        oppnt_move_action_onehot = oppnt_move_action_onehot.detach().cpu().numpy()
        oppnt_comm_actions[-1] = oppnt_comm_action_onehot
        oppnt_comm_action_onehot = oppnt_comm_action_onehot.detach().cpu().numpy()

        agent_actions = np.concatenate((agent_move_action_onehot, agent_comm_action_onehot))
        actions = np.stack((agent_states, oppnt_actions))

        state, reward, done, _ = env.step(actions)
        agent_state, oppnt_state = state[0], state[1]

        oppnt_state = torch.from_numpy(oppnt_state).reshape(1, oppnt_state_dim).to(device=device, dtype=torch.float32)
        cur_state = torch.from_numpy(agent_state).to(device=device).reshape(1, agent_state_dim)
        agent_states = torch.cat([agent_states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        agent_state_dim,
        agent_move_act_dim,
        agent_comm_act_dim,
        oppnt_state_dim,
        oppnt_move_act_dim,
        oppnt_comm_act_dim,
        agent_model,
        oppnt_model,
        max_ep_len=1000,
        scale=1000.,
        agent_state_mean=0.,
        agent_state_std=1.,
        oppnt_state_mean=0.,
        oppnt_state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        model_opponent_states=False,
        model_opponent_actions=False,
        opponent=0,
    ):
    # TODO: Add Accuracy metric for 20th timestep
    # TODO: Add Accuracy metric for each timestep
    agent_model.eval()
    agent_model.to(device=device)
    oppnt_model.eval()
    oppnt_model.to(device=device)

    agent_state_mean = torch.from_numpy(agent_state_mean).to(device=device)
    agent_state_std = torch.from_numpy(agent_state_std).to(device=device)
    oppnt_state_mean = torch.from_numpy(oppnt_state_mean).to(device=device)
    oppnt_state_std = torch.from_numpy(oppnt_state_std).to(device=device)

    num_envs = env.num_envs
    state = env.reset()
    agent_state, oppnt_state = state[1], state[0]

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    agent_states = torch.from_numpy(agent_state).reshape(num_envs, agent_state_dim).to(device=device, dtype=torch.float32)
    agent_move_actions = torch.zeros((0, agent_move_act_dim), device=device, dtype=torch.float32)
    agent_comm_actions = torch.zeros((0, agent_comm_act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    target_oppnt = torch.ones((num_envs, 1), device=device, dtype=torch.long) * opponent

    oppnt_states, oppnt_move_actions, oppnt_comm_actions = None, None, None
    if model_opponent_states:
        oppnt_states = torch.zeros((0, oppnt_state_dim), device=device, dtype=torch.float32)
    if model_opponent_actions:
        oppnt_move_actions = torch.zeros((0, oppnt_move_act_dim), device=device, dtype=torch.float32)
        oppnt_comm_actions = torch.zeros((0, oppnt_comm_act_dim), device=device, dtype=torch.float32)

    oppnt_state = torch.from_numpy(oppnt_state).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_hidden = torch.zeros((num_envs, oppnt_model.recurrent_dim, oppnt_model.hidden_dim)).to(device=device, dtype=torch.float32)
    oppnt_mask = torch.ones((num_envs, 1)).to(device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(num_envs, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(num_envs, 1)

    sim_states = []

    episode_return, episode_length, oppnt_move_action_matches, oppnt_comm_action_matches, oppnt_id_matches = 0, 0, 0, 0, 0
    end_eps_oppnt_move_action_match = 0
    end_eps_oppnt_comm_action_match = 0
    for t in range(max_ep_len):

        # add padding
        agent_move_actions = torch.cat([agent_move_actions, torch.zeros((num_envs, agent_move_act_dim), device=device)], dim=0)
        agent_comm_actions = torch.cat([agent_comm_actions, torch.zeros((num_envs, agent_comm_act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(num_envs, device=device)])

        oppnt_action, oppnt_probs, oppnt_hidden = oppnt_model.get_action(oppnt_state, oppnt_hidden, oppnt_mask)
        oppnt_move_action_onehot = torch.zeros((oppnt_move_act_dim),)
        oppnt_move_action_onehot[oppnt_action[0, 0]] = 1.0
        oppnt_move_action_onehot = oppnt_move_action_onehot.detach().cpu().numpy()
        oppnt_comm_action_onehot = torch.zeros((oppnt_comm_act_dim),)
        oppnt_comm_action_onehot[oppnt_action[0, 1]] = 1.0
        oppnt_comm_action_onehot = oppnt_comm_action_onehot.detach().cpu().numpy()
        oppnt_actions = np.concatenate((oppnt_move_action_onehot, oppnt_comm_action_onehot))

        agent_move_action, agent_comm_action, oppnt_pred, oppnt_state, oppnt_move_action, oppnt_comm_action = agent_model.get_action(
            agent_states=(agent_states.to(dtype=torch.float32) - agent_state_mean) / agent_state_std,
            agent_move_actions=agent_move_actions.to(dtype=torch.float32),
            agent_comm_actions=agent_comm_actions.to(dtype=torch.float32),
            rewards=rewards.to(dtype=torch.float32),
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps.to(dtype=torch.long),
        )

        agent_move_action_onehot = to_onehot(agent_move_action)
        agent_comm_action_onehot = to_onehot(agent_comm_action)

        agent_move_actions[-1] = agent_move_action_onehot
        agent_move_action_onehot = agent_move_action_onehot.detach().cpu().numpy()
        agent_comm_actions[-1] = agent_comm_action_onehot
        agent_comm_action_onehot = agent_comm_action_onehot.detach().cpu().numpy()
        agent_actions = np.concatenate((agent_move_action_onehot, agent_comm_action_onehot))
        actions = np.stack((oppnt_actions, agent_actions))

        state, reward, done, _ = env.step(actions)
        agent_state, oppnt_state = state[1], state[0]
        agent_reward = reward[1][0]
        agent_done = done[1]

        oppnt_state = torch.from_numpy(oppnt_state).reshape(1, oppnt_state_dim).to(device=device, dtype=torch.float32)
        cur_state = torch.from_numpy(agent_state).to(device=device).reshape(1, agent_state_dim)
        agent_states = torch.cat([agent_states, cur_state], dim=0)
        rewards[-1] = agent_reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (agent_reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        oppnt_id_matches += int(oppnt_pred == target_oppnt)
        episode_return += agent_reward
        episode_length += 1

        if agent_done:
            break

    return episode_return, episode_length, oppnt_move_action_matches, oppnt_comm_action_matches, oppnt_id_matches, end_eps_oppnt_move_action_match, end_eps_oppnt_comm_action_match


@torch.no_grad()
def vec_evaluate_episode_rtg(
    env,
    agent_state_dim,
    agent_act_dim,
    oppnt_state_dim,
    oppnt_act_dim,
    agent_model,
    oppnt_model,
    gamma=0.99,
    gae_lambda=0.95,
    max_ep_len=1000,
    scale=1000.,
    state_mean=0.,
    state_std=1.,
    device='cuda',
    target_return=None,
    opponent=0,
    finetune=False,
    model_opponent=True,
    agent_critic=None
):
    agent_model.eval()
    agent_model.to(device=device)
    oppnt_model.eval()
    oppnt_model.to(device=device)

    agent_state_mean = torch.from_numpy(state_mean).to(device=device)
    agent_state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = env.num_envs
    state = env.reset()
    agent_state, oppnt_state = state[1], state[0]

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(agent_state).reshape(num_envs, 1, agent_state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((num_envs, 0, agent_act_dim), device=device, dtype=torch.float32)
    oppnt_states = torch.from_numpy(oppnt_state).reshape(num_envs, 1, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_actions = torch.zeros((num_envs, 0, oppnt_act_dim), device=device, dtype=torch.float32)
    oppnt_rewards = torch.zeros(num_envs, 1, 1, device=device, dtype=torch.float32)
    probs = torch.zeros((num_envs, 0, agent_act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(num_envs, 1, 1, device=device, dtype=torch.float32)
    values = torch.zeros(num_envs, 1, 1, device=device, dtype=torch.float32)
    dones = torch.zeros(num_envs, 1, 1, device=device, dtype=torch.long)

    ep_return = target_return
    target_return = torch.ones((num_envs, 1, 1), device=device, dtype=torch.float32) * target_return
    target_oppnt = torch.ones((num_envs), device=device, dtype=torch.long) * opponent
    timesteps = torch.zeros((num_envs, 1), device=device, dtype=torch.long)

    oppnt_state = torch.from_numpy(oppnt_state).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_hidden = torch.zeros((num_envs, oppnt_model.recurrent_dim, oppnt_model.hidden_dim)).to(device=device, dtype=torch.float32)
    oppnt_mask = torch.ones((num_envs, 1)).to(device=device, dtype=torch.float32)

    episode_return, oppnt_id_matches = 0, 0
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((num_envs, 1, agent_act_dim), device=device)], dim=1)
        oppnt_actions = torch.cat([oppnt_actions, torch.zeros((num_envs, 1, oppnt_act_dim), device=device)], dim=1)
        probs = torch.cat([probs, torch.zeros((num_envs, 1, agent_act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros(num_envs, 1, 1, device=device)], dim=1)
        oppnt_rewards = torch.cat([oppnt_rewards, torch.zeros(num_envs, 1, 1, device=device)], dim=1)
        dones = torch.cat([dones, torch.zeros(num_envs, 1, 1, device=device)], dim=1)

        oppnt_action, oppnt_probs, oppnt_hidden = oppnt_model.get_action(oppnt_state, oppnt_hidden, oppnt_mask)
        oppnt_actions_onehot = F.one_hot(oppnt_action.long().squeeze(), oppnt_act_dim)

        action_pred, oppnt_outputs = agent_model.get_action(
            states=(states.to(dtype=torch.float32) - agent_state_mean) / agent_state_std,
            actions=actions.to(dtype=torch.float32),
            rewards=rewards.to(dtype=torch.float32),
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps.to(dtype=torch.long),
            oppnt_ids=target_oppnt.reshape(num_envs, 1),
            num_envs=num_envs
        )
        if agent_critic is not None:
            values = torch.cat([values, torch.zeros(num_envs, 1, 1, device=device)], dim=1)
            value = agent_critic.get_value(
                states=(states.to(dtype=torch.float32) - agent_state_mean) / agent_state_std,
                actions=actions.to(dtype=torch.float32),
                rewards=rewards.to(dtype=torch.float32),
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps.to(dtype=torch.long),
                oppnt_ids=target_oppnt.reshape(num_envs, 1),
                num_envs=num_envs
            )
            values[:, -1] = value

        if finetune:
            action_dist = torch.distributions.Categorical(probs=action_pred)
            action = action_dist.sample()
        else:
            action = action_pred.argmax(dim=-1)
        agent_actions_onehot = F.one_hot(action.long().squeeze(), agent_act_dim)

        actions[:, -1] = agent_actions_onehot
        oppnt_actions[:, -1] = oppnt_actions_onehot
        probs[:, -1] = action_pred
        agent_actions_onehot = agent_actions_onehot.squeeze().detach().cpu().numpy()
        oppnt_actions_onehot = oppnt_actions_onehot.squeeze().detach().cpu().numpy()

        env_actions = []
        for oppnt_act, agent_act in zip(oppnt_actions_onehot, agent_actions_onehot):
            env_actions.append([oppnt_act, agent_act])

        state, reward, done, _ = env.step(env_actions)
        next_oppnt_state = state[0]
        next_agent_state = state[1]
        oppnt_reward = np.asarray([r[0] for r in reward])
        agent_reward = np.asarray([r[1] for r in reward])
        agent_done = np.asarray([d[1] for d in done])

        next_oppnt_state = torch.from_numpy(next_oppnt_state).reshape(num_envs, 1, oppnt_state_dim).to(device=device, dtype=torch.float32)
        next_agent_state = torch.from_numpy(next_agent_state).reshape(num_envs, 1, agent_state_dim).to(device=device, dtype=torch.float32)
        states = torch.cat([states, next_agent_state], dim=1)
        oppnt_states = torch.cat([oppnt_states, next_oppnt_state], dim=1)
        cur_done = torch.from_numpy(agent_done).reshape(num_envs, 1).to(device=device, dtype=torch.long)
        rewards[:, -1] = torch.from_numpy(agent_reward).reshape(num_envs, 1).to(device=device, dtype=torch.float32)
        oppnt_rewards[:, -1] = torch.from_numpy(agent_reward).reshape(num_envs, 1).to(device=device, dtype=torch.float32)
        dones[:, -1] = cur_done

        pred_return = target_return[:, -1] - (rewards[:, -1] / scale)
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, 1, 1)], dim=1
        )
        timesteps = torch.cat(
            [
                timesteps, torch.ones((num_envs, 1), device=device, dtype=torch.long) * (t + 1)
            ], dim=1
        )

        episode_return += agent_reward

    oppnt_id_accuracy = oppnt_id_matches / (num_envs * max_ep_len)

    trajectories = []
    for i in range(num_envs):
        terminals = np.zeros(max_ep_len)
        terminals[-1] = 1
        opponent_ids = np.ones(max_ep_len) * opponent
        # task = np.ones(max_ep_len) * int(finetune)
        if finetune:
            task = "finetune"
            returns, advantages = compute_advantages(
                rewards,
                values,
                dones,
                value,
                cur_done,
                gamma,
                gae_lambda
            )
        else:
            task = "opponent_classification"
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
        traj = {
            "observations": states[i].detach().cpu().numpy()[:max_ep_len],
            "actions": actions[i].detach().cpu().numpy()[:max_ep_len],
            "opponent_observations": oppnt_states[i].detach().cpu().numpy()[:max_ep_len],
            "opponent_actions": oppnt_actions[i].detach().cpu().numpy()[:max_ep_len],
            "opponent_rewards": oppnt_rewards[i].detach().cpu().numpy()[:max_ep_len],
            "probs": probs[i].detach().cpu().numpy()[:max_ep_len],
            "rewards": rewards[i].detach().cpu().numpy()[:max_ep_len],
            "returns": returns[i].detach().cpu().numpy()[:max_ep_len],
            "advantages": advantages[i].cpu().numpy()[:max_ep_len],
            "terminals": terminals,
            "opponent": opponent_ids,
            "task": task
        }
        trajectories.append(traj)

    return episode_return, oppnt_id_accuracy, trajectories


@torch.no_grad()
def vec_evaluate_episode_vae(
    env,
    agent_state_dim,
    agent_act_dim,
    oppnt_state_dim,
    oppnt_act_dim,
    agent_model,
    oppnt_model,
    gamma=0.99,
    gae_lambda=0.95,
    max_ep_len=1000,
    scale=1000.,
    state_mean=0.,
    state_std=1.,
    device='cuda',
    target_return=None,
    opponent=0,
    finetune=False,
    model_opponent=True,
    agent_critic=None
):
    agent_model.eval()
    agent_model.to(device=device)
    oppnt_model.eval()
    oppnt_model.to(device=device)

    agent_state_mean = torch.from_numpy(state_mean).to(device=device)
    agent_state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = env.num_envs
    state = env.reset()
    agent_state, oppnt_state = state[1], state[0]

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(agent_state).reshape(num_envs, 1, agent_state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((num_envs, 0, agent_act_dim), device=device, dtype=torch.float32)
    oppnt_states = torch.from_numpy(oppnt_state).reshape(num_envs, 1, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_actions = torch.zeros((num_envs, 0, oppnt_act_dim), device=device, dtype=torch.float32)
    oppnt_rewards = torch.zeros(num_envs, 1, 1, device=device, dtype=torch.float32)
    probs = torch.zeros((num_envs, 0, agent_act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(num_envs, 1, 1, device=device, dtype=torch.float32)
    values = torch.zeros(num_envs, 1, 1, device=device, dtype=torch.float32)
    dones = torch.zeros(num_envs, 1, 1, device=device, dtype=torch.long)
    hidden1 = torch.zeros(agent_model.oppnt_model.num_layers, num_envs, agent_model.oppnt_model.hidden_dim, device=device, dtype=torch.float32)
    hidden2 = torch.zeros(agent_model.oppnt_model.num_layers, num_envs, agent_model.oppnt_model.hidden_dim, device=device, dtype=torch.float32)
    hidden = (hidden1, hidden2)

    ep_return = target_return
    target_return = torch.ones((num_envs, 1, 1), device=device, dtype=torch.float32) * target_return
    target_oppnt = torch.ones((num_envs), device=device, dtype=torch.long) * opponent
    timesteps = torch.zeros((num_envs, 1), device=device, dtype=torch.long)

    oppnt_state = torch.from_numpy(oppnt_state).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_hidden = torch.zeros((num_envs, oppnt_model.recurrent_dim, oppnt_model.hidden_dim)).to(device=device, dtype=torch.float32)
    oppnt_mask = torch.ones((num_envs, 1)).to(device=device, dtype=torch.float32)

    episode_return, oppnt_id_matches = 0, 0
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((num_envs, 1, agent_act_dim), device=device)], dim=1)
        oppnt_actions = torch.cat([oppnt_actions, torch.zeros((num_envs, 1, oppnt_act_dim), device=device)], dim=1)
        probs = torch.cat([probs, torch.zeros((num_envs, 1, agent_act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros(num_envs, 1, 1, device=device)], dim=1)
        oppnt_rewards = torch.cat([oppnt_rewards, torch.zeros(num_envs, 1, 1, device=device)], dim=1)
        dones = torch.cat([dones, torch.zeros(num_envs, 1, 1, device=device)], dim=1)

        oppnt_action, oppnt_probs, oppnt_hidden = oppnt_model.get_action(oppnt_state, oppnt_hidden, oppnt_mask)
        oppnt_actions_onehot = F.one_hot(oppnt_action.long().squeeze(), oppnt_act_dim)

        action_pred, oppnt_outputs = agent_model.get_action(
            states=(states.to(dtype=torch.float32) - agent_state_mean) / agent_state_std,
            actions=actions.to(dtype=torch.float32),
            rewards=rewards.to(dtype=torch.float32),
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps.to(dtype=torch.long),
            oppnt_ids=target_oppnt.reshape(num_envs, 1),
            num_envs=num_envs,
            rnn_state=hidden
        )
        hidden = oppnt_outputs['rnn_state']
        if agent_critic is not None:
            values = torch.cat([values, torch.zeros(num_envs, 1, 1, device=device)], dim=1)
            value = agent_critic.get_value(
                states=(states.to(dtype=torch.float32) - agent_state_mean) / agent_state_std,
                actions=actions.to(dtype=torch.float32),
                rewards=rewards.to(dtype=torch.float32),
                returns_to_go=target_return.to(dtype=torch.float32),
                timesteps=timesteps.to(dtype=torch.long),
                oppnt_ids=target_oppnt.reshape(num_envs, 1),
                num_envs=num_envs
            )
            values[:, -1] = value

        if finetune:
            action_dist = torch.distributions.Categorical(probs=action_pred)
            action = action_dist.sample()
        else:
            action = action_pred.argmax(dim=-1)
        agent_actions_onehot = F.one_hot(action.long().squeeze(), agent_act_dim)

        actions[:, -1] = agent_actions_onehot
        oppnt_actions[:, -1] = oppnt_actions_onehot
        probs[:, -1] = action_pred
        agent_actions_onehot = agent_actions_onehot.squeeze().detach().cpu().numpy()
        oppnt_actions_onehot = oppnt_actions_onehot.squeeze().detach().cpu().numpy()

        env_actions = []
        for oppnt_act, agent_act in zip(oppnt_actions_onehot, agent_actions_onehot):
            env_actions.append([oppnt_act, agent_act])

        state, reward, done, _ = env.step(env_actions)
        next_oppnt_state = state[0]
        next_agent_state = state[1]
        oppnt_reward = np.asarray([r[0] for r in reward])
        agent_reward = np.asarray([r[1] for r in reward])
        agent_done = np.asarray([d[1] for d in done])

        next_oppnt_state = torch.from_numpy(next_oppnt_state).reshape(num_envs, 1, oppnt_state_dim).to(device=device, dtype=torch.float32)
        next_agent_state = torch.from_numpy(next_agent_state).reshape(num_envs, 1, agent_state_dim).to(device=device, dtype=torch.float32)
        states = torch.cat([states, next_agent_state], dim=1)
        oppnt_states = torch.cat([oppnt_states, next_oppnt_state], dim=1)
        cur_done = torch.from_numpy(agent_done).reshape(num_envs, 1).to(device=device, dtype=torch.long)
        rewards[:, -1] = torch.from_numpy(agent_reward).reshape(num_envs, 1).to(device=device, dtype=torch.float32)
        oppnt_rewards[:, -1] = torch.from_numpy(agent_reward).reshape(num_envs, 1).to(device=device, dtype=torch.float32)
        dones[:, -1] = cur_done

        pred_return = target_return[:, -1] - (rewards[:, -1] / scale)
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, 1, 1)], dim=1
        )
        timesteps = torch.cat(
            [
                timesteps, torch.ones((num_envs, 1), device=device, dtype=torch.long) * (t + 1)
            ], dim=1
        )

        episode_return += agent_reward

    oppnt_id_accuracy = oppnt_id_matches / (num_envs * max_ep_len)

    trajectories = []
    for i in range(num_envs):
        terminals = np.zeros(max_ep_len)
        terminals[-1] = 1
        opponent_ids = np.ones(max_ep_len) * opponent
        # task = np.ones(max_ep_len) * int(finetune)
        if finetune:
            task = "finetune"
            returns, advantages = compute_advantages(
                rewards,
                values,
                dones,
                value,
                cur_done,
                gamma,
                gae_lambda
            )
        else:
            task = "opponent_classification"
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
        traj = {
            "observations": states[i].detach().cpu().numpy()[:max_ep_len],
            "actions": actions[i].detach().cpu().numpy()[:max_ep_len],
            "opponent_observations": oppnt_states[i].detach().cpu().numpy()[:max_ep_len],
            "opponent_actions": oppnt_actions[i].detach().cpu().numpy()[:max_ep_len],
            "opponent_rewards": oppnt_rewards[i].detach().cpu().numpy()[:max_ep_len],
            "probs": probs[i].detach().cpu().numpy()[:max_ep_len],
            "rewards": rewards[i].detach().cpu().numpy()[:max_ep_len],
            "returns": returns[i].detach().cpu().numpy()[:max_ep_len],
            "advantages": advantages[i].cpu().numpy()[:max_ep_len],
            "terminals": terminals,
            "opponent": opponent_ids,
            "task": task
        }
        trajectories.append(traj)

    return episode_return, oppnt_id_accuracy, trajectories


@torch.no_grad()
def vec_evaluate_episode_liam(
    env,
    agent_state_dim,
    agent_move_act_dim,
    agent_comm_act_dim,
    oppnt_state_dim,
    oppnt_move_act_dim,
    oppnt_comm_act_dim,
    hidden_dim,
    agent_model,
    oppnt_model,
    max_ep_len=1000,
    gamma=0.99,
    gae_lambda=0.95,
    device='cuda',
):
    agent_model.eval()
    agent_model.to(device=device)
    oppnt_model.eval()
    oppnt_model.to(device=device)

    num_envs = env.num_envs
    act_dim = agent_move_act_dim + agent_comm_act_dim
    state = env.reset()
    agent_state, oppnt_state = state[:, 1], state[:, 0]

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(agent_state).reshape(num_envs, agent_state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((num_envs, act_dim), device=device, dtype=torch.float32)
    hidden1 = torch.zeros(1, num_envs, hidden_dim, device=device, dtype=torch.float32)
    hidden2 = torch.zeros(1, num_envs, hidden_dim, device=device, dtype=torch.float32)
    hidden = (hidden1, hidden2)

    oppnt_state = torch.from_numpy(oppnt_state).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_hidden = torch.zeros((num_envs, oppnt_model.recurrent_dim, oppnt_model.hidden_dim)).to(device=device, dtype=torch.float32)
    oppnt_mask = torch.ones((num_envs, 1)).to(device=device, dtype=torch.float32)

    episode_return, oppnt_move_matches, oppnt_comm_matches = 0, 0, 0
    for t in range(max_ep_len):

        oppnt_action, oppnt_probs, oppnt_hidden = oppnt_model.get_action(oppnt_state, oppnt_hidden, oppnt_mask)
        oppnt_move_action_onehot = F.one_hot(oppnt_action[:, 0].long(), oppnt_move_act_dim)
        oppnt_comm_action_onehot = F.one_hot(oppnt_action[:, 1].long(), oppnt_comm_act_dim)
        oppnt_move_action_onehot = oppnt_move_action_onehot.detach().cpu().numpy()
        oppnt_comm_action_onehot = oppnt_comm_action_onehot.detach().cpu().numpy()
        oppnt_actions = np.concatenate((oppnt_move_action_onehot, oppnt_comm_action_onehot), axis=1)

        move_action, comm_action, oppnt_move_pred, oppnt_comm_pred, hidden = agent_model.get_action(
            obs=states.to(dtype=torch.float32),
            action=actions.to(dtype=torch.float32),
            hidden=hidden,
        )

        if t == 20:
            # print("T = 20: Opponent move preds: ", oppnt_move_pred)
            # print("T = 20: Opponent comm preds: ", oppnt_comm_pred)
            # print("T = 20: Opponent move targets: ", oppnt_move_action_onehot.argmax(-1))
            # print("T = 20: Opponent comm targets: ", oppnt_comm_action_onehot.argmax(-1))

            oppnt_move_matches = (oppnt_move_pred.cpu().numpy() == oppnt_move_action_onehot.argmax(-1)).sum()
            oppnt_comm_matches = (oppnt_comm_pred.cpu().numpy() == oppnt_comm_action_onehot.argmax(-1)).sum()

        move_action_onehot = F.one_hot(move_action.long().squeeze(), agent_move_act_dim)
        comm_action_onehot = F.one_hot(comm_action.long().squeeze(), agent_comm_act_dim)

        actions = torch.cat((move_action_onehot, comm_action_onehot), dim=-1)

        move_action_onehot = move_action_onehot.detach().cpu().numpy()
        comm_action_onehot = comm_action_onehot.detach().cpu().numpy()
        agent_actions = np.concatenate((move_action_onehot, comm_action_onehot), axis=1)

        env_actions = []
        for oppnt_act, agent_act in zip(oppnt_actions, agent_actions):
            env_actions.append([oppnt_act, agent_act])

        state, reward, done, _ = env.step(env_actions)
        oppnt_states = np.asarray([s[0] for s in state])
        agent_states = np.asarray([s[1] for s in state])
        agent_reward = np.asarray([r[1] for r in reward])
        agent_done = np.asarray([d[1] for d in done])

        oppnt_state = torch.from_numpy(oppnt_states).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
        states = torch.from_numpy(agent_states).reshape(num_envs, agent_state_dim).to(device=device, dtype=torch.float32)

        episode_return += agent_reward

    oppnt_move_accuracy = oppnt_move_matches / (num_envs)
    oppnt_comm_accuracy = oppnt_comm_matches / (num_envs)

    return episode_return, oppnt_move_accuracy, oppnt_comm_accuracy


@torch.no_grad()
def vec_evaluate_episode_cql(
    env,
    agent_state_dim,
    agent_move_act_dim,
    agent_comm_act_dim,
    oppnt_state_dim,
    oppnt_move_act_dim,
    oppnt_comm_act_dim,
    agent_model,
    oppnt_model,
    max_ep_len=1000,
    gamma=0.99,
    gae_lambda=0.95,
    device='cuda',
):
    agent_model.eval()
    agent_model.to(device=device)
    oppnt_model.eval()
    oppnt_model.to(device=device)

    num_envs = env.num_envs
    act_dim = agent_move_act_dim + agent_comm_act_dim
    state = env.reset()
    agent_state, oppnt_state = state[:, 1], state[:, 0]

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(agent_state).reshape(num_envs, agent_state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((num_envs, act_dim), device=device, dtype=torch.float32)

    oppnt_state = torch.from_numpy(oppnt_state).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_hidden = torch.zeros((num_envs, oppnt_model.recurrent_dim, oppnt_model.hidden_dim)).to(device=device, dtype=torch.float32)
    oppnt_mask = torch.ones((num_envs, 1)).to(device=device, dtype=torch.float32)

    episode_return, oppnt_id_matches = 0, 0
    for t in range(max_ep_len):

        oppnt_action, oppnt_probs, oppnt_hidden = oppnt_model.get_action(oppnt_state, oppnt_hidden, oppnt_mask)
        oppnt_move_action_onehot = F.one_hot(oppnt_action[:, 0].long(), oppnt_move_act_dim)
        oppnt_comm_action_onehot = F.one_hot(oppnt_action[:, 1].long(), oppnt_comm_act_dim)
        oppnt_move_action_onehot = oppnt_move_action_onehot.detach().cpu().numpy()
        oppnt_comm_action_onehot = oppnt_comm_action_onehot.detach().cpu().numpy()
        oppnt_actions = np.concatenate((oppnt_move_action_onehot, oppnt_comm_action_onehot), axis=1)

        move_action, comm_action = agent_model.get_action(
            obs=states.to(dtype=torch.float32),
        )

        move_action_onehot = F.one_hot(move_action.long().squeeze(), agent_move_act_dim)
        comm_action_onehot = F.one_hot(comm_action.long().squeeze(), agent_comm_act_dim)

        actions = torch.cat((move_action_onehot, comm_action_onehot), dim=-1)

        move_action_onehot = move_action_onehot.detach().cpu().numpy()
        comm_action_onehot = comm_action_onehot.detach().cpu().numpy()
        agent_actions = np.concatenate((move_action_onehot, comm_action_onehot), axis=1)

        env_actions = []
        for oppnt_act, agent_act in zip(oppnt_actions, agent_actions):
            env_actions.append([oppnt_act, agent_act])

        state, reward, done, _ = env.step(env_actions)
        oppnt_states = np.asarray([s[0] for s in state])
        agent_states = np.asarray([s[1] for s in state])
        agent_reward = np.asarray([r[1] for r in reward])
        agent_done = np.asarray([d[1] for d in done])

        oppnt_state = torch.from_numpy(oppnt_states).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
        states = torch.from_numpy(agent_states).reshape(num_envs, agent_state_dim).to(device=device, dtype=torch.float32)

        episode_return += agent_reward

    return episode_return



@torch.no_grad()
def vec_evaluate_episode_cbom(
    env,
    agent_state_dim,
    agent_move_act_dim,
    agent_comm_act_dim,
    oppnt_state_dim,
    oppnt_move_act_dim,
    oppnt_comm_act_dim,
    opponent,
    agent_model,
    oppnt_model,
    max_ep_len=1000,
    gamma=0.99,
    gae_lambda=0.95,
    device='cuda',
    finetune=False,
):
    agent_model.eval()
    agent_model.to(device=device)
    oppnt_model.eval()
    oppnt_model.to(device=device)

    num_envs = env.num_envs
    act_dim = agent_move_act_dim + agent_comm_act_dim
    state = env.reset()
    agent_state, oppnt_state = state[:, 1], state[:, 0]

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(agent_state).reshape(num_envs, agent_state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((num_envs, act_dim), device=device, dtype=torch.float32)

    all_states = np.zeros((num_envs, max_ep_len, agent_state_dim), dtype=np.float32)
    all_move_actions = np.zeros((num_envs, max_ep_len, agent_move_act_dim), dtype=np.float32)
    all_comm_actions = np.zeros((num_envs, max_ep_len, agent_comm_act_dim), dtype=np.float32)
    all_rewards = np.zeros((num_envs, max_ep_len, 1), dtype=np.float32)
    all_dones = np.zeros((num_envs, max_ep_len), dtype=bool)

    target_oppnt = torch.ones((num_envs), device=device, dtype=torch.long) * opponent

    oppnt_state = torch.from_numpy(oppnt_state).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
    oppnt_hidden = torch.zeros((num_envs, oppnt_model.recurrent_dim, oppnt_model.hidden_dim)).to(device=device, dtype=torch.float32)
    oppnt_mask = torch.ones((num_envs, 1)).to(device=device, dtype=torch.float32)

    episode_return, oppnt_id_matches = 0, 0
    for t in range(max_ep_len):
        all_states[:, t] = states.cpu().numpy()

        oppnt_action, oppnt_probs, oppnt_hidden = oppnt_model.get_action(oppnt_state, oppnt_hidden, oppnt_mask)
        oppnt_move_action_onehot = F.one_hot(oppnt_action[:, 0].long(), oppnt_move_act_dim)
        oppnt_comm_action_onehot = F.one_hot(oppnt_action[:, 1].long(), oppnt_comm_act_dim)
        oppnt_move_action_onehot = oppnt_move_action_onehot.detach().cpu().numpy()
        oppnt_comm_action_onehot = oppnt_comm_action_onehot.detach().cpu().numpy()
        oppnt_actions = np.concatenate((oppnt_move_action_onehot, oppnt_comm_action_onehot), axis=1)

        move_action, comm_action, oppnt_pred = agent_model.get_action(
            state=states.to(dtype=torch.float32),
            prev_action=actions.to(dtype=torch.long),
            opponent_id=target_oppnt
        )

        move_action_onehot = F.one_hot(move_action.long().squeeze(), agent_move_act_dim)
        comm_action_onehot = F.one_hot(comm_action.long().squeeze(), agent_comm_act_dim)
        all_move_actions[:, t] = move_action_onehot.cpu().numpy()
        all_comm_actions[:, t] = comm_action_onehot.cpu().numpy()

        actions = torch.cat((move_action_onehot, comm_action_onehot), dim=-1)

        move_action_onehot = move_action_onehot.detach().cpu().numpy()
        comm_action_onehot = comm_action_onehot.detach().cpu().numpy()
        agent_actions = np.concatenate((move_action_onehot, comm_action_onehot), axis=1)

        env_actions = []
        for oppnt_act, agent_act in zip(oppnt_actions, agent_actions):
            env_actions.append([oppnt_act, agent_act])

        state, reward, done, _ = env.step(env_actions)
        oppnt_states = np.asarray([s[0] for s in state])
        agent_states = np.asarray([s[1] for s in state])
        agent_reward = np.asarray([r[1] for r in reward])
        agent_done = np.asarray([d[1] for d in done])
        all_rewards[:, t] = agent_reward
        all_dones[:, t] = agent_done

        oppnt_state = torch.from_numpy(oppnt_states).reshape(num_envs, oppnt_state_dim).to(device=device, dtype=torch.float32)
        states = torch.from_numpy(agent_states).reshape(num_envs, agent_state_dim).to(device=device, dtype=torch.float32)

        episode_return += agent_reward
        oppnt_id_matches += int((oppnt_pred.argmax(-1) == target_oppnt).sum())

    oppnt_id_accuracy = oppnt_id_matches / (num_envs * max_ep_len)

    trajectories = []
    for i in range(num_envs):
        terminals = np.zeros(max_ep_len, dtype=bool)
        terminals[-1] = 1
        # print("Terminals type = ", terminals.dtype)
        opponent_ids = np.ones(max_ep_len, dtype=np.uint8) * opponent
        if finetune:
            task = "finetune"
        else:
            task = "opponent_classification"
        traj = {
            "observations": all_states[i],
            "move_actions": all_move_actions[i],
            "comm_actions": all_comm_actions[i],
            "rewards": all_rewards[i],
            "terminals": terminals,
            "opponent": opponent_ids,
            "task": task
        }
        trajectories.append(traj)

    return episode_return, oppnt_id_accuracy, trajectories
