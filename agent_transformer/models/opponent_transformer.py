import numpy as np
import transformers
import torch
import torch.nn as nn

from opponent_transformer.models.trajectory_gpt2 import GPT2Model
from opponent_transformer.models.utils import stack_inputs, stack_attention_mask


class OpponentTransformerActor(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_dim,
        action_tanh,
        max_length=None,
        max_ep_len=4096,
        oppnt_model=None,
        **kwargs
    ):
        super().__init__()
        gpt2_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_dim,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # self.encoder = BertModel(bert_config)
        self.decoder = GPT2Model(gpt2_config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_return = torch.nn.Linear(1, hidden_dim)
        self.embed_state = torch.nn.Linear(state_dim, hidden_dim)
        self.embed_action = torch.nn.Linear(act_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)

        self.oppnt_model = oppnt_model
        if self.oppnt_model is not None:
            self.embed_oppnt = torch.nn.Linear(self.oppnt_model.embedding_dim, hidden_dim)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_dim, state_dim)
        self.predict_return = torch.nn.Linear(hidden_dim, 1)
        self.predict_value = torch.nn.Linear(hidden_dim, 1)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_dim, act_dim)] + ([nn.Tanh()] if action_tanh else [nn.Softmax(dim=-1)]))
        )

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

    def forward(
        self,
        states,
        actions,
        returns_to_go,
        timesteps,
        oppnt_outputs=None,
        attention_mask=None,
        **kwargs
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        if oppnt_outputs is not None:
            oppnt_embeddings = self.embed_oppnt(oppnt_outputs)

            state_embeddings = state_embeddings + oppnt_embeddings
            action_embeddings = action_embeddings + oppnt_embeddings
            returns_embeddings = returns_embeddings + oppnt_embeddings

            # state_embeddings = state_embeddings + oppnt_outputs
            # action_embeddings = action_embeddings + oppnt_outputs
            # returns_embeddings = returns_embeddings + oppnt_outputs

        embeddings = (returns_embeddings, state_embeddings, action_embeddings)

        stacked_inputs = stack_inputs(embeddings)
        stacked_attention_mask = stack_attention_mask(attention_mask, len(embeddings), batch_size, seq_length)

        transformer_outputs = self.decoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, len(embeddings), self.hidden_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given agent state, opponent state, opponent action and agent action
        action_preds = self.predict_action(x[:, 1])  # predict next action given agent state, opponent state and opponent action

        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states,
        actions,
        returns_to_go,
        timesteps,
        num_envs=1,
        **kwargs
    ):
        # we don't care about the past rewards in this model
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(num_envs, -1, 1)
        timesteps = timesteps.reshape(num_envs, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(num_envs, self.max_length-states.shape[1]), torch.ones(num_envs, states.shape[1])], dim=1)
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(num_envs, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32
            )
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32
            )
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32
            )
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        if self.oppnt_model is not None:
            oppnt_outputs = self.oppnt_model(states=states, actions=actions, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs)
        else:
            oppnt_outputs = {}

        _, action_preds, _ = self.forward(
            states=states, actions=actions, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, oppnt_outputs=oppnt_outputs.get('embeddings'), **kwargs
        )

        return action_preds[:, -1], oppnt_outputs


# TODO: Generalize the Decision Transformer to work with any type of opponent model
# TODO: Implement Opponent Models in a separate module
# TODO: Include VAE, MLP Classifier, Meta-RL, Hierarchical-RL

class OpponentTransformerCritic(nn.Module):
    def __init__(
        self,
        hidden_dim,
        **kwargs
    ):
        del kwargs['model_opponent']
        super().__init__()

        gpt2_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_dim,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # self.encoder = BertModel(bert_config)
        self.decoder = GPT2Model(gpt2_config)

        self.predict_value = torch.nn.Linear(hidden_dim, 1)

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        oppnt_ids=None,
        attention_mask=None
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings, action_embeddings, returns_embeddings, time_embeddings = self.compute_embeddings(states, actions, returns_to_go, timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        embeddings = (returns_embeddings, state_embeddings, action_embeddings)

        stacked_inputs = self.stack_inputs(embeddings, batch_size, len(embeddings), seq_length)
        stacked_attention_mask = self.stack_attention(attention_mask, batch_size, len(embeddings), seq_length)

        transformer_outputs = self.decoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, len(embeddings), self.hidden_dim).permute(0, 2, 1, 3)

        value_preds = self.predict_value(x[:, 2])

        return value_preds

    def get_value(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        oppnt_ids=None,
        num_envs=1,
        **kwargs
    ):
        # we don't care about the past rewards in this model
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(num_envs, -1, 1)
        timesteps = timesteps.reshape(num_envs, -1)
        oppnt_ids = oppnt_ids.reshape(num_envs, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]
            oppnt_ids = oppnt_ids[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(num_envs, self.max_length-states.shape[1]), torch.ones(num_envs, states.shape[1])], dim=1)
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(num_envs, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32
            )
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32
            )
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32
            )
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
            oppnt_ids = torch.cat(
                [torch.zeros((oppnt_ids.shape[0], self.max_length-oppnt_ids.shape[1]), device=oppnt_ids.device), oppnt_ids],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        value_preds = self.forward(
            states=states, actions=actions, oppnt_ids=oppnt_ids, rewards=None, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs
        )

        return value_preds[:, -1]
