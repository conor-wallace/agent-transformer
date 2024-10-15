import numpy as np
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):

    def __init__(
        self,
        state_dim,
        act_dim,
        max_length=None
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])


class OpponentTransformer(nn.Module):
    def __init__(
            self,
            num_opponents,
            state_dim,
            act_dim,
            oppnt_state_dim,
            oppnt_act_dim,
            hidden_dim,
            max_length=None,
            max_ep_len=4096,
            model_opponent_policy=True,
            model_opponent_states=False,
            model_opponent_actions=False,
            model_opponent_returns=False,
            **kwargs
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=kwargs['n_head'], activation=kwargs['activation_function'], dropout=kwargs['resid_pdrop']
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=kwargs['n_layer'])

        # bert_config = transformers.BertConfig(
        #     vocab_size=1,  # doesn't matter -- we don't use the vocab
        #     hidden_size=hidden_dim,
        #     num_hidden_layers=kwargs['n_layer'],
        #     num_attention_heads=kwargs['n_head'],
        #     intermediate_size=kwargs['n_inner'],
        #     hidden_act=kwargs['activation_function'],
        #     hidden_dropout_prob=kwargs['resid_pdrop'],
        #     attention_probs_dropout_prob=kwargs['attn_pdrop'],
        #     max_position_embeddings=kwargs['n_positions']
        # )
        # self.encoder = BertModel(bert_config)

        # gpt2_config = transformers.GPT2Config(
        #     vocab_size=1,  # doesn't matter -- we don't use the vocab
        #     n_embd=hidden_dim,
        #     **kwargs
        # )
        # self.encoder = GPT2Model(gpt2_config)

        self.oppnt_embedding_dim = 0
        if model_opponent_policy:
            self.oppnt_embedding_dim += num_opponents
            self.predict_oppnt_policy = nn.Sequential(torch.nn.Linear(hidden_dim, num_opponents), nn.Softmax(dim=-1))
        if model_opponent_states:
            self.oppnt_embedding_dim += oppnt_state_dim
            self.predict_oppnt_states = torch.nn.Linear(hidden_dim, oppnt_state_dim)
        if model_opponent_actions:
            self.oppnt_embedding_dim += oppnt_act_dim
            self.predict_oppnt_actions = nn.Sequential(torch.nn.Linear(hidden_dim, oppnt_act_dim), nn.Softmax(dim=-1))
        if model_opponent_returns:
            self.oppnt_embedding_dim += 1
            self.predict_oppnt_returns = torch.nn.Linear(hidden_dim, 1)

        if self.oppnt_embedding_dim == 0:
            raise ValueError("Opponent embedding dim must be greater than zero.")

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_return = torch.nn.Linear(1, hidden_dim)
        self.embed_state = torch.nn.Linear(state_dim, hidden_dim)
        self.embed_action = torch.nn.Linear(act_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states,
        actions,
        returns_to_go,
        timesteps,
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

        embeddings = (returns_embeddings, state_embeddings, action_embeddings)

        stacked_inputs = stack_inputs(embeddings)
        stacked_attention_mask = stack_attention_mask(attention_mask, len(embeddings), batch_size, seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask
        )
        x = encoder_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, len(embeddings), self.hidden_dim).permute(0, 2, 1, 3)

        outputs = {'embeddings': x[:, 2]}

        return outputs

    def predict_opponent(self, embeddings):
        oppnt_preds = {}
        if hasattr(self, 'predict_oppnt_policy'):
            oppnt_preds['policy'] = self.predict_oppnt_policy(embeddings)
        if hasattr(self, 'predict_oppnt_states'):
            oppnt_preds['states'] = self.predict_oppnt_states(embeddings)
        if hasattr(self, 'predict_oppnt_actions'):
            oppnt_preds['actions'] = self.predict_oppnt_actions(embeddings)
        if hasattr(self, 'predict_oppnt_returns'):
            oppnt_preds['returns'] = self.predict_oppnt_returns(embeddings)

        return oppnt_preds