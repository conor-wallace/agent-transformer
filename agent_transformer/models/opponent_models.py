import os
import numpy as np
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint

from opponent_transformer.models.transformer import BERTEncoder
from opponent_transformer.models.trajectory_bert import BertModel
from opponent_transformer.models.trajectory_gpt2 import GPT2Model
from opponent_transformer.models.utils import stack_inputs, stack_attention_mask


class OpponentTransformer(nn.Module):
    def __init__(
            self,
            num_opponents,
            state_dim,
            act_dim,
            oppnt_state_dims,
            oppnt_act_dims,
            hidden_dim,
            embedding_dim,
            max_length=None,
            max_ep_len=4096,
            model_opponent_policy=True,
            model_opponent_states=False,
            model_opponent_actions=False,
            model_opponent_returns=False,
            use_embedding_layer=True,
            fuse_embeddings=False,
            **kwargs
    ):
        super().__init__()

        self.obs_dim = state_dim
        self.act_dim = act_dim
        self.seq_length = max_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_embedding_layer = use_embedding_layer
        self.fuse_embeddings = fuse_embeddings

        if self.use_embedding_layer:
            if self.fuse_embeddings:
                self.embedding = nn.Linear(3 * hidden_dim, embedding_dim)
            else:
                self.embedding = nn.Linear(hidden_dim, embedding_dim)
        else:
            if self.fuse_embeddings:
                embedding_dim = 3 * hidden_dim
            else:
                embedding_dim = hidden_dim

        self.oppnt_embedding_dim = 0
        if model_opponent_policy:
            self.oppnt_embedding_dim += num_opponents
            self.predict_oppnt_policy = nn.Sequential(torch.nn.Linear(embedding_dim, num_opponents), nn.Softmax(dim=-1))
        if model_opponent_states:
            self.oppnt_embedding_dim += sum(oppnt_state_dims)
            self.predict_oppnt_states = torch.nn.Linear(embedding_dim, sum(oppnt_state_dims))
        if model_opponent_actions:
            self.predict_oppnt_actions = []
            for oppnt_act_dim in oppnt_act_dims:
                self.oppnt_embedding_dim += oppnt_act_dim
                self.predict_oppnt_actions.append(nn.Sequential(torch.nn.Linear(embedding_dim, oppnt_act_dim), nn.Softmax(dim=-1)))
            self.predict_oppnt_actions = nn.ModuleList(self.predict_oppnt_actions)
        if model_opponent_returns:
            self.oppnt_embedding_dim += 1
            self.predict_oppnt_returns = torch.nn.Linear(embedding_dim, 1)

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
        
        x = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask
        )

        x = x.reshape(batch_size, seq_length, len(embeddings), self.hidden_dim).permute(0, 2, 1, 3)
        r_embeddings = x[:, 0]
        o_embeddings = x[:, 1]
        a_embeddings = x[:, 2]

        if self.fuse_embeddings:
            embeddings = torch.cat((r_embeddings, o_embeddings, a_embeddings), dim=-1)
        else:
            embeddings = o_embeddings

        if self.use_embedding_layer:
            embeddings = self.embedding(embeddings)

        return embeddings

    def predict(
        self,
        states,
        actions,
        returns_to_go,
        timesteps,
        attention_mask=None
    ):
        # pad sequences
        batch_size, tlen, device = states.shape[0], states.shape[1], states.device
        states = torch.cat(
            (torch.zeros((batch_size, max(self.seq_length - tlen, 0), self.obs_dim), device=device), states), dim=1
        )
        actions = torch.cat(
            (torch.ones((batch_size, max(self.seq_length - tlen, 0), self.act_dim), device=device), actions), dim=1
        )
        returns_to_go = torch.cat(
            (torch.zeros((batch_size, max(self.seq_length - tlen, 0), 1), device=device), returns_to_go), dim=1
        )
        timesteps = torch.cat(
            (torch.zeros((batch_size, max(self.seq_length - tlen, 0)), device=device), timesteps), dim=1
        ).to(torch.long)
        mask = torch.cat((torch.zeros((batch_size, max(self.seq_length - tlen, 0)), device=device), torch.ones((batch_size, tlen), device=device, dtype=torch.float32)), dim=1)

        outputs = self(
            states[:, :self.seq_length],
            actions[:, :self.seq_length],
            returns_to_go[:, :self.seq_length],
            timesteps[:, :self.seq_length],
            mask[:, :self.seq_length]
        )

        return outputs['embeddings'][:, -1]

    def predict_opponent(self, embeddings):
        opp_obs_preds = self.predict_oppnt_states(embeddings)
        opp_act_preds = []
        for predict_oppnt_action in self.predict_oppnt_actions:
            opp_act_preds.append(predict_oppnt_action(embeddings).unsqueeze(2))
        opp_act_preds = torch.cat(opp_act_preds, dim=2)

        return opp_obs_preds, opp_act_preds
    
    def save_params(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)
    
    def load_params(self, path):
        save_dict = torch.load(path)
        self.load_state_dict(save_dict)


class BERTOpponentTransformer(OpponentTransformer):
    def __init__(
            self,
            hidden_dim,
            **kwargs
    ):
        super().__init__(hidden_dim=hidden_dim, **kwargs)

        bert_config = transformers.BertConfig(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            hidden_size=hidden_dim,
            num_hidden_layers=kwargs['n_layer'],
            num_attention_heads=kwargs['n_head'],
            intermediate_size=kwargs['n_inner'],
            hidden_act=kwargs['activation_function'],
            hidden_dropout_prob=kwargs['resid_pdrop'],
            attention_probs_dropout_prob=kwargs['attn_pdrop'],
            max_position_embeddings=kwargs['n_positions']
        )
        self.encoder = BertModel(bert_config)


class BERTRAOOpponentTransformer(OpponentTransformer):
    def __init__(
            self,
            hidden_dim,
            **kwargs
    ):
        super().__init__(hidden_dim=hidden_dim, **kwargs)

        bert_config = transformers.BertConfig(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            hidden_size=hidden_dim,
            num_hidden_layers=kwargs['n_layer'],
            num_attention_heads=kwargs['n_head'],
            intermediate_size=kwargs['n_inner'],
            hidden_act=kwargs['activation_function'],
            hidden_dropout_prob=kwargs['resid_pdrop'],
            attention_probs_dropout_prob=kwargs['attn_pdrop'],
            max_position_embeddings=kwargs['n_positions']
        )
        self.encoder = BertModel(bert_config)

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

        embeddings = (returns_embeddings, action_embeddings, state_embeddings)

        stacked_inputs = stack_inputs(embeddings)
        stacked_attention_mask = stack_attention_mask(attention_mask, len(embeddings), batch_size, seq_length)
        
        x = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask
        )

        x = x.reshape(batch_size, seq_length, len(embeddings), self.hidden_dim).permute(0, 2, 1, 3)
        r_embeddings = x[:, 0]
        a_embeddings = x[:, 1]
        o_embeddings = x[:, 2]

        if self.fuse_embeddings:
            embeddings = torch.cat((r_embeddings, a_embeddings, o_embeddings), dim=-1)
        else:
            embeddings = o_embeddings

        if self.use_embedding_layer:
            embeddings = self.embedding(embeddings)

        return embeddings


class BERTDiscriminativeOpponentTransformer(OpponentTransformer):
    def __init__(
            self,
            hidden_dim,
            **kwargs
    ):
        super().__init__(hidden_dim=hidden_dim, **kwargs)

        bert_config = transformers.BertConfig(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            hidden_size=hidden_dim,
            num_hidden_layers=kwargs['n_layer'],
            num_attention_heads=kwargs['n_head'],
            intermediate_size=kwargs['n_inner'],
            hidden_act=kwargs['activation_function'],
            hidden_dropout_prob=kwargs['resid_pdrop'],
            attention_probs_dropout_prob=kwargs['attn_pdrop'],
            max_position_embeddings=kwargs['n_positions']
        )
        self.encoder = BertModel(bert_config)

        self.avg_pool = nn.AvgPool1d(kernel_size=self.seq_length)

    def predict_opponent(self, embeddings):
        opp_obs_preds = self.predict_oppnt_states(embeddings)
        opp_act_preds = []
        for predict_oppnt_action in self.predict_oppnt_actions:
            opp_act_preds.append(predict_oppnt_action(embeddings).unsqueeze(2))
        opp_act_preds = torch.cat(opp_act_preds, dim=2)

        trajectory_embeddings = self.avg_pool(embeddings.permute((0, 2, 1))).squeeze(-1)

        opp_policy_preds = self.predict_oppnt_policy(trajectory_embeddings)

        return opp_obs_preds, opp_act_preds, opp_policy_preds


class GPTOpponentTransformer(OpponentTransformer):
    def __init__(
            self,
            hidden_dim,
            **kwargs
    ):
        super().__init__(hidden_dim=hidden_dim, **kwargs)

        gpt2_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_dim,
            **kwargs
        )
        self.encoder = GPT2Model(gpt2_config)


class LIAMVAE(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        oppnt_state_dim,
        oppnt_act_dims,
        hidden_dim,
        embedding_dim,
        num_layers: int = 1,
        use_recurrent_encoder: bool = True,
        model_rewards: bool = False
    ):
        super().__init__()
        if model_rewards:
            input_dim = state_dim + act_dim + 1
        else:
            input_dim = state_dim + act_dim

        if use_recurrent_encoder:
            self.encoder = VAELSTMEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        else:
            self.encoder = VAEEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        self.decoder = VAEDecoder(embedding_dim, hidden_dim, oppnt_state_dim, oppnt_act_dims)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

    def forward(
        self,
        states,
        actions,
        rnn_state=None,
        rewards=None,
        **kwargs
    ):
        if rewards is not None:
            input_tensor = torch.cat((states, actions, rewards), dim=-1)
        else:
            input_tensor = torch.cat((states, actions), dim=-1)

        if rnn_state is None:
            embedding = self.encoder(input_tensor)

            return embedding
        else:
            embedding, rnn_state = self.encoder(input_tensor, rnn_state)

            return embedding, rnn_state

    def predict_opponent(self, embeddings):
        oppnt_state_preds, opp_act1_pred, opp_act2_pred, opp_act3_pred = self.decoder(embeddings)

        return oppnt_state_preds, opp_act1_pred, opp_act2_pred, opp_act3_pred

    def save_params(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)


class VAELSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(VAELSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        h, hidden = self.lstm(x, hidden)
        h = F.relu(self.fc1(h))
        embedding = self.m_z(h)
        return embedding, hidden


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(VAEEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embedding = self.m_z(x)
        return embedding


class VAEDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, oppnt_state_dim, oppnt_act_dims):
        super(VAEDecoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.predict_oppnt_states = nn.Linear(hidden_dim, oppnt_state_dim)

        self.predict_oppnt_actions = []
        for oppnt_act_dim in oppnt_act_dims:
            self.predict_oppnt_actions.append(nn.Sequential(torch.nn.Linear(hidden_dim, oppnt_act_dim), nn.Softmax(dim=-1)))
        self.predict_oppnt_actions = nn.ModuleList(self.predict_oppnt_actions)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        oppnt_state_preds = self.predict_oppnt_states(h)

        opp_act1_pred = F.softmax(self.predict_oppnt_actions[0](h), dim=-1)
        opp_act2_pred = F.softmax(self.predict_oppnt_actions[1](h), dim=-1)
        opp_act3_pred = F.softmax(self.predict_oppnt_actions[2](h), dim=-1)

        return oppnt_state_preds, opp_act1_pred, opp_act2_pred, opp_act3_pred



class SpreadLIAMVAE(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        oppnt_state_dim,
        oppnt_act_dims,
        hidden_dim,
        embedding_dim,
        num_layers: int = 1,
        use_recurrent_encoder: bool = True,
        model_rewards: bool = False
    ):
        super().__init__()
        if model_rewards:
            input_dim = state_dim + act_dim + 1
        else:
            input_dim = state_dim + act_dim

        if use_recurrent_encoder:
            self.encoder = SpreadVAELSTMEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        else:
            self.encoder = SpreadVAEEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        self.decoder = SpreadVAEDecoder(embedding_dim, hidden_dim, oppnt_state_dim, oppnt_act_dims)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

    def forward(
        self,
        states,
        actions,
        rnn_state=None,
        rewards=None,
        **kwargs
    ):
        if rewards is not None:
            input_tensor = torch.cat((states, actions, rewards), dim=-1)
        else:
            input_tensor = torch.cat((states, actions), dim=-1)

        if rnn_state is None:
            embedding = self.encoder(input_tensor)

            return embedding
        else:
            embedding, rnn_state = self.encoder(input_tensor, rnn_state)

            return embedding, rnn_state

    def predict_opponent(self, embeddings):
        oppnt_state_preds, opp_act1_pred, opp_act2_pred = self.decoder(embeddings)

        return oppnt_state_preds, opp_act1_pred, opp_act2_pred

    def save_params(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)


class SpreadVAELSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SpreadVAELSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        h, hidden = self.lstm(x, hidden)
        h = F.relu(self.fc1(h))
        embedding = self.m_z(h)
        return embedding, hidden


class SpreadVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SpreadVAEEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embedding = self.m_z(x)
        return embedding


class SpreadVAEDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, oppnt_state_dim, oppnt_act_dims):
        super(SpreadVAEDecoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.predict_oppnt_states = nn.Linear(hidden_dim, oppnt_state_dim)

        self.predict_oppnt_actions = []
        for oppnt_act_dim in oppnt_act_dims:
            self.predict_oppnt_actions.append(nn.Sequential(torch.nn.Linear(hidden_dim, oppnt_act_dim), nn.Softmax(dim=-1)))
        self.predict_oppnt_actions = nn.ModuleList(self.predict_oppnt_actions)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        oppnt_state_preds = self.predict_oppnt_states(h)

        opp_act1_pred = F.softmax(self.predict_oppnt_actions[0](h), dim=-1)
        opp_act2_pred = F.softmax(self.predict_oppnt_actions[1](h), dim=-1)

        return oppnt_state_preds, opp_act1_pred, opp_act2_pred