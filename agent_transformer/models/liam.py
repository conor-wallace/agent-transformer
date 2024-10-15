from torch.distributions import OneHotCategorical
import torch
import torch.nn as nn
import torch.nn.functional as F

from opponent_transformer.models.distributions import Categorical


class DDQN(nn.Module):
    def __init__(
        self,
        state_size,
        embedding_size,
        move_act_dim,
        comm_act_dim,
        hidden_size
    ):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act1 = nn.Linear(hidden_size, move_act_dim)
        self.act2 = nn.Linear(hidden_size, comm_act_dim)

    def forward(self, input):
        """
        
        """
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        act1 = self.act1(x)
        act2 = self.act2(x)

        return act1, act2


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, hidden = self.lstm(x, hidden)
        h = F.relu(self.fc1(h))
        embedding = self.m_z(h)
        return embedding, hidden


class Decoder(nn.Module):
    def __init__(self, input_dim1, hidden_dim, output_dim1, output_dim2, output_dim3):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim1)
        self.fc4 = nn.Linear(hidden_dim, output_dim2)
        self.fc5 = nn.Linear(hidden_dim, output_dim3)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        probs1 = F.softmax(self.fc4(h), dim=-1)
        probs2 = F.softmax(self.fc5(h), dim=-1)

        return out, probs1, probs2


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, norm_in=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # create network layers
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(self.in_fn(x)))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out


class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, norm_in=True):
        super(RNN, self).__init__()

        # mlp base
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)) 
        self.fc_h = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.fc2 = nn.ModuleList([copy.deepcopy(self.fc_h)])

        # rnn layer
        self.rnn = nn.GRU(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # categorical action layer
        self.pi = Categorical(hidden_dim, output_dim)

    def forward(self, x, h):
        x = self.fc1(x)
        x = self.fc2(x)
        x, h = self.rnn(x, h)
        x = self.norm(x)
        act = self.pi(x)
        return act


class CQL(nn.Module):
    def __init__(
        self,
        state_dim,
        move_act_dim,
        comm_act_dim,
        hidden_size,
    ):
        super().__init__()

        self.network = DDQN(
            state_size=state_dim,
            embedding_size=0,
            move_act_dim=move_act_dim,
            comm_act_dim=comm_act_dim,
            hidden_size=hidden_size
        )

        self.target_net = DDQN(
            state_size=state_dim,
            embedding_size=0,
            move_act_dim=move_act_dim,
            comm_act_dim=comm_act_dim,
            hidden_size=hidden_size
        )

    def get_action(self, obs):
        act1, act2 = self.network(obs)

        act1 = act1.argmax(dim=-1)
        act2 = act2.argmax(dim=-1)

        return act1, act2


class CBOM(nn.Module):
    def __init__(
        self,
        state_dim,
        move_act_dim,
        comm_act_dim,
        hidden_dim,
        num_opponents,
    ):
        super(CBOM, self).__init__()

        self.network = DDQN(
            state_size=state_dim,
            embedding_size=num_opponents,
            move_act_dim=move_act_dim,
            comm_act_dim=comm_act_dim,
            hidden_size=hidden_dim
        )

        self.target_net = DDQN(
            state_size=state_dim,
            embedding_size=num_opponents,
            move_act_dim=move_act_dim,
            comm_act_dim=comm_act_dim,
            hidden_size=hidden_dim
        )

        input_dim = state_dim + move_act_dim + comm_act_dim
        self.opponent_model = nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, num_opponents),
            nn.Softmax(dim=-1)
        )
        # self.opponent_model = nn.Sequential(
        #     torch.nn.Linear(input_dim, num_opponents),
        #     nn.Softmax(dim=-1)
        # )

        self.num_opponents = num_opponents

    def get_action(self, state, prev_action, opponent_id=None):
        if opponent_id is None:
            oppnt_pred = self.predict_opponent(state, prev_action)
            opponent_id = F.one_hot(oppnt_pred.argmax(-1), self.num_opponents).detach()
        else:
            opponent_id = F.one_hot(opponent_id, self.num_opponents)

        input_tensor = torch.cat((state, opponent_id), dim=-1)
        move_action, comm_action = self.network(input_tensor)

        move_action = move_action.argmax(dim=-1)
        comm_action = comm_action.argmax(dim=-1)

        return move_action, comm_action, opponent_id

    def predict_opponent(self, state, prev_action):
        input_tensor = torch.cat((state, prev_action), dim=-1)
        oppnt_preds = self.opponent_model(input_tensor)
        return oppnt_preds


class LIAM(nn.Module):
    def __init__(
        self,
        state_dim,
        move_act_dim,
        comm_act_dim,
        hidden_size,
        embedding_size,
    ):
        super().__init__()

        self.network = DDQN(
            state_size=state_dim,
            embedding_size=embedding_size,
            move_act_dim=move_act_dim,
            comm_act_dim=comm_act_dim,
            hidden_size=hidden_size
        )

        self.target_net = DDQN(
            state_size=state_dim,
            embedding_size=embedding_size,
            move_act_dim=move_act_dim,
            comm_act_dim=comm_act_dim,
            hidden_size=hidden_size
        )   
    
        self.encoder = Encoder(state_dim + move_act_dim + comm_act_dim, hidden_size, embedding_size)
        self.decoder = Decoder(embedding_size, hidden_size, state_dim, move_act_dim, comm_act_dim)
        self.embedding_size = embedding_size

    def compute_embedding(self, obs, action, hidden):
        input_tensor = torch.cat((obs, action), dim=-1)
        embedding, hidden = self.encoder(input_tensor, hidden)
        return embedding, hidden

    def get_action(self, obs, action, hidden):
        embedding, hidden = self.compute_embedding(obs, action, hidden)
        input_tensor = torch.cat((obs.unsqueeze(0), embedding), dim=-1)
        act1, act2 = self.network(input_tensor)
        _, probs1, probs2 = self.predict_opponent(embedding)

        act1 = act1.argmax(dim=-1)
        act2 = act2.argmax(dim=-1)
        oppnt_act1 = probs1.argmax(dim=-1)
        oppnt_act2 = probs2.argmax(dim=-1)

        return act1, act2, oppnt_act1, oppnt_act2, hidden

    def predict_opponent(self, embeddings):
        out1, probs1, probs2 = self.decoder(embeddings)

        return out1, probs1, probs2

    def eval_decoding(self, embeddings, modelled_obs, modelled_act):
        out1, probs1, probs2 = self.predict_opponent(embeddings)

        # print("Move acc preds: ", probs1.reshape(-1, 5).argmax(dim=-1))
        # print("Comm acc preds: ", probs2.reshape(-1, 10).argmax(dim=-1))
        # print("Move acc targets: ", modelled_act[:, :, :5].reshape(-1, 5).argmax(dim=-1))
        # print("Comm acc targets: ", modelled_act[:, :, 5:].reshape(-1, 10).argmax(dim=-1))

        # recon_loss1 = 0.5 * ((out1 - modelled_obs) ** 2).sum(-1)
        # recon_loss2 = -torch.log(torch.sum(modelled_act[:, :, :5] * probs1, dim=-1))
        # recon_loss2 -=torch.log(torch.sum(modelled_act[:, :, 5:] * probs2, dim=-1))

        recon_loss1 = F.mse_loss(out1.reshape(-1, 21), modelled_obs.reshape(-1, 21))
        recon_loss2_move = F.cross_entropy(probs1.reshape(-1, 5), modelled_act[:, :, :5].reshape(-1, 5).argmax(dim=-1))
        recon_loss2_comm = F.cross_entropy(probs2.reshape(-1, 10), modelled_act[:, :, 5:].reshape(-1, 10).argmax(dim=-1))
        recon_loss2 = recon_loss2_move + recon_loss2_comm

        move_acc = torch.sum(probs1.reshape(-1, 5).argmax(dim=-1) == modelled_act[:, :, :5].reshape(-1, 5).argmax(dim=-1)).detach().cpu().item() / (probs1.shape[0] * probs1.shape[1])
        comm_acc = torch.sum(probs2.reshape(-1, 10).argmax(dim=-1) == modelled_act[:, :, 5:].reshape(-1, 10).argmax(dim=-1)).detach().cpu().item() / (probs2.shape[0] * probs2.shape[1])

        return recon_loss1, recon_loss2, move_acc, comm_acc

