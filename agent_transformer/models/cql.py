class DQN(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_dim,
        embedding_dim: int = 0
    ):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Linear(hidden_dim, act_dim)

    def forward(self, x, embedding = None):
        """
        """
        if embedding is not None:
            x = torch.cat((x, embedding), dim=-1)
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        act = self.act(x)

        return act


class CQL(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_dim,
        embedding_dim = 0
    ):
        super().__init__()

        self.network = DQN(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim
        )

        self.target_net = DQN(
            state_size=state_dim,
            embedding_size=0,
            move_act_dim=move_act_dim,
            comm_act_dim=comm_act_dim,
            hidden_size=hidden_size
        )