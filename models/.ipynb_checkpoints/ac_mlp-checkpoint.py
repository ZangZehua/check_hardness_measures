import torch
import torch.nn as nn
from torch.distributions import Categorical


class ContinuousMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ContinuousMLP, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        mu = self.actor(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)
        v = self.critic(x)
        return mu, std, log_std, v

    def act(self, x):
        mu = self.actor(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)
        return mu, std


class DiscreteMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiscreteMLP, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        v = self.critic(x)
        return dist, v

    def act(self, x):
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        return dist


if __name__ == "__main__":
    Discrete(128, 128, 2)
    print(Discrete)

