import os
import math
import torch
import numpy as np
import torch.optim as optim

from models.ac_mlp import ContinuousMLP, DiscreteMLP
from models.ac_cnn import ContinuousCNN, DiscreteCNN
from config import PPOConfig

args = PPOConfig()


class PPO:
    def __init__(self, action_dim, continuous_action=False, cnn=False):
        self.continuous_action = continuous_action
        if continuous_action:
            if cnn:
                self.model = ContinuousCNN(args.in_channels, args.feature_dim, args.hidden_dim, action_dim)
            else:
                self.model = ContinuousMLP(args.observation_dim, args.hidden_dim, action_dim)
        else:
            if cnn:
                self.model = DiscreteCNN(args.in_channels, args.feature_dim, args.hidden_dim, action_dim)
            else:
                self.model = DiscreteMLP(args.observation_dim, args.hidden_dim, action_dim)

        if torch.cuda.is_available():
            self.model.cuda()

        self.optim = optim.Adam(self.model.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)

        self.state_memory = torch.Tensor([]).cuda()
        self.action_memory = torch.Tensor([]).cuda()
        self.reward_memory = torch.Tensor([]).cuda()
        self.done_memory = torch.Tensor([]).cuda()

    def clear_memory(self):
        if torch.cuda.is_available():
            self.state_memory = torch.Tensor([]).cuda()
            self.action_memory = torch.Tensor([]).cuda()
            self.reward_memory = torch.Tensor([]).cuda()
            self.done_memory = torch.Tensor([]).cuda()
        else:
            self.state_memory = torch.Tensor([])
            self.action_memory = torch.Tensor([])
            self.reward_memory = torch.Tensor([])
            self.done_memory = torch.Tensor([])

    def update_memory(self, state, action, reward, done):
        self.state_memory = torch.cat((self.state_memory, state), dim=0)
        self.action_memory = torch.cat((self.action_memory, action), dim=0)
        self.reward_memory = torch.cat((self.reward_memory, reward), dim=0)
        self.done_memory = torch.cat((self.done_memory, done), dim=0)

    def select_action(self, state):
        if self.continuous_action:
            mu, std = self.model.act(state)
            action = torch.normal(mu, std).detach()
        else:
            dist = self.model.act(state)
            action = dist.sample()
        return action

    def set_mode(self, mode=True):
        self.model.train(mode)

    def log_density(self, x, mu, std, log_std):
        """
        求正态分布的概率密度函数的log，把正态分布的概率密度函数外面套一个log然后化简就是本函数
        :param x: action
        :param mu: \mu
        :param std: \sigma
        :param log_std: log\sigma
        :return: the probability of action x happened in Normal(mu, std)
        """
        var = std.pow(2)
        log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std

        return log_density.sum(1, keepdim=True)

    def reward_scaling(self):
        self.reward_memory = (self.reward_memory - self.reward_memory.min())/(self.reward_memory.max() - self.reward_memory.min())
    
    def learn(self):
        self.reward_scaling()
        if self.continuous_action:
            self.learn_continuous()
        else:
            self.learn_discrete()

    def learn_discrete(self):
        old_dist, value = self.model(self.state_memory)
        # ----------------------------
        # step 1: get returns and GAEs and log probability of old policy
        returns, advants = self.get_gae(self.reward_memory, self.done_memory, value)
        old_policy = old_dist.log_prob(self.action_memory).detach()
        criterion = torch.nn.MSELoss()
        n = self.done_memory.shape[0]
        arr = np.arange(n)
        # ----------------------------
        # step 2: get value loss and actor loss and update actor & critic
        for epoch in range(args.k_epoch):
            np.random.shuffle(arr)
            for i in range(n // args.batch_size):
                batch_index = arr[args.batch_size * i: args.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                states_samples = self.state_memory[batch_index]
                returns_samples = returns[batch_index]
                advants_samples = advants.unsqueeze(1)[batch_index]
                actions_samples = self.action_memory[batch_index]
                old_policy_samples = old_policy[batch_index]

                sur_loss, ratio, values, dist = self.surrogate_loss_discrete(advants_samples, states_samples,
                                                                             old_policy_samples, actions_samples)
                # critic loss
                critic_loss = criterion(values, returns_samples)
                # actor loss
                clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)
                clipped_loss = clipped_ratio * advants_samples
                actor_loss = -torch.min(sur_loss, clipped_loss).mean()
                # entropy bonus
                entropy_bonus = torch.mean(dist.entropy())

                loss = actor_loss + args.c1 * critic_loss + args.c2 * entropy_bonus

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        self.clear_memory()


    def learn_continuous(self):
        mu, std, log_std, value = self.model(self.state_memory)
        # ----------------------------
        # step 1: get returns and GAEs and log probability of old policy
        returns, advants = self.get_gae(self.reward_memory, self.done_memory, value)
        old_policy = self.log_density(self.action_memory, mu, std, log_std).detach()
        criterion = torch.nn.MSELoss()
        n = self.done_memory.shape[0]
        arr = np.arange(n)
        # ----------------------------
        # step 2: get value loss and actor loss and update actor & critic
        for epoch in range(args.k_epoch):
            np.random.shuffle(arr)
            for i in range(n // args.batch_size):
                batch_index = arr[args.batch_size * i: args.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                states_samples = self.state_memory[batch_index]
                returns_samples = returns[batch_index]
                advants_samples = advants.unsqueeze(1)[batch_index]
                actions_samples = self.action_memory[batch_index]
                old_policy_samples = old_policy[batch_index]

                sur_loss, ratio, values = self.surrogate_loss_continuous(advants_samples, states_samples, old_policy_samples,
                                                              actions_samples)

                critic_loss = criterion(values, returns_samples)

                clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)
                clipped_loss = clipped_ratio * advants_samples
                actor_loss = -torch.min(sur_loss, clipped_loss).mean()

                loss = actor_loss + args.c1 * critic_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        self.clear_memory()

    def get_gae(self, rewards, masks, values):
        returns = torch.zeros_like(rewards).cuda()
        advants = torch.zeros_like(rewards).cuda()

        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + args.gamma * running_returns * masks[t]
            running_tderror = rewards[t] + args.gamma * previous_value * masks[t] - values.data[t]
            running_advants = running_tderror + args.gamma * args.lamda * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def surrogate_loss_continuous(self, advants, states, old_policy, actions):
        mu, std, log_std, values = self.model(states)
        new_policy = self.log_density(actions, mu, std, log_std)

        ratio = torch.exp(new_policy - old_policy)
        surrogate = ratio * advants
        return surrogate, ratio, values

    def surrogate_loss_discrete(self, advants, states, old_policy, actions):
        dist, values = self.model(states)
        new_policy = dist.log_prob(actions)

        ratio = torch.exp(new_policy - old_policy)
        surrogate = ratio * advants
        return surrogate, ratio, values, dist

    def save_model(self, epoch, save_path_base, agent_type):
        save_path = os.path.join(save_path_base, "epoch_" + str(epoch) + "_" + agent_type + ".pth")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), save_path)

