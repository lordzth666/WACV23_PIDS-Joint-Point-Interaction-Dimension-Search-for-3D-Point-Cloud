import logging
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn

from tianshou.data import to_torch
from tianshou.env.worker import EnvWorker
from tianshou.policy import BasePolicy, PPOPolicy
from nasflow.algo.optimization.rl_env import RLEnv

_logger = logging.getLogger(__name__)

class Preprocessor(nn.Module):
    def __init__(self, obs_space, hidden_dim=64, num_layers=1):
        super().__init__()
        self.action_dim = obs_space['action_history'].nvec[0]
        self.hidden_dim = hidden_dim
        # first token is [SOS]
        self.embedding = nn.Embedding(self.action_dim + 1, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, obs):
        #print('action_history', obs['action_history'])
        seq = nn.functional.pad(obs['action_history'] + 1, (1, 1))  # pad the start token and end token
        # end token is used to avoid out-of-range of v_s_. Will not actually affect BP.
        seq = self.embedding(seq.long())
        feature, _ = self.rnn(seq)
        return feature[torch.arange(len(feature), device=feature.device), obs['cur_step'].long() + 1]


class Actor(nn.Module):
    def __init__(self, action_space, preprocess):
        super().__init__()
        self.preprocess = preprocess
        self.action_dim = action_space.n
        self.linear = nn.Linear(self.preprocess.hidden_dim, self.action_dim)

    def forward(self, obs, **kwargs):
        obs = to_torch(obs, device=self.linear.weight.device)
        out = self.linear(self.preprocess(obs))
        # to take care of choices with different number of options
        mask = torch.arange(self.action_dim).expand(len(out), self.action_dim) >= obs['action_dim'].unsqueeze(1)
        out[mask.to(out.device)] = float('-inf')
        oput = nn.functional.softmax(out, dim=-1)
        #print('actor output', oput)
        return oput, kwargs.get('state', None)

class Critic(nn.Module):
    def __init__(self, preprocess):
        super().__init__()
        self.preprocess = preprocess
        self.linear = nn.Linear(self.preprocess.hidden_dim, 1)

    def forward(self, obs, **kwargs):
        obs = to_torch(obs, device=self.linear.weight.device)
        return self.linear(self.preprocess(obs)).squeeze(-1)

def default_policy_fn(env:RLEnv):
        net = Preprocessor(env.observation_space)
        # If all the elements in the encoding is operatable, then action_space = search_space
        actor = Actor(env.action_space, net)
        critic = Critic(net)
        optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=1e-4)
        return PPOPolicy(actor, critic, optim, torch.distributions.Categorical,
                         discount_factor=1., action_space=env.action_space)