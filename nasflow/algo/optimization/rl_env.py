import logging
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn

from tianshou.data import to_torch
from tianshou.env.worker import EnvWorker

from nasflow.algo.optimization.decoder import BaseDecoder

_logger = logging.getLogger(__name__)

class RLEnv(gym.Env):
    def __init__(self, eval_fn, encoding_space):
        #self.base_encoding = base_encoding
        self.eval_fn = eval_fn
        self.encoding_space = encoding_space
        self.action_dim = max(self.encoding_space)
        self.num_steps = len(self.encoding_space)
        # In this implementation, the action_space is a vector.
        # In the orginal repo, the input is search_space {dict}.
        # We translate search_space into action_space in main_search.py
        # We also simplify "mutator". 
        # The mutator function in the original repo is for compute_graph construction.
        #self.mutators = mutators
        #self.search_space = search_space
        #self.ss_keys = list(self.search_space.keys())
        #self.action_dim = max(map(lambda v: len(v), self.search_space.values()))

    @property
    def observation_space(self):
        return spaces.Dict({
            'action_history': spaces.MultiDiscrete([self.action_dim] * self.num_steps),
            'cur_step': spaces.Discrete(self.num_steps + 1),
            'action_dim': spaces.Discrete(self.action_dim + 1)
        })

    @property
    def action_space(self):
        return spaces.Discrete(self.action_dim)

    def reset(self):
        self.action_history = np.zeros(self.num_steps, dtype=np.int32)
        self.cur_step = 0
        self.candidate = []
        return {
            'action_history': self.action_history,
            'cur_step': self.cur_step,
            #'action_dim': len(self.search_space[self.ss_key[self.cur_step]]
            'action_dim': self.encoding_space[self.cur_step]
        }
    def step(self, action):
        # action is 0/1 in one-hot coding at one position
        #cur_key = self.ss_keys[self.cur_step]
        assert action >= 0
        assert action < self.encoding_space[self.cur_step], \
            f'Current action {action} out of range {self.encoding_space[cur_step]}.'
        self.action_history[self.cur_step] = action
        self.candidate.append(action)
        self.cur_step += 1
        obs = {
            'action_history': self.action_history,
            'cur_step': self.cur_step,
            'action_dim': self.encoding_space[self.cur_step] \
                if self.cur_step < self.num_steps else self.action_dim
        }
        if self.cur_step == self.num_steps:
            rew = self.eval_fn([self.candidate])
            print('New model created:', self.candidate)
            print('Objectives:', rew)
            #_logger.info(f'New model created: {self.candidate}')
            return obs, rew, True, {}
        else:
            return obs, 0., False, {}
    

        
'''


Original step function
def step(self, action):
        cur_key = self.ss_keys[self.cur_step]
        assert action < len(self.search_space[cur_key]), \
            f'Current action {action} out of range {self.search_space[cur_key]}.'
        self.action_history[self.cur_step] = action
        self.sample[cur_key] = self.search_space[cur_key][action]
        self.cur_step += 1
        obs = {
            'action_history': self.action_history,
            'cur_step': self.cur_step,
            'action_dim': len(self.search_space[self.ss_keys[self.cur_step]]) \
                if self.cur_step < self.num_steps else self.action_dim
        }
        if self.cur_step == self.num_steps:
            rew = self.eval_fn(action, *self.args)
            #rew = eval(apply(action, model))
            return obs, rew, True, {}
        else:
            return obs, 0., False, {}


class MultiThreadEnvWorker(EnvWorker):
    def __init__(self, env_fn):
        self.env = env_fn()
        self.pool = ThreadPool(processes=1)
        super().__init__(env_fn)

    def __getattr__(self, key):
        return getattr(self.env, key)

    def reset(self):
        return self.env.reset()

    @staticmethod
    def wait(*args, **kwargs):
        raise NotImplementedError('Async collect is not supported yet.')

    def send_action(self, action) -> None:
        # self.result is actually a handle
        self.result = self.pool.apply_async(self.env.step, (action,))

    def get_result(self):
        return self.result.get()

    def seed(self, seed):
        super().seed(seed)
        return self.env.seed(seed)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.pool.terminate()
        return self.env.close()
'''