import numpy as np
import random

from nasflow.io_utils.base_parser import parse_args_from_kwargs
from nasflow.algo.optimization.optimizer import BaseOptimizer
from nasflow.algo.optimization.sampler import BaseSampler
from nasflow.algo.optimization.decoder import BaseDecoder
from nasflow.algo.optimization.rl_env import RLEnv
import logging

try:
    has_tianshou = True
    import torch
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import BaseVectorEnv
except ImportError:
    has_tianshou = False

_logger = logging.getLogger(__name__)

class PolicyBasedRLSearch(BaseOptimizer):
    def __init__(
            self,
            eval_fn,
            sampler:BaseSampler,
            decoder:BaseDecoder,
            search_space,
            env,
            policy_fn,
            *args,
            **kwargs):
        super(PolicyBasedRLSearch, self).__init__(eval_fn, sampler, decoder, *args, **kwargs)
        
        if not has_tianshou:
            raise ImportError('`tianshou` is required to run RL-based strategy. '
                              'Please use "pip install tianshou" to install it beforehand.')

        # For concise purpose, we remove the mutator interface.
        # For example, current_model [0,1,2,1,1,1], action [0,1,0,0,0,0], mutator = add(),
        #              we can next_model = mutator.apply(base_model, action)
        self.policy = policy_fn
        self.env = env
        self.search_space = search_space
        #self.base_encoding = parse_args_from_kwargs(kwargs, "base_encoding", self.sample()['encoding'])         
        self.buffer_size = parse_args_from_kwargs(kwargs, "buffer_size", 20000)         
        self.encode_type = parse_args_from_kwargs(kwargs, "encode_type", 'ordinal') #'one-hot'

    def minimize(self, **kwargs):
        return_history = parse_args_from_kwargs(kwargs, 'return_history', False)

        max_collect = parse_args_from_kwargs(kwargs, 'max_collect', 2)
        trial_per_collect = parse_args_from_kwargs(kwargs, 'trial_per_collect', 10)
        update_batch_size = parse_args_from_kwargs(kwargs, 'update_batch_size', 64)
        update_repeat = parse_args_from_kwargs(kwargs, 'update_repeat', 5)

        idx = 0
        collector = self._create_collector()
        sample_rews = []
        sample_encodings = []
        for cur_collect in range(1, max_collect + 1):
            print('Collect Running...', cur_collect)
            #_logger.info('Collect [%d] Running...', cur_collect)
            result = collector.collect(n_episode=trial_per_collect)
            for i in range(len(result['rews'])):
                sample_rews.append(result['rews'][i])
                sample_encodings.append(collector.buffer.act[result['idxs'][i] : result['idxs'][i] + result['lens'][i]])
                print(sample_rews[-1])
                print(sample_encodings[-1])
            print('Collect Result:', cur_collect, str(result))
            # results #
            #    * ``n/ep`` collected number of episodes.
            #    * ``n/st`` collected number of steps.
            #    * ``rews`` array of episode reward over collected episodes.
            #    * ``lens`` array of episode length over collected episodes.
            #    * ``idxs`` array of episode start index in buffer over collected episodes.
            print('Policy Updating...')
            #_logger.info('Collect [%d] Result: %s', cur_collect, str(result))
            self.policy.update(0, collector.buffer, batch_size=update_batch_size, repeat=update_repeat)
            #print(len(collector.buffer))
            #print(min(collector.buffer.rew))
            # The following info is stored into a batch at each step t
            # All the batches are stored into buffer
            # Details: https://tianshou.readthedocs.io/en/master/tutorials/concepts.html
            # obs: the observation of step t
            # act: the action of step t
            # rew: the reward of step t
            # done: the done flag of step t
            # obs_next: the next observation of step t+1
            # info: the info of step t (the output in RLEnv.step())
            # policy: the data computed by policy at step t
            
            # collect logic:
            # while(True):
            #   act = get_next_action()
            #   rew, done = env.step(act)
            #   store act, rew, done in buffer
            #   if done:
            #       i+=1
            #       calculate statistics
            #       env.reset()
            #   i >= n_episode ? break

        sample_history = {'best_encodings': sample_encodings,
                               'best_rewards': sample_rews}
        return self.get_best_sample(sample_rews, sample_encodings, sample_history, return_history)
    def get_best_sample(self, sample_rews, sample_encodings, sample_history, return_history):
        
        best_sample_idx = np.argmax(sample_rews)
        if not return_history:
            return sample_encodings[best_sample_idx], sample_rews[best_sample_idx][0]
        else:
            return sample_encodings[best_sample_idx], sample_rews[best_sample_idx][0], sample_history

        #self.search_space.decode(str_encoding = best_sample_encoding)
        #encode_list = self.gene_choice.encode(style=self.gene_encode_type)
        #block_args = self.gene_choice.str_encode()
        # Need to generate block_arg here
        #return {'block_args': block_args, 'encoding': encode_list}


    def _create_collector(self):
        return Collector(self.policy, self.env, 
                         VectorReplayBuffer(self.buffer_size,1))
        #return Collector(self.policy, self.env, 
        #                 VectorReplayBuffer(self.buffer_size, len(self.env)))
    
    def _test(self):
        # Not implemented
        pass

