import numpy as np

from nasflow.algo.optimization.sampler import BaseSampler
from nasflow.io_utils.base_parser import parse_args_from_kwargs

from pids_search_space.pids_architect import (
    PIDS_Space_Cls,
    PIDS_Space_Seg,
)

class PIDS_Cls_Sampler(BaseSampler):
    def __init__(
            self,
            pids_cls_space: PIDS_Space_Cls,
            **kwargs):
        super().__init__(**kwargs)
        self.search_space = pids_cls_space
        self.encode_style = parse_args_from_kwargs(self.kwargs, 'encode_style', 'ordinal')

    def sample(self, **kwargs):
        self.search_space.__seed__()
        block_args = self.search_space.str_encode()
        encode_list = self.search_space.encode(style=self.encode_style)
        return {
            'block_args': block_args,
            'encoding': encode_list
        }

class PIDS_Cls_MutationSampler(PIDS_Cls_Sampler):
    def __init__(
            self,
            pids_cls_space: PIDS_Space_Cls,
            **kwargs):
        super().__init__(pids_cls_space, **kwargs)

    def sample(self, **kwargs):
        cur_block_args = parse_args_from_kwargs(kwargs, "block_args", None)
        assert cur_block_args is not None, \
            ValueError("You should not specify an empty 'block_args' for mutation.")
        self.search_space.decode(str_encoding=cur_block_args)
        idx = np.random.choice(len(self.search_space))
        self.search_space.__seed_one__(idx)
        encode_list = self.search_space.encode(style=self.encode_style)
        block_args = self.search_space.str_encode()
        return {
            'block_args': block_args,
            'encoding': encode_list
        }

class PIDS_Seg_Sampler(BaseSampler):
    def __init__(
            self,
            pids_seg_space: PIDS_Space_Seg,
            **kwargs):
        super().__init__(**kwargs)
        self.search_space = pids_seg_space
        self.encode_style = parse_args_from_kwargs(self.kwargs, 'encode_style', 'ordinal')
        self.aggregate = parse_args_from_kwargs(self.kwargs, "aggregate", True)

    def sample(self, **kwargs):
        self.search_space.__seed__()
        block_args = self.search_space.str_encode()
        encode_list = self.search_space.encode(style=self.encode_style, aggregate=self.aggregate)
        return {
            'block_args': block_args,
            'encoding': encode_list
        }

class PIDS_Seg_MutationSampler(PIDS_Seg_Sampler):
    def __init__(
            self,
            pids_seg_space: PIDS_Space_Seg,
            **kwargs):
        super().__init__(pids_seg_space, **kwargs)

    def sample(self, **kwargs):
        cur_block_args = parse_args_from_kwargs(kwargs, "block_args", None)
        assert cur_block_args is not None, \
            ValueError("You should not specify an empty 'block_args' for mutation.")
        self.search_space.decode(str_encoding=cur_block_args)
        idx = np.random.choice(len(self.search_space))
        self.search_space.__seed_one__(idx)
        encode_list = self.search_space.encode(style=self.encode_style, aggregate=self.aggregate)
        block_args = self.search_space.str_encode()
        return {
            'block_args': block_args,
            'encoding': encode_list
        }
