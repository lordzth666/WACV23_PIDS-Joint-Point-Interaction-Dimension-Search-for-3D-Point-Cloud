from typing import (
    List,
    Tuple,
    Any
)
import numpy as np
# Import two base search spaces from NASFlow.
from nasflow.search_space.comb_search_space import BaseCombSearchSpace
from nasflow.io_utils.base_parser import parse_args_from_kwargs
from pids_search_space.pids_module import PIDS_Search_Space

class PIDS_Space_Cls(BaseCombSearchSpace):
    """Combinatorial Search Space that combines each search space to make a large one.
        Encode the architecture in the PIDS search space for Classification.
    """
    def str_encode(self):
        encoding = []
        for i in range(self.num_search_spaces):
            encoding.append(self.search_spaces[i].str_encode())
        return encoding

    def size(self):
        size_ = 1
        for i in range(self.num_search_spaces):
            size_ *= self.search_spaces[i].size()
        return size_

    def generate_argscope(self):
        argscope = []
        for i in range(self.num_search_spaces):
            argscope += [self.search_spaces[i].generate_argscope()]
        return argscope

    @staticmethod
    def generate_arch_specs_from_argscope(argscope,
                                          num_stem_k: int = 7,
                                          num_stem_inp: int = 1,
                                          num_stem_oup: int = 16,
                                          insert_global_avg_pool=True):
        archs = ["simple_k{}_i{}_o{}".format(num_stem_k, num_stem_inp, num_stem_oup)]
        inp = num_stem_oup
        oup = -1
        for arg in argscope:
            k, e, s, r, c, d, a = arg
            for j in range(int(r)):
                if d == 0:
                    arch_str = "pids_"
                else:
                    arch_str = "pids_deformable_"       # This inherits the KPConv deformable option.
                if s != 1 and j == 0:
                    arch_str += 'strided_'
                oup = c
                arch_str += "e{}_k{}_i{}_o{}_a{}".format(e, k, inp, oup, a)
                inp = oup
                archs.append(arch_str)
        # Finally, append global average pooling.
        if insert_global_avg_pool:
            archs.append("global_average")
        return archs

    def __seed__(self, **kwargs):
        for i in range(self.num_search_spaces):
            self.search_spaces[i].__seed__(**kwargs)

    def __seed_one__(self, idx, **kwargs):
        self.search_spaces[idx].__seed__(**kwargs)

    def encode(self, **kwargs):
        style = parse_args_from_kwargs(kwargs, "style", "one-hot")
        encoding = []
        for i in range(self.num_search_spaces):
            code_ = self.search_spaces[i].encode(style=style)
            encoding.extend(code_)
        return encoding

    def decode(self, **kwargs):
        decode_method = parse_args_from_kwargs(kwargs, "decoding_method", "decode_str_encoding")
        if decode_method == "decode_str_encoding":
            str_encoding = parse_args_from_kwargs(kwargs, "str_encoding", )
            assert str_encoding is not None, "You must pass an 'encoding' to materize this \
                'decode()' function."
            self.decode_str_encoding(str_encoding)
        else:
            raise NotImplementedError("Decoding method {} not implemented!".format(decode_method))

    def decode_str_encoding(self, str_encoding: List[str]):
        for i in range(self.num_search_spaces):
            self.search_spaces[i].decode_str_encoding(str_encoding[i])

class PIDS_Space_Seg(BaseCombSearchSpace):
    """
     A 'CombSearchSpace' combines each search space to make a large one. \
         This instance is the KPFCN search space (Segmentation).
    """
    def __init__(self, search_spaces, **kwargs):
        """
        Args:
            avail_lists ([type]): [description]
            num_cnn_search_cfg ([type]): [description]
            num_fcn_search_cfg ([type]): [description]
        """
        super().__init__(search_spaces, **kwargs)
        self.num_cnn_search_spaces = parse_args_from_kwargs(kwargs, "num_cnn_search_spaces")
        self.num_fcn_search_spaces = parse_args_from_kwargs(kwargs, "num_fcn_search_spaces")

    def str_encode(self):
        encoding = []
        for i in range(self.num_search_spaces):
            encoding.append(self.search_spaces[i].str_encode())
        return encoding

    def size(self):
        size_ = 1
        for i in range(self.num_search_spaces):
            size_ *= self.search_spaces[i].size()
        return size_

    def generate_argscope(self):
        argscope = []
        for i in range(self.num_search_spaces):
            argscope += [self.search_spaces[i].generate_argscope()]
        return argscope

    def __seed__(self, **kwargs):
        for i in range(self.num_search_spaces):
            self.search_spaces[i].__seed__(**kwargs)

    def __seed_one__(self, idx, **kwargs):
        self.search_spaces[idx].__seed__(**kwargs)

    def encode(self, **kwargs):
        """
        Encode the architecture in the PIDS search space for Segmentation.
        Args:
            style: The way to perform encoding to represent the architecture.
            aggregate: Whether to aggregate the encoding for both CNN and FCN part.
        Returns:
        """
        style = parse_args_from_kwargs(kwargs, "style", "one-hot")
        aggregate = parse_args_from_kwargs(kwargs, 'aggregate', False)
        if aggregate:
            encoding = []
            for i in range(self.num_search_spaces):
                code_ = self.search_spaces[i].encode(style=style)
                encoding.extend(code_)
            return encoding
        # Do encoding for both KPCNN and KPFCN.
        cnn_encoding = []
        fcn_encoding = []
        for i in range(self.num_cnn_search_spaces):
            code_ = self.search_spaces[i].encode(style=style)
            cnn_encoding.extend(code_)
        for i in range(self.num_fcn_search_spaces):
            code_ = self.search_spaces[i+self.num_cnn_search_spaces].encode(style=style)
            fcn_encoding.extend(code_)
        return cnn_encoding, fcn_encoding

    def decode(self, **kwargs):
        decode_method = parse_args_from_kwargs(kwargs, "decoding_method", "decode_str_encoding")
        if decode_method == "decode_str_encoding":
            str_encoding = parse_args_from_kwargs(kwargs, "str_encoding", )
            assert str_encoding is not None, "You must pass an 'encoding' to materize this \
                'decode()' function."
            self.decode_str_encoding(str_encoding)
        else:
            raise NotImplementedError("Decoding method {} not implemented!".format(decode_method))

    def decode_str_encoding(self, str_encoding: List[str]):
        for i in range(self.num_search_spaces):
            self.search_spaces[i].decode_str_encoding(str_encoding[i])

    def generate_arch_specs_from_argscope(self,
                                          argscope,
                                          num_stem_k: int = 7,
                                          num_stem_inp: int = 1,
                                          num_stem_oup: int = 16,
                                          fcn_pool_position: Tuple[int] = (0, 1, 2, 3)):
        archs = ["simple_k{}_i{}_o{}".format(num_stem_k, num_stem_inp, num_stem_oup)]
        inp = num_stem_oup
        oup = -1
        num_layers = 0
        num_inp_fcn = []

        # Next, encode for RNN search space.
        for arg in argscope[:self.num_cnn_search_spaces]:
            k, e, s, r, c, d, a = arg
            if s != 1:
                num_layers += 1
                num_inp_fcn.append(inp)
            for j in range(int(r)):
                if d == 0:
                    arch_str = "pids_"
                else:
                    arch_str = "pids_deformable_"
                if s != 1 and j == 0:
                    arch_str += 'strided_'
                oup = c
                arch_str += "e{}_k{}_i{}_o{}_a{}".format(e, k, inp, oup, a)
                inp = oup
                archs.append(arch_str)

        num_inp_fcn.append(oup)
        # Now, add FCN layers
        for arg in argscope[self.num_cnn_search_spaces:]:
            archs.append("nearest_upsample")
            num_layers -= 1
            k, e, s, r, c, d, a = arg
            if num_layers in fcn_pool_position:
                inp += num_inp_fcn[num_layers]
            for j in range(r):
                arch_str = "pids_"
                oup = c
                arch_str += "e{}_k{}_i{}_o{}_a{}".format(e, k, inp, oup, a)
                inp = oup
                archs.append(arch_str)
        return archs

def build_pids_comb_search_space_from_genotype(search_space_cfgs):
    search_spaces = [PIDS_Search_Space(avail_dict=cfg) for cfg in search_space_cfgs]
    return search_spaces
