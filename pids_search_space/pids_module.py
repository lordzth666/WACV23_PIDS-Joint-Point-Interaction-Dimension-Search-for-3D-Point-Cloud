import copy
from collections import OrderedDict
from math import log
import numpy as np

from nasflow.search_space.search_space import BaseSearchSpace
from nasflow.io_utils.base_parser import parse_args_from_kwargs
from nasflow.algo.encoding_utils import encode_fn

all_positional_dense_features = [
    'c', 'r'
]
all_architectural_dense_features = [
    'e'
]
all_sparse_features = [
    'k', 'a'
]

dense_transform = lambda x: log(1 + x)

def dense_sparse_encode_pids(state_dict, avail_dict):
    positional_dense_enc = []
    architectural_dense_enc = []
    sparse_enc = []
    for key in state_dict.keys():
        if key in all_positional_dense_features:
            positional_dense_enc.append(dense_transform(state_dict[key]))
        elif key in all_architectural_dense_features:
            architectural_dense_enc.append(dense_transform(state_dict[key]))
        elif key in all_sparse_features:
            sparse_enc += encode_fn['ordinal'](state_dict[key], avail_dict[key])
        else:
            pass
    return [positional_dense_enc, architectural_dense_enc, sparse_enc]

def dense_encode_pids(state_dict):
    dense_enc = []
    for key in state_dict.keys():
        dense_enc.append(dense_transform(state_dict[key]))
    return dense_enc

class PIDS_Search_Space(BaseSearchSpace):
    """[summary]
    'PIDS_Backbone_Cls' class that inherits the base class.
    """
    def __init__(self,
                 **kwargs):
        """
        Args:
        **kwargs: Optional Arguments.
        Important Kwargs:
        avail_dict (Dict[Any])): Dictionary that holds the search space.
        """
        super().__init__(**kwargs)
        avail_dict = parse_args_from_kwargs(kwargs, "avail_dict")
        self._avail_dict = copy.deepcopy(avail_dict)
        self.state_dict = OrderedDict()
        for key in self._avail_dict.keys():
            self.state_dict[key] = -1
        self.__seed__()

    def __seed__(self, **kwargs):
        for key in self._avail_dict.keys():
            self.state_dict[key] = np.random.choice(self._avail_dict[key])

    def str_encode(self):
        item_list = []
        for key in self.state_dict.keys():
            if key != 'e':
                item_list.append("%s%d" % (key, self.state_dict[key]))
            else:
                item_list.append("%s%.2f" % (key, self.state_dict[key]))
        return "_".join(item_list)

    def encode(self, **kwargs):
        style = parse_args_from_kwargs(kwargs, "style", "one-hot")
        if style == 'dense-sparse-encoding':
            return dense_sparse_encode_pids(self.state_dict, self._avail_dict)
        elif style == "dense-encoding":
            return dense_encode_pids(self.state_dict)
        encoding = []
        for key in sorted(self.state_dict.keys()):
            code_ = encode_fn[style](self.state_dict[key], self._avail_dict[key])
            encoding.extend(code_)
        return encoding

    def decode(self, **kwargs):
        decode_method = parse_args_from_kwargs(kwargs, "decoding_method", "decode_str_encoding")
        if decode_method == "decode_str_encoding":
            str_encoding = parse_args_from_kwargs(kwargs, "str_encoding", )
            assert str_encoding is not None, "You must pass an 'encoding' to materize this \
                'decode()' function."
        elif decode_method == 'decode_ordinal_encoding':
            self.decode_ordinal_encoding(str_encoding)
        else:
            raise NotImplementedError("Decoding method {} not implemented!".format(decode_method))

    def decode_ordinal_encoding(self, ordinal_encoding, **kwargs):
        assert len(ordinal_encoding) == len(self.state_dict.keys()), ValueError(
            "Oridnal must have same length as search space keys.")
        for idx, key in enumerate(sorted(self.state_dict.keys())):
            self.state_dict[key] = self._avail_dict[key][ordinal_encoding[idx]]

    def size(self):
        size_ = 1
        for key in self._avail_dict.keys():
            size_ *= len(self._avail_dict[key])
        return size_

    def generate_argscope(self):
        argscope = []
        for key in self.state_dict.keys():
            argscope.append(self.state_dict[key])
        return argscope

    def decode_str_encoding(self, str_encoding: str):
        def _split_var_values(target_str):
            num_idx = 0
            while not (target_str[num_idx] >= '0' and target_str[num_idx] <= '9'
                       or target_str[num_idx] == '-'):
                num_idx += 1
            return target_str[:num_idx], float(target_str[num_idx:])

        items_ = str_encoding.split("_")
        for item in items_:
            key, value = _split_var_values(item)
            if key != 'e':
                value = int(value)
            self.state_dict[key] = value
