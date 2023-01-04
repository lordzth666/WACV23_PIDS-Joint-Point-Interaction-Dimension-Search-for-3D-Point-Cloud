from typing import (
    List,
    Any)

import copy

from nasflow.search_space.search_space import BaseSearchSpace

class BaseCombSearchSpace:
    """Combinatorial Search Space that combines each search space to make a large one.
    This instance is the KPCNN search space.
    """

    def __init__(self, search_spaces: List[Any], **kwargs):
        self.kwargs = kwargs
        # Do a sanity check of the search space.
        for idx, space in enumerate(search_spaces):
            assert isinstance(space, BaseSearchSpace), \
                TypeError("Search Space instance {} should be 'SearchSpace' \
                    for NAS Flow to execute!".format(idx))
        self.search_spaces = copy.deepcopy(search_spaces)
        self.num_search_spaces = len(self.search_spaces)

    def __len__(self):
        return self.num_search_spaces

    def __getitem__(self, key: int):
        return self.search_spaces[key]

    def __seed__(self, **kwargs):
        for i in range(self.num_search_spaces):
            self.search_spaces[i].__seed__(**kwargs)

    def encode(self, **kwargs):
        """Note: Encode one architecture.
        """
        raise BaseException(
            "You must override the 'encode() function if your search space inherits \
            '{}'.".format(type(self)))

    def decode(self, **kwargs):
        """Note: Decode one architecture from a given encoding.
        """
        raise BaseException(
            "You must override the 'decode() function if your search space inherits \
            '{}'.".format(type(self)))
