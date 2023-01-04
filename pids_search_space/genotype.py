from collections import OrderedDict

# Intepretation of keys:
# k (Kernel size): Kernel size of the point interaction in the current stage.
# e (Expansion factor): Expansion factor of IRB in the current stage.
# s (Stride): Stride of the current series of blocks for the current stage.
# r (Repeats): Number of repeats of blocks for the current stage.
# c (Channel): Channel number of the current stage.
# d (Deformable): Deformable Conv or not. (Disabled for all layers)
# a (Attention): Attention option for the current stage.

# Segmentation Benchmark search spaces.
def PIDS_seg_search_space_cfgs():
    search_cfg1 = OrderedDict({
        'k': [5, 7, 13],
        'e': [1],
        's': [1],
        'r': [1],
        'c': [16],
        'd': [0],
        'a': [0],
    })
    search_cfg2 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [2],
        'r': [2, 3],
        'c': [16, 24],
        'd': [0],
        'a': [0],
    })
    search_cfg3 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [2],
        'r': [2, 3, 4],
        'c': [24, 32],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg4 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [2],
        'r': [3, 4, 5],
        'c': [24, 32, 40],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg5 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [1],
        'r': [2, 3, 4],
        'c': [40, 56, 72],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg6 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [2],
        'r': [3, 4, 5],
        'c': [64, 80, 96],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg7 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [1],
        'r': [1],
        'c': [160],
        'd': [0],
        'a': [0, 1],
    })
    # 4 KPConv blocks, which replaces aformentioned Unary blocks in the paper.
    search_cfg8 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3],
        's': [1],
        'r': [1, 2],
        'c': [64, 80, 96],
        'd': [0],
        'a': [0, 1],
    })

    search_cfg9 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3],
        's': [1],
        'r': [1, 2],
        'c': [40, 56, 72],
        'd': [0],
        'a': [0, 1],
    })

    search_cfg10 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3],
        's': [1],
        'r': [1, 2],
        'c': [24, 32, 40],
        'd': [0],
        'a': [0, 1],
    })

    search_cfg11 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3],
        's': [1],
        'r': [1, 2],
        'c': [16, 24],
        'd': [0],
        'a': [0],
    })

    search_cfg_list_cnn = [search_cfg1,
                           search_cfg2,
                           search_cfg3,
                           search_cfg4,
                           search_cfg5,
                           search_cfg6,
                           search_cfg7,]
    search_cfg_list_fcn = [search_cfg8,
                           search_cfg9,
                           search_cfg10,
                           search_cfg11,]

    return search_cfg_list_cnn, search_cfg_list_fcn


# Classificatiion Benchmark search spaces (e.g., ModelNet40)
def PIDS_cls_search_space_cfgs():
    search_cfg1 = OrderedDict({
        'k': [5, 7, 13],
        'e': [1],
        's': [1],
        'r': [1],
        'c': [16],
        'd': [0],
        'a': [0],
    })
    search_cfg2 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [2],
        'r': [2, 3],
        'c': [16, 24],
        'd': [0],
        'a': [0],
    })
    search_cfg3 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [2],
        'r': [2, 3, 4],
        'c': [24, 32],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg4 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [2],
        'r': [3, 4, 5],
        'c': [24, 32, 40],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg5 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [1],
        'r': [2, 3, 4],
        'c': [40, 56, 72],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg6 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [2],
        'r': [3, 4, 5],
        'c': [64, 80, 96],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg7 = OrderedDict({
        'k': [5, 7, 13],
        'e': [2, 3, 4],
        's': [1],
        'r': [1],
        'c': [160],
        'd': [0],
        'a': [0, 1],
    })
    search_cfg_list = [search_cfg1,
                       search_cfg2,
                       search_cfg3,
                       search_cfg4,
                       search_cfg5,
                       search_cfg6,
                       search_cfg7]
    return search_cfg_list