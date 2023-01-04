import torch

def get_unary_block_params(block_arg):
    items = block_arg.split("_")
    return_dict = {}
    for item in items:
        if item[0] == 'i':
            return_dict.update({"i": int(item[1:])})
        elif item[0] == 'o':
            return_dict.update({"o": int(item[1:])})
        else:
            pass
    return return_dict

def get_resnet_block_params(block_arg):
    items = block_arg.split("_")
    return_dict = {}
    for item in items:
        if item[0] =='k':
            return_dict.update({'k': int(item[1:])})
        elif item[0] == 'i':
            return_dict.update({"i": int(item[1:])})
        elif item[0] == 'o':
            return_dict.update({"o": int(item[1:])})
        else:
            pass
    return return_dict

def get_simple_block_params(block_arg):
    items = block_arg.split("_")
    return_dict = {}
    for item in items:
        if item[0] =='k':
            return_dict.update({'k': int(item[1:])})
        elif item[0] == 'i':
            return_dict.update({"i": int(item[1:])})
        elif item[0] == 'o':
            return_dict.update({"o": int(item[1:])})
        else:
            pass
    return return_dict


def get_PIDS_params(block_arg):
    items = block_arg.split("_")
    return_dict = {}
    for item in items:
        if item[0] == 'e':
            return_dict.update({'e': float(item[1:])})
        elif item[0] =='k':
            return_dict.update({'k': int(item[1:])})
        elif item[0] == 'i':
            return_dict.update({"i": int(item[1:])})
        elif item[0] == 'o':
            return_dict.update({"o": int(item[1:])})
        elif item[0] == 'a':
            return_dict.update({'a': int(item[1:])})
        else:
            pass
    return return_dict

def drop_connect_impl(inputs, p, training):
    """Drop connect.
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training or p == 0:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output
