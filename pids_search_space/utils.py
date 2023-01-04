def scale_and_make_divisible(c_num, multiplier=1.0, multiply_by=8):
    return round(c_num * multiplier / multiply_by) * multiply_by

def compound_scaling(block_args, k_multiplier, d_multiplier, c_multiplier):
    new_block_args = []
    for block in block_args:
        item_dict = {}
        for item in block.split("_"):
            key = item[0]
            if key == 'e':
                value = float(item[1:])
            else:
                value = int(item[1:])
            if key == 'k':
                value = scale_and_make_divisible(value, k_multiplier, 1)
            elif key == 'c':
                value = scale_and_make_divisible(value, c_multiplier, 4)
            elif key == 'r':
                value = scale_and_make_divisible(value, d_multiplier, 1)
            else:
                pass
            item_dict.update({key: value})
        new_block_arg = "k{}_e{}_s{}_r{}_c{}_d{}_a{}".format(
            item_dict['k'],
            item_dict['e'],
            item_dict['s'],
            item_dict['r'],
            item_dict['c'],
            item_dict['d'],
            item_dict['a'],)
        new_block_args.append(new_block_arg)
    return new_block_args
