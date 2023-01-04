from pids_core.models.blocks import (
    UnaryBlock,
    SimpleBlock,
    ResnetBottleneckBlock,
    MaxPoolBlock,
    GlobalAverageBlock,
    NearestUpsampleBlock,
    PointOperator,
)

def block_decider(block_name, radius, in_dim, out_dim, layer_ind, config, **kwargs):
    if block_name.startswith('unary'):
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum, config.batch_norm_epsilon)

    elif block_name.startswith("simple"):
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config, k=kwargs['k'])

    elif block_name.startswith('resnetb'):
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config, k=kwargs['k'])

    elif block_name.startswith("pids"):
        # Should decode settings for mobilenetv2-like blocks
        return PointOperator(
            block_name,
            in_dim,
            out_dim,
            radius,
            layer_ind,
            config,
            k=kwargs['k'],
            expand=kwargs['e'],
            use_attn=kwargs['use_attn']
            )
    elif block_name in ['max_pool', 'max_pool_wide']:
        return MaxPoolBlock(layer_ind)
    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)
