from nasflow.algo.optimization.decoder import BaseDecoder

class PIDSVanillaDecoder(BaseDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def decode(self, x, **kwargs):
        return x['encoding']

class PIDSDenseSparseDecoder(BaseDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def decode(self, x, **kwargs):
        encode_list = x['encoding']
        assert len(encode_list) % 3 == 0, \
            ValueError("Encode list after 'dense-sparse-encoding' must be divisible by 3!")
        dense_pos_enc, dense_arch_enc, sparse_attn_enc = [], [], []
        for idx in range(len(encode_list) // 3):
            dense_pos_enc.extend(encode_list[idx * 3])
            dense_arch_enc.extend(encode_list[idx * 3 + 1])
            sparse_attn_enc.extend(encode_list[idx * 3 + 2])
        full_dense_sparse_enc = dense_pos_enc + dense_arch_enc + sparse_attn_enc
        return full_dense_sparse_enc
