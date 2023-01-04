import numpy as np

def get_onehot_encoding(value, all_values):
    m_ = np.shape(all_values)[0]
    vec = np.zeros(m_, dtype=np.float32)
    vec[np.arange(m_, dtype=np.int)[value == np.asarray(all_values)]] = 1
    return vec.tolist()

def get_ordinal_encoding(value, all_values):
    m_ = np.shape(all_values)[0]
    return [int(np.arange(m_)[value == np.asarray(all_values)])]

def get_real_encoding(value, _):
    return [np.log(min(1, value + 1))]

encode_fn = {
    'one-hot': get_onehot_encoding,
    'ordinal': get_ordinal_encoding,
    'real': get_real_encoding,
}
