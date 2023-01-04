import numpy as np

from scipy.stats import kendalltau
from scipy.stats import pearsonr

def get_pearson_coefficient(X, y):
    # Get covariance first.
    # cov = np.mean((X-np.mean(X))*(y-np.mean(y)))
    # return cov / (np.std(X) * np.std(y))
    r, _ = pearsonr(X, y)
    return r

def get_kendalltau_coefficient(X, y):
    def get_rank(array):
        temp = array.argsort()
        ranks = np.arange(len(array))[temp.argsort()]
        return ranks
    r, _ = kendalltau(get_rank(X), get_rank(y))
    return r
