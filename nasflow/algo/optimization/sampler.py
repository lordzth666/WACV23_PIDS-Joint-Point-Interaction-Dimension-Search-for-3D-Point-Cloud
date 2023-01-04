class BaseSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def sample(self, **kwargs):
        raise BaseException("Base sampler case. Not implemented!")
