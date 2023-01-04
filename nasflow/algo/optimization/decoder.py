class BaseDecoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def decode(self, x, **kwargs):
        raise BaseException("Base sampler case. Not implemented!")
