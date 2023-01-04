class BaseSearchSpace:
    """    'BaseSearchSpace' class that defines the search space for one certain stage. """

    def __init__(self,
                 **kwargs):
        """
        Args:
            avail_dict (Dict[Any])): Dictionary that holds the search space.
        """
        self._kwargs = kwargs

    def __seed__(self, **kwargs):
        """
        This function is needed to randomly grab candidates from the search space.
        """
        raise BaseException(
            "You must override the '__seed__() function if your search space inherits \
            '{}'.".format(type(self)))

    def encode(self, **kwargs):
        """
        This function is needed to generate encodings for a given architecture in the search
        space.
        """
        raise BaseException(
            "You must override the 'encode() function if your search space inherits \
            '{}'.".format(type(self)))

    def decode(self, **kwargs):
        """
        This function is needed to decode the encodings and get an architecture in the search
        space.
        """
        raise BaseException(
            "You must override the 'decode() function if your search space inherits \
            '{}'.".format(type(self)))
