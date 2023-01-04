from nasflow.io_utils.base_io import maybe_load_json_file

class Hparams:
    def __init__(self, json_file):
        data = maybe_load_json_file(json_file)
        self.learning_rate = data['learning_rate']
        self.weight_decay = data['weight_decay']
        self.ranking_loss_coef = data['ranking_loss_coef']
        self.margin = data['margin']

    def load_from_json_file(self, json_file):
        self.__init__(json_file)
