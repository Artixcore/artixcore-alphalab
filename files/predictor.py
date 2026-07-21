class Predictor:
    """Minimal base class used by the reconstructed local AlphaLab runner.

    AlphaNova submissions subclass this class and implement ``train`` and
    ``predict``. The official platform may provide its own equivalent class.
    """

    def train(self, features, target):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError
