# from recognizer import UnionBuildingRecognizer

class UnionBuildingRecognizer():
    """
    A Python class that simulates a recognizer towards buildings
    across Union College's main campus, based in Schenectady, NY.
    The recognizer is expected to take an input of the data, and
    it includes up to 3 (types) of models to work on, in order to
    process the amount all over the place. pass in any amount of hyperparameters to train Python class
    """

    def __init__(self, data, **kwargs):
        """
        Initialize the recognizer.
        :param data: the data to train the recognizer.
        """
        self.data = data
        # unpack kwargs

    def get_data(self):
        """
        Return whatever the recognizer has so far.
        """
        return self.data

    def set_data(self, data):
        """
        Set the recognizer with some new training data.
        :param data: the new training data to set with.
        """
        self.data = data

    def mlr(self):
        """

        """

    def decision_tree(self):
        """

        """

    def svm(self):
        """

        """

    def random_forest(self):
        """

        """

    def pca(self):
        """

        """

    def netvlad(self):
        """

        """

    def g_cnn(self):
        """

        """

    def places_cnn(self):
        """

        """
