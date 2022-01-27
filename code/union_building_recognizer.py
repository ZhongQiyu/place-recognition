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
        self.classes = []
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

    def get_classes(self):
        """
        Return the classes to predict.
        :return: the classes to predict by the recognizer.
        """
        return self.classes

    def set_classes(self, new_classes):
        """
        Set a series of new classes to predict,
        with a given collection of classes.
        :param new_classes: the new classes to set for predictions.
        """
        self.classes = new_classes

    def confmat_val(self, conf_mat):
        """
        Validate the results of the confusion matrices,
        so that a collection of buildings would be reasonable
        enough to predict with.
        """

    def log_reg(self):
        """

        """

    def lasso_reg(self):
        """

        """

    def ridge_reg(self):
        """

        """

    def hog_svm(self):
        """
        Recognize the buildings with the help of histograms of gradients
        (HOG) descriptor and the support vector machine (SVM) model.
        """

    def bovw_svm(self):
        """

        """

    def decision_tree(self):
        """

        """

    def random_forest(self):
        """

        """

    def netvlad(self):
        """

        """

    def patch_netvlad(self):
        """

        """
