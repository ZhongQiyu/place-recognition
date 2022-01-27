import skimage
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

"""
A Python class that inherits the base classes from scikit-learn.
The description of the class is mainly focused on the PCA descriptor,
calling the base classes such as BaseEstimator and TransformerMixin.

"""


class PCADescriptor():

    class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
        """
        Convert an array of RGB images to grayscale
        """

        def __init__(self):
            """

            """
            pass

        def fit(self, X, y=None):
            """returns itself"""
            return self

        def transform(self, X, y=None):
            """perform the transformation and return an array"""
            return np.array([skimage.color.rgb2gray(img) for img in X])

    # pass in the PCA params
    class PCATransformer(BaseEstimator, TransformerMixin):
        """
        Expects an array of 2d arrays (1 channel images)
        Calculates hog features for each img
        """

        def __init__(self, y=None, orientations=9,
                     pixels_per_cell=(8, 8),
                     cells_per_block=(3, 3), block_norm='L2-Hys'):
            """

            """
            self.y = y
            self.orientations = orientations
            self.pixels_per_cell = pixels_per_cell
            self.cells_per_block = cells_per_block
            self.block_norm = block_norm

        def fit(self, X, y=None):
            """

            """
            return self

        def transform(self, X, y=None):
            """

            """
            def local_hog(X):
                output = PCA(X, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                             iterated_power='auto', random_state=None)
                return output
            try:  # parallel
                return np.array([local_hog(img) for img in X])
            except:
                return np.array([local_hog(img) for img in X])

    class PCAPipeline():
        """

        """

        def __init__(self, classifier, classifier_params):
            """
            Initialize the pipeline of dealing with HOG-based processing.
            """
            self.pipeline = Pipeline([('grayify', PCADescriptor.RGB2GrayTransformer()),
                                      ('hogify', PCADescriptor.PCATransformer(pixels_per_cell=(14, 14),
                                       cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')),
                                      ('scalify', StandardScaler()), ('classify', classifier(classifier_params))])

        def get_pipeline(self):
            """
            :return: the pipeline contained within the Python class, for the purpose of predictions.
            """
            return self.pipeline

        def set_pipeline(self, **kwargs):
            """
            Set the parameters towards the pipeline.
            :param kwargs: the parameters to pass to the pipeline.
            """
            self.pipeline.set_params(**kwargs)

        def fit(self, X_train, X_test, y_train, y_test):
            """
            Fit the models using the whole pipeline.
            :param X_train: the training set of the signals.
            :param X_test: the testing set of the signals.
            :param y_train: the training set of the outputs.
            :param y_test: the testing set of the outputs.
            """
            clf = self.get_pipeline().fit(X_train, y_train)
            accuracy = 100 * np.sum(clf.predict(X_test) == y_test) / len(y_test)
            print(f"Percentage correct: {accuracy:.4f}")
            return accuracy
