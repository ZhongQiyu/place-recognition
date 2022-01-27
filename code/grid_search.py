import os
import joblib
import shutil
import pprint
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from constants import Constants


class GridSearcher():

    def __init__(self, pipeline):
        """
        # add parameters for grid search so that the classifier
        # would be automatically configured
        """
        # check robustness
        param_grid = [
            {
                'hogify__orientations': [8, 9],
                'hogify__cells_per_block': [(2, 2), (3, 3)],
                'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)]
            },
            {
                'hogify__orientations': [8],
                'hogify__cells_per_block': [(3, 3)],
                'hogify__pixels_per_cell': [(8, 8)],
                'classify': [
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
                    svm.SVC(kernel='linear')
                ]
            }
        ]
        self.constants = Constants()
        self.searcher = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1,
                                     return_train_score=True)

    def get_constants(self):
        """
        :return: the constants that are used for compiling the results.
        """
        return self.constants

    def get_searcher(self):
        """
        :return: the grid searcher for the parameters.
        """
        return self.searcher

    def fit(self, X_train, y_train):
        """
        Fit a grid searcher with a group of training data,
        and return the fitted grid searcher.
        :param X_train: the signals from the training data.
        :param y_train: the outputs from the training data.
        :return: the fitted grid searcher based on the data.
        """
        return self.get_searcher().fit(X_train, y_train)

    def fit_dump(self, X_train, y_train):
        """
        Fit the grid searcher so that the parameters
        for the model training would be optimized.
        :param X_train: the signals from the training data.
        :param y_train: the outputs from the training data.
        """
        searcher = self.fit(X_train, y_train)
        constants = self.get_constants()
        grid_res_results = 'hog_sgd_model.pkl'
        if os.path.exists(os.path.join(constants.REPO, grid_res_results)):
            print("The pickle package has already exists.")
        else:
            joblib.dump(searcher, grid_res_results)
            shutil.move(grid_res_results, constants.REPO)

    def pred_val(self, X_test, y_test):
        """
        Make predictions based on the grid searcher, and
        validate the results of the predictions, in terms
        of confusion matrices. Return the accuracy of the
        predictions as well as the gotten confusion matrices.
        """
        searcher = self.get_searcher()
        # description of the best performing object, a pipeline in our case.
        print("The best estimator has the following settings:")
        print(searcher.best_estimator_)
        # the highscore during the search
        print(f"Out best search score is {searcher.best_score_}.")
        # show the best parameters
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(searcher.best_params_)
        # predict based on the trained SVM classifier
        best_pred = searcher.predict(X_test)
        accuracy = 100*np.sum(best_pred == y_test)/len(y_test)
        print(f"Percentage correct: {accuracy}")
        # compute the confusion matrices
        cmx = confusion_matrix(y_test, best_pred)
        return accuracy, cmx
