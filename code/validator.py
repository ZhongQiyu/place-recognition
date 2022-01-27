import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
A Python class that models a data validator for the image input
involved through the pipeline.
"""


class Validator():

    def __init__(self):
        """
        Initialize a validator for the data to train model with.
        """
        self.data = None
        self.method = ""

    def plot_confusion_matrix(self, cmx, vmax1=None, vmax2=None, vmax3=None):
        """
        Plot three types of confusion matrices, the as is,
        the percentage-based, and the zerofied version of
        the previous one. The data of confusion matrix and
        ranges of plotting the three matrices are given.
        :param cmx: the data of the confusion matrix.
        :param vmax1: the range of the as is confusion matrix.
        :param vmax2: the range of the percentage-based confusion matrix.
        :param vmax3: the range of the percentage-based and zerofied confusion matrix.
        """
        # normalize the confusion data
        cmx_norm = 100 * cmx / cmx.sum(axis=1, keepdims=True)
        cmx_zero_diag = cmx_norm.copy()
        # align the inputs with zero fillers
        np.fill_diagonal(cmx_zero_diag, 0)
        # set the dimensions of axes for plotting
        fig, ax = plt.subplots(ncols=3)
        fig.set_size_inches(12, 3)
        [a.set_xticks(range(len(cmx) + 1)) for a in ax]
        [a.set_yticks(range(len(cmx) + 1)) for a in ax]
        # set the titles of the axes for plotting
        im1 = ax[0].imshow(cmx, vmax=vmax1)
        ax[0].set_title('as is')
        im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
        ax[1].set_title('%')
        im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
        ax[2].set_title('% and 0 diagonal')
        # set the dividers for the axes
        dividers = [make_axes_locatable(a) for a in ax]
        cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) for divider in dividers]
        # get the layout for the colored plots
        fig.colorbar(im1, cax=cax1)
        fig.colorbar(im2, cax=cax2)
        fig.colorbar(im3, cax=cax3)
        fig.tight_layout()


def main():
    """
    The driver function of the class.
    """
    y_test = None
    # the types appear in this order
    print(sorted(np.unique(y_test)))


if __name__ == "__main__":
    main()
